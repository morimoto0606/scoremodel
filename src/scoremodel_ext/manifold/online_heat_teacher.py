"""Online Heat-teacher training for the independent Upstream-style path.

The existing fixed-dataset trainer remains untouched.  This module changes
only batch construction: selected observed points receive freshly sampled
physical times and GRW noises on every optimizer update.
"""

from __future__ import annotations

import copy
from typing import Callable, Dict

import torch
from torch.utils.data import DataLoader, TensorDataset

from scoremodel_ext.malliavin.models import (
    learning_rate_for_update,
    learning_rate_trace_indices,
    update_ema_model,
)

from .beta_schedule import LinearBetaSchedule
from .s2_malliavin import (
    s2_batched_grw_endpoint,
    s2_batched_heat_kernel_score,
)
from .upstream_style_score import build_upstream_style_score_model


Tensor = torch.Tensor


def build_online_heat_teacher_batch(
    initial_points: Tensor,
    *,
    beta_schedule: LinearBetaSchedule,
    minimum_time: float,
    maximum_time: float,
    n_steps: int,
    heat_terms: int,
    generator: torch.Generator,
) -> Dict[str, Tensor]:
    """Generate one fresh online DSM batch without changing the Heat target."""

    if initial_points.ndim != 2 or initial_points.shape[1] != 3:
        raise ValueError("initial_points must have shape [batch, 3]")
    if initial_points.shape[0] < 1:
        raise ValueError("initial_points must not be empty")
    if not 0.0 < minimum_time < maximum_time:
        raise ValueError("expected 0 < minimum_time < maximum_time")
    if n_steps < 1:
        raise ValueError("n_steps must be positive")

    batch_size = initial_points.shape[0]
    times = torch.empty(
        batch_size,
        dtype=initial_points.dtype,
        device=initial_points.device,
    ).uniform_(minimum_time, maximum_time, generator=generator)
    noises = torch.randn(
        batch_size,
        n_steps,
        3,
        dtype=initial_points.dtype,
        device=initial_points.device,
        generator=generator,
    )
    brownian_times = beta_schedule.rescale_t(times)
    with torch.no_grad():
        endpoints = s2_batched_grw_endpoint(
            initial_points,
            noises,
            brownian_times,
        )
        score_target = s2_batched_heat_kernel_score(
            initial_points,
            endpoints,
            brownian_times,
            n_terms=heat_terms,
        )
    return {
        "initial_point": initial_points.detach(),
        "time": times.detach(),
        "noise": noises.detach(),
        "endpoint": endpoints.detach(),
        "score_target": score_target.detach(),
    }


def train_s2_upstream_style_score_model_online_heat(
    initial_points: Tensor,
    normalization_dataset: Dict[str, Tensor],
    *,
    beta_schedule: LinearBetaSchedule,
    minimum_time: float,
    maximum_time: float,
    n_steps: int,
    heat_terms: int,
    online_teacher_seed: int,
    n_epochs: int = 1000,
    batch_size: int = 128,
    learning_rate: float = 2e-4,
    weight_decay: float = 0.0,
    hidden: int = 512,
    n_blocks: int = 6,
    num_frequencies: int = 16,
    device: str = "cuda",
    return_history: bool = False,
    training_unit: str = "updates",
    updates: int = 0,
    warmup_updates: int = 0,
    lr_scheduler: str = "constant",
    ema_rate: float = 0.0,
    checkpoint_every_updates: int = 0,
    checkpoint_callback: Callable[[Dict[str, object]], None] | None = None,
    return_training_state: bool = False,
):
    """Train with fresh ``(t, noise, endpoint, Heat score)`` every update.

    ``normalization_dataset`` is used only to preserve the fixed experiment's
    input normalization and to report losses on a stable reference set.  Its
    teacher triples are never selected for optimizer updates.
    """

    required = {"time", "endpoint", "score_target"}
    missing = required.difference(normalization_dataset)
    if missing:
        raise KeyError(f"normalization_dataset is missing fields: {sorted(missing)}")
    if training_unit != "updates":
        raise ValueError("online Heat training currently requires training_unit='updates'")
    if updates < 1 or n_epochs < 1 or batch_size < 1:
        raise ValueError("updates, n_epochs, and batch_size must be positive")
    if not 0.0 <= ema_rate < 1.0:
        raise ValueError("ema_rate must be in [0, 1)")
    if warmup_updates > updates:
        raise ValueError("warmup_updates must not exceed total optimizer updates")

    initial_points = initial_points.detach().to(device)
    reference_time = normalization_dataset["time"].detach().to(device)
    reference_endpoint = normalization_dataset["endpoint"].detach().to(device)
    reference_score = normalization_dataset["score_target"].detach().to(device)
    x_mean = reference_endpoint.mean(dim=0, keepdim=True)
    x_std = reference_endpoint.std(dim=0, keepdim=True).clamp_min(1e-6)
    time_column = reference_time[:, None]
    t_mean = time_column.mean(dim=0, keepdim=True)
    t_std = time_column.std(dim=0, keepdim=True).clamp_min(1e-6)

    model = build_upstream_style_score_model(
        x_mean=x_mean,
        x_std=x_std,
        t_mean=t_mean,
        t_std=t_std,
        hidden=hidden,
        n_blocks=n_blocks,
        num_frequencies=num_frequencies,
        beta_schedule=beta_schedule,
        device=device,
        dtype=reference_endpoint.dtype,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    ema_model = copy.deepcopy(model) if ema_rate > 0.0 else None
    if ema_model is not None:
        ema_model.requires_grad_(False)

    n_samples = initial_points.shape[0]
    index_loader = DataLoader(
        TensorDataset(torch.arange(n_samples, dtype=torch.long)),
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=0,
    )
    updates_per_epoch = len(index_loader)
    loader_iterator = iter(index_loader)
    teacher_generator = torch.Generator(device=device)
    teacher_generator.manual_seed(online_teacher_seed)
    trace_indices = set(
        learning_rate_trace_indices(updates, warmup_updates=warmup_updates)
    )
    sampled_updates: list[int] = []
    sampled_losses: list[float] = []
    learning_rate_trace: list[dict[str, float | int]] = []
    initial_learning_rate = None
    peak_learning_rate = 0.0
    final_learning_rate = None
    teacher_examples_generated = 0

    with torch.no_grad():
        initial_reference_loss = float(
            model.score_loss(reference_time, reference_endpoint, reference_score)
        )
    best_online_batch_loss = float("inf")
    normalization_state = {
        "x_mean": x_mean.detach().cpu().clone(),
        "x_std": x_std.detach().cpu().clone(),
        "t_mean": t_mean.detach().cpu().clone(),
        "t_std": t_std.detach().cpu().clone(),
    }

    for update_index in range(updates):
        current_lr = learning_rate_for_update(
            update_index,
            total_updates=updates,
            base_learning_rate=learning_rate,
            warmup_updates=warmup_updates,
            scheduler=lr_scheduler,
        )
        for group in optimizer.param_groups:
            group["lr"] = current_lr
        if initial_learning_rate is None:
            initial_learning_rate = current_lr
        peak_learning_rate = max(peak_learning_rate, current_lr)
        final_learning_rate = current_lr
        if update_index in trace_indices:
            learning_rate_trace.append(
                {"update": update_index, "learning_rate": current_lr}
            )

        try:
            (indices,) = next(loader_iterator)
        except StopIteration:
            loader_iterator = iter(index_loader)
            (indices,) = next(loader_iterator)
        indices = indices.to(device=device)
        online_batch = build_online_heat_teacher_batch(
            initial_points[indices],
            beta_schedule=beta_schedule,
            minimum_time=minimum_time,
            maximum_time=maximum_time,
            n_steps=n_steps,
            heat_terms=heat_terms,
            generator=teacher_generator,
        )
        teacher_examples_generated += int(indices.numel())

        loss = model.score_loss(
            online_batch["time"],
            online_batch["endpoint"],
            online_batch["score_target"],
        )
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if ema_model is not None:
            update_ema_model(ema_model, model, ema_rate)

        detached_loss = float(loss.detach())
        best_online_batch_loss = min(best_online_batch_loss, detached_loss)
        if update_index in trace_indices:
            sampled_updates.append(update_index + 1)
            sampled_losses.append(detached_loss)

        completed_updates = update_index + 1
        if (
            checkpoint_every_updates > 0
            and completed_updates % checkpoint_every_updates == 0
            and checkpoint_callback is not None
        ):
            checkpoint_callback(
                {
                    "complete_online_model_state_dict": {
                        key: value.detach().cpu().clone()
                        for key, value in model.state_dict().items()
                    },
                    "complete_ema_model_state_dict": (
                        {
                            key: value.detach().cpu().clone()
                            for key, value in ema_model.state_dict().items()
                        }
                        if ema_model is not None
                        else None
                    ),
                    "optimizer_state_dict": copy.deepcopy(optimizer.state_dict()),
                    "scheduler_state": {
                        "lr_scheduler": lr_scheduler,
                        "base_learning_rate": float(learning_rate),
                        "warmup_updates": warmup_updates,
                        "total_updates": updates,
                        "current_update": completed_updates,
                        "last_learning_rate": current_lr,
                    },
                    "current_update": completed_updates,
                    "current_epoch": completed_updates // updates_per_epoch,
                    "requested_total_updates": updates,
                    "warmup_updates": warmup_updates,
                    "lr_scheduler": lr_scheduler,
                    "ema_rate": ema_rate,
                    "normalization_state": normalization_state,
                    "teacher_sampling": "online",
                    "online_teacher_seed": online_teacher_seed,
                    "online_teacher_rng_state": teacher_generator.get_state().cpu(),
                    "teacher_examples_generated": teacher_examples_generated,
                }
            )

    with torch.no_grad():
        final_reference_loss = float(
            model.score_loss(reference_time, reference_endpoint, reference_score)
        )
    history = {
        "epochs": sampled_updates,
        "updates": sampled_updates,
        "train_loss": sampled_losses,
        "initial_train_loss": initial_reference_loss,
        "final_train_loss": final_reference_loss,
        "best_train_loss": best_online_batch_loss,
        "loss_definition": "mean(||sigma(t) * (s_pred - s_teacher)||_2^2)",
        "network_output": "sigma(t) * effective_score",
        "teacher_sampling": "online",
        "full_loss_dataset": "fixed_normalization_reference",
        "teacher_examples_generated": teacher_examples_generated,
    }
    training_state = {
        "ema_model": ema_model,
        "optimizer_state_dict": copy.deepcopy(optimizer.state_dict()),
        "scheduler_state": {
            "lr_scheduler": lr_scheduler,
            "base_learning_rate": float(learning_rate),
            "warmup_updates": warmup_updates,
            "total_updates": updates,
            "current_update": updates,
            "last_learning_rate": float(final_learning_rate),
        },
        "current_update": updates,
        "current_epoch": updates // updates_per_epoch,
        "requested_total_updates": updates,
        "actual_optimizer_updates": updates,
        "updates_per_epoch": updates_per_epoch,
        "effective_epochs": updates / updates_per_epoch,
        "initial_learning_rate": float(initial_learning_rate),
        "peak_learning_rate": float(peak_learning_rate),
        "final_learning_rate": float(final_learning_rate),
        "learning_rate_trace": learning_rate_trace,
        "normalization_state": normalization_state,
        "legacy_training_path": False,
        "score_parameterization": "upstream_scaled_score",
        "teacher_sampling": "online",
        "online_teacher_seed": online_teacher_seed,
        "online_teacher_rng_state": teacher_generator.get_state().cpu(),
        "teacher_examples_generated": teacher_examples_generated,
    }

    if not return_history and not return_training_state:
        return model
    if return_history and return_training_state:
        return model, history, training_state
    if return_history:
        return model, history
    return model, training_state
