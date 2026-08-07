"""Upstream-style scaled-score training for direct S2 score teachers.

This module is deliberately separate from the existing direct-score training
path.  Its network predicts ``sigma(t) * score`` while its public forward
method returns the effective score consumed by the existing reverse sampler.
"""

from __future__ import annotations

import copy
from typing import Callable, Dict

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from scoremodel_ext.malliavin.models import (
    MirafzaliSkorokhodNet,
    learning_rate_for_update,
    learning_rate_trace_indices,
    update_ema_model,
)

from .beta_schedule import LinearBetaSchedule


Tensor = torch.Tensor


def upstream_score_standard_deviation(
    time: Tensor,
    beta_schedule: LinearBetaSchedule,
) -> Tensor:
    r"""Return ``sigma(t) = sqrt(1-exp(-tau(t)))``."""

    tau = beta_schedule.rescale_t(time)
    return torch.sqrt(1.0 - torch.exp(-tau))


def upstream_style_score_loss(
    raw_output: Tensor,
    teacher_score: Tensor,
    time: Tensor,
    beta_schedule: LinearBetaSchedule,
) -> Tensor:
    r"""Return ``mean_batch ||raw - sigma(t) * teacher_score||_2^2``.

    Since ``effective_prediction = raw / sigma(t)``, this is exactly
    ``mean_batch ||sigma(t) * (s_pred - s_teacher)||_2^2``.
    """

    if raw_output.shape != teacher_score.shape or raw_output.ndim != 2:
        raise ValueError("raw_output and teacher_score must share [sample, dimension]")
    if time.ndim != 1 or time.shape[0] != raw_output.shape[0]:
        raise ValueError("time must have shape [sample]")
    sigma = upstream_score_standard_deviation(time, beta_schedule)
    residual = raw_output - sigma[:, None] * teacher_score
    return residual.square().sum(dim=1).mean()


class UpstreamStyleScoreModel(nn.Module):
    """Normalize model inputs, emit raw scaled score, expose effective score."""

    def __init__(
        self,
        net: nn.Module,
        x_mean: Tensor,
        x_std: Tensor,
        t_mean: Tensor,
        t_std: Tensor,
        *,
        beta_0: float = 0.001,
        beta_f: float = 5.0,
        beta_t0: float = 0.0,
        beta_tf: float = 1.0,
    ) -> None:
        super().__init__()
        schedule = LinearBetaSchedule(
            beta_0=beta_0,
            beta_f=beta_f,
            t0=beta_t0,
            tf=beta_tf,
        )
        self.net = net
        self.register_buffer("x_mean", x_mean)
        self.register_buffer("x_std", x_std)
        self.register_buffer("t_mean", t_mean)
        self.register_buffer("t_std", t_std)
        self.register_buffer(
            "beta_0", torch.tensor(schedule.beta_0, dtype=x_mean.dtype)
        )
        self.register_buffer(
            "beta_f", torch.tensor(schedule.beta_f, dtype=x_mean.dtype)
        )
        self.register_buffer("beta_t0", torch.tensor(schedule.t0, dtype=x_mean.dtype))
        self.register_buffer("beta_tf", torch.tensor(schedule.tf, dtype=x_mean.dtype))

    def score_standard_deviation(self, time: Tensor) -> Tensor:
        u = (time - self.beta_t0) / (self.beta_tf - self.beta_t0)
        tau = self.beta_0 * u + 0.5 * (self.beta_f - self.beta_0) * u.square()
        return torch.sqrt(1.0 - torch.exp(-tau))

    def raw_output(self, time: Tensor, x: Tensor) -> Tensor:
        time_column = time[:, None] if time.ndim == 1 else time
        normalized_time = (
            (time_column - self.t_mean) / self.t_std.clamp_min(1e-6)
        ).squeeze(-1)
        normalized_x = (x - self.x_mean) / self.x_std.clamp_min(1e-6)
        return self.net(normalized_time, normalized_x)

    def forward(self, time: Tensor, x: Tensor) -> Tensor:
        raw = self.raw_output(time, x)
        time_vector = time.squeeze(-1) if time.ndim != 1 else time
        sigma = self.score_standard_deviation(time_vector)
        return raw / sigma[:, None]

    def score_loss(self, time: Tensor, x: Tensor, teacher_score: Tensor) -> Tensor:
        raw = self.raw_output(time, x)
        time_vector = time.squeeze(-1) if time.ndim != 1 else time
        sigma = self.score_standard_deviation(time_vector)
        residual = raw - sigma[:, None] * teacher_score
        return residual.square().sum(dim=1).mean()


def build_upstream_style_score_model(
    *,
    x_mean: Tensor,
    x_std: Tensor,
    t_mean: Tensor,
    t_std: Tensor,
    hidden: int,
    n_blocks: int,
    num_frequencies: int,
    beta_schedule: LinearBetaSchedule,
    device: str,
    dtype: torch.dtype,
) -> UpstreamStyleScoreModel:
    net = MirafzaliSkorokhodNet(
        x_dim=3,
        out_dim=3,
        hidden=hidden,
        n_blocks=n_blocks,
        num_frequencies=num_frequencies,
    ).to(device=device, dtype=dtype)
    return UpstreamStyleScoreModel(
        net,
        x_mean,
        x_std,
        t_mean,
        t_std,
        beta_0=beta_schedule.beta_0,
        beta_f=beta_schedule.beta_f,
        beta_t0=beta_schedule.t0,
        beta_tf=beta_schedule.tf,
    ).to(device=device, dtype=dtype)


def train_s2_upstream_style_score_model(
    dataset: Dict[str, Tensor],
    *,
    beta_schedule: LinearBetaSchedule,
    n_epochs: int = 1000,
    batch_size: int = 2048,
    learning_rate: float = 2e-4,
    weight_decay: float = 1e-5,
    hidden: int = 512,
    n_blocks: int = 6,
    num_frequencies: int = 16,
    device: str = "cuda",
    return_history: bool = False,
    training_unit: str = "epochs",
    updates: int = 0,
    warmup_updates: int = 0,
    lr_scheduler: str = "constant",
    ema_rate: float = 0.0,
    checkpoint_every_updates: int = 0,
    checkpoint_callback: Callable[[Dict[str, object]], None] | None = None,
    return_training_state: bool = False,
):
    """Train an independent Upstream-style scaled-score path."""

    required = {"time", "endpoint", "score_target"}
    missing = required.difference(dataset)
    if missing:
        raise KeyError(f"dataset is missing fields: {sorted(missing)}")
    if training_unit not in {"epochs", "updates"}:
        raise ValueError(f"unknown training_unit: {training_unit!r}")
    if training_unit == "updates" and updates < 1:
        raise ValueError("updates must be positive when training_unit='updates'")
    if n_epochs < 1 or batch_size < 1:
        raise ValueError("n_epochs and batch_size must be positive")
    if not 0.0 <= ema_rate < 1.0:
        raise ValueError("ema_rate must be in [0, 1)")
    if checkpoint_every_updates < 0:
        raise ValueError("checkpoint_every_updates must be non-negative")

    total_updates = updates if training_unit == "updates" else n_epochs
    if warmup_updates > total_updates:
        raise ValueError("warmup_updates must not exceed total optimizer updates")
    time = dataset["time"].detach().to(device)
    endpoint = dataset["endpoint"].detach().to(device)
    teacher_score = dataset["score_target"].detach().to(device)
    x_mean = endpoint.mean(dim=0, keepdim=True)
    x_std = endpoint.std(dim=0, keepdim=True).clamp_min(1e-6)
    time_column = time[:, None]
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
        dtype=endpoint.dtype,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    ema_model = copy.deepcopy(model) if ema_rate > 0.0 else None
    if ema_model is not None:
        ema_model.requires_grad_(False)

    n_samples = endpoint.shape[0]
    index_loader = DataLoader(
        TensorDataset(torch.arange(n_samples, dtype=torch.long)),
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=0,
    )
    updates_per_epoch = len(index_loader)
    loader_iterator = iter(index_loader)
    trace_indices = set(
        learning_rate_trace_indices(total_updates, warmup_updates=warmup_updates)
    )
    sampled_updates: list[int] = []
    sampled_losses: list[float] = []
    learning_rate_trace: list[dict[str, float | int]] = []
    initial_learning_rate = None
    peak_learning_rate = 0.0
    final_learning_rate = None

    with torch.no_grad():
        initial_full_loss = float(model.score_loss(time, endpoint, teacher_score))
    best_train_loss = initial_full_loss

    normalization_state = {
        "x_mean": x_mean.detach().cpu().clone(),
        "x_std": x_std.detach().cpu().clone(),
        "t_mean": t_mean.detach().cpu().clone(),
        "t_std": t_std.detach().cpu().clone(),
    }

    for update_index in range(total_updates):
        current_lr = learning_rate_for_update(
            update_index,
            total_updates=total_updates,
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

        if training_unit == "updates":
            try:
                (indices,) = next(loader_iterator)
            except StopIteration:
                loader_iterator = iter(index_loader)
                (indices,) = next(loader_iterator)
            indices = indices.to(device=device)
        else:
            indices = torch.randint(0, n_samples, (batch_size,), device=device)

        loss = model.score_loss(
            time[indices], endpoint[indices], teacher_score[indices]
        )
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if ema_model is not None:
            update_ema_model(ema_model, model, ema_rate)

        detached_loss = float(loss.detach())
        best_train_loss = min(best_train_loss, detached_loss)
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
                        "total_updates": total_updates,
                        "current_update": completed_updates,
                        "last_learning_rate": current_lr,
                    },
                    "current_update": completed_updates,
                    "current_epoch": (
                        completed_updates
                        if training_unit == "epochs"
                        else completed_updates // updates_per_epoch
                    ),
                    "requested_total_updates": (
                        updates if training_unit == "updates" else 0
                    ),
                    "warmup_updates": warmup_updates,
                    "lr_scheduler": lr_scheduler,
                    "ema_rate": ema_rate,
                    "normalization_state": normalization_state,
                }
            )

    with torch.no_grad():
        final_full_loss = float(model.score_loss(time, endpoint, teacher_score))
    history = {
        "epochs": sampled_updates,
        "updates": sampled_updates,
        "train_loss": sampled_losses,
        "initial_train_loss": initial_full_loss,
        "final_train_loss": final_full_loss,
        "best_train_loss": best_train_loss,
        "loss_definition": "mean(||sigma(t) * (s_pred - s_teacher)||_2^2)",
        "network_output": "sigma(t) * effective_score",
    }
    training_state = {
        "ema_model": ema_model,
        "optimizer_state_dict": copy.deepcopy(optimizer.state_dict()),
        "scheduler_state": {
            "lr_scheduler": lr_scheduler,
            "base_learning_rate": float(learning_rate),
            "warmup_updates": warmup_updates,
            "total_updates": total_updates,
            "current_update": total_updates,
            "last_learning_rate": float(final_learning_rate),
        },
        "current_update": total_updates,
        "current_epoch": (
            total_updates
            if training_unit == "epochs"
            else total_updates // updates_per_epoch
        ),
        "requested_total_updates": updates if training_unit == "updates" else 0,
        "actual_optimizer_updates": total_updates,
        "updates_per_epoch": updates_per_epoch,
        "effective_epochs": total_updates / updates_per_epoch,
        "initial_learning_rate": float(initial_learning_rate),
        "peak_learning_rate": float(peak_learning_rate),
        "final_learning_rate": float(final_learning_rate),
        "learning_rate_trace": learning_rate_trace,
        "normalization_state": normalization_state,
        "legacy_training_path": False,
        "score_parameterization": "upstream_scaled_score",
    }

    if not return_history and not return_training_state:
        return model
    if return_history and return_training_state:
        return model, history, training_state
    if return_history:
        return model, history
    return model, training_state
