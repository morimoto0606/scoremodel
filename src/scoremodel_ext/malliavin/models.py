import copy
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

class TimeScoreMLP(nn.Module):
    def __init__(self, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, t, y):
        if t.ndim == 1:
            t = t[:, None]
        if y.ndim == 1:
            y = y[:, None]
        return self.net(torch.cat([t, y], dim=1))


class ScoreMLP2D(nn.Module):
    def __init__(self, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 2),
        )

    def forward(self, x):
        return self.net(x)


class TimeScoreMLP2D(nn.Module):
    def __init__(self, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, hidden),   # t, x1, x2
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 2),
        )

    def forward(self, t, x):
        if t.ndim == 1:
            t = t[:, None]
        inp = torch.cat([t, x], dim=1)
        return self.net(inp)


class FourierFeatures(nn.Module):
    def __init__(self, in_dim: int, num_frequencies: int = 16, scale: float = 10.0):
        super().__init__()
        B = torch.randn(in_dim, num_frequencies) * scale
        self.register_buffer("B", B)

    def forward(self, x):
        proj = 2.0 * math.pi * x @ self.B
        return torch.cat([torch.sin(proj), torch.cos(proj)], dim=-1)


class ResidualBlock(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
        )
        self.act = nn.SiLU()

    def forward(self, x):
        return self.act(x + self.net(x))


class MirafzaliSkorokhodNet(nn.Module):
    """
    N_theta(t, x) ≈ E[delta_t(u_t) | X_t=x]
    For practical memory, default hidden=512. For faithful large run, use hidden=4096.
    """
    def __init__(
        self,
        x_dim: int = 2,
        out_dim: int | None = None,
        hidden: int = 512,
        n_blocks: int = 6,
        num_frequencies: int = 16,
        fourier_scale: float = 10.0,
    ):
        super().__init__()
        out_dim = x_dim if out_dim is None else out_dim
        in_dim = x_dim + 1
        self.ff = FourierFeatures(in_dim, num_frequencies, fourier_scale)
        ff_dim = 2 * num_frequencies
        self.in_layer = nn.Sequential(
            nn.Linear(in_dim + ff_dim, hidden),
            nn.SiLU(),
        )
        self.blocks = nn.Sequential(*[ResidualBlock(hidden) for _ in range(n_blocks)])
        self.out_layer = nn.Linear(hidden, out_dim)

    def forward(self, t, x):
        if t.ndim == 1:
            t = t[:, None]
        z = torch.cat([t, x], dim=1)
        zff = self.ff(z)
        h = torch.cat([z, zff], dim=1)
        h = self.in_layer(h)
        h = self.blocks(h)
        return self.out_layer(h)


class NormalizedSkorokhodModel(nn.Module):
    def __init__(self, net, x_mean, x_std, t_mean, t_std, y_mean, y_std):
        super().__init__()
        self.net = net
        self.register_buffer("x_mean", x_mean)
        self.register_buffer("x_std", x_std)
        self.register_buffer("t_mean", t_mean)
        self.register_buffer("t_std", t_std)
        self.register_buffer("y_mean", y_mean)
        self.register_buffer("y_std", y_std)

    def forward(self, t, x):
        if t.ndim == 1:
            t_col = t[:, None]
        else:
            t_col = t
        tn = ((t_col - self.t_mean) / self.t_std.clamp_min(1e-6)).squeeze(-1)
        xn = (x - self.x_mean) / self.x_std.clamp_min(1e-6)
        yn = self.net(tn, xn)
        return yn * self.y_std + self.y_mean


def learning_rate_for_update(
    update_index: int,
    *,
    total_updates: int,
    base_learning_rate: float,
    warmup_updates: int,
    scheduler: str,
) -> float:
    """Return the LR used by zero-based optimizer update ``update_index``."""

    if total_updates < 1:
        raise ValueError("total_updates must be positive")
    if not 0 <= update_index < total_updates:
        raise ValueError("update_index is outside the training range")
    if not 0 <= warmup_updates <= total_updates:
        raise ValueError("warmup_updates must be in [0, total_updates]")
    if scheduler not in {"constant", "cosine"}:
        raise ValueError(f"unknown lr scheduler: {scheduler!r}")
    if update_index < warmup_updates:
        return base_learning_rate * (update_index + 1) / warmup_updates
    if scheduler == "constant":
        return base_learning_rate
    progress = (update_index - warmup_updates) / max(
        1,
        total_updates - warmup_updates,
    )
    return 0.5 * base_learning_rate * (1.0 + math.cos(math.pi * progress))


def learning_rate_trace_indices(
    total_updates: int,
    *,
    warmup_updates: int,
    maximum_points: int = 1000,
) -> list[int]:
    """Choose at most ``maximum_points`` diagnostic update indices."""

    if total_updates < 1 or maximum_points < 1:
        return []
    priority = set(range(min(10, total_updates)))
    priority.update(range(max(0, total_updates - 10), total_updates))
    priority.update(
        index
        for index in range(warmup_updates - 2, warmup_updates + 3)
        if 0 <= index < total_updates
    )
    if len(priority) >= maximum_points:
        return sorted(priority)[:maximum_points]
    remaining = maximum_points - len(priority)
    if remaining > 0 and total_updates > 1:
        for sample_index in range(remaining):
            index = round(sample_index * (total_updates - 1) / max(1, remaining - 1))
            priority.add(index)
            if len(priority) >= maximum_points:
                break
    return sorted(priority)


@torch.no_grad()
def update_ema_model(
    ema_model: nn.Module,
    online_model: nn.Module,
    ema_rate: float,
) -> None:
    """EMA trainable parameters and exactly synchronize all buffers."""

    if not 0.0 < ema_rate < 1.0:
        raise ValueError("ema_rate must lie strictly between zero and one")
    online_parameters = dict(online_model.named_parameters())
    for name, ema_parameter in ema_model.named_parameters():
        online_parameter = online_parameters[name]
        ema_parameter.mul_(ema_rate).add_(online_parameter, alpha=1.0 - ema_rate)
    online_buffers = dict(online_model.named_buffers())
    for name, ema_buffer in ema_model.named_buffers():
        ema_buffer.copy_(online_buffers[name])


def _normalization_state(
    *,
    x_mean: torch.Tensor,
    x_std: torch.Tensor,
    t_mean: torch.Tensor,
    t_std: torch.Tensor,
    y_mean: torch.Tensor,
    y_std: torch.Tensor,
) -> dict[str, torch.Tensor]:
    return {
        "x_mean": x_mean.detach().cpu().clone(),
        "x_std": x_std.detach().cpu().clone(),
        "t_mean": t_mean.detach().cpu().clone(),
        "t_std": t_std.detach().cpu().clone(),
        "y_mean": y_mean.detach().cpu().clone(),
        "y_std": y_std.detach().cpu().clone(),
    }


def train_mirafzali_skorokhod_net(
    t,
    x,
    delta,
    n_epochs=1000,
    batch_size=2048,
    lr=2e-4,
    weight_decay=1e-5,
    hidden=512,
    n_blocks=6,
    num_frequencies=16,
    device="cuda",
    return_history=False,
    training_unit="epochs",
    updates=0,
    warmup_updates=0,
    lr_scheduler="constant",
    ema_rate=0.0,
    checkpoint_every_updates=0,
    checkpoint_callback=None,
    return_training_state=False,
):
    """
    Algorithm 6 style:
        input  : (X_t, t)
        target : delta_t(u_t)
        output : E[delta_t(u_t) | X_t]
    """
    advanced_training = (
        training_unit != "epochs"
        or warmup_updates != 0
        or lr_scheduler != "constant"
        or ema_rate != 0.0
        or checkpoint_every_updates != 0
    )
    if advanced_training:
        return _train_mirafzali_skorokhod_net_advanced(
            t,
            x,
            delta,
            n_epochs=n_epochs,
            batch_size=batch_size,
            lr=lr,
            weight_decay=weight_decay,
            hidden=hidden,
            n_blocks=n_blocks,
            num_frequencies=num_frequencies,
            device=device,
            return_history=return_history,
            training_unit=training_unit,
            updates=updates,
            warmup_updates=warmup_updates,
            lr_scheduler=lr_scheduler,
            ema_rate=ema_rate,
            checkpoint_every_updates=checkpoint_every_updates,
            checkpoint_callback=checkpoint_callback,
            return_training_state=return_training_state,
        )

    # Algorithm-6 inputs are a fixed teacher dataset.  Detach them before
    # constructing the normalization tensors so only the network parameters
    # participate in each epoch's newly-created autograd graph.
    t = t.detach().to(device)
    x = x.detach().to(device)
    delta = delta.detach().to(device)

    x_mean = x.mean(dim=0, keepdim=True)
    x_std = x.std(dim=0, keepdim=True).clamp_min(1e-6)
    t_col = t[:, None]
    t_mean = t_col.mean(dim=0, keepdim=True)
    t_std = t_col.std(dim=0, keepdim=True).clamp_min(1e-6)
    y_mean = delta.mean(dim=0, keepdim=True)
    y_std = delta.std(dim=0, keepdim=True).clamp_min(1e-6)

    x_n = (x - x_mean) / x_std
    t_n = ((t_col - t_mean) / t_std).squeeze(-1)
    y_n = (delta - y_mean) / y_std

    net = MirafzaliSkorokhodNet(
        x_dim=x.shape[1],
        out_dim=delta.shape[1],
        hidden=hidden,
        n_blocks=n_blocks,
        num_frequencies=num_frequencies,
    ).to(device=device, dtype=x.dtype)

    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_epochs)

    n = x.shape[0]
    best_loss = float("inf")
    best_state = None
    history_epochs = []
    history_train_loss = []
    learning_rates = []

    with torch.no_grad():
        initial_full_loss = F.mse_loss(net(t_n, x_n), y_n).item()

    for ep in range(1, n_epochs + 1):
        learning_rates.append(float(opt.param_groups[0]["lr"]))
        idx = torch.randint(0, n, (batch_size,), device=device)
        pred = net(t_n[idx], x_n[idx])
        loss = F.mse_loss(pred, y_n[idx])

        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()

        if return_history:
            with torch.no_grad():
                full_loss = F.mse_loss(net(t_n, x_n), y_n).item()
            history_epochs.append(ep)
            history_train_loss.append(full_loss)
            if full_loss < best_loss:
                best_loss = full_loss
                best_state = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}

        if ep % 500 == 0:
            with torch.no_grad():
                # validation proxy on random subset to avoid huge full pass
                vidx = torch.randint(0, n, (min(20000, n),), device=device)
                vpred = net(t_n[vidx], x_n[vidx])
                vloss = F.mse_loss(vpred, y_n[vidx]).item()

            if (not return_history) and vloss < best_loss:
                best_loss = vloss
                best_state = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
                print(f"  *** mirafzali best updated: {best_loss:.6e}")

            print(f"  epoch={ep:5d}  loss={vloss:.6e}  best={best_loss:.6e}")

            sched.step()
        else:
            sched.step()

    if best_state is not None:
        net.load_state_dict(best_state)

    wrapped = NormalizedSkorokhodModel(
        net,
        x_mean.detach(),
        x_std.detach(),
        t_mean.detach(),
        t_std.detach(),
        y_mean.detach(),
        y_std.detach(),
    ).to(device)

    if not return_history and not return_training_state:
        return wrapped

    history = {
        "epochs": history_epochs,
        "train_loss": history_train_loss,
        "initial_train_loss": float(initial_full_loss),
        "final_train_loss": float(history_train_loss[-1] if history_train_loss else initial_full_loss),
        "best_train_loss": float(best_loss),
    }
    updates_per_epoch = max(1, math.ceil(n / batch_size))
    trace_indices = learning_rate_trace_indices(
        len(learning_rates),
        warmup_updates=0,
    )
    training_state = {
        "ema_model": None,
        "optimizer_state_dict": copy.deepcopy(opt.state_dict()),
        "scheduler_state": {
            "implementation": "legacy_cosine_annealing",
            "state_dict": copy.deepcopy(sched.state_dict()),
            "current_update": n_epochs,
            "last_learning_rate": (
                learning_rates[-1] if learning_rates else float(lr)
            ),
        },
        "current_update": n_epochs,
        "current_epoch": n_epochs,
        "requested_total_updates": 0,
        "actual_optimizer_updates": n_epochs,
        "updates_per_epoch": updates_per_epoch,
        "effective_epochs": n_epochs / updates_per_epoch,
        "initial_learning_rate": (
            learning_rates[0] if learning_rates else float(lr)
        ),
        "peak_learning_rate": (
            max(learning_rates) if learning_rates else float(lr)
        ),
        "final_learning_rate": (
            learning_rates[-1] if learning_rates else float(lr)
        ),
        "learning_rate_trace": [
            {"update": index, "learning_rate": learning_rates[index]}
            for index in trace_indices
        ],
        "normalization_state": _normalization_state(
            x_mean=x_mean,
            x_std=x_std,
            t_mean=t_mean,
            t_std=t_std,
            y_mean=y_mean,
            y_std=y_std,
        ),
        "legacy_training_path": True,
    }
    if return_history and return_training_state:
        return wrapped, history, training_state
    if return_history:
        return wrapped, history
    return wrapped, training_state


def _train_mirafzali_skorokhod_net_advanced(
    t,
    x,
    delta,
    *,
    n_epochs,
    batch_size,
    lr,
    weight_decay,
    hidden,
    n_blocks,
    num_frequencies,
    device,
    return_history,
    training_unit,
    updates,
    warmup_updates,
    lr_scheduler,
    ema_rate,
    checkpoint_every_updates,
    checkpoint_callback,
    return_training_state,
):
    """Opt-in update-driven training with warmup, cosine decay, and EMA."""

    if training_unit not in {"epochs", "updates"}:
        raise ValueError(f"unknown training_unit: {training_unit!r}")
    if training_unit == "updates" and updates < 1:
        raise ValueError("updates must be positive when training_unit='updates'")
    if n_epochs < 1 or batch_size < 1:
        raise ValueError("n_epochs and batch_size must be positive")
    if ema_rate < 0.0 or ema_rate >= 1.0:
        raise ValueError("ema_rate must be zero or lie in (0, 1)")
    if checkpoint_every_updates < 0:
        raise ValueError("checkpoint_every_updates must be non-negative")

    total_updates = updates if training_unit == "updates" else n_epochs
    if warmup_updates > total_updates:
        raise ValueError("warmup_updates must not exceed total optimizer updates")

    t = t.detach().to(device)
    x = x.detach().to(device)
    delta = delta.detach().to(device)
    x_mean = x.mean(dim=0, keepdim=True)
    x_std = x.std(dim=0, keepdim=True).clamp_min(1e-6)
    t_col = t[:, None]
    t_mean = t_col.mean(dim=0, keepdim=True)
    t_std = t_col.std(dim=0, keepdim=True).clamp_min(1e-6)
    y_mean = delta.mean(dim=0, keepdim=True)
    y_std = delta.std(dim=0, keepdim=True).clamp_min(1e-6)
    x_n = (x - x_mean) / x_std
    t_n = ((t_col - t_mean) / t_std).squeeze(-1)
    y_n = (delta - y_mean) / y_std

    net = MirafzaliSkorokhodNet(
        x_dim=x.shape[1],
        out_dim=delta.shape[1],
        hidden=hidden,
        n_blocks=n_blocks,
        num_frequencies=num_frequencies,
    ).to(device=device, dtype=x.dtype)
    optimizer = torch.optim.AdamW(
        net.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )
    ema_net = copy.deepcopy(net) if ema_rate > 0.0 else None
    if ema_net is not None:
        ema_net.requires_grad_(False)

    n_samples = x.shape[0]
    index_loader = DataLoader(
        TensorDataset(torch.arange(n_samples, dtype=torch.long)),
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=0,
    )
    updates_per_epoch = len(index_loader)
    loader_iterator = iter(index_loader)
    selected_trace_indices = set(
        learning_rate_trace_indices(
            total_updates,
            warmup_updates=warmup_updates,
        )
    )
    learning_rate_trace = []
    sampled_updates = []
    sampled_train_losses = []
    initial_learning_rate = None
    peak_learning_rate = 0.0
    final_learning_rate = None
    best_train_loss = torch.full((), float("inf"), dtype=x.dtype, device=device)

    with torch.no_grad():
        initial_full_loss = F.mse_loss(net(t_n, x_n), y_n).item()

    normalization_state = _normalization_state(
        x_mean=x_mean,
        x_std=x_std,
        t_mean=t_mean,
        t_std=t_std,
        y_mean=y_mean,
        y_std=y_std,
    )

    for update_index in range(total_updates):
        current_lr = learning_rate_for_update(
            update_index,
            total_updates=total_updates,
            base_learning_rate=lr,
            warmup_updates=warmup_updates,
            scheduler=lr_scheduler,
        )
        for parameter_group in optimizer.param_groups:
            parameter_group["lr"] = current_lr
        if initial_learning_rate is None:
            initial_learning_rate = current_lr
        peak_learning_rate = max(peak_learning_rate, current_lr)
        final_learning_rate = current_lr
        if update_index in selected_trace_indices:
            learning_rate_trace.append(
                {"update": update_index, "learning_rate": current_lr}
            )

        if training_unit == "updates":
            try:
                (batch_indices,) = next(loader_iterator)
            except StopIteration:
                loader_iterator = iter(index_loader)
                (batch_indices,) = next(loader_iterator)
            batch_indices = batch_indices.to(device=device)
        else:
            batch_indices = torch.randint(
                0,
                n_samples,
                (batch_size,),
                device=device,
            )

        prediction = net(t_n[batch_indices], x_n[batch_indices])
        loss = F.mse_loss(prediction, y_n[batch_indices])
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        optimizer.step()
        if ema_net is not None:
            update_ema_model(ema_net, net, ema_rate)

        detached_loss = loss.detach()
        best_train_loss = torch.minimum(best_train_loss, detached_loss)
        if update_index in selected_trace_indices:
            sampled_updates.append(update_index + 1)
            sampled_train_losses.append(float(detached_loss))

        completed_updates = update_index + 1
        if (
            checkpoint_every_updates > 0
            and completed_updates % checkpoint_every_updates == 0
            and checkpoint_callback is not None
        ):
            scheduler_state = {
                "lr_scheduler": lr_scheduler,
                "base_learning_rate": float(lr),
                "warmup_updates": warmup_updates,
                "total_updates": total_updates,
                "current_update": completed_updates,
                "last_learning_rate": current_lr,
            }
            checkpoint_callback(
                {
                    "online_network_state_dict": {
                        key: value.detach().cpu().clone()
                        for key, value in net.state_dict().items()
                    },
                    "ema_network_state_dict": (
                        {
                            key: value.detach().cpu().clone()
                            for key, value in ema_net.state_dict().items()
                        }
                        if ema_net is not None
                        else None
                    ),
                    "optimizer_state_dict": copy.deepcopy(optimizer.state_dict()),
                    "scheduler_state": scheduler_state,
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
        final_full_loss = F.mse_loss(net(t_n, x_n), y_n).item()

    wrapped = NormalizedSkorokhodModel(
        net,
        x_mean.detach(),
        x_std.detach(),
        t_mean.detach(),
        t_std.detach(),
        y_mean.detach(),
        y_std.detach(),
    ).to(device)
    ema_wrapped = None
    if ema_net is not None:
        ema_wrapped = NormalizedSkorokhodModel(
            ema_net,
            x_mean.detach().clone(),
            x_std.detach().clone(),
            t_mean.detach().clone(),
            t_std.detach().clone(),
            y_mean.detach().clone(),
            y_std.detach().clone(),
        ).to(device)
        for ema_buffer, online_buffer in zip(
            ema_wrapped.buffers(),
            wrapped.buffers(),
        ):
            ema_buffer.copy_(online_buffer)

    history = {
        "epochs": sampled_updates,
        "updates": sampled_updates,
        "train_loss": sampled_train_losses,
        "initial_train_loss": float(initial_full_loss),
        "final_train_loss": float(final_full_loss),
        "best_train_loss": float(best_train_loss),
    }
    scheduler_state = {
        "lr_scheduler": lr_scheduler,
        "base_learning_rate": float(lr),
        "warmup_updates": warmup_updates,
        "total_updates": total_updates,
        "current_update": total_updates,
        "last_learning_rate": float(final_learning_rate),
    }
    training_state = {
        "ema_model": ema_wrapped,
        "optimizer_state_dict": copy.deepcopy(optimizer.state_dict()),
        "scheduler_state": scheduler_state,
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
    }

    if not return_history and not return_training_state:
        return wrapped
    if return_history and return_training_state:
        return wrapped, history, training_state
    if return_history:
        return wrapped, history
    return wrapped, training_state
