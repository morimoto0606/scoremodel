from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

try:
    from .sde_2d import knn_nw_teacher_2d
except ImportError:
    from sde_2d import knn_nw_teacher_2d


_RESIDUAL_METHODS = (
    "mirafzali_residual_binned",
    "mirafzali_residual_nw",
    "mirafzali_residual_knn_nw",
)


@torch.no_grad()
def compute_residuals_nl(
    model,
    t_tr: torch.Tensor,
    x_tr: torch.Tensor,
    H_tr: torch.Tensor,
    device: str = "cpu",
    batch_size: int = 4_096,
) -> Tuple[torch.Tensor, dict]:
    """
    Compute per-sample residuals  r_i = H_i − model(t_i, x_i).

    Parameters
    ----------
    model     : trained score model; must support (t: Tensor(b,), x: Tensor(b,2))
    t_tr      : (n,)   training time labels
    x_tr      : (n, 2) training positions
    H_tr      : (n, 2) Malliavin score weights (targets)
    device    : device to run model inference on
    batch_size: mini-batch size for model inference

    Returns
    -------
    r_tr        : (n, 2) CPU tensor of residuals
    diagnostics : dict with var_H, var_residual, mean_residual_norm
    """
    model.eval()
    model = model.to(device)
    preds = []
    n = t_tr.shape[0]
    for i in range(0, n, batch_size):
        t_b = t_tr[i : i + batch_size].to(device)
        x_b = x_tr[i : i + batch_size].to(device)
        preds.append(model(t_b, x_b).cpu())
    pred_all = torch.cat(preds)
    r_tr = H_tr.cpu() - pred_all

    diagnostics = {
        "var_H": float(H_tr.var().item()),
        "var_residual": float(r_tr.var().item()),
        "mean_residual_norm": float(r_tr.norm(dim=1).mean().item()),
    }
    return r_tr, diagnostics


@torch.no_grad()
def _precompute_residual_bins(
    X_T: torch.Tensor,
    R: torch.Tensor,
    n_bins: int,
) -> dict:
    """Build a 2-D histogram of bin-averaged residuals (CPU tensors)."""
    q = torch.tensor([0.005, 0.995])
    x_min, x_max = torch.quantile(X_T[:, 0], q)
    y_min, y_max = torch.quantile(X_T[:, 1], q)

    x_edges = torch.linspace(x_min.item(), x_max.item(), n_bins + 1)
    y_edges = torch.linspace(y_min.item(), y_max.item(), n_bins + 1)
    n_cells = n_bins * n_bins

    ix = (torch.bucketize(X_T[:, 0].contiguous(), x_edges) - 1).clamp(0, n_bins - 1)
    iy = (torch.bucketize(X_T[:, 1].contiguous(), y_edges) - 1).clamp(0, n_bins - 1)
    flat = ix * n_bins + iy

    counts = torch.bincount(flat, minlength=n_cells).float()
    avg_r0 = torch.bincount(flat, weights=R[:, 0].contiguous(), minlength=n_cells) / counts.clamp_min(1.0)
    avg_r1 = torch.bincount(flat, weights=R[:, 1].contiguous(), minlength=n_cells) / counts.clamp_min(1.0)

    return {
        "x_edges": x_edges,
        "y_edges": y_edges,
        "avg_r0": avg_r0,
        "avg_r1": avg_r1,
        "n_bins": n_bins,
    }


@torch.no_grad()
def _apply_bin_residual(query_x: torch.Tensor, bt: dict) -> torch.Tensor:
    """Look up bin-averaged residual correction at *query_x* positions."""
    n_bins = bt["n_bins"]
    x_edges = bt["x_edges"]
    y_edges = bt["y_edges"]
    ix = (torch.bucketize(query_x[:, 0].contiguous(), x_edges) - 1).clamp(0, n_bins - 1)
    iy = (torch.bucketize(query_x[:, 1].contiguous(), y_edges) - 1).clamp(0, n_bins - 1)
    flat = ix * n_bins + iy
    return torch.stack([bt["avg_r0"][flat], bt["avg_r1"][flat]], dim=1)


@torch.no_grad()
def _nw_residual(
    query_x: torch.Tensor,
    X_ref: torch.Tensor,
    R_ref: torch.Tensor,
    bandwidth_scale: float = 1.0,
    batch_size: int = 512,
) -> torch.Tensor:
    """
    Nadaraya-Watson estimate of E[R | X = query_x] from reference set (X_ref, R_ref).

    Bandwidth = bandwidth_scale × Silverman's rule (d=2).
    """
    n_ref = X_ref.shape[0]
    sigma_mean = float(X_ref.std(dim=0).mean().clamp_min(1e-3))
    h = bandwidth_scale * sigma_mean * float(n_ref) ** (-1.0 / 6.0)

    nq = query_x.shape[0]
    out = torch.zeros(nq, 2, device=query_x.device, dtype=query_x.dtype)
    for i in range(0, nq, batch_size):
        xq = query_x[i : i + batch_size]
        diff = (xq[:, None, :] - X_ref[None, :, :]) / h
        kw = torch.exp(-0.5 * (diff * diff).sum(-1))
        denom = kw.sum(-1, keepdim=True).clamp_min(1e-12)
        out[i : i + batch_size] = (kw[:, :, None] * R_ref[None, :, :]).sum(1) / denom
    return out


@torch.no_grad()
def _knn_nw_residual(
    query_x: torch.Tensor,
    X_ref: torch.Tensor,
    R_ref: torch.Tensor,
    k: int = 256,
    bandwidth_scale: float = 1.0,
    batch_size: int = 256,
) -> torch.Tensor:
    """
    kNN-adaptive Nadaraya-Watson estimate of E[R | X = query_x].

    Bandwidth per query = bandwidth_scale × distance to k-th nearest reference point.
    """
    return knn_nw_teacher_2d(
        X_ref, R_ref, query_x,
        k=k, bandwidth_scale=bandwidth_scale, batch_size=batch_size,
    )


class ResidualCorrectionModel:
    """
    Wraps a trained base score model with a per-time-step nonparametric
    residual correction:

        score(t, x) = base_model(t, x) + alpha * r̂(t, x)

    r̂(t, x) ≈ E[r | X_t = x] is estimated from training residuals stored at
    each discrete time step.  At inference the nearest stored time is used.

    This is a plain Python callable (not ``nn.Module``), with ``eval()`` and
    ``to()`` pass-throughs for compatibility with ``reverse_euler_nl``.

    Parameters
    ----------
    base_model          : trained score model, (t: Tensor(n,), x: Tensor(n,2)) → Tensor(n,2)
    times               : list of discrete training times (sorted ascending)
    X_T_list            : training positions per time step (CPU tensors)
    R_list              : training residuals per time step (CPU tensors)
    mode                : ``'binned'`` | ``'nw'`` | ``'knn_nw'``
    alpha               : residual shrinkage weight (1.0 = full correction)
    n_bins              : spatial bins per axis (for 'binned' mode)
    nw_n_ref            : max reference points per time step (for 'nw'/'knn_nw')
    nw_bandwidth_scale  : Silverman bandwidth multiplier (for 'nw' mode)
    knn_k               : number of nearest neighbours (for 'knn_nw' mode)
    knn_bandwidth_scale : local bandwidth multiplier (for 'knn_nw' mode)
    nw_batch            : query batch size for NW/knn_nw inference
    """

    def __init__(
        self,
        base_model,
        times: List[float],
        X_T_list: List[torch.Tensor],
        R_list: List[torch.Tensor],
        mode: str = "binned",
        alpha: float = 1.0,
        n_bins: int = 60,
        nw_n_ref: int = 5_000,
        nw_bandwidth_scale: float = 1.0,
        knn_k: int = 256,
        knn_bandwidth_scale: float = 1.0,
        nw_batch: int = 512,
    ) -> None:
        if mode not in ("binned", "nw", "knn_nw"):
            raise ValueError(f"ResidualCorrectionModel: unknown mode {mode!r}.")
        self.base_model = base_model
        self.times = list(times)
        self.mode = mode
        self.alpha = alpha
        self.nw_batch = nw_batch
        self.nw_bandwidth_scale = nw_bandwidth_scale
        self.knn_k = knn_k
        self.knn_bandwidth_scale = knn_bandwidth_scale

        if mode == "binned":
            # Precompute bin tables on CPU; tensors transferred to device in forward().
            self._bin_tables = [
                _precompute_residual_bins(Xt.cpu(), R.cpu(), n_bins)
                for Xt, R in zip(X_T_list, R_list)
            ]
            self._X_T_list = None
            self._R_list = None
        else:  # nw or knn_nw
            self._bin_tables = None
            # Subsample reference set to nw_n_ref per time step.
            self._X_T_list = []
            self._R_list = []
            for Xt, R in zip(X_T_list, R_list):
                n = Xt.shape[0]
                idx = torch.randperm(n)[: min(n, nw_n_ref)]
                self._X_T_list.append(Xt[idx].cpu())
                self._R_list.append(R[idx].cpu())

    # ── pass-throughs for nn.Module-style API ──────────────────────────────
    def eval(self):
        self.base_model.eval()
        return self

    def to(self, device):
        self.base_model = self.base_model.to(device)
        return self

    # ── core ──────────────────────────────────────────────────────────────
    def _nearest_idx(self, t_val: float) -> int:
        return min(range(len(self.times)), key=lambda i: abs(self.times[i] - t_val))

    def __call__(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        base_score = self.base_model(t, x)
        if self.alpha == 0.0:
            return base_score
        device = x.device
        t_val = float(t.reshape(-1)[0].item())
        idx = self._nearest_idx(t_val)

        if self.mode == "binned":
            bt = {
                k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                for k, v in self._bin_tables[idx].items()
            }
            r_hat = _apply_bin_residual(x, bt)
        elif self.mode == "nw":
            X_ref = self._X_T_list[idx].to(device)
            R_ref = self._R_list[idx].to(device)
            r_hat = _nw_residual(
                x, X_ref, R_ref,
                bandwidth_scale=self.nw_bandwidth_scale,
                batch_size=self.nw_batch,
            )
        else:  # knn_nw
            X_ref = self._X_T_list[idx].to(device)
            R_ref = self._R_list[idx].to(device)
            r_hat = _knn_nw_residual(
                x, X_ref, R_ref,
                k=self.knn_k,
                bandwidth_scale=self.knn_bandwidth_scale,
                batch_size=self.nw_batch,
            )

        return base_score + self.alpha * r_hat


def _plot_residual_field(
    corrected_model: ResidualCorrectionModel,
    T: float,
    device: str,
    outpath: Path,
    n_grid: int = 30,
) -> None:
    """
    Save a quiver plot of the base, corrected and residual score fields at
    time *T* over a regular 2-D spatial grid.
    """
    # Build grid from [-4, 4]² (covers swissroll / 8gmm / checkerboard)
    xs = torch.linspace(-4.0, 4.0, n_grid, device=device)
    ys = torch.linspace(-4.0, 4.0, n_grid, device=device)
    gX, gY = torch.meshgrid(xs, ys, indexing="ij")
    grid = torch.stack([gX.reshape(-1), gY.reshape(-1)], dim=1)  # (n_grid², 2)
    t_vec = torch.full((grid.shape[0],), T, device=device)

    with torch.no_grad():
        base = corrected_model.base_model(t_vec, grid).cpu().numpy()
        corrected = corrected_model(t_vec, grid).cpu().numpy()
    residual = corrected - base
    xy = grid.cpu().numpy()

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    scale = max(float(np.abs(corrected).max()), 1.0) * 5.0
    for ax, field, title in zip(
        axes,
        [base, corrected, residual],
        ["Base model", "Corrected model", "Residual correction"],
    ):
        ax.quiver(
            xy[:, 0], xy[:, 1], field[:, 0], field[:, 1],
            angles="xy", scale_units="xy", scale=scale, alpha=0.75,
        )
        ax.set_title(title)
        ax.set_aspect("equal")
        ax.set_xlim(-4.5, 4.5)
        ax.set_ylim(-4.5, 4.5)
    fig.suptitle(f"Residual correction field  T={T:.2f}", fontsize=13)
    fig.tight_layout()
    fig.savefig(outpath, dpi=100)
    plt.close(fig)
