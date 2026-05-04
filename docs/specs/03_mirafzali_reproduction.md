# Spec 03: Reproduce Mirafzali Section 4

Status: In Progress  
Priority: High

## Phase Status

| Phase | Description | Status |
|-------|-------------|--------|
| A | Linear SDEs (VE / VP / sub-VP) + 8GMM/checkerboard/swissroll | ✅ Done |
| B | **Nonlinear SDE (Appendix C)** | 🔲 In Progress |

---

## Goal

Faithfully reproduce numerical results of:

Malliavin Calculus for Score-based Diffusion Models

Section 4.

---

## Datasets

- Checkerboard
- Swiss Roll
- 8-GMM

Train size:

8000

---

## Linear SDE

Implement:

- VE
- VP
- sub-VP

Settings:

```python
T = 1.0
dt = 0.004
n_steps = 250

---

## Phase B — Nonlinear SDE (Appendix C)  ← PRIORITY

### SDE

```
dX_t = -k β(t) (X_t - a) / (1 + (X_t - a)²) dt + σ √β(t) dW_t
```

Acts **component-wise** on R² data.

### Default config

```python
k        = 1.0
sigma    = 1.0
a        = 0.0
beta_min = 1.0
beta_max = 25.0
T        = 1.0
dt       = 0.004   # n_steps = round(T / dt) = 250
```

### Malliavin weight

For this nonlinear SDE the Itô–Malliavin formula gives (same algorithm as
the existing `simulate_2d_malliavin_ito`, with the new drift and Jacobian):

```
H = -δ  where δ = Σ_s U_s^T dW_s,  U_s = (D_s X_T)^T γ^{-1}
γ = Y_T (∫_0^T Y_s^{-1} σ²β(s) (Y_s^{-1})^T ds) Y_T^T
```

The Jacobian of the drift is **diagonal**:

```
∂b_i/∂x_i = -k β(t) (1 - (x_i - a)²) / (1 + (x_i - a)²)²
```

### Teachers

| Label   | Method |
|---------|--------|
| `raw`     | Subsample (X_T^i, H^i) directly (Skorokhod integral estimator) |
| `binned`  | 2D histogram binning → bin-averaged score |
| `nw`      | Nadaraya-Watson (Silverman bandwidth) on binned query grid |
| `knn_nw`  | kNN-adaptive NW on binned query grid |

### Reverse SDE (Euler)

```
dX = [-b(X,t) + σ²β(t) s(X,t)] dt + σ√β(t) dW̄   (time T → 0)
```

Starting distribution: samples from p_T via fresh forward simulation.

### Datasets (in order)

1. 8-GMM (radius=2, std=0.08, 8 modes)
2. Checkerboard
3. Swiss Roll

### Metrics

- MMD (RBF kernel, σ=1)
- Sliced Wasserstein (200 random projections)
- Mode coverage + nearest-mode distance (8GMM only)
- nan_rate

### Output layout

```
results/mirafzali_nonlinear/{dataset}/{teacher_method}/
  teacher_field.png
  reverse_samples.png
  metrics.json
  model.pt
```

### Implementation files

| File | Role |
|------|------|
| `src/scoremodel_ext/malliavin/sde_nonlinear.py` | SDE config, drift, Jacobian, Malliavin sim, reverse Euler |
| `src/scoremodel_ext/malliavin/experiment_mirafzali_nonlinear.py` | Teacher application, dataset builder, training loop, metrics |
| `tests/test_mirafzali_nonlinear.py` | Smoke tests |

### Implementation order

1. `raw` teacher + 8GMM ✅ target first
2. `binned` teacher
3. `nw` teacher
4. `knn_nw` teacher
5. Checkerboard + Swiss Roll
