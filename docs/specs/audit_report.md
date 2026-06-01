# Mirafzali Reproduction Audit

> Paper: **Malliavin Calculus for Score-based Diffusion Models** (arXiv:2503.16917v3)
> Audit date: 2026-06-01

---

## 1. Paper algorithm summary

### Algorithm 4

- **Inputs:** drift `b(t,x)`, diffusion `σ(t)`, horizon `T`, step `dt`, `n_paths`, dimension `d`
- **Outputs:** stored trajectories `{X_k(i), Y_k(i), Z_k(i)}` for all steps `k` and paths `i`
- **Mathematical operations:**
  1. Sample `X_0 ∼ p_data`
  2. Initialise `Y_0 = I_d`, `Z_0 = 0 ∈ R^{d×d×d}`
  3. For each Euler step:
     - `X_k = X_{k-1} + b(t_{k-1}, X_{k-1}) dt + σ(t_{k-1}) dB`
     - First variation: `J = ∇_x b(t_{k-1}, X_{k-1})`;  `Y_k = Y_{k-1} + J · Y_{k-1} dt`
     - **Second variation:** `H_xx = ∇_xx b(t_{k-1}, X_{k-1})`;  
       `Z_k = Z_{k-1} + [H_xx (Y_{k-1} ⊗ Y_{k-1}) + J · Z_{k-1}] dt`
- **Role in pipeline:** Produces `(X, Y, Z)` tuples needed by Algorithm 5 to construct the Malliavin covariance matrix and Skorokhod integral.

### Algorithm 5

- **Inputs:** simulated `{X_t, Y_t, Z_t}`, diffusion `σ(t)`, dimension `d`
- **Outputs:** Malliavin covariances `{γ_{X_t}}`, Skorokhod integrals `{δ_t(u_t)}`
- **Mathematical operations:**
  1. Malliavin covariance:  
     `D_v X_t = Y_t (Y_v)^{-1} σ(v)`;  `γ_{X_t} = ∫_0^t D_v X_t (D_v X_t)^T dv`
  2. Skorokhod integral — auxiliary processes (using **both Y_T and Z_T**):
     - `W_s = Y_T (Y_s)^{-1} σ(s)`
     - `Ω(t) = Z_T (Y_t)^{-1} σ(t) − Y_T (Y_t)^{-1} Z_t (Y_t)^{-1} σ(t)`
     - `Θ(t,s)` = further compound involving `Z_s, Z_t, Y_s, Y_t`
  3. Interaction terms `I_1(t,s)`, `I_2(t,s)` built from `Ω, Θ, W_s`
  4. Correction terms `A(u,t)`, `B(u,t)`, `C(u,t)` using integrals of `I_1, I_2`
  5. **Stochastic term:** `S = γ^{-1}_{X_{t_k}} Y_{t_k} ∫_0^{t_k} (Y_u^{-1} σ(u))^T dB_u`
  6. **Deterministic correction:** `D = ∫_0^{t_k} (Y_u^{-1} σ(u))^T [A − B − C](u,t_k) du`
  7. **Skorokhod integral:** `δ_{t_k}(u_{t_k}) = S − D`
- **Role in pipeline:** Produces the per-path Skorokhod integral `δ_t(u_t)` that serves directly as the score label for Algorithm 6.

### Algorithm 6

- **Inputs:** dataset `{(X_t, t, δ_t(u_t))}`, epochs, batch size `B`, learning rate `η`
- **Outputs:** trained network `N_θ` approximating `E[δ_t(u_t) | X_t]`
- **Mathematical operations:**
  1. Normalise features and targets
  2. MSE loss with L2 regularisation: `L = (1/B) Σ ||δ̂ − δ||² + λ||θ||²`
  3. AdamW + cosine annealing
  4. Architecture: Fourier feature embedding + MLP with 4096 hidden units + 6 residual blocks
- **Role in pipeline:** Trains the score network using Skorokhod integrals from Algorithm 5 as direct regression targets.

### Algorithm 7

- **Inputs:** trained `N_θ`, diffusion `σ(t)`, reverse drift `f(t,x)`, `n_samples`, steps `M`
- **Outputs:** generated samples from `p_data`
- **Mathematical operations:**
  1. Sample from stationary distribution `x ∼ p_s(x)`
  2. For each reverse step (T → 0):
     - `δ̂ = N_θ(x, t)`;  `∇ log p_t(x) = −δ̂`
     - Drift: `µ = f(t,x) + σ(t)² ∇ log p_t(x)`
     - `x ← x + µ dt + σ(t) dB`
- **Role in pipeline:** Reverse-time sampler; uses the trained Skorokhod network as score function.

---

## 2. Current code path

```
dataset (get_sampler)
    ↓
forward SDE + Malliavin weight H
    [sde_nonlinear.py :: simulate_malliavin_nl]
    ↓
teacher dataset (optional smoothing)
    [experiment_mirafzali_nonlinear.py :: apply_teacher_nl]
    ↓
neural network training
    [models.py :: train_mirafzali_skorokhod_net]
    ↓
reverse sampling
    [sde_nonlinear.py :: reverse_euler_nl]
    ↓
metrics (Wasserstein, MMD, NLL)
    [experiment_mirafzali_nonlinear.py :: run_one_experiment_nl]
```

---

## 3. Malliavin weight H comparison

| Component | Paper (Alg 4 + 5) | Current code | Match? | Notes |
|---|---|---|---|---|
| X simulation | Euler–Maruyama | Euler–Maruyama (`simulate_malliavin_nl` lines 183–190) | ✅ Yes | Identical |
| First variation Y | `Y_k = Y_{k-1} + J · Y_{k-1} dt` | `Y = Y + bmm(J, Y) * dt` (line 214) | ✅ Yes | Identical |
| **Second variation Z** | `Z_k = Z_{k-1} + [H_xx(Y⊗Y) + J·Z] dt` | **Not implemented** | ❌ No | `Z` is never allocated, tracked, or stored anywhere in `sde_nonlinear.py` |
| Malliavin covariance γ | `γ = ∫ D_v X_t (D_v X_t)^T dv` assembled via `Y_T (∫ Y_s^{-1} σ² (Y_s^{-1})^T ds) Y_T^T` | Same formula, `core += g**2 * bmm(invYs, invYs.T) * dt` (lines 222–228) | ✅ Yes | Correct |
| **Stochastic term S** | `S = γ^{-1}_{X_t} Y_t ∫₀ᵗ (Y_u^{-1} σ(u))^T dB_u` | Only `Σ U_s^T dW_s` with `U_s = (D_s X_T)^T γ^{-1}` (lines 233–237) | ❌ Partial | The current stochastic integral omits the `[A−B]` correction — note: S itself has no [A−B] weight; the correction enters only in D |
| **Deterministic correction D** | `D = ∫₀ᵗ (Y_u^{-1} σ(u))^T [A−B−C](u,t) du` | **Not implemented** | ❌ No | No deterministic correction term subtracted from δ |
| **A/B/C correction terms** | Computed from `Ω(t), Θ(t,s), I_1(t,s), I_2(t,s)` using both Y_T and Z_T | **Not implemented** | ❌ No | Requires Z, which is absent |
| Final sign of H | `H = −δ = −(S − D)` | `H = −delta` (line 239) | ✅ Yes (sign) | Correct sign, but `delta` is the wrong quantity |

**Summary of Algorithm 4/5 compliance:**
- Algorithm 4: **Partially implemented** — X and Y are correct; Z (second variation) is never computed.
- Algorithm 5: **Not implemented** — The Skorokhod integral is missing the correction terms Ω, Θ, I₁, I₂, A, B, C and the deterministic part D. The stochastic part S is approximated as a plain Itô integral without the nonlinear correction.

---

## 4. Conditional expectation comparison

| Component | Paper (Alg 6) | Current code | Match? | Notes |
|---|---|---|---|---|
| H samples | Per-path Skorokhod integral `δ_t(u_t)` from Alg 5 | `H = −delta` from simplified stochastic integral | ⚠️ Approximate | Label is computed but is not the correct Alg 5 Skorokhod integral |
| Kernel / NW smoothing | Not in Alg 6 — all paths used directly | Available as `method="nw"`, `"knn_nw"` | N/A | Extra heuristic beyond the paper |
| Binned smoothing | Not in Alg 6 | Available as `method="binned"` | N/A | Extra heuristic beyond the paper |
| Target for NN | `δ_t(u_t)` — Algorithm 5 output, all paths | `method="mirafzali"` uses all paths; `method="raw"` subsamples | ⚠️ Partial | `mirafzali` method matches the intent of Alg 6 if labels were correct |
| Direct noisy regression | Yes — Alg 6 trains NN directly on raw per-path δ values | **Yes** (`method="mirafzali"` or `method="raw"`) | ✅ Yes (intent) | The NN learns `E[δ | X_t]` via MSE, consistent with Alg 6 |

**Training target classification:** The current code falls into **Case B** (direct noisy regression): it trains on raw `(X_T, H)` pairs where the NN learns `E[H | X_T]` implicitly through MSE — which is consistent with Algorithm 6. However, since `H` is not the correct Algorithm 5 Skorokhod integral, the labels are wrong.

---

## 5. Algorithm 6/7 training comparison

| Component | Paper | Current code | Match? | Notes |
|---|---|---|---|---|
| Architecture | Fourier features + MLP + 6 residual blocks + hidden 4096 | `MirafzaliSkorokhodNet`: Fourier features + MLP + 6 residual blocks + hidden 512 (default) | ⚠️ Partial | Structure matches; hidden dim is 512 vs paper's 4096 |
| Hidden units | 4096 | 512 (default); configurable via `hidden=` arg | ⚠️ Smaller | `hidden=4096` flag exists in docstring but default is 512 |
| Number of layers | 6 residual blocks | 6 residual blocks (`n_blocks=6`) | ✅ Yes | |
| Input variables | `(X_t, t)` | `(t, x)` concatenated then Fourier embedded | ✅ Yes | |
| Fourier features | Yes | Yes — `FourierFeatures(in_dim, num_frequencies=16, scale=10.0)` | ✅ Yes | In paper; also in code |
| Loss | MSE + L2 regularisation: `(1/B)Σ||δ̂−δ||² + λ||θ||²` | `F.mse_loss(pred, y_n[idx])` + `weight_decay=1e-5` via AdamW | ✅ Yes | L2 regularisation via AdamW weight decay |
| Optimizer | AdamW (lr=2e-4, weight_decay=1e-5) | AdamW (lr=2e-4, weight_decay=1e-5) | ✅ Yes | Exact match |
| LR schedule | Cosine annealing | `CosineAnnealingLR(T_max=n_epochs)` | ✅ Yes | |
| Epochs | 1000 | 1000 (default `n_epochs=1000`) | ✅ Yes | |
| Batch size | 2048 per device | `batch_size=2048` | ✅ Yes | |
| Input/output normalisation | Yes (paper section 4) | Yes — `x_mean/x_std`, `y_mean/y_std` normalisation in `train_mirafzali_skorokhod_net` | ✅ Yes | |
| Reverse sampler | Algorithm 7: sample from `p_s(x)`, reverse SDE with `f(t,x) + σ²∇log p_t` | `reverse_euler_nl`: `x = x + (−fwd_drift + σ²β · score) dt + g · dW` | ✅ Yes | Correct formula |
| Initial distribution for reverse | Stationary distribution `p_s(x)` | `sample_stationary_nl` — Cauchy sampler | ✅ Yes | Correct for `k/σ²=1` case |

---

## 6. Final classification

**Classification: B — Faithful Algorithm 6/7 training, but approximate Algorithm 4/5 Malliavin weight**

**Justification:**

The training loop (Algorithm 6) is faithfully reproduced: same architecture family (Fourier + residual MLP), same MSE+L2 loss, same AdamW + cosine annealing, same normalisation strategy, matching epochs and batch size. The reverse sampler (Algorithm 7) is also faithful: correct reverse drift formula, stationary distribution initialisation, and Euler–Maruyama integration.

However, the Malliavin weight `H` supplied as training labels is **not** the Algorithm 5 Skorokhod integral. Algorithm 5 requires:
1. The **second variation process Z_t** (tracks Hessian effects), which is **never computed**.
2. Auxiliary processes `Ω(t)`, `Θ(t,s)`, interaction terms `I_1, I_2`, and correction factors `A, B, C`.
3. A **deterministic correction term D** subtracted from the stochastic term S to form the full Skorokhod integral `δ = S − D`.

The current code computes only the first-order stochastic term `Σ_s U_s^T dW_s` (a simplified Itô integral), which misses the nonlinear corrections that distinguish the Mirafzali nonlinear formulation from the simpler linear case. This is **Algorithm 5 partially implemented** (Malliavin covariance is correct, but the Skorokhod integral is the linear-case formula applied to a nonlinear SDE).

---

## 7. Required changes

| Priority | File | Function | Required change | Reason | Difficulty |
|---|---|---|---|---|---|
| 1 (Critical) | `sde_nonlinear.py` | `simulate_malliavin_nl` | Add Hessian computation `∇_xx b` and second variation `Z_k = Z_{k-1} + [H_xx(Y_{k-1}⊗Y_{k-1}) + J·Z_{k-1}] dt`; store `Z_list` | Algorithm 4 requires Z for all subsequent Alg 5 calculations | High |
| 2 (Critical) | `sde_nonlinear.py` | `simulate_malliavin_nl` | Compute auxiliary processes `W_s`, `Ω(t)`, `Θ(t,s)` from stored `Y_T, Z_T, Y_s, Z_s` | These are prerequisites for A/B/C correction terms in Alg 5 | High |
| 3 (Critical) | `sde_nonlinear.py` | `simulate_malliavin_nl` | Compute interaction terms `I_1(t,s)`, `I_2(t,s)` and correction terms `A(u,t)`, `B(u,t)`, `C(u,t)` | Algorithm 5, lines 16–22 | High |
| 4 (Critical) | `sde_nonlinear.py` | `simulate_malliavin_nl` | Replace current stochastic sum with correct `S = γ^{-1}_{X_t} Y_t ∫₀ᵗ (Y_u^{-1} σ(u))^T dB_u`; add deterministic correction `D = ∫₀ᵗ (Y_u^{-1} σ(u))^T [A−B−C](u,t) du`; set `δ = S − D` | Algorithm 5, lines 23–26; missing D means δ is systematically biased | High |
| 5 (Critical) | `sde_nonlinear.py` | `jac_drift_nl` or new function | Add `hess_drift_nl` returning `∇_xx b` (d×d×d tensor) | Required for Z update in Algorithm 4, step 16 | Medium |
| 6 (Low) | `models.py` | `MirafzaliSkorokhodNet.__init__` | Change default `hidden` from 512 to 4096 | Paper specifies 4096 hidden units | Low |
| 7 (Low) | `experiment_mirafzali_nonlinear.py` | `run_one_experiment_nl` | Use `method="mirafzali"` as the primary/default method when labels are correct | Ensures Alg 6 pipeline uses all paths with no pre-smoothing | Low |

---

## 8. Recommended next implementation plan

### Step 1: Implement `hess_drift_nl` (Algorithm 4 prerequisite)

In `sde_nonlinear.py`, add a function returning the Hessian tensor `∇_xx b(t,x)` of shape `(n, d, d, d)`. For the Mirafzali nonlinear drift (component-wise), the Hessian is also diagonal: only `∂²b_i/∂x_i²` is nonzero.

$$\frac{\partial^2 b_i}{\partial x_i^2} = -k\beta(t) \cdot \frac{-2u_i(3 - u_i^2)}{(1+u_i^2)^3}, \quad u_i = x_i - a$$

### Step 2: Add second variation Z to `simulate_malliavin_nl` (Algorithm 4, step 15–17)

Inside the Euler loop, allocate `Z` as a `(n, d, d, d)` tensor and update each step:
```
Z_k = Z_{k-1} + [H_xx(Y_{k-1} ⊗ Y_{k-1}) + J · Z_{k-1}] dt
```
Store `Z_list` alongside `Y_list`.

### Step 3: Implement Algorithm 5 Skorokhod correction

After the forward pass, extract terminal values `Y_T, Z_T = Y_list[-1], Z_list[-1]` and implement the following per time step:
1. Compute `W_s = Y_T @ inv(Y_s) @ σ(s)` for all stored `s`
2. Compute `Ω(t) = Z_T @ inv(Y_t) @ σ(t) − Y_T @ inv(Y_t) @ Z_t @ inv(Y_t) @ σ(t)` for each `s=t`
3. Compute `Θ(t,s)` using the formula in Algorithm 5, lines 14–15
4. Compute `I_1(t,s)`, `I_2(t,s)` (lines 17–18)
5. Integrate `A(u,t)`, `B(u,t)`, `C(u,t)` (lines 20–22)
6. Compute `S = γ^{-1}_{X_t} Y_t ∫₀ᵗ (Y_u^{-1} σ(u))^T dB_u` (plain stochastic integral) and `D = ∫₀ᵗ (Y_u^{-1} σ(u))^T [A−B−C](u,t) du` (deterministic correction)
7. Return `δ = S − D`, then `H = −δ`

### Step 4: Validate numerically

Before re-running the full experiment, validate on the 1D version of the SDE:
- Check that `H` from the new implementation matches the analytically known score `∇ log p_t(x)` on a small grid.
- Verify that `E[H | X_T = x]` converges to the correct score as `n_paths → ∞`.

### Step 5: Update default hidden dimension (low priority)

In `models.py`, change `MirafzaliSkorokhodNet` default `hidden=512` to `hidden=4096` to match the paper (or pass `hidden=4096` explicitly in experiment runner).

### Step 6: Re-run experiments with `method="mirafzali"`

Once the Skorokhod labels are correct, run the full experiment with `method="mirafzali"` (all paths, no pre-smoothing) to replicate Algorithm 6 faithfully, and compare metrics against the existing results in `results/mirafzali_nonlinear_*/`.
