"""
S² De Bortoli Teacher vs Variational Score Comparison

Compare the De Bortoli method (grad_marginal_log_prob) with the variational method
on the 2-sphere using geodesic random walk.

Sweeps:
- times = [0.01, 0.05, 0.10, 0.50, 1.00]
- n_max_list = [5, 10, 20, 40]
- thresh_list = [0.0, 0.5]

Saves results to results/s2_debortoli_teacher_check/
"""

import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
from geomstats.geometry.hypersphere import Hypersphere
import json
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

M = Hypersphere(dim=2)
key = jax.random.PRNGKey(0)

n = 1000

# Generate random base points on S²
x0 = M.random_uniform(key, n)

def sphere_grw_step(key, x, t):
    """Generate geodesic random walk samples on S²"""
    key, subkey = jax.random.split(key)
    z = jax.random.normal(subkey, x.shape)
    
    # Project ambient Gaussian to tangent space T_x S^2
    z_tan = M.to_tangent(z, x)
    
    # exp_x(sqrt(t) z_tan)
    tangent_vec = jnp.sqrt(t)[:, None] * z_tan
    xt = M.metric.exp(tangent_vec, x)
    return key, xt

# Configuration
times = [0.01, 0.05, 0.10, 0.50, 1.00]
n_max_list = [5, 10, 20, 40]
thresh_list = [0.0, 0.5]

# Results storage
results = []
summary_by_t = {}

# Main sweep loop
for t0 in times:
    print(f"\n=== Time t = {t0:.2f} ===")
    t = t0 * jnp.ones((n,))
    key, xt = sphere_grw_step(key, x0, t)
    
    # Variational score: log_{xt}(x0) / t
    s_var = M.metric.log(x0, xt) / t[:, None]
    norm_var = jnp.linalg.norm(s_var, axis=1)
    mean_norm_var = jnp.mean(norm_var)
    
    for thresh in thresh_list:
        for n_max in n_max_list:
            # De Bortoli score
            s_db = M.grad_marginal_log_prob(
                x0, xt, t,
                thresh=thresh,
                n_max=n_max,
            )
            
            norm_db = jnp.linalg.norm(s_db, axis=1)
            mean_norm_db = jnp.mean(norm_db)
            
            # Compute errors
            err = jnp.linalg.norm(s_db - s_var, axis=1)
            rmse = float(jnp.sqrt(jnp.mean(err**2)))
            relative_error = float(jnp.mean(err / (norm_db + 1e-8)))
            
            result = {
                't': t0,
                'n_max': n_max,
                'thresh': thresh,
                'rmse': rmse,
                'relative_error': relative_error,
                'mean_norm_db': float(mean_norm_db),
                'mean_norm_var': float(mean_norm_var),
            }
            results.append(result)
            
            print(f"  n_max={n_max:2d} thresh={thresh:.1f}  "
                  f"rmse={rmse:.6f}  rel_err={relative_error:.6f}")
    
    # Store summary for this time
    summary_by_t[t0] = {
        'mean_norm_var': float(mean_norm_var),
    }

# Save raw results to CSV
result_dir = Path("results/s2_debortoli_teacher_check")
result_dir.mkdir(parents=True, exist_ok=True)

df = pd.DataFrame(results)
csv_path = result_dir / "raw_results.csv"
df.to_csv(csv_path, index=False)
print(f"\n✓ Saved raw results to {csv_path}")

# Save summary statistics
summary = {
    'n_samples': n,
    'times': times,
    'n_max_list': n_max_list,
    'thresh_list': thresh_list,
    'summary_by_t': summary_by_t,
}
json_path = result_dir / "summary.json"
with open(json_path, 'w') as f:
    json.dump(summary, f, indent=2)
print(f"✓ Saved summary to {json_path}")

# Create visualization: RMSE vs time for best parameters
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Plot 1: RMSE vs t for different n_max (thresh=0.0)
df_t00 = df[df['thresh'] == 0.0]
for n_max in n_max_list:
    df_subset = df_t00[df_t00['n_max'] == n_max].sort_values('t')
    axes[0].plot(df_subset['t'], df_subset['rmse'], marker='o', label=f'n_max={n_max}')
axes[0].set_xlabel('Time t')
axes[0].set_ylabel('RMSE')
axes[0].set_title('RMSE vs Time (thresh=0.0)')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Plot 2: Relative Error vs t for different n_max (thresh=0.0)
for n_max in n_max_list:
    df_subset = df_t00[df_t00['n_max'] == n_max].sort_values('t')
    axes[1].plot(df_subset['t'], df_subset['relative_error'], marker='s', label=f'n_max={n_max}')
axes[1].set_xlabel('Time t')
axes[1].set_ylabel('Relative Error')
axes[1].set_title('Relative Error vs Time (thresh=0.0)')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
png_path = result_dir / "rmse_vs_t.png"
plt.savefig(png_path, dpi=100, bbox_inches='tight')
print(f"✓ Saved plot to {png_path}")

print("\n" + "="*60)
print("S² Teacher Comparison Complete")
print("="*60)