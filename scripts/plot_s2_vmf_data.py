import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt

from geomstats.geometry.hypersphere import Hypersphere

M = Hypersphere(dim=2)
key = jax.random.PRNGKey(0)

n = 4096
mu = jnp.array([1.0, 0.0, 0.0])
kappa = 15.0

x = M.random_von_mises_fisher(mu=mu, kappa=kappa, n_samples=n)
x = np.asarray(x)

print("shape =", x.shape)
print("norm mean =", np.linalg.norm(x, axis=1).mean())
print("mean =", x.mean(axis=0))

fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection="3d")
ax.scatter(x[:, 0], x[:, 1], x[:, 2], s=3, alpha=0.6)

ax.set_title("Target vMF samples on $S^2$")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("z")
ax.set_xlim([-1, 1])
ax.set_ylim([-1, 1])
ax.set_zlim([-1, 1])

plt.tight_layout()
plt.savefig("results/debortoli_reproduction/target_vmf_samples.png", dpi=200)
np.save("results/debortoli_reproduction/target_vmf_samples.npy", x)