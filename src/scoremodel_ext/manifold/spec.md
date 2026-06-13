Goal:

Tonight, verify that the De Bortoli / Riemannian Score-SDE S² setup works in our repository.

Do NOT implement manifold Malliavin yet.

Environment is already configured. Just activate:

source ~/github/scoremodel/.venv-riemannian/bin/activate

This activation already sets:

JAX_ENABLE_X64=True
GEOMSTATS_BACKEND=jax
PYTHONPATH=~/github/scoremodel/upstream/geomstats:$PYTHONPATH

Do not modify:

upstream/geomstats
upstream/riemannian-score-sde

Tasks:

1. Clean up and extend:

src/scoremodel_ext/manifold/s2_teacher_compare.py

Compare:

s_var = M.metric.log(x0, xt) / t[:, None]

and

s_db = M.grad_marginal_log_prob(
    x0, xt, t,
    thresh=thresh,
    n_max=n_max,
)

Use:

M = Hypersphere(dim=2)

Since M.random_walk() returns None for S², generate xt manually:

z = jax.random.normal(key, x0.shape)
z_tan = M.to_tangent(z, x0)
xt = M.metric.exp(jnp.sqrt(t)[:, None] * z_tan, x0)

Sweep:

times = [0.01, 0.05, 0.10, 0.50, 1.00]
n_max_list = [5, 10, 20, 40]
thresh_list = [0.0, 0.5]

Save:

results/s2_debortoli_teacher_check/raw_results.csv
results/s2_debortoli_teacher_check/summary.json
results/s2_debortoli_teacher_check/rmse_vs_t.png

Report:

rmse
relative_error
mean_norm_db
mean_norm_var

2. Reproduce the original De Bortoli S² toy experiment.

We already inspected the config:

upstream/riemannian-score-sde/config/experiment/s2_toy.yaml

It says the command is:

python main.py experiment=s2_toy

First run a smoke test:

cd ~/github/scoremodel/upstream/riemannian-score-sde
python main.py experiment=s2_toy steps=10 batch_size=32 eval_batch_size=32

If the smoke test succeeds, run a slightly larger check:

python main.py experiment=s2_toy steps=500 batch_size=256 eval_batch_size=256

Only if that also succeeds, optionally start the default run:

python main.py experiment=s2_toy

3. Record all commands and logs.

Save under:

results/debortoli_reproduction/

Include:

command.txt
smoke_stdout.log
smoke_stderr.log
run_status.json

If the original experiment fails, do not spend too much time fixing it. Save the error log and summarize the failure.

4. Do not implement any Malliavin code.

The manifold Malliavin formula has not yet been derived. The purpose tonight is only:

De Bortoli S² teacher works
Original S² toy experiment starts/runs
Outputs are saved for inspection