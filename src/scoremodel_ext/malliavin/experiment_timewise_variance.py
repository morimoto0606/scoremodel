from scoremodel_ext.malliavin.mirafzali_teacher import simulate_all_times_nl
from scoremodel_ext.malliavin.sde_nonlinear import NonlinearSDEConfig

cfg = NonlinearSDEConfig()
times=[
    0.005, 0.01, 0.02, 0.035,
    0.05, 0.075, 0.10,
    0.20, 0.35, 0.50, 0.75, 1.00,
]

for corr in ["approx", "a_correction", "mirafzali_full"]:
    print("\n", corr)
    cache = simulate_all_times_nl(
        times=times,
        dataset_name="swissroll",
        cfg=cfg,
        n_paths=25000,
        n_steps_per_unit=250,
        device="cuda",
        correction=corr,
    )