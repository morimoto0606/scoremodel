from scoremodel_ext.malliavin.experiment_mirafzali_nonlinear import run_residual_multiseed_eval

#run_residual_multiseed_eval(
#    dataset="swissroll",
#    seeds=[0],
#    configs=[
#        {"_key": "full_big", "method": "mirafzali", "correction": "mirafzali_full"},
#    ],
#    hidden=2048,
#    n_blocks=6,
#    n_paths=25000,
#    n_epochs=5000,
#    batch_size=4096,
#    lr=2e-4,
#    n_steps_per_unit=250,
#    outbase="results/mirafzali_full_swissroll_big_1seed",
#)

run_residual_multiseed_eval(
    dataset="swissroll",
    seeds=[0,1,2,3,4],
    configs=[
        {
            "_key": "approx_lowt_stationary",
            "method": "mirafzali",
            "correction": "approx",
            "n_steps_rev": 1000,
        },
        {
            "_key": "a_correction_lowt_stationary",
            "method": "mirafzali",
            "correction": "a_correction",
            "n_steps_rev": 1000,
        },
        {
            "_key": "full_lowt_stationary",
            "method": "mirafzali",
            "correction": "mirafzali_full",
            "n_steps_rev": 1000,
        },
    ],
    times=[
        0.005, 0.01, 0.02, 0.035,
        0.05, 0.075, 0.10,
        0.20, 0.35, 0.50, 0.75, 1.00,
    ],
    reverse_init="stationary",
    hidden=2048,
    n_blocks=6,
    n_paths=25000,
    n_epochs=5000,
    batch_size=4096,
    lr=2e-4,
    n_steps_per_unit=250,
    outbase="results/mirafzali_variance_diag_5seed",
)