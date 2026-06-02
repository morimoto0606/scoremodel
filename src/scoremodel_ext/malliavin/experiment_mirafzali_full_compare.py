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
    seeds=[0],
    configs=[
        {
            "_key": "full_big_forward_init",
            "method": "mirafzali",
            "correction": "mirafzali_full",
            "n_steps_rev": 1000,
        },
    ],

    reverse_init= "forward_terminal",
    hidden=2048,
    n_blocks=6,
    n_paths=25000,
    n_epochs=5000,
    batch_size=4096,
    lr=2e-4,
    n_steps_per_unit=250,
    outbase="results/mirafzali_full_swissroll_big_forward_init_rev1000_1seed",
)