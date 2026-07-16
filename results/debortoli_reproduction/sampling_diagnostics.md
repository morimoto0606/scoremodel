# Sampling Diagnostics: S2 Toy

## Scope
Investigate why generated S2 samples look near-uniform compared to vMF target around mu=[1,0,0], without changing training code.

## 1) Generated sample diagnostics
Generated file:
- results/debortoli_reproduction/generated_samples.npy

Statistics (N=4096):
- shape: (4096, 3)
- norm mean/std: 1.0 / 2.346690421579347e-16
- mean vector: [-0.04128893, -0.28489410, 0.18785515]
- mean resultant length ||mean||: 0.3437426116074615

Target file (found):
- results/debortoli_reproduction/target_vmf_samples.npy

Target statistics (N=4096):
- shape: (4096, 3)
- norm mean/std: 1.0 / 2.353093408709757e-17
- mean vector: [0.93503124, 0.00380494, 0.00210829]
- mean resultant length ||mean||: 0.9350413554065017

Interpretation:
- Generated samples are on S2 (norm ~1) but concentration is much weaker than target vMF.

## 2) Checkpoint restore verification
Checkpoint used:
- upstream/riemannian-score-sde/results/s2_toy/batch_size=32,eval_batch_size=32,steps=5000,train_plot=false,warmup_steps=100/0/ckpt

Existence and size:
- ckpt exists: True
- arrays.npy size: 33,757,952 bytes
- tree.pkl size: 766 bytes

Restore stats:
- restored train_state leaves: 53
- params_ema leaves: 12
- params leaves: 12
- model_state leaves: 0

Basic value stats:
- params_ema mean/std/L2/count: (4.570142971807826e-05, 0.04326026070006, 44.42818616184212, 1054723)
- params mean/std/L2/count: (4.570142971807826e-05, 0.04326026070006, 44.42818616184212, 1054723)
- model_state mean/std/L2/count: (0.0, 0.0, 0.0, 1)

Restored vs fresh init:
- fresh_vs_restored_param_absdiff_mean: 0.06050154767515431
- fresh_vs_restored_param_absdiff_max: 0.5029440437191657
- fresh_vs_restored_nonzero: True

Conclusion:
- Checkpoint restore is valid and non-trivial. Restored parameters clearly differ from fresh initialization.

## 3) Sampler path and parameter wiring
Training/generation path references:
- run.py generate_plots uses model_w_dicts=(model, train_state.params_ema, train_state.model_state)
- run.py calls pushforward.get_sampler(..., train=False, N=100, eps=cfg.eps, predictor="GRW")
- flow.py SDEPushForward.get_sampler builds reverse SDE sampler via get_pc_sampler

Standalone script wiring:
- scripts/plot_s2_samples.py builds model_w_dicts=(model, train_state.params_ema, train_state.model_state)
- Then calls pushforward.get_sampler(..., train=False, N=..., eps=cfg.eps, predictor="GRW")

EMA vs raw params:
- Script already uses EMA params, same as run.py.

Conclusion:
- Script uses restored train_state correctly, with EMA params as intended.

## 4) Sampling direction check
From flow.py:
- get_sampler default reverse=True.
- In SDEPushForward, reverse=True means sampling from base distribution then reverse SDE to data space.

Empirical before/after comparison:
- base_z (before sampler):
  - mean ~ [-0.00016716, -0.00469873, 0.00844381]
  - resultant ~ 0.009664568940099512 (near uniform)
- trained_samples (after sampler):
  - mean ~ [-0.06484781, -0.27561302, 0.17682643]
  - resultant ~ 0.33381936022561154

Conclusion:
- Script is not accidentally outputting raw base/prior samples. Sampler changes distribution, but not enough toward target vMF concentration.

## 5) Fresh vs trained sampling comparison
Same pipeline, same settings (N=100, predictor=GRW), context=None:
- fresh_samples resultant: 0.2499356692279599
- trained_samples resultant: 0.33381936022561154
- fresh_vs_trained_l2_mean: 1.3313320725486752
- fresh_vs_trained_meanvec_l2: 0.3951321058805901

Conclusion:
- Trained model differs materially from untrained model.
- But trained output is still far from target vMF (resultant ~0.33 vs ~0.94).

## Additional check: sampling step count N
Using restored model, context=None:
- N=20 -> resultant 0.33884985762039316
- N=50 -> resultant 0.3424439103346275
- N=100 -> resultant 0.33153306814307487
- N=200 -> resultant 0.3345873365007582
- N=500 -> resultant 0.3311658237619564

Conclusion:
- Increasing sampler steps does not fix concentration mismatch.

## Root-cause assessment
No bug found in scripts/plot_s2_samples.py parameter wiring or sampling direction.
Most likely cause is model quality/config mismatch rather than sampling script logic:
- training run used batch_size=32 (default s2_toy is 512)
- objective settings in this run include thresh=0.5, n_max=5 (from resolved config), which may underfit concentration for kappa=15
- final generated concentration is improved over base/fresh but still far from target

## Script fix decision
Because no script bug was identified in the restore/sampler path, scripts/plot_s2_samples.py was not modified in this diagnostics pass.
