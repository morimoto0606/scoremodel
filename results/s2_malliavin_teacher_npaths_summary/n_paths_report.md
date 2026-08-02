# Phase 2A sweep: n_paths

Condition number columns use S2 tangent-restricted definitions.

- unregularized_tangent: kappa_tan = lambda3 / max(lambda2, eps), with per-path eigenvalue sort lambda1 <= lambda2 <= lambda3

- regularized_tangent: kappa_tan_lambda = (lambda3 + lambda) / (lambda2 + lambda)

## Per-run metrics

| n_paths | seed | rmse | cosine | cond_mean | cond_median | cond_max | gen_s | total_s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 64 | 0 | 0.6111216909 | 0.9566090549 | nan | nan | nan | 101.2793433 | 102.5966343 |
| 128 | 0 | 0.4706006622 | 0.9743431354 | nan | nan | nan | 206.0793311 | 208.443904 |
| 256 | 0 | 0.3389474282 | 0.9866838896 | nan | nan | nan | 423.3815097 | 427.4943599 |
| 512 | 0 | 0.2949761631 | 0.9839739644 | nan | nan | nan | 823.1896823 | 836.6146246 |
| 1024 | 0 | 0.2181929938 | 0.9898193222 | nan | nan | nan | 2001.057199 | 2016.743441 |

## Grouped summary (mean/std/min/max)

| n_paths | n_runs | rmse_mean | rmse_std | cosine_mean | cosine_std | cond_mean | cond_median | cond_max | gen_s_mean | total_s_mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 64 | 1 | 0.6111216909 | 0 | 0.9566090549 | 0 | nan | nan | nan | 101.2793433 | 102.5966343 |
| 128 | 1 | 0.4706006622 | 0 | 0.9743431354 | 0 | nan | nan | nan | 206.0793311 | 208.443904 |
| 256 | 1 | 0.3389474282 | 0 | 0.9866838896 | 0 | nan | nan | nan | 423.3815097 | 427.4943599 |
| 512 | 1 | 0.2949761631 | 0 | 0.9839739644 | 0 | nan | nan | nan | 823.1896823 | 836.6146246 |
| 1024 | 1 | 0.2181929938 | 0 | 0.9898193222 | 0 | nan | nan | nan | 2001.057199 | 2016.743441 |
