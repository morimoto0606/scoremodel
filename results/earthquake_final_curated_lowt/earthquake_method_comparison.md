# Earthquake Smoke Method Comparison

| teacher | final_train_loss | validation_loss | teacher_generation_seconds | training_seconds | reverse_sampling_seconds | s2_rbf_mmd | nearest_neighbor_geodesic_mean | nearest_neighbor_geodesic_median | nearest_neighbor_geodesic_max | generated_sample_norm_error |
|---|---|---|---|---|---|---|---|---|---|---|
| heat | 0.0013421124 | 36.354905 | 153.58568 | 1544.0973 | 481.37165 | 0.014554785 | 0.10529889 | 0.068955176 | 0.67076635 | 1.3959103e-17 |
| varadhan | 0.001265622 | 37.849423 | 75.86608 | 1548.2759 | 387.54896 | 0.016463333 | 0.1003784 | 0.064556338 | 0.68960141 | 1.4419889e-17 |
| malliavin | 0.0014021893 | 35.967651 | 35150.857 | 1536.4039 | 414.45804 | 0.014212869 | 0.099906991 | 0.060436252 | 0.6908874 | 1.4094628e-17 |
