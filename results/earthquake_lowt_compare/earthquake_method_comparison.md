# Earthquake Smoke Method Comparison

| teacher | final_train_loss | validation_loss | teacher_generation_seconds | training_seconds | reverse_sampling_seconds | s2_rbf_mmd | nearest_neighbor_geodesic_mean | nearest_neighbor_geodesic_median | nearest_neighbor_geodesic_max | generated_sample_norm_error |
|---|---|---|---|---|---|---|---|---|---|---|
| heat | 0.20202203 | 12.718839 | 91.524076 | 78.573646 | 78.196488 | 0.013845314 | 0.14858327 | 0.11026881 | 0.83880133 | 1.4582519e-17 |
| varadhan | 0.19731278 | 13.044752 | 19.486917 | 78.661463 | 77.274705 | 0.018357181 | 0.15381022 | 0.11251216 | 0.86375375 | 1.496199e-17 |
| malliavin | 0.18816866 | 10.181742 | 5658.4634 | 77.748704 | 92.185325 | 0.019034798 | 0.13963546 | 0.10222778 | 0.79980727 | 1.4203048e-17 |
