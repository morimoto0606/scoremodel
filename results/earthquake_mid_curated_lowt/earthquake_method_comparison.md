# Earthquake Smoke Method Comparison

| teacher | final_train_loss | validation_loss | teacher_generation_seconds | training_seconds | reverse_sampling_seconds | s2_rbf_mmd | nearest_neighbor_geodesic_mean | nearest_neighbor_geodesic_median | nearest_neighbor_geodesic_max | generated_sample_norm_error |
|---|---|---|---|---|---|---|---|---|---|---|
| heat | 0.1452855 | 35.982567 | 61.682709 | 77.874139 | 77.980678 | 0.021038364 | 0.16051895 | 0.11771111 | 0.8729384 | 1.3986208e-17 |
| varadhan | 0.16607267 | 39.675039 | 25.690698 | 78.308831 | 78.246217 | 0.019983074 | 0.15559575 | 0.11243676 | 0.82167127 | 1.4311469e-17 |
| malliavin | 0.17238743 | 39.463332 | 4334.8925 | 77.749164 | 77.456552 | 0.019369524 | 0.15800013 | 0.1114309 | 0.8323655 | 1.4148838e-17 |
