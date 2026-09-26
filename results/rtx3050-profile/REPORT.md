# Nsight Compute kernel measurements

BA-100k, F=16, layers 16,8, mean aggregation. Instrumented first inference; not used for application speedup.

| Backend | Kernel | Duration (us) | SM throughput (%) | DRAM throughput (%) | Registers/thread | Static shared (bytes) |
|---|---|---:|---:|---:|---:|---:|
| cuda-edge | aggregate_edge_feature_kernel | 3321.024 | 49.761 | 19.106 | 21.000 | 0.000 |
| cuda-edge | dense_transform_kernel | 332.192 | 89.774 | 21.924 | 35.000 | 0.000 |
| cuda-edge | aggregate_edge_feature_kernel | 3325.728 | 50.053 | 19.082 | 21.000 | 0.000 |
| cuda-edge | dense_transform_kernel | 168.736 | 88.415 | 33.087 | 35.000 | 0.000 |
| cuda-edge-tiled | aggregate_edge_feature_kernel | 3326.976 | 49.764 | 18.992 | 21.000 | 0.000 |
| cuda-edge-tiled | dense_tiled_kernel | 269.920 | 87.513 | 26.746 | 40.000 | 2048.000 |
| cuda-edge-tiled | aggregate_edge_feature_kernel | 3325.888 | 49.991 | 18.968 | 21.000 | 0.000 |
| cuda-edge-tiled | dense_tiled_kernel | 270.880 | 87.454 | 20.492 | 40.000 | 2048.000 |
| cuda-tiled | aggregate_vertex_feature_kernel | 859.968 | 48.787 | 64.680 | 40.000 | 0.000 |
| cuda-tiled | dense_tiled_kernel | 270.112 | 87.510 | 26.923 | 40.000 | 2048.000 |
| cuda-tiled | aggregate_vertex_feature_kernel | 856.864 | 48.845 | 64.744 | 40.000 | 0.000 |
| cuda-tiled | dense_tiled_kernel | 270.464 | 86.958 | 20.613 | 40.000 | 2048.000 |
| cuda-vertex | aggregate_vertex_feature_kernel | 857.728 | 48.756 | 64.717 | 40.000 | 0.000 |
| cuda-vertex | dense_transform_kernel | 332.352 | 89.760 | 21.992 | 35.000 | 0.000 |
| cuda-vertex | aggregate_vertex_feature_kernel | 858.112 | 48.830 | 64.654 | 40.000 | 0.000 |
| cuda-vertex | dense_transform_kernel | 168.736 | 88.281 | 33.114 | 35.000 | 0.000 |
