# Measured experiment summary

Timing: full inference call, excluding setup/loading and verification. Error bars are sample standard deviation, not confidence intervals. Threads/schedule apply only to OpenMP; other backends ignore those requested settings.

| Dataset | Layers | Backend | Threads | Schedule | Median ms | SD ms | CPU speedup |
|---|---|---|---:|---|---:|---:|---:|
| ogbn-arxiv.gnni | 16;8 | cuda-tiled | n/a | n/a | 37.3915 | 0.5397 | 29.294 |
| ogbn-arxiv.gnni | 16;8 | omp | 4 | static | 443.4758 | 14.0237 | 2.470 |
| ogbn-arxiv.gnni | 16;8 | cuda-vertex | n/a | n/a | 38.1061 | 0.6839 | 28.745 |
| ogbn-arxiv.gnni | 16;8 | cuda-edge | n/a | n/a | 60.1166 | 0.3159 | 18.220 |
| ogbn-arxiv.gnni | 16;8 | cpu | 1 | n/a | 1095.3474 | 77.9543 | 1.000 |
| ogbn-arxiv.gnni | 16;8 | cuda-edge-tiled | n/a | n/a | 59.0631 | 0.2502 | 18.545 |
