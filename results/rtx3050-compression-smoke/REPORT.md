# Measured experiment summary

Timing: full inference call, excluding setup/loading and verification. Error bars are sample standard deviation, not confidence intervals. Threads/schedule apply only to OpenMP; other backends ignore those requested settings.

| Dataset | Layers | Backend | Threads | Schedule | Median ms | SD ms | CPU speedup |
|---|---|---|---:|---|---:|---:|---:|
| er-n1000-f16-seed42.gnni | 16;8 | cuda-edge-tiled | n/a | n/a | 0.2638 | 0.0183 | n/a |
| er-n1000-f16-seed42.gnni | 16;8 | cuda-vertex-fp16 | n/a | n/a | 0.3652 | 0.0181 | n/a |
| er-n1000-f16-seed42.gnni | 16;8 | cuda-edge-fp16 | n/a | n/a | 0.3844 | 0.0238 | n/a |
| er-n1000-f16-seed42.gnni | 16;8 | cuda-edge-tiled-fp16 | n/a | n/a | 0.3925 | 0.0237 | n/a |
| er-n1000-f16-seed42.gnni | 16;8 | cuda-edge | n/a | n/a | 0.2731 | 0.0184 | n/a |
| er-n1000-f16-seed42.gnni | 16;8 | cuda-tiled-fp16 | n/a | n/a | 0.3793 | 0.0364 | n/a |
| er-n1000-f16-seed42.gnni | 16;8 | cuda-vertex | n/a | n/a | 0.2008 | 0.0077 | n/a |
| er-n1000-f16-seed42.gnni | 16;8 | cuda-tiled | n/a | n/a | 0.2152 | 0.0163 | n/a |
| ba-n1000-f16-seed42.gnni | 16;8 | cuda-edge-tiled | n/a | n/a | 0.2596 | 0.0170 | n/a |
| ba-n1000-f16-seed42.gnni | 16;8 | cuda-edge-tiled-fp16 | n/a | n/a | 0.4023 | 0.0215 | n/a |
| ba-n1000-f16-seed42.gnni | 16;8 | cuda-edge | n/a | n/a | 0.2866 | 0.0189 | n/a |
| ba-n1000-f16-seed42.gnni | 16;8 | cuda-vertex | n/a | n/a | 0.2266 | 0.0154 | n/a |
| ba-n1000-f16-seed42.gnni | 16;8 | cuda-vertex-fp16 | n/a | n/a | 0.3511 | 0.0179 | n/a |
| ba-n1000-f16-seed42.gnni | 16;8 | cuda-edge-fp16 | n/a | n/a | 0.8090 | 0.1100 | n/a |
| ba-n1000-f16-seed42.gnni | 16;8 | cuda-tiled-fp16 | n/a | n/a | 0.3542 | 0.0311 | n/a |
| ba-n1000-f16-seed42.gnni | 16;8 | cuda-tiled | n/a | n/a | 0.2600 | 0.0763 | n/a |
| ws-n1000-f16-seed42.gnni | 16;8 | cuda-edge-tiled | n/a | n/a | 0.2771 | 0.0147 | n/a |
| ws-n1000-f16-seed42.gnni | 16;8 | cuda-tiled-fp16 | n/a | n/a | 0.3538 | 0.0133 | n/a |
| ws-n1000-f16-seed42.gnni | 16;8 | cuda-edge | n/a | n/a | 0.2624 | 0.0124 | n/a |
| ws-n1000-f16-seed42.gnni | 16;8 | cuda-vertex-fp16 | n/a | n/a | 0.3375 | 0.0249 | n/a |
| ws-n1000-f16-seed42.gnni | 16;8 | cuda-tiled | n/a | n/a | 0.2002 | 0.0136 | n/a |
| ws-n1000-f16-seed42.gnni | 16;8 | cuda-edge-fp16 | n/a | n/a | 0.4007 | 0.1447 | n/a |
| ws-n1000-f16-seed42.gnni | 16;8 | cuda-edge-tiled-fp16 | n/a | n/a | 0.3834 | 0.0188 | n/a |
| ws-n1000-f16-seed42.gnni | 16;8 | cuda-vertex | n/a | n/a | 0.2614 | 0.0112 | n/a |

## Feature compression

Speedup is median end-to-end time with FP32 features divided by median time with FP16 features. Values above 1 indicate faster inference with compression. Device bytes saved includes all reported device allocations.

| Dataset | Layers | CUDA backend | FP32 ms | FP16 ms | Speedup | Device bytes saved | FP16 max absolute error vs CPU |
|---|---|---|---:|---:|---:|---:|---:|
| er-n1000-f16-seed42.gnni | 16;8 | cuda-vertex | 0.2008 | 0.3652 | 0.550 | 64000 | 8.27312469482e-05 |
| er-n1000-f16-seed42.gnni | 16;8 | cuda-edge | 0.2731 | 0.3844 | 0.710 | 64000 | 8.27312469482e-05 |
| er-n1000-f16-seed42.gnni | 16;8 | cuda-edge-tiled | 0.2638 | 0.3925 | 0.672 | 64000 | 8.27312469482e-05 |
| er-n1000-f16-seed42.gnni | 16;8 | cuda-tiled | 0.2152 | 0.3793 | 0.567 | 64000 | 8.27312469482e-05 |
| ba-n1000-f16-seed42.gnni | 16;8 | cuda-edge-tiled | 0.2596 | 0.4023 | 0.645 | 64000 | 8.53836536407e-05 |
| ba-n1000-f16-seed42.gnni | 16;8 | cuda-vertex | 0.2266 | 0.3511 | 0.645 | 64000 | 8.53836536407e-05 |
| ba-n1000-f16-seed42.gnni | 16;8 | cuda-edge | 0.2866 | 0.8090 | 0.354 | 64000 | 8.53836536407e-05 |
| ba-n1000-f16-seed42.gnni | 16;8 | cuda-tiled | 0.2600 | 0.3542 | 0.734 | 64000 | 8.53836536407e-05 |
| ws-n1000-f16-seed42.gnni | 16;8 | cuda-tiled | 0.2002 | 0.3538 | 0.566 | 64000 | 0.000140696763992 |
| ws-n1000-f16-seed42.gnni | 16;8 | cuda-vertex | 0.2614 | 0.3375 | 0.775 | 64000 | 0.000140696763992 |
| ws-n1000-f16-seed42.gnni | 16;8 | cuda-edge | 0.2624 | 0.4007 | 0.655 | 64000 | 0.000140696763992 |
| ws-n1000-f16-seed42.gnni | 16;8 | cuda-edge-tiled | 0.2771 | 0.3834 | 0.723 | 64000 | 0.000140696763992 |
