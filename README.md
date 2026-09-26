# Project Q1: Graph Neural Network Inference on GPUs

Build, run and reproduce experiments using [README.txt](README.txt).

- [Design and evaluation](DOCUMENTATION.md)
- [Full RTX 3050 experiment results](results/rtx3050-full/REPORT.md)
- [Validation record](docs/VALIDATION.md)
- [20-minute presentation](docs/Q1-presentation-v2.pptx)
- [Larger public graph results](results/rtx3050-arxiv/REPORT.md)
- [Nsight Compute analysis](results/rtx3050-profile/REPORT.md)

The root `gnni_q1` executable integrates sequential CPU, OpenMP and four CUDA
configurations with shared inputs, fixed weights and CPU-reference verification.
The older module entry points remain available for their original fixtures.
