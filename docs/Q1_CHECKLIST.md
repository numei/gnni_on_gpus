# Q1 deliverable mapping

| Requirement | Implementation / evidence |
|---|---|
| One or two GNN architectures, inference only | GCN-style multilayer model; sum/mean/symmetric aggregation; fixed generated or loaded weights |
| Sequential C/C++ | CPU/src/backends/cpu_sequential.cpp |
| Parallel CPU | CPU/src/backends/cpu_openmp.cpp; 1/2/4/8-thread and scheduling experiments |
| Efficient CSR/CSC representation | Common versioned CSR format, dense row-major FP32 features/weights |
| CUDA versions and mappings | cuda/src/gcn_cuda.cu and alternative_kernels.cuh; 2x2 mapping/tiling comparison |
| Throughput, memory and scalability | Unified per-call CSV, host/device array accounting, 10k/100k/1m sweeps |
| Feature dimensions and model depth | F16/64/128 and L1/2/4/8 controlled sweeps |
| Repeated timings | Two warmups, seven samples per configuration, raw CSV and statistics |
| Synthetic and public graphs | ER/BA/WS generators; measured Cora and ogbn-arxiv |
| Skewed degree distributions | Static/dynamic/guided OpenMP, edge alternative, degree statistics and analysis |
| C/C++ and CUDA source | CPU/, common/, cuda/, integration/ |
| Plain ASCII README | README.txt |
| Design and experimental documentation | DOCUMENTATION.md, docs/RESULTS_ANALYSIS.md, results/*/REPORT.md |
| Tables and plots | Raw CSV, summary CSV/Markdown and PNG/PDF plots in results/ |
| 20-minute slides | docs/Q1-presentation-v2.pptx, 13 slides with timed speaker notes |

Correctness uses the real CPU reference and literal fixtures. Twenty combined
CTest tests passed on Windows/RTX3050; GPU memory/race checks and four Nsight
profiles are recorded. Optional GraphSAGE, attention, feature compression,
dense adjacency, training and multi-GPU extensions are not implemented.
No classification-accuracy result is claimed for generated weights.

Before submission, add the students' names/IDs required by the course and
rehearse the oral presentation. Teacher grading and any unstated submission
rules are outside this technical checklist. Changes have not been pushed.
