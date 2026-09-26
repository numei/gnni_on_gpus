# Interpretation of the measured results

All values below are from the retained raw CSV and use median full-call
inference time. Verification and load/setup are excluded, while each GPU
call includes input/weight H2D and final-output D2H. Seven samples follow
two warmups per configuration. Threads apply to OpenMP only.

## Representative end-to-end results

| Dataset | Sequential ms | OpenMP (4 threads) ms | Fastest measured GPU | GPU ms | CPU/GPU speedup |
|---|---:|---:|---|---:|---:|
| ER, 1 million nodes, F16 | 2877.65 | 1063.01 | cuda-tiled | 59.63 | 48.26x |
| BA, 1 million nodes, F16 | 2767.41 | 1560.19 | cuda-tiled | 57.58 | 48.06x |
| WS, 1 million nodes, F16 | 1022.42 | 766.78 | cuda-vertex | 43.46 | 23.53x |
| Cora, 2708 nodes, F1433 | 122.45 | 68.37 | cuda-vertex | 5.12 | 23.91x |
| ogbn-arxiv, 169343 nodes, F128 | 1095.35 | 443.48 | cuda-tiled | 37.39 | 29.29x |

These cases use layer widths 16,8 and mean aggregation. Cora has 13,264 CSR
entries and ogbn-arxiv 2,484,941 after symmetrization and self-loop insertion.
They are execution benchmarks using generated fixed weights; no classification
accuracy is implied. Small differences between vertex and tiled variants may
be comparable to run variability and should not be overinterpreted.

## Aggregation mapping

The atomic edge implementation is slower than the vertex implementation on
the measured cases. On million-node BA, vertex takes 58.64 ms and edge takes
93.68 ms. Distributing a high-degree row is not enough to overcome atomic
accumulation, aggregate zeroing and extra indexing costs in this version.

Nsight on BA-100k measures approximately 858 us per vertex aggregation kernel
versus 3321-3326 us for edge aggregation. Vertex aggregation reaches about
65% DRAM throughput versus 19% for edge aggregation. This is consistent with
additional atomic/instruction overhead; the basic counter set alone does not
prove which individual stall mechanism dominates. Edge destination storage
also adds approximately 4 bytes per CSR entry on device.

## Shared-memory dense transform

Tiling does not uniformly improve end-to-end latency. Nsight's first dense
layer (16 to 16) takes about 332 us with the baseline and 270 us tiled. The
second layer (16 to 8) takes about 169 us baseline and 270 us tiled: half of
the tile's output columns are inactive, and shared-memory synchronization
still has a cost. Both tiled layers use 2048 static shared bytes per block
and 40 registers/thread; baseline dense uses 35 registers/thread.

This explains why a narrow-output two-layer model can show little gain or a
regression. For BA-10k/F128, tiled full-call time improves from 2.291 to 2.166
ms, but Cora regresses from 5.122 to 5.880 ms. A future adaptive implementation
could select a tile shape by output width and retain the simple dense kernel
for narrow outputs. That adaptive selection is not claimed as implemented.

## CPU scaling and skew

On BA-100k, static OpenMP improves from 202.11 ms at one thread to 52.76 ms
at eight threads (3.83x within OpenMP). Dynamic scheduling takes 233.74 and
81.90 ms respectively; guided takes 172.83 and 70.59 ms. Fine-grained dynamic
scheduling overhead can outweigh load-balancing benefits. The graph has
maximum degree 1381 and degree coefficient of variation about 1.39.

Serial-to-OpenMP speedup includes a different dense loop ordering, so it is
not a pure measure of parallel scaling. The thread-count sweep isolates
thread changes within the same OpenMP implementation. Timings vary across
the earlier graph-size case and the later schedule sweep; compare samples
within the same declared case and retain the sample standard deviations.

## Transfer, setup and memory costs

Kernel improvements need not produce proportional end-to-end speedups because
feature/weight uploads, final downloads and host work remain. The stage chart
shows GPU H2D, kernel and D2H medians separately. These independently computed
medians need not sum exactly to median end-to-end time, which also includes
host work. Graph load/validation and context creation are separately recorded
and can dominate a single cold run. The headline speedups therefore describe
repeated inference with an already prepared static graph, not first-use latency.

The memory chart compares live host array estimates and device buffer capacity
for a representative million-node case. It excludes process/driver overhead,
and must not be labeled total RAM/VRAM usage. CSR avoids the multi-terabyte
dense adjacency required at this graph scale.

## Figures and reproducibility

- [Graph size, BA](../results/rtx3050-full/plots/graph-size-ba.png)
- [Feature size](../results/rtx3050-full/plots/feature-size-ba.png)
- [Model depth](../results/rtx3050-full/plots/depth-ba.png)
- [CPU scaling](../results/rtx3050-full/plots/cpu-scaling.png)
- [GPU stages](../results/rtx3050-full/plots/gpu-stages.png)
- [Memory estimates](../results/rtx3050-full/plots/memory.png)
- [Complete tables](../results/rtx3050-full/REPORT.md)
- [ogbn-arxiv table](../results/rtx3050-arxiv/REPORT.md)
- [Kernel profiling table](../results/rtx3050-profile/REPORT.md)

The run metadata includes commands and hashes. Error bars in the scientific
plots are sample standard deviations. Laptop power management, shared system
activity and limited repetitions constrain generalization; this is not a
comparison with cuSPARSE, cuBLAS, PyTorch Geometric or DGL.
