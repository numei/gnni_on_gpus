# Role A Semantic Contract

This document is the README for the Role A implementation. It defines what the CPU program computes, how data is represented, and what its command-line/benchmark results mean. The project statement in `problem.pdf` remains the source requirement; this file records the reasonable concrete choices made for the baseline.

## 1. Required baseline

The implementation is one full-batch, inference-only mean-aggregation GCN. It does not train, back-propagate, or process mini-batches.

For layer `l`, let `H(l)` have shape `N x F_in`, let `W(l)` have shape `F_in x F_out`, and let `M(l)` be the aggregated node representation. The layer computes:

```text
H(0) = X

M(l)[v, f] =
    sum(H(l)[u, f] for u in incoming_neighbors(v)) / in_degree(v)
    when in_degree(v) > 0,
    otherwise 0

Z(l) = M(l) * W(l)
H(l + 1) = ReLU(Z(l))       for hidden layers
H(L) = Z(L - 1)             for the final layer
```

`H(L)` is returned as logits or embeddings. Softmax is not part of the required GCN layer.

This matches the PDF's explicit-neighbour mean aggregation: the program does not add self-loops. The zero-degree rule is a stated boundary choice that avoids division by zero.

## 2. Graph semantics

The graph uses compressed sparse row (CSR) storage:

```text
num_nodes
num_edges
row_ptr[num_nodes + 1]
col_idx[num_edges]
```

CSR row `v` spans `p in [row_ptr[v], row_ptr[v + 1])`. At each such position, `u = col_idx[p]` represents the directed edge `(u -> v)`. In other words, each row holds the incoming neighbours of its destination node.

The loader validates the file-local CSR invariants: monotonic offsets, correct endpoints, valid node identifiers, no explicit self-loops, no duplicate neighbours, and increasing neighbour IDs within each row. An undirected graph is stored as two directed CSR entries per undirected edge.

## 3. Dense data and model layout

Features, layer weights, intermediate values, and outputs use finite IEEE-754 `float32` values in row-major storage:

```text
H[v, f]   = H.values[v * H.cols + f]
W[f, out] = W.values[f * W.cols + out]
```

Every layer must satisfy `H.cols == W.rows`; consecutive model layers must satisfy `W(l).cols == W(l + 1).rows`. Formal benchmark inputs are expected to satisfy `features.rows == graph.num_nodes` and `features.cols == model.layers[0].weights.rows`.

The in-memory objects are `CSRGraph`, `DenseMatrix`, and `Model`. Their byte-level binary representation is defined separately in [file_formats.md](file_formats.md).

## 4. CPU implementations

`cpu` is deliberately straightforward and sequential. Each layer has three visible stages:

```text
mean aggregation: H -> M
linear transform: M * W -> Z
activation:       Z -> H_next
```

It is the correctness reference. `omp` implements the same result with OpenMP node-parallel work: a destination row is written by one iteration, so aggregation needs no atomics. The supported OpenMP schedules are configuration choices, not semantic changes.

The approximate per-layer work is `O(E * F_in)` for aggregation and `O(N * F_in * F_out)` for the linear transform.

## 5. Correctness policy

`compare_outputs` compares an implementation result against the sequential result using absolute and relative tolerances. It reports shape mismatch, maximum errors, RMSE, and non-finite values.

File loaders perform their own structural checks before large allocation. The program intentionally has no separate cross-object validation module; benchmark input files are assumed to obey the shape contract above. This keeps the current Role A baseline focused on loading and inference while making the assumption explicit.

## 6. CLI and benchmarking

The executable accepts graph, feature, and model paths plus a backend:

```text
--backend cpu|omp
```

`--verify` compares the selected backend to the sequential reference. Benchmarking supports warm-up runs, repeated timed runs, min/mean/median/standard-deviation summaries, text output, and CSV output.

For a model with `L` layers and compute time `t` seconds, reported throughput is:

```text
nodes_per_second = N * L / t
edges_per_second = E * L / t
```

These measure node-layer and edge-layer work. File loading, command-line parsing, and optional verification are outside the timed inference samples.

## 7. Module boundary

```text
include/gnni/types.hpp          Public graph, matrix, model, and configuration types
include/gnni/api.hpp            Inference, comparison, metrics, and dispatch declarations
include/gnni/io.hpp             Graph, dense-matrix, and model loader declarations

src/io/graph_loader.cpp         Binary CSR graph loading and local graph checks
src/io/matrix_loader.cpp        Binary dense-matrix loading
src/io/model_loader.cpp         Binary model descriptor and weight loading
src/core/compare.cpp            Tolerance-based output comparison

src/backends/cpu_kernels.cpp    Sequential aggregation, linear transform, activation
src/backends/cpu_sequential.cpp Sequential reference inference
src/backends/cpu_openmp.cpp     OpenMP inference
src/backends/dispatch.cpp       cpu/omp backend dispatch and capability checks

src/cli/main.cpp                Executable entry point
src/cli/cli_options.cpp         CLI parsing and workflow orchestration
src/benchmark/runner.cpp        Warm-up, repeats, timing, statistics
src/benchmark/report.cpp        Terminal and CSV report formatting
```

## 8. Optional extensions

The following are deliberately not baseline requirements:

- **Softmax:** a numerically stable row-wise post-processing step for final logits.
- **Edge features:** if enabled, `edge_features[p, ef]` belongs to the same CSR entry as `col_idx[p]`; a new aggregation rule must state exactly how it changes messages.
- **Attention:** requires a separately defined score function, parameter shapes, head count, normalisation, zero-degree rule, and sequential reference.

An optional feature must never silently change the mean-GCN baseline. It needs an explicit configuration choice, documented semantics, and tests.
