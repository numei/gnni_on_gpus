# Role A Implementation Checklist

## Completed baseline

- [x] C++17 CMake build with optional OpenMP support.
- [x] Public graph, dense-matrix, model, inference, and benchmark types.
- [x] Versioned binary dense-matrix loader with checked-size and finite-value checks.
- [x] Versioned binary CSR graph loader with local CSR checks and optional edge-feature payload loading.
- [x] Versioned binary model loader with descriptor, activation, weight-size, and finite-value checks.
- [x] Sequential CPU mean-GCN inference.
- [x] OpenMP CPU inference with thread and schedule configuration.
- [x] Tolerance-based output comparison.
- [x] CLI parsing and end-to-end loading/inference workflow.
- [x] Warm-up/repeat benchmarking plus text and CSV reports.
- [x] Unit and end-to-end tests for loaders, CPU/OMP agreement, CLI, and reports.

## Deliberately excluded from the current baseline

- A standalone cross-object validation module. Loaders retain file-local validation; official benchmark inputs must obey the documented shape contract.
- Text fixture parsers and fully automatic text/binary format detection.
- Training, back-propagation, bias terms, self-loop insertion, or any extra GNN architecture.

## Optional future Role A work

- Stable row-wise softmax for final logits.
- Semantics and CPU reference for using edge features during aggregation.
- Attention aggregation and its sequential CPU reference.
- Text fixture loading for small hand-written test data.
- Additional CPU profiling and optimization after preserving reference equivalence.
