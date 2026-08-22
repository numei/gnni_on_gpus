# GNNI — Role A CPU Inference

This repository contains the Role A implementation of a full-batch GNN inference baseline:

- versioned binary graph, feature, and model loading;
- sequential CPU inference as the correctness reference;
- OpenMP CPU inference;
- output comparison, benchmark summaries, text/CSV reporting, and CLI orchestration.

The mathematical and data-layout choices are explained in [docs/semantic_contract.md](docs/semantic_contract.md). The on-disk binary formats are specified in [docs/file_formats.md](docs/file_formats.md).

## Build and test

```text
cmake -S . -B build
cmake --build build
ctest --test-dir build --output-on-failure
```

## Run

```text
./build/gnni \
  --graph graph.gnnigraph \
  --features features.gnnimatrix \
  --model model.gnnimodel \
  --backend cpu \
  --repeat 5 --verify
```

To use the OpenMP implementation instead:

```text
./build/gnni --graph graph.gnnigraph --features features.gnnimatrix \
  --model model.gnnimodel --backend omp --threads 4 --omp-schedule static
```

Use `--help` for the complete list of options. Binary input is the implemented baseline; text fixtures remain a future convenience feature.

## Current baseline

- Full-batch mean-aggregation GCN inference.
- CSR rows store incoming neighbours: an entry in row `v` represents an edge `(u -> v)`.
- Only explicit neighbours are averaged; no self-loops are inserted.
- A node with zero in-degree receives an all-zero aggregate.
- Dense matrices and weights use row-major `float32` storage.
- Hidden layers use ReLU; the final layer uses Identity and returns logits/embeddings.
- `cpu` is the sequential reference; `omp` is checked against it.

Softmax, edge features, and attention are deliberately optional extensions. They are not required to run the baseline.
