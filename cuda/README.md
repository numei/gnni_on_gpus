# CUDA vertex-centric GCN baseline

This directory contains the role-2 CUDA baseline for full-batch GCN inference.
It keeps a static CSR graph and reusable inference workspaces on the GPU,
separates neighbor aggregation from dense transformation, and provides CUDA
Event timings and CPU-reference correctness tests.

## Requirements

- CMake 3.24 or newer
- A C++17 host compiler
- CUDA Toolkit 12.8 or newer
- An NVIDIA GPU supported by the installed Toolkit and driver

The examples below use the Toolkit installed on the development machine. If
CUDA is installed elsewhere, replace the compiler path. The default CUDA
architecture is `native`; a build server or distributable binary can override
it with `-DCMAKE_CUDA_ARCHITECTURES=<architecture-list>`.

## Configure, build, and test

From the repository root:

```bash
cmake --fresh -S cuda -B cuda/build-release \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda-13.1/bin/nvcc
cmake --build cuda/build-release --parallel
ctest --test-dir cuda/build-release --output-on-failure
```

To target the RTX 5070 explicitly while retaining an overridable CMake cache
setting, add `-DCMAKE_CUDA_ARCHITECTURES=120` to the configure command.

Run the detailed correctness harness directly with:

```bash
./cuda/build-release/gcn_cuda_test
```

When Compute Sanitizer is installed:

```bash
/usr/local/cuda-13.1/bin/compute-sanitizer --tool memcheck \
  --leak-check full ./cuda/build-release/gcn_cuda_test
```

## Benchmark

The default benchmark uses 100,000 vertices, degree 16, 64 input features, 32
output features, 5 warm-up calls, and 20 measured calls:

```bash
./cuda/build-release/gcn_cuda_benchmark
```

Every size and repetition count is configurable:

```bash
./cuda/build-release/gcn_cuda_benchmark \
  --vertices 200000 --degree 16 --features 128 --outputs 64 \
  --warmup 5 --iterations 20
```

The output reports the one-time static-graph H2D transfer separately, followed
by average input-plus-weight H2D, kernel, final D2H, their summed CUDA pipeline,
and synchronized end-to-end time. End-to-end covers the complete public
inference call, including its host validation and weight-container work, but
intentionally excludes the context's one-time graph upload. Nodes/s and
edges/s use end-to-end time.

## Public API and semantics

Include `gnni/gcn_cuda.hpp` and link `gnni_cuda`. For one layer:

```cpp
gnni::GcnRunResult result = gnni::run_gcn_cuda_vertex_centric(
    graph, features, weights,
    {gnni::Aggregation::Mean, gnni::Activation::Relu});
```

For repeated or multi-layer inference, construct `gnni::CudaGcnContext` once
and call `run_layer` or `run_network`. The context uploads CSR once, uploads
input features once per call, packs and uploads all layer weights once per
call, retains intermediate matrices on device, and copies back only the final
output.

- CSR uses `row_offsets` of length `N + 1` and zero-based `column_indices`.
- Features and outputs are row-major `N x F` FP32 matrices.
- Weights are row-major `F_in x F_out` FP32 matrices.
- Each aggregation thread owns one `(vertex, input_feature)` result.
- Each dense thread owns one `(vertex, output_feature)` result.
- Sum, mean, and symmetric GCN normalization are explicit options.
- Symmetric normalization is `1 / sqrt(degree(v) * degree(u))` using CSR row
  degrees. A zero-degree neighbor contributes zero.
- The implementation never inserts self-loops; callers must include them in
  CSR when required by their model.
- Isolated vertices produce zero outputs because the defined update is
  `activation(W * aggregate)` without a bias term.

`include/gnni/device_buffer.hpp` provides the move-only `DeviceBuffer<T>` RAII
utility for later CUDA modules. CUDA runtime failures throw `gnni::CudaError`;
kernel launches are checked immediately and completion errors surface at the
synchronized timing/output boundary.

Ubuntu 26.04 with glibc 2.42 exposes C23 `rsqrt` declarations through
`_GNU_SOURCE`, which collide with CUDA 13.1 device declarations. The local
CMake configuration applies `-U_GNU_SOURCE` to CUDA compilation and its
compiler-detection checks.

## VS Code build, test, benchmark, and CUDA debugging

1. Install the Remote - WSL extension.
2. From WSL, enter the repository and run `code cuda` so `cuda/` is
   `${workspaceFolder}`.
3. Accept the recommended C/C++, CMake Tools, and Nsight extensions.
4. Press `Ctrl+Shift+B` to run the default **CMake: build** task.
5. Use **Tasks: Run Task** then **GCN CUDA: test** for the complete CTest suite.
6. Use **GCN CUDA: benchmark** for the baseline benchmark.
7. Select **GCN CUDA: debug correctness** and press `F5` to debug the CUDA
   correctness executable with `cuda-gdb`.

The Debug task uses `-G -g` for CUDA breakpoints and is not suitable for
performance measurements. Use the Release commands above for benchmarks.


## Dataset pipeline

All prepared graphs use the versioned `.gnni` format. A CSR row is a message
destination and contains its source vertices. Rows are sorted, duplicate-free,
and self-loops are stored only when explicitly requested. Generated files,
downloads, caches, and sidecars under `cuda/data/` are ignored by Git.

Generate the three synthetic graph families:

```bash
./cuda/build-release/gnni_dataset_generator er \
  --vertices 10000 --average-degree 16 --features 64 --seed 20260815 \
  --self-loops --output cuda/data/er-10k.gnni
./cuda/build-release/gnni_dataset_generator ba \
  --vertices 100000 --edges-per-new-vertex 8 --features 64 --seed 20260815 \
  --self-loops --output cuda/data/ba-100k.gnni
./cuda/build-release/gnni_dataset_generator ws \
  --vertices 1000000 --neighbors 16 --rewire-probability 0.1 \
  --features 64 --seed 20260815 --self-loops \
  --output cuda/data/ws-1m.gnni
```

Download and convert the public datasets using only Python's standard
library. Cora has labels but no split in the LINQS archive; ogbn-arxiv keeps
its official time split. `--make-undirected` is explicit because the source
edge directions are preserved by default.

```bash
python3 cuda/tools/prepare_public_dataset.py cora \
  --make-undirected --self-loops --output cuda/data/cora.gnni
python3 cuda/tools/prepare_public_dataset.py ogbn-arxiv \
  --make-undirected --self-loops --keep-cache \
  --cache-dir cuda/data/cache --output cuda/data/ogbn-arxiv.gnni
```

Use `--archive FILE` to convert an already downloaded official archive and
`--force` to intentionally replace an output. Each command writes a JSON
provenance sidecar containing source/output SHA-256 values or generator
parameters.

Run deterministic full-batch inference on any prepared dataset:

```bash
./cuda/build-release/gcn_dataset_runner \
  --dataset cuda/data/cora.gnni --layer-widths 64,32 \
  --warmup 5 --iterations 20
```

The runner reports `key=value` metadata, load and transfer times, kernel and
end-to-end time, throughput, output shape, and checksum. It uses deterministic
fixed weights for repeatable performance measurements. Labels and splits are
preserved in the file, but this runner does not train a classifier and does
not claim classification accuracy.

### `.gnni` version 1 layout

The 64-byte little-endian header contains `GNNIDAT1`, version `1`, header
length `64`, flags, a zero reserved word, `N`, CSR entry count `E`, feature
width `F`, class count, and exact payload bytes. The payload is:

1. `uint64 row_offsets[N+1]`;
2. `uint32 column_indices[E]`;
3. row-major `float32 features[N*F]`;
4. optional `int32 labels[N]`;
5. optional one-byte splits (`0=none`, `1=train`, `2=valid`, `3=test`).

Flag bits are directed, exact self-loop per vertex, labels present, and splits
present. The loader rejects unknown flags, inconsistent sizes, malformed CSR,
non-finite features, invalid labels/splits, truncation, and trailing bytes.
