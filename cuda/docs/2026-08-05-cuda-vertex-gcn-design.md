# CUDA Vertex-Centric GCN Baseline Design

## Scope and semantics

The CUDA module implements full-batch inference for GCN-style layers:

1. aggregate CSR-row neighbors into an `N x F_in` matrix;
2. multiply the aggregate by a row-major `F_in x F_out` weight matrix;
3. optionally apply ReLU.

CSR rows define the neighbor set exactly as supplied. The module never adds
self-loops or changes edge direction. Sum, arithmetic mean, and symmetric GCN
normalization are explicit options. Symmetric normalization uses
`1 / sqrt(degree(v) * degree(u))`, where degrees are CSR row lengths. An
isolated row produces zeros for every aggregation mode.

## Public boundary

`CSRGraph` and `DenseMatrix` are small owning host types because no role-1
interface exists yet. `CudaGcnContext` owns a device copy of one graph and
reuses feature, weight, aggregate, and output buffers across calls. It exposes
single-layer and multi-layer execution. The convenience
`run_gcn_cuda_vertex_centric` function creates a temporary context for callers
that do not need reuse.

All CUDA-specific ownership stays behind the context implementation, so role 1
can later adapt its graph and matrix types without changing kernels.

## Device data flow

- CSR row offsets and column indices are uploaded once per context.
- Input features are uploaded once per inference call.
- Each layer's weights are uploaded once; intermediate node matrices remain on
  the device and use ping-pong buffers.
- One thread owns one `(vertex, input_feature)` aggregation output.
- One thread owns one `(vertex, output_feature)` dense-transform output.
- No kernel uses atomics or unified memory.
- Only the final node matrix is copied back to the host.

All allocations use a move-only `DeviceBuffer<T>` RAII wrapper. Every runtime
call, event operation, memory transfer, and kernel launch is checked and CUDA
failures become descriptive `CudaError` exceptions.

## Timing

CUDA events measure graph H2D, per-run input/weight H2D, all kernels, and final
D2H after correct synchronization. End-to-end time is a synchronized host wall
clock measurement and includes validation, allocation/reuse, transfers, and
kernels. The benchmark performs warm-up calls before measured repetitions.

## Verification

The test executable contains a deliberately independent CPU reference and
literal graph fixtures. It covers a single vertex, chain, star, isolated
vertex, zero-edge graph, dimensions not divisible by the CUDA block size,
normalization modes, activations, multi-layer inference, repeated context use,
and invalid inputs. CUDA results use `atol=1e-5` and `rtol=1e-4`.

