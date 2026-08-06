# CUDA Vertex-Centric GCN Baseline Implementation Plan

**Goal:** Deliver a tested CUDA baseline with explicit device ownership,
stable host APIs, correct GCN semantics, and reproducible timings.

**Architecture:** Build a focused `gnni_cuda` library around a persistent graph
context. Keep aggregation and dense update as separate vertex-owned kernels,
and keep all intermediate layer matrices on the GPU.

**Tech stack:** C++17, CUDA 17, CMake/CTest, CUDA Runtime API.

## Tasks

1. Add `tests/gcn_cuda_test.cu` with an independent CPU reference and all
   required graph/shape fixtures. Wire it to CTest and confirm the missing API
   makes the build fail.
2. Add `include/gnni/cuda_error.hpp` and `include/gnni/device_buffer.hpp` with
   checked CUDA calls, checked launches, move-only allocation ownership, and
   reusable capacity.
3. Add `include/gnni/gcn_cuda.hpp` and `src/gcn_cuda.cu` with input validation,
   persistent CSR storage, aggregation/dense kernels, single/multi-layer APIs,
   timing, and output transfer. Build and run the correctness test.
4. Add `bench/gcn_benchmark.cu` with deterministic synthetic CSR/features,
   warm-up, repetitions, and split timing output. Update `cuda/README.md`.
5. Run a fresh Release configure/build, all CTest tests, Compute Sanitizer
   memcheck, and the benchmark. Record exact commands and observed results.

