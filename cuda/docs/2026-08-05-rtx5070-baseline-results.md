# RTX 5070 CUDA Baseline Results

## Environment

- GPU: NVIDIA GeForce RTX 5070, compute capability 12.0
- Driver reported by WSL `nvidia-smi`: 590.57 (Windows driver 591.86)
- CUDA compiler: 13.1, nvcc 13.1.115
- Compute Sanitizer: 2025.4.1.0
- CMake: 4.2.3
- Release architecture flag:
  `--generate-code=arch=compute_120,code=[compute_120,sm_120]`

`nvcc` is not on this WSL user's default `PATH`; all commands use the
installed compiler's absolute path.

## Commands executed

```bash
cmake --fresh -S cuda -B cuda/build-release \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda-13.1/bin/nvcc \
  -DCMAKE_CUDA_ARCHITECTURES=120
cmake --build cuda/build-release --parallel
ctest --test-dir cuda/build-release --output-on-failure
```

CTest passed 6 of 6 tests: GCN correctness, benchmark smoke, CUDA/glibc
compatibility, project structure, VS Code configuration, and obsolete-smoke
artifact checks.

```bash
/usr/local/cuda-13.1/bin/compute-sanitizer --tool memcheck \
  --leak-check full ./cuda/build-release/gcn_cuda_test
```

Compute Sanitizer reported `ERROR SUMMARY: 0 errors` and
`LEAK SUMMARY: 0 bytes leaked in 0 allocations` after all ten correctness
checks passed.

## Correctness sample

For a one-vertex graph with an explicit self-edge, features `[2, -1, 3]`, and
weights

```text
[ 1.00,  2.00]
[-1.00,  0.50]
[ 0.25, -1.00]
```

the independent expected output and CUDA output were both `[3.75, 0.50]`.
Across all fixtures, the largest observed absolute CPU/CUDA difference in the
Release sanitizer run was `2.38419e-7`, below `atol=1e-5` and `rtol=1e-4`.

## Initial timing

Command:

```bash
./cuda/build-release/gcn_cuda_benchmark \
  --vertices 100000 --degree 16 --features 64 --outputs 32 \
  --warmup 5 --iterations 20
```

The deterministic regular graph contained 1,600,000 directed CSR entries.

| Measurement | Result |
|---|---:|
| One-time graph H2D | 2.257600 ms |
| Average input + weights H2D | 2.664797 ms |
| Average aggregation + dense kernels | 0.488378 ms |
| Average final-output D2H | 1.412290 ms |
| Average summed CUDA pipeline | 4.565464 ms |
| Average synchronized full-call end-to-end | 5.104111 ms |
| Throughput | 19,592,052 nodes/s |
| Throughput | 313,472,833 edges/s |

These are first-pass baseline numbers from one run, not a statistical GPU
performance study. End-to-end excludes the context's one-time static graph
upload but includes public-call weight-container work, validation, workspace
reuse, transfers, kernels, event handling, and final synchronization.

## Current boundary

No role-1 graph or matrix interface exists in the repository, so the CUDA
module currently provides small neutral owning `CSRGraph` and `DenseMatrix`
types. The CSR neighborhood direction, self-loop policy, aggregation,
normalization, and activation are explicit and are not silently changed. Role
1 can adapt its eventual public types at the host boundary without changing
the CUDA kernels.

