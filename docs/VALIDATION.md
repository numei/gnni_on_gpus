# Validation record

Date: 26 September 2026. Branch: `feat/q1-integration`.

## Machine and tools

- AMD Ryzen 9 5900HS, 8 physical cores / 16 logical processors.
- NVIDIA GeForce RTX 3050 Laptop GPU, 4096 MiB, driver 616.92, WDDM.
- Windows 10 build 19045; MSVC 19.29.30159; CMake 4.2.3.
- NVIDIA CUDA 12.6.85 compiler (12.6.3 redistribution), target SM 86.
- Nsight Compute 2024.3.2; matching CUDA Compute Sanitizer distribution.
- Python 3.11 for tests/experiments; matplotlib 3.11.2 for plots.

The installed CUDA_PATH variables pointed to missing directories. The minimal
official CUDA redistribution was downloaded to ignored `tmp/cuda-toolkit/`
and each archive was SHA-256 checked against NVIDIA's redistribution manifest:
https://developer.download.nvidia.com/compute/cuda/redist/redistrib_12.6.3.json.
No driver install or system CUDA setting change was performed. Local Ninja
builds loaded the existing Visual Studio x64 development environment.

## Completed checks

1. CPU-only Visual Studio Release build: 15/15 CTest tests passed.
2. Combined CPU/CUDA Ninja Release build: 20/20 CTest tests passed.
3. Additional supplied `.gnnimodel` CLI fixture passed on all six backends:
   independent expected checksum `4*2 + 5*3 = 23`.
4. Compute Sanitizer memcheck on `q1_integration_test`: **0 errors** and
   **0 bytes leaked in 0 allocations**. See `results/q1-memcheck.log`.
5. Compute Sanitizer racecheck: **0 hazards, 0 errors, 0 warnings**.
   See `results/q1-racecheck.log`.
6. Full synthetic/Cora matrix: **114 backend configurations, 798 samples**,
   each configuration verified against CPU before measurement.
7. ogbn-arxiv: **6 backend configurations, 42 samples**, also verified.
8. Four Nsight Compute reports successfully collected. Original `.ncu-rep`
   files, raw metric CSV, logs and readable summaries are retained.

Maximum absolute comparison error was approximately `2.38e-7` in the full
synthetic/Cora suite and `1.25e-6` on ogbn-arxiv, below the configured combined
absolute/relative tolerance. A passing comparison is a numerical correctness
claim, not a claim about predictive accuracy of generated weights.

## Reproduction commands for this machine

From an x64 Visual Studio development command prompt:

```text
chcp 65001
set VSLANG=1033
cmake -S . -B build-cuda-ninja -G Ninja -DCMAKE_BUILD_TYPE=Release -DGNNI_ENABLE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=86 -DCMAKE_CUDA_COMPILER=<toolkit>/bin/nvcc.exe -DPython3_EXECUTABLE=<python.exe>
cmake --build build-cuda-ninja --parallel 4
ctest --test-dir build-cuda-ninja --output-on-failure
compute-sanitizer --tool memcheck --leak-check full --error-exitcode 99 build-cuda-ninja/q1_integration_test.exe
compute-sanitizer --tool racecheck --error-exitcode 99 build-cuda-ninja/q1_integration_test.exe
```

Experiment commands and source/executable/dataset hashes are recorded in each
run's JSON metadata. Profiling launch commands are in `results/rtx3050-profile/status.json`.
The Git commit recorded there is the parent commit: measurements include the
uncommitted integration work, as the recorded Git status explicitly shows.
Additional malformed-input overflow guards and test/documentation changes
were made after measurement; the numerical kernels and measured semantics
were not changed. Do not describe the parent commit alone as reproducing
these results.

## Limits of validation

- No Linux runtime was available. Linux-only syscall-hook/symlink tests were
  preserved but not executed on this Windows machine.
- Measurements describe one laptop and power/thermal state. They are not
  universal GPU performance claims or comparisons with optimized libraries.
- Host memory values estimate live array payloads, not RSS. Device values
  count owned buffers, not driver/context allocations.
- Performance data uses generated fixed weights. External weight loading is
  tested, but pretrained node-classification accuracy is not evaluated.
- The PowerPoint is generated with the bundled presentation runtime and
  inspected through rendered slides. It has not been opened in PowerPoint.

## Presentation build

`docs/build_presentation.mjs` uses the supplied `@oai/artifact-tool` runtime.
Set `RUNTIME_NODE_MODULES`, `RUNTIME_PYTHON`, and `SKILL_DIR` to the installed
presentation runtime/skill paths before running it with that runtime's Node.
It builds 13 slides with speaker notes totaling 20 minutes, renders previews,
and validates the package and editable native charts before writing
`docs/Q1-presentation-v2.pptx`. The finalizer requires a new output path for a
revision. Ordinary users can directly open the delivered PPTX.
