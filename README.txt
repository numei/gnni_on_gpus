Project Q1: Graph Neural Network Inference on GPUs
=================================================

The root project integrates the existing C++ CPU implementation, shared
datasets and CUDA inference. No training is performed. See DOCUMENTATION.md
for semantics, design decisions, timing definitions and measured results.

Requirements
------------
CMake >= 3.24, C++17 compiler, Python >= 3.11 for experiment tools/tests.
OpenMP is optional; unavailable backends fail explicitly. GPU builds require
an NVIDIA GPU and compatible CUDA toolkit. CUDA 12.6 / MSVC 2019 / RTX 3050
is the tested Windows configuration. RTX 50-series requires a newer toolkit.
Plotting requires matplotlib (experiments/requirements.txt).

Build and test on Linux
-----------------------
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DGNNI_ENABLE_CUDA=ON
cmake --build build --parallel
ctest --test-dir build --output-on-failure

Use -DGNNI_ENABLE_CUDA=OFF for CPU-only builds. For an explicit GPU target,
add -DCMAKE_CUDA_ARCHITECTURES=86 (RTX 3050); choose your GPU architecture.
Use -DCMAKE_CUDA_COMPILER=/path/to/nvcc if it is not on PATH.

Build on Windows
----------------
Use an x64 Native Tools Command Prompt and an installed CUDA toolkit:
chcp 65001
set VSLANG=1033
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DGNNI_ENABLE_CUDA=ON
cmake --build build --parallel
ctest --test-dir build --output-on-failure

For Visual Studio CPU-only builds, omit -G Ninja, build with --config Release,
run ctest with -C Release, and use executables in the Release subdirectories.
Pass -DPython3_EXECUTABLE=C:/path/to/python.exe if necessary.
Keep the UTF-8 code page in the build terminal so Ninja recognizes localized
MSVC header dependency messages. After changing compiler locale, reconfigure with
cmake --fresh and rebuild with --clean-first.

Generate a dataset
------------------
build/common/gnni_dataset_generator ba --vertices 10000 --edges-per-new-vertex 8 --features 64 --seed 42 --self-loops --output graph.gnni

Other graph families: er (--average-degree 16), ws (--neighbors 16
--rewire-probability 0.1). Use --help on the generator for complete options.

Public datasets (downloaded from original providers)
---------------------------------------------------
python common/tools/prepare_public_dataset.py cora --make-undirected --self-loops --output cora.gnni
python common/tools/prepare_public_dataset.py ogbn-arxiv --make-undirected --self-loops --output arxiv.gnni

Run the unified inference CLI
-----------------------------
build/gnni_q1 --dataset graph.gnni --layers 64,32 --backend cpu --verify
build/gnni_q1 --dataset graph.gnni --layers 64,32 --backend omp --threads 4 --schedule guided --verify
build/gnni_q1 --dataset graph.gnni --layers 64,32 --backend cuda-tiled --verify

Backends: cpu, omp, cuda-vertex, cuda-tiled, cuda-edge, cuda-edge-tiled.
Use --list-backends to list compiled backends. GPU hardware availability is
checked when a GPU context is created, not by --list-backends.

Parameters:
  --dataset FILE        GNNIDAT1 dataset (CSR, FP32 features, optional labels).
  --layers 64,32        Fixed generated weights, hidden ReLU, final Identity.
  --model FILE          Load existing GNNIMDL binary weights instead of --layers.
  --seed N              Deterministic generated weight seed (default 42).
  --aggregation MODE    mean (default), sum, or symmetric.
  --backend NAME        Implementation (default cpu).
  --threads N           OpenMP thread count (0 = runtime default).
  --schedule NAME       static, dynamic, guided (OpenMP only).
  --warmup N            Untimed warmups (default 3).
  --repeat N            Measured runs (default 10).
  --verify              Compare against the actual sequential CPU backend.
  --csv FILE            Write raw CSV to a new file; default is stdout.

The CLI never trains weights or claims predictive accuracy for generated
weights. Weights stay fixed across measured runs. Existing .gnnimodel files
load through the CPU model loader described in CPU/docs/file_formats.md.

Data contract
-------------
common/README.md describes the GNNIDAT1 on-disk format. A row v contains
incoming neighbors u (u -> v). Arrays are row-major FP32. Self-loops must be
explicit in the file, matching its metadata. No backend inserts them.
The integrated path uses .gnni; the legacy CPU CLI still reads its original
three-file format, whose loader disallows self-loops. Do not mix formats.

Experiments and plots
---------------------
python -m pip install -r experiments/requirements.txt
python experiments/run_experiments.py --exe build/gnni_q1 --generator build/common/gnni_dataset_generator --suite full --public cora.gnni --public arxiv.gnni --output results/my-run
python experiments/analyze.py results/my-run

The output directory must be new. The script records raw per-run CSV,
configuration commands, dataset hashes and software/hardware metadata.
The full suite includes 10k/100k/1m-node ER/BA/WS graphs, feature and depth
sweeps, and thread/schedule comparisons. Each configuration is verified
before warmup and timed runs. Use --suite smoke for a quick small check.
Seven repetitions and two warmups are the experiment script defaults.
Large graphs need adequate RAM/VRAM. Failed configurations abort the run;
only a successful full run receives a COMPLETE marker.

Profiling and memory checking (optional external tools)
------------------------------------------------------
python experiments/profile.py --exe build/gnni_q1 --dataset graph.gnni --output results/my-profile
compute-sanitizer --tool memcheck --leak-check full --error-exitcode 99 build/q1_integration_test
compute-sanitizer --tool racecheck --error-exitcode 99 build/q1_integration_test

The profile script uses ncu from PATH, or pass --ncu /path/to/ncu.
It saves errors if hardware counters are unavailable and never changes
driver settings. Do not use instrumented profiling times as benchmark times.

Repository layout
-----------------
CPU/             Existing CPU backend, legacy loaders/CLI and tests.
common/          Shared CSR datasets, generators and converters.
cuda/            Vertex/edge aggregation, baseline/tiled dense kernels.
integration/     Common model/data adapter, CLI and integration tests.
experiments/     Reproducible evaluation, plots and profiling scripts.
results/         Recorded experiments (large datasets/builds are ignored).
docs/            Design, validation and presentation material.

CPU types now live in namespace gnni_cpu to avoid conflicting definitions
with common/CUDA gnni types. The adapter converts once during setup.
