# GNN Dataset Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a deterministic, dependency-light pipeline that generates ER/BA/WS synthetic datasets, converts Cora and ogbn-arxiv, validates and imports a shared `.gnni` format, and runs the imported graphs through the existing CUDA GCN implementation.

**Architecture:** A CUDA-free `gnni_dataset` static library owns core graph types, validation, binary I/O, and synthetic generation. A native generator writes synthetic `.gnni` files; a Python-standard-library converter safely downloads and converts public datasets into the same format; a separate CUDA runner loads any `.gnni` file and reports stable `key=value` inference metrics. Generated data and caches remain ignored, while tests use only small deterministic local fixtures.

**Tech Stack:** C++17, CUDA 13.x, CMake/CTest, Python 3 standard library, Git.

## Global Constraints

- The approved specification is `docs/superpowers/specs/2026-08-15-dataset-pipeline-design.md`; implementation choices must not contradict it.
- Do not add pip, Conda, PyTorch, PyG, DGL, OGB, NetworkX, NumPy, or a JSON dependency.
- A stored edge `(u, v)` means CSR row `v` contains column `u`; rows are sorted and duplicate-free.
- Synthetic graphs are simple and undirected, represented by symmetric CSR entries. Self-loops exist only when requested and then exactly once per vertex.
- Version 1 files are little-endian, use a fixed 64-byte header, and reject unknown flags, overflow, malformed CSR, non-finite features, inconsistent labels/splits, truncation, and trailing bytes.
- Never overwrite a dataset or sidecar unless `--force` is present. Use same-directory temporary files plus atomic rename.
- All random behavior is seed-controlled and reproducible. Avoid standard-library distributions whose mapping can differ across implementations; derive bounded integers and unit doubles explicitly from `std::mt19937_64` output.
- Full Cora/ogbn-arxiv downloads are opt-in verification commands, never ordinary CTest fixtures.
- Before every commit, run the focused tests named in that task and inspect `git diff --check` plus `git status --short`.

## File and Target Map

### New C++ production files

- `cuda/include/gnni/graph_types.hpp` — CUDA-independent `CSRGraph` and `DenseMatrix`.
- `cuda/include/gnni/dataset.hpp` — host dataset model, metadata, split enum, validation and accounting.
- `cuda/include/gnni/dataset_io.hpp` — `.gnni` version/flag constants and load/save API.
- `cuda/include/gnni/synthetic_graph.hpp` — ER/BA/WS options and generator API.
- `cuda/src/graph_types.cpp` — checked `DenseMatrix` constructors.
- `cuda/src/dataset.cpp` — dataset semantic validation and host byte accounting.
- `cuda/src/dataset_io.cpp` — strict little-endian binary reader/writer.
- `cuda/src/synthetic_graph.cpp` — deterministic graph and feature generation.
- `cuda/apps/dataset_generator.cpp` — synthetic generator CLI and JSON provenance.
- `cuda/apps/gcn_dataset_runner.cu` — file-backed multilayer CUDA inference runner.

### New Python production files

- `cuda/tools/gnni_format.py` — dependency-free `.gnni` writer, CSR canonicalization, atomic file helpers.
- `cuda/tools/prepare_public_dataset.py` — safe download/archive handling and Cora/ogbn-arxiv conversion CLI.

### New tests and documentation

- `cuda/tests/graph_types_test.cpp`
- `cuda/tests/dataset_validation_test.cpp`
- `cuda/tests/dataset_io_test.cpp`
- `cuda/tests/synthetic_graph_test.cpp`
- `cuda/tests/check_dataset_generator.py`
- `cuda/tests/test_public_converter.py`
- `cuda/tests/check_dataset_runner.py`
- `cuda/docs/dataset-format.md`

### Existing files to modify

- `cuda/include/gnni/gcn_cuda.hpp` — include shared graph types instead of defining them.
- `cuda/src/gcn_cuda.cu` — remove moved `DenseMatrix` constructors.
- `cuda/CMakeLists.txt` — add the host library, executables, tests and compiler warnings.
- `cuda/README.md` — document build, generation, conversion, runner and expected output.
- `cuda/.gitignore` — ignore build products, downloaded archives, generated datasets, sidecars and temporary files.
- `cuda/tests/check_cuda_project.sh` — assert the new target/file structure without accepting generated artifacts.

---

## Task 1: Extract CUDA-independent graph types

**Files:**

- Create: `cuda/include/gnni/graph_types.hpp`
- Create: `cuda/src/graph_types.cpp`
- Create: `cuda/tests/graph_types_test.cpp`
- Modify: `cuda/include/gnni/gcn_cuda.hpp`
- Modify: `cuda/src/gcn_cuda.cu`
- Modify: `cuda/CMakeLists.txt`

- [ ] **Step 1: Write the failing core-type regression test**

  Test valid zero-filled construction, construction from supplied values, length mismatch, and `rows * cols` overflow. The essential assertions are:

  ```cpp
  gnni::DenseMatrix zeros(2, 3);
  expect(zeros.values == std::vector<float>(6, 0.0F));
  expect_throw([] { gnni::DenseMatrix(2, 2, {1.0F}); });
  expect_throw([] {
    gnni::DenseMatrix(std::numeric_limits<std::size_t>::max(), 2);
  });
  ```

- [ ] **Step 2: Configure and prove the new test is not yet buildable**

  Run:

  ```bash
  cmake -S cuda -B cuda/build -DBUILD_TESTING=ON -DCMAKE_BUILD_TYPE=Debug
  cmake --build cuda/build --target graph_types_test
  ```

  Expected: failure because `graph_types.hpp`, `graph_types.cpp`, or the target does not exist.

- [ ] **Step 3: Move the types without changing the CUDA API**

  Put these declarations in `graph_types.hpp`:

  ```cpp
  struct CSRGraph {
    int num_vertices = 0;
    std::vector<int> row_offsets;
    std::vector<int> column_indices;
  };

  struct DenseMatrix {
    std::size_t rows = 0;
    std::size_t cols = 0;
    std::vector<float> values;
    DenseMatrix() = default;
    DenseMatrix(std::size_t rows, std::size_t cols);
    DenseMatrix(std::size_t rows, std::size_t cols, std::vector<float> values);
  };
  ```

  Implement checked multiplication in `graph_types.cpp`. Make `gcn_cuda.hpp` include this header, remove the old duplicate declarations and constructor definitions, and introduce `gnni_dataset` with `graph_types.cpp`. Link `gnni_cuda PUBLIC gnni_dataset`.

- [ ] **Step 4: Run focused and existing CUDA regression tests**

  ```bash
  cmake --build cuda/build --target graph_types_test gcn_cuda_test
  ctest --test-dir cuda/build --output-on-failure -R 'graph_types|gcn_cuda_correctness'
  ```

  Expected: both tests pass and downstream includes of `gcn_cuda.hpp` remain source-compatible.

- [ ] **Step 5: Commit the extraction**

  ```bash
  git add cuda/include/gnni/graph_types.hpp cuda/include/gnni/gcn_cuda.hpp cuda/src/graph_types.cpp cuda/src/gcn_cuda.cu cuda/tests/graph_types_test.cpp cuda/CMakeLists.txt
  git commit -m "refactor: extract host graph types"
  ```

## Task 2: Add the validated host dataset model

**Files:**

- Create: `cuda/include/gnni/dataset.hpp`
- Create: `cuda/src/dataset.cpp`
- Create: `cuda/tests/dataset_validation_test.cpp`
- Modify: `cuda/CMakeLists.txt`

- [ ] **Step 1: Write failing validation tests**

  Cover a valid directed dataset, a valid symmetric dataset, zero-edge CSR, malformed offset count, nonzero first offset, decreasing offsets, final-offset mismatch, unsorted/duplicate/out-of-range columns, asymmetric undirected input, wrong feature rows, zero feature width, NaN/Inf, label count/range, split-without-label, invalid split byte, class-count inconsistency, and self-loop-flag mismatch.

- [ ] **Step 2: Define the stable public model**

  Use this interface in `dataset.hpp`:

  ```cpp
  enum class NodeSplit : std::uint8_t {
    Unassigned = 0, Train = 1, Validation = 2, Test = 3
  };

  struct DatasetMetadata {
    bool directed = false;
    bool self_loops = false;
    std::uint64_t num_classes = 0;
  };

  struct Dataset {
    CSRGraph graph;
    DenseMatrix features;
    std::vector<std::int32_t> labels;
    std::vector<NodeSplit> splits;
    DatasetMetadata metadata;
  };

  void validate_dataset(const Dataset& dataset);
  std::uint64_t dataset_host_bytes(const Dataset& dataset);
  ```

- [ ] **Step 3: Implement one strict semantic validator**

  Make validation reusable by both `save_dataset()` and tests. Error messages must identify the violated invariant, for example `CSR row 7 is not strictly increasing` or `undirected CSR is missing reverse entry (3,9)`. Use checked arithmetic for byte accounting.

- [ ] **Step 4: Build and run focused tests**

  ```bash
  cmake --build cuda/build --target dataset_validation_test
  ctest --test-dir cuda/build --output-on-failure -R dataset_validation
  ```

- [ ] **Step 5: Commit the model**

  ```bash
  git add cuda/include/gnni/dataset.hpp cuda/src/dataset.cpp cuda/tests/dataset_validation_test.cpp cuda/CMakeLists.txt
  git commit -m "feat: add validated host dataset model"
  ```

## Task 3: Implement strict version 1 binary I/O

**Files:**

- Create: `cuda/include/gnni/dataset_io.hpp`
- Create: `cuda/src/dataset_io.cpp`
- Create: `cuda/tests/dataset_io_test.cpp`
- Modify: `cuda/CMakeLists.txt`

- [ ] **Step 1: Write failing round-trip and golden-byte tests**

  Test all four valid optional-section combinations: neither labels nor splits, labels only, labels plus splits, and directed/undirected flags. Include a zero-edge graph. For a one-vertex fixture, compare the first 64 bytes byte-for-byte to a literal header containing `GNNIDAT1`, version `1`, header length `64`, and the expected payload byte count.

- [ ] **Step 2: Write failing corruption and overwrite tests**

  Mutate one property at a time: magic, version, header length, unknown flag, reserved word, split-without-label flag, N/E above `INT_MAX`, payload overflow, truncated section, trailing byte, malformed disk CSR, invalid label/split and existing output with `overwrite=false`.

- [ ] **Step 3: Implement field-wise little-endian I/O**

  Expose this API and constants:

  ```cpp
  inline constexpr std::uint32_t kDatasetFormatVersion = 1;
  inline constexpr std::uint32_t kDatasetHeaderBytes = 64;
  inline constexpr std::uint32_t kDirectedFlag = 1U << 0U;
  inline constexpr std::uint32_t kSelfLoopsFlag = 1U << 1U;
  inline constexpr std::uint32_t kLabelsFlag = 1U << 2U;
  inline constexpr std::uint32_t kSplitsFlag = 1U << 3U;

  Dataset load_dataset(const std::filesystem::path& path);
  void save_dataset(const std::filesystem::path& path,
                    const Dataset& dataset,
                    bool overwrite = false);
  ```

  Read/write each integer and float explicitly; do not serialize a C++ struct. Write to `<name>.tmp-<pid>`, flush and close, then rename. On failure, remove only that exact temporary file.

- [ ] **Step 4: Run I/O tests under sanitizers where supported**

  ```bash
  cmake --build cuda/build --target dataset_io_test
  ctest --test-dir cuda/build --output-on-failure -R dataset_io
  ```

- [ ] **Step 5: Commit binary I/O**

  ```bash
  git add cuda/include/gnni/dataset_io.hpp cuda/src/dataset_io.cpp cuda/tests/dataset_io_test.cpp cuda/CMakeLists.txt
  git commit -m "feat: add versioned dataset binary io"
  ```

## Task 4: Implement deterministic feature and ER generation

**Files:**

- Create: `cuda/include/gnni/synthetic_graph.hpp`
- Create: `cuda/src/synthetic_graph.cpp`
- Create: `cuda/tests/synthetic_graph_test.cpp`
- Modify: `cuda/CMakeLists.txt`

- [ ] **Step 1: Write failing deterministic primitive tests**

  Fix exact golden values for the first eight `synthetic_feature(seed=42, v, f)` results and the bounded-integer sampler. Assert every feature is finite and in `[-1,1]`, and changing seed or coordinate changes the sequence.

- [ ] **Step 2: Write failing ER invariant tests**

  Assert `p=0` yields no logical edges, `p=1` yields the complete graph, average degree converts to `p=d/(n-1)`, fixed seed/options yield identical CSR, different seeds alter a nontrivial graph, rows are canonical and symmetric, and optional self-loops are exact. Reject invalid probabilities, degree, sizes and `N*F` overflow.

- [ ] **Step 3: Implement the generator API and ER skip algorithm**

  The public boundary is:

  ```cpp
  struct GeneratedGraph {
    CSRGraph graph;
    std::uint64_t logical_edges = 0;
  };

  GeneratedGraph generate_erdos_renyi(
      int vertices, double probability, std::uint64_t seed,
      bool self_loops);
  DenseMatrix generate_synthetic_features(
      int vertices, std::size_t feature_dim, std::uint64_t seed);
  Dataset make_synthetic_dataset(GeneratedGraph graph,
                                 std::size_t feature_dim,
                                 std::uint64_t feature_seed);
  ```

  Implement Batagelj-Brandes skipping for `0 < p < 1`; special-case `p=0` and `p=1` to avoid logarithm edge cases. Canonicalize once after edge generation.

- [ ] **Step 4: Run focused tests twice to prove repeatability**

  ```bash
  cmake --build cuda/build --target synthetic_graph_test
  ctest --test-dir cuda/build --output-on-failure -R synthetic_graph
  ctest --test-dir cuda/build --output-on-failure -R synthetic_graph
  ```

- [ ] **Step 5: Commit ER generation**

  ```bash
  git add cuda/include/gnni/synthetic_graph.hpp cuda/src/synthetic_graph.cpp cuda/tests/synthetic_graph_test.cpp cuda/CMakeLists.txt
  git commit -m "feat: generate deterministic er datasets"
  ```

## Task 5: Add BA and WS generation

**Files:**

- Modify: `cuda/include/gnni/synthetic_graph.hpp`
- Modify: `cuda/src/synthetic_graph.cpp`
- Modify: `cuda/tests/synthetic_graph_test.cpp`

- [ ] **Step 1: Write failing BA tests**

  Validate exact edge count `m*(m+1)/2 + m*(n-m-1)`, the initial `m+1` clique, distinct targets per new vertex, symmetry/canonical rows, determinism and invalid `m`. Use a small golden CSR to freeze the explicit RNG-to-index mapping.

- [ ] **Step 2: Implement BA preferential attachment**

  Add:

  ```cpp
  GeneratedGraph generate_barabasi_albert(
      int vertices, int edges_per_new_vertex,
      std::uint64_t seed, bool self_loops);
  ```

  Maintain an endpoint table containing each vertex once per current degree. Sample with the project bounded-integer helper and reject targets already selected by the new vertex. Append new edge endpoints only after selecting all distinct targets so one new edge does not bias the remaining selections.

- [ ] **Step 3: Write failing WS tests**

  Assert `beta=0` is the exact ring lattice, fixed-seed `beta=1` is deterministic, logical edge count is always `n*k/2`, no duplicate/self edge is created before optional loops, degree input is even, and invalid `beta`, `k=0`, odd `k`, or `k>=n` fails.

- [ ] **Step 4: Implement WS rewiring and rerun the full generator suite**

  Add:

  ```cpp
  GeneratedGraph generate_watts_strogatz(
      int vertices, int neighbors, double rewire_probability,
      std::uint64_t seed, bool self_loops);
  ```

  Visit each clockwise lattice edge once. When rewiring, sample until the replacement is neither the source nor already adjacent; update the adjacency set atomically.

  Run:

  ```bash
  cmake --build cuda/build --target synthetic_graph_test
  ctest --test-dir cuda/build --output-on-failure -R synthetic_graph
  ```

- [ ] **Step 5: Commit BA and WS generation**

  ```bash
  git add cuda/include/gnni/synthetic_graph.hpp cuda/src/synthetic_graph.cpp cuda/tests/synthetic_graph_test.cpp
  git commit -m "feat: generate ba and ws datasets"
  ```

## Task 6: Build the synthetic dataset CLI and provenance sidecar

**Files:**

- Create: `cuda/apps/dataset_generator.cpp`
- Create: `cuda/tests/check_dataset_generator.py`
- Modify: `cuda/CMakeLists.txt`
- Modify: `cuda/.gitignore`

- [ ] **Step 1: Write a failing black-box CLI test**

  The Python test receives `$<TARGET_FILE:gnni_dataset_generator>` and `$<TARGET_FILE:dataset_io_test>`, creates a temporary directory, and checks `er`, `ba`, and `ws`. It must verify successful files/sidecars, identical SHA-256 for repeated commands, changed output for a changed seed, refusal to overwrite, `--force`, invalid/missing/duplicate options, and no orphan `.tmp-*` file.

- [ ] **Step 2: Implement explicit subcommand parsing**

  Support these exact forms:

  ```text
  gnni_dataset_generator er --vertices N (--probability P|--average-degree D) --features F --seed S [--self-loops] --output FILE [--force]
  gnni_dataset_generator ba --vertices N --edges-per-new-vertex M --features F --seed S [--self-loops] --output FILE [--force]
  gnni_dataset_generator ws --vertices N --neighbors K --rewire-probability B --features F --seed S [--self-loops] --output FILE [--force]
  ```

  Reject ambiguous ER probability options and any option unused by the chosen model. No silent defaults for vertices, features, seed, or model parameters.

- [ ] **Step 3: Write JSON provenance without adding a library**

  Emit valid UTF-8 JSON with escaped strings, model parameters, seed, logical edge count, CSR entries, direction, self-loop policy, format version and output byte count. Write `<dataset>.json` atomically after the binary succeeds; with `--force`, preserve the old pair until both new temporary outputs are ready.

- [ ] **Step 4: Build and run the CLI test**

  ```bash
  cmake --build cuda/build --target gnni_dataset_generator dataset_io_test
  ctest --test-dir cuda/build --output-on-failure -R dataset_generator
  ```

- [ ] **Step 5: Commit the CLI**

  ```bash
  git add cuda/apps/dataset_generator.cpp cuda/tests/check_dataset_generator.py cuda/CMakeLists.txt cuda/.gitignore
  git commit -m "feat: add synthetic dataset cli"
  ```

## Task 7: Implement the Python format writer and safe archive utilities

**Files:**

- Create: `cuda/tools/gnni_format.py`
- Create: `cuda/tools/prepare_public_dataset.py`
- Create: `cuda/tests/test_public_converter.py`
- Modify: `cuda/CMakeLists.txt`

- [ ] **Step 1: Write failing Python unit tests for binary compatibility**

  Build a tiny directed dataset in Python, write it, inspect its exact 64-byte header with `struct.unpack('<8sIIIIQQQQQ', header)`, and invoke:

  ```text
  dataset_io_test --verify-python-fixture FILE N E F LABELS SPLITS
  ```

  Add that verification mode to the C++ test executable without weakening its normal self-tests.

- [ ] **Step 2: Write failing archive-safety and atomicity tests**

  Create local miniature `.tar.gz` and `.zip` files in `tempfile.TemporaryDirectory`. Reject absolute paths, `..` traversal, symlinks, hardlinks and destination escape. Test `file://`/`--archive` input, cache reuse, SHA-256 calculation, existing-output refusal, forced replacement and cleanup after a deliberately malformed row.

- [ ] **Step 3: Implement the dependency-free Python primitives**

  `gnni_format.py` must expose:

  ```python
  def canonical_csr(num_vertices: int, edges: Iterable[tuple[int, int]],
                    make_undirected: bool,
                    self_loops: bool) -> tuple[list[int], list[int]]:
      rows = [set() for _ in range(num_vertices)]
      for source, destination in edges:
          if not 0 <= source < num_vertices or not 0 <= destination < num_vertices:
              raise ValueError("edge endpoint is outside the vertex range")
          rows[destination].add(source)
          if make_undirected:
              rows[source].add(destination)
      if self_loops:
          for vertex, row in enumerate(rows):
              row.add(vertex)
      columns: list[int] = []
      row_offsets = [0]
      for row in rows:
          columns.extend(sorted(row))
          row_offsets.append(len(columns))
      return row_offsets, columns

  def write_dataset(path: pathlib.Path, *, directed: bool, self_loops: bool,
                    row_offsets: Sequence[int], columns: Sequence[int],
                    features: Iterable[Sequence[float]], num_classes: int,
                    labels: Sequence[int] | None,
                    splits: Sequence[int] | None, force: bool) -> int:
      header = build_validated_header(
          directed=directed, self_loops=self_loops,
          row_offsets=row_offsets, columns=columns,
          feature_rows=features, num_classes=num_classes,
          labels=labels, splits=splits)
      return atomic_write_payload(path, header, row_offsets, columns,
                                  features, labels, splits, force)
  ```

  `build_validated_header()` owns size/flag validation and returns the packed
  64-byte header. `atomic_write_payload()` writes each section in the fixed
  format order and returns the final file size.

  The implementation may stream features/labels rather than retaining ogbn-arxiv twice, but it must precompute and verify the exact payload size. Use `array` only after checking/byteswapping host endianness, or pack explicit little-endian chunks.

- [ ] **Step 4: Implement safe download/extraction helpers**

  Use `urllib.request`, `hashlib`, `tempfile`, `tarfile`, `zipfile`, `gzip`, `csv` and `pathlib`. Download to a cache temporary file, hash while streaming, `os.replace` only after success, and extract only regular files beneath the resolved temporary root.

- [ ] **Step 5: Run cross-language and safety tests, then commit**

  ```bash
  cmake --build cuda/build --target dataset_io_test
  ctest --test-dir cuda/build --output-on-failure -R 'dataset_io|public_converter'
  git add cuda/tools/gnni_format.py cuda/tools/prepare_public_dataset.py cuda/tests/test_public_converter.py cuda/tests/dataset_io_test.cpp cuda/CMakeLists.txt
  git commit -m "feat: add safe public dataset conversion core"
  ```

## Task 8: Add Cora conversion

**Files:**

- Modify: `cuda/tools/prepare_public_dataset.py`
- Modify: `cuda/tests/test_public_converter.py`

- [ ] **Step 1: Build a failing miniature Cora fixture test**

  Package local `cora/cora.content` and `cora/cora.cites` fixtures with shuffled string IDs, seven sparse binary feature columns, two textual classes, a duplicate citation and an unknown citation ID case. Verify deterministic sorted ID/class mapping, destination-row CSR semantics, duplicate removal, direction preservation, `--make-undirected`, `--self-loops`, absent splits, and exact labels/features.

- [ ] **Step 2: Implement the Cora parser and CLI**

  Support:

  ```text
  python3 cuda/tools/prepare_public_dataset.py cora --output FILE [--archive FILE|--url URL] [--cache-dir DIR] [--keep-cache] [--make-undirected] [--self-loops] [--force]
  ```

  Default URL is `https://linqs-data.soe.ucsc.edu/public/lbc/cora.tgz`. For the official archive require 2,708 nodes, 1,433 features, 7 classes and 5,429 citation rows before deduplication; fixture mode records fixture counts but does not pretend they are official.

- [ ] **Step 3: Implement and verify Cora provenance**

  The sidecar includes source URL/path, source SHA-256, original citation row count, stored CSR entries, transform flags, format version, output byte count and output SHA-256. It explicitly records `splits: absent_from_linqs_archive`.

- [ ] **Step 4: Run the Cora fixture tests**

  ```bash
  ctest --test-dir cuda/build --output-on-failure -R public_converter
  ```

- [ ] **Step 5: Commit Cora conversion**

  ```bash
  git add cuda/tools/prepare_public_dataset.py cuda/tests/test_public_converter.py
  git commit -m "feat: convert cora public dataset"
  ```

## Task 9: Add ogbn-arxiv conversion and official split preservation

**Files:**

- Modify: `cuda/tools/prepare_public_dataset.py`
- Modify: `cuda/tests/test_public_converter.py`

- [ ] **Step 1: Build a failing miniature OGB archive test**

  Create a local zip with the official relative layout:

  ```text
  ogbn_arxiv/raw/edge.csv.gz
  ogbn_arxiv/raw/node-feat.csv.gz
  ogbn_arxiv/raw/node-label.csv.gz
  ogbn_arxiv/split/time/train.csv.gz
  ogbn_arxiv/split/time/valid.csv.gz
  ogbn_arxiv/split/time/test.csv.gz
  ogbn_arxiv/RELEASE_v1.txt
  ```

  Verify directed destination-row CSR, 128-like fixture feature rows, labels, mutually exclusive complete splits, optional undirecting/self-loops, release marker checks, out-of-range node IDs, duplicate/missing split IDs and row-count failures.

- [ ] **Step 2: Implement streaming CSV/GZip parsing**

  Support:

  ```text
  python3 cuda/tools/prepare_public_dataset.py ogbn-arxiv --output FILE [--archive FILE|--url URL] [--cache-dir DIR] [--keep-cache] [--make-undirected] [--self-loops] [--force]
  ```

  Default URL is `http://snap.stanford.edu/ogb/data/nodeproppred/arxiv.zip`. Read edges as `(source,destination)` and store source in the destination row. Read node features row-by-row and reject non-finite or non-128-wide official rows.

- [ ] **Step 3: Enforce official release metadata**

  For a downloaded/default official archive require release v1, 169,343 nodes, 1,166,243 input directed edges, 128 features and 40 classes. Convert split index lists into one byte per node (`1/2/3`), rejecting overlap or omission. Local test fixtures use an explicit internal fixture flag available only to the test harness.

- [ ] **Step 4: Verify provenance and run converter tests**

  Sidecar fields mirror Cora and additionally report each split count and `release: 1`. Run:

  ```bash
  ctest --test-dir cuda/build --output-on-failure -R public_converter
  ```

- [ ] **Step 5: Commit ogbn-arxiv conversion**

  ```bash
  git add cuda/tools/prepare_public_dataset.py cuda/tests/test_public_converter.py
  git commit -m "feat: convert ogbn arxiv dataset"
  ```

## Task 10: Build the file-backed CUDA dataset runner

**Files:**

- Create: `cuda/apps/gcn_dataset_runner.cu`
- Create: `cuda/tests/check_dataset_runner.py`
- Modify: `cuda/CMakeLists.txt`

- [ ] **Step 1: Write a failing black-box runner test**

  Generate a tiny `.gnni` fixture, execute the new runner, parse every output line as one unique `key=value`, and assert metadata, positive finite timing/throughput values, output dimensions and finite checksum. Also test bad width lists, zero iterations, missing dataset and corrupted dataset.

- [ ] **Step 2: Implement strict options and deterministic weights**

  Support exactly:

  ```text
  gcn_dataset_runner --dataset FILE --layer-widths 64,32 [--warmup N] [--iterations N]
  ```

  Defaults are warmup `5` and iterations `20`. Each layer uses `SymmetricGcn`; hidden layers use ReLU and the final layer uses no activation. Derive each FP32 weight from `(layer,row,col)` with the same fixed 64-bit mixer so results do not depend on RNG library distributions.

- [ ] **Step 3: Measure the complete pipeline**

  Time `load_dataset()` with `steady_clock`, construct one `CudaGcnContext`, warm up, then average existing CUDA timing fields over measured iterations. Emit exactly these required keys, once each: `dataset`, `directed`, `self_loops`, `vertices`, `csr_entries`, `feature_dim`, `labels_present`, `splits_present`, `load_ms`, `host_bytes`, `graph_h2d_ms`, `avg_h2d_ms`, `avg_kernel_ms`, `avg_d2h_ms`, `avg_cuda_pipeline_ms`, `avg_end_to_end_ms`, `nodes_per_second`, `csr_entries_per_second`, `output_rows`, `output_cols`, and `output_checksum`. Boolean values are `0` or `1`; counts are decimal integers; timings, throughput and checksum are finite decimal floating-point values.

- [ ] **Step 4: Build and run the GPU smoke test**

  ```bash
  cmake --build cuda/build --target gcn_dataset_runner gnni_dataset_generator
  ctest --test-dir cuda/build --output-on-failure -R dataset_runner
  ```

- [ ] **Step 5: Commit the runner**

  ```bash
  git add cuda/apps/gcn_dataset_runner.cu cuda/tests/check_dataset_runner.py cuda/CMakeLists.txt
  git commit -m "feat: run cuda gcn on imported datasets"
  ```

## Task 11: Document the format and end-to-end workflows

**Files:**

- Create: `cuda/docs/dataset-format.md`
- Modify: `cuda/README.md`
- Modify: `cuda/.gitignore`
- Modify: `cuda/tests/check_cuda_project.sh`
- Modify: `cuda/CMakeLists.txt`

- [ ] **Step 1: Add a failing structure/documentation check**

  Extend `check_cuda_project.sh` to require the library, generator, converter, runner, format document and tests, and to reject tracked `.gnni`, downloaded archives, cache directories, `.tmp-*`, `__pycache__`, and in-source build outputs.

- [ ] **Step 2: Write the binary format reference**

  Document exact offsets, types, flags, payload formulas, message direction, canonical CSR requirements, label/split encodings, limits, error behavior and a worked one-vertex byte-layout example. The document must be sufficient to build another compatible writer.

- [ ] **Step 3: Write reproducible user workflows**

  Add copy-paste commands for release configuration, all three synthetic families at 10k/100k/1m scale, Cora, ogbn-arxiv, the runner, output interpretation, disk/GPU sizing, cache behavior, `--force`, and the explicit inference-not-accuracy boundary.

- [ ] **Step 4: Run all non-download tests**

  ```bash
  cmake -S cuda -B cuda/build-release -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=ON
  cmake --build cuda/build-release --parallel
  ctest --test-dir cuda/build-release --output-on-failure
  git diff --check
  ```

- [ ] **Step 5: Commit documentation and hygiene rules**

  ```bash
  git add cuda/docs/dataset-format.md cuda/README.md cuda/.gitignore cuda/tests/check_cuda_project.sh cuda/CMakeLists.txt
  git commit -m "docs: add dataset pipeline workflows"
  ```

## Task 12: Run full end-to-end acceptance and clean the tree

**Files:**

- Modify only if verification exposes a defect in the files above.
- Generate ignored artifacts under: `cuda/data/`
- Record stable benchmark evidence in: `cuda/docs/2026-08-15-dataset-pipeline-results.md`

- [ ] **Step 1: Generate and run all synthetic families**

  Use explicit seeds and self-loops:

  ```bash
  cuda/build-release/gnni_dataset_generator er --vertices 10000 --average-degree 16 --features 64 --seed 20260815 --self-loops --output cuda/data/er-10k.gnni
  cuda/build-release/gnni_dataset_generator ba --vertices 100000 --edges-per-new-vertex 8 --features 64 --seed 20260815 --self-loops --output cuda/data/ba-100k.gnni
  cuda/build-release/gnni_dataset_generator ws --vertices 1000000 --neighbors 16 --rewire-probability 0.1 --features 64 --seed 20260815 --self-loops --output cuda/data/ws-1m.gnni
  cuda/build-release/gcn_dataset_runner --dataset cuda/data/er-10k.gnni --layer-widths 64,32 --warmup 5 --iterations 20
  cuda/build-release/gcn_dataset_runner --dataset cuda/data/ba-100k.gnni --layer-widths 64,32 --warmup 5 --iterations 20
  cuda/build-release/gcn_dataset_runner --dataset cuda/data/ws-1m.gnni --layer-widths 64,32 --warmup 5 --iterations 20
  ```

  Regenerate one dataset to a second temporary path and compare SHA-256 hashes.

- [ ] **Step 2: Download, convert, reload and run Cora**

  ```bash
  python3 cuda/tools/prepare_public_dataset.py cora --self-loops --make-undirected --output cuda/data/cora.gnni
  cuda/build-release/gcn_dataset_runner --dataset cuda/data/cora.gnni --layer-widths 64,32 --warmup 5 --iterations 20
  ```

  Confirm 2,708 vertices, 1,433 features, 7 classes, labels present, splits absent, finite output and positive timing values.

- [ ] **Step 3: Download, convert, reload and run ogbn-arxiv**

  ```bash
  python3 cuda/tools/prepare_public_dataset.py ogbn-arxiv --self-loops --make-undirected --output cuda/data/ogbn-arxiv.gnni
  cuda/build-release/gcn_dataset_runner --dataset cuda/data/ogbn-arxiv.gnni --layer-widths 64,32 --warmup 5 --iterations 20
  ```

  Confirm 169,343 vertices, 128 features, 40 classes, labels/splits present, finite output and positive timing values. Verify sidecar source/output hashes.

- [ ] **Step 4: Run final correctness and memory checks**

  ```bash
  ctest --test-dir cuda/build-release --output-on-failure
  compute-sanitizer --tool memcheck cuda/build-release/gcn_dataset_runner --dataset cuda/data/er-10k.gnni --layer-widths 16,8 --warmup 1 --iterations 1
  git diff --check
  git status --short --ignored
  ```

  Generated datasets, archives, caches and build directories must appear only as ignored items. No unexpected untracked file may remain.

- [ ] **Step 5: Record evidence and perform final review**

  Create `cuda/docs/2026-08-15-dataset-pipeline-results.md` containing environment, exact commands, dataset metadata, timings/checksums, full test result and compute-sanitizer result. Review all diffs for duplicated helpers, unclear names, unchecked conversions and stale comments. Then run:

  ```bash
  git add cuda/docs/2026-08-15-dataset-pipeline-results.md
  git commit -m "test: verify dataset pipeline end to end"
  git status --short
  ```

  Expected final status: empty output.

## Specification Coverage Check

- Three required synthetic models: Tasks 4–6 and acceptance Task 12.
- Public real datasets Cora and ogbn-arxiv: Tasks 8–9 and acceptance Task 12.
- One shared versioned format and cross-language compatibility: Tasks 2–3 and 7.
- Import into existing CUDA types: Tasks 1–3 and 10.
- Labels/splits preserved without false accuracy claims: Tasks 2, 8–10 and documentation Task 11.
- Safe downloads, archive extraction, atomic writes, provenance and overwrite rules: Tasks 3, 6–9.
- Determinism, graph semantics, overflow/corruption rejection and clean repository: covered by focused tests in Tasks 1–11 and final checks in Task 12.

