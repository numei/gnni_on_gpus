# GNN Dataset Pipeline Design

## 1. Objective

Extend the existing CUDA GCN baseline with a reproducible dataset pipeline that:

1. generates the three synthetic graph families named by the assignment;
2. downloads and converts representative public graph datasets;
3. stores every prepared dataset in one versioned, validated format;
4. imports that format into the existing `CSRGraph` and `DenseMatrix` types;
5. runs full-batch CUDA inference and reports stable, machine-readable results;
6. keeps generated data, caches, and temporary files out of version control.

The delivered pipeline must run without a Python package manager or third-party
Python packages. C++17, the C++ standard library, Python 3's standard library,
CUDA, and the project's existing CMake toolchain are the only runtime
dependencies.

## 2. Scope

### 2.1 Included datasets

Synthetic datasets:

- Erdős-Rényi `G(n,p)` random graphs;
- Barabási-Albert preferential-attachment graphs;
- Watts-Strogatz small-world graphs.

Public datasets:

- Cora from the LINQS `cora.tgz` distribution, used as a small functional
  public dataset;
- `ogbn-arxiv` from the official OGB node-property archive, used as the main
  realistic performance dataset.

### 2.2 Inference and accuracy boundary

The pipeline preserves labels and standard split information when the source
provides them. The first delivered runner measures inference with deterministic
preloaded weights and does not claim classification accuracy. Training and
fetching third-party pretrained model weights are outside this scope.

This boundary is reversible: a later classifier can reuse the stored labels,
splits, graph, and features without changing the on-disk format.

### 2.3 Non-goals

- GNN training or backpropagation;
- mini-batch sampling;
- `ogbn-papers100M`, which cannot fit the current three-workspace FP32 design
  on the available 12 GB GPU;
- embedding Python, PyTorch, PyG, OGB, NetworkX, or a JSON library in the C++
  runtime;
- silently changing graph direction, inserting self-loops, or normalizing
  features.

## 3. Architecture

The system has four deliberately separate components.

### 3.1 Dataset model and binary I/O library

`gnni_dataset` owns the host-side dataset representation, validation, binary
reader, and binary writer. It has no CUDA dependency. Its public boundary
contains:

- a graph and FP32 node-feature matrix compatible with the existing CUDA API;
- optional `int32` labels;
- optional one-byte node split values;
- graph flags and basic metadata;
- strict `load_dataset()` and `save_dataset()` functions.

CUDA code consumes the loaded `CSRGraph` and `DenseMatrix`; it does not know
whether the input was synthetic, Cora, or OGB.

### 3.2 Native synthetic generator

`gnni_dataset_generator` is a C++ executable backed by generator functions in
the dataset library. It creates one synthetic graph, deterministic FP32 node
features, and an optional self-loop per vertex, then writes one `.gnni` file.

Model-specific command-line parameters remain explicit instead of being hidden
behind one overloaded "degree" option:

- `er`: `--probability` or `--average-degree`;
- `ba`: `--edges-per-new-vertex`;
- `ws`: `--neighbors` and `--rewire-probability`.

Every command also accepts `--vertices`, `--features`, `--seed`, `--self-loops`,
and `--output`.

### 3.3 Standard-library public-data converter

`prepare_public_dataset.py` performs download, archive validation, safe
extraction, conversion, and provenance output using only Python's standard
library.

- Cora parsing reads `cora.content` and `cora.cites` from `cora.tgz`.
- `ogbn-arxiv` parsing reads the compressed raw edge, node-feature, label, and
  official split files from the OGB archive.
- Source string identifiers are deterministically remapped to contiguous
  integer IDs.
- The converter writes the same binary format as the C++ writer.
- Cross-language compatibility is verified by C++ loading files produced by
  the Python test fixture.

The converter downloads to a cache, extracts into a temporary directory, and
atomically replaces only a fully written output. `--keep-cache` is opt-in;
temporary extraction is always removed.

### 3.4 Dataset inference runner

`gcn_dataset_runner` loads a `.gnni` file, creates deterministic row-major
weights, constructs one `CudaGcnContext`, performs warm-up runs, and measures
repeated full-batch inference.

Layer widths are explicit, for example `--layer-widths 64,32`. The input width
comes from the dataset; every listed number is an output width for one GCN
layer. The runner reports:

- dataset path and stored graph flags;
- vertices, CSR entries, feature width, labels, and split availability;
- file load time and host dataset bytes;
- one-time graph upload time;
- average H2D, kernel, D2H, CUDA-pipeline, and end-to-end times;
- nodes/s and CSR entries/s;
- output shape and a deterministic checksum.

Output remains `key=value` text so it is both human-readable and scriptable.

## 4. Graph Semantics

### 4.1 CSR message direction

The stored CSR row for destination vertex `v` contains source vertices `u`
whose features contribute messages to `v`. A source edge `(u, v)` is therefore
stored as row `v`, column `u`.

This rule matches the existing aggregation kernel:

```text
aggregate[v] = sum(input[u] for u in CSR_row(v))
```

Every stored row is canonical: column indices are strictly increasing. The
format contains no duplicate `(row,column)` entries.

### 4.2 Directed and undirected modes

Public converters preserve source direction unless `--make-undirected` is
specified. Undirected conversion inserts both message directions, sorts each
CSR row, and removes duplicate entries.

Synthetic graph models generate simple undirected graphs. Every undirected
edge is stored as two CSR entries. Reported `csr_entries` therefore counts
directed storage entries, while provenance also records the logical undirected
edge count.

### 4.3 Self-loops

Self-loops are never implicit. `--self-loops` adds exactly one `(v,v)` entry per
vertex after deduplication. The runner reports the stored policy and does not
modify it. End-to-end GCN examples use undirected graphs with explicit
self-loops; directed source-preserving variants remain available for
experiments.

## 5. Synthetic Algorithms

All generators use `std::mt19937_64` and require an explicit seed in provenance.
The same executable, parameters, compiler-independent integer sampling rules,
and format version must reproduce the same graph and feature values.

### 5.1 Erdős-Rényi

Generate a simple undirected `G(n,p)` graph with the Batagelj-Brandes skip
method so expected work is `O(N + E)`, not `O(N²)`. If average degree `d` is
provided, compute `p = d / (n - 1)` after validating `0 <= d < n`.

### 5.2 Barabási-Albert

Start from a complete graph on `m + 1` vertices. Each new vertex selects `m`
distinct existing vertices with probability proportional to current degree.
The implementation maintains an endpoint sampling table and rejects duplicate
targets for the new vertex. Validate `1 <= m < n`.

### 5.3 Watts-Strogatz

Build an undirected ring lattice in which every vertex has even degree `k`.
Visit each clockwise lattice edge once and rewire its destination with
probability `beta`, rejecting self-loops and duplicate edges. Validate
`0 <= beta <= 1` and `0 < k < n` with even `k`.

### 5.4 Synthetic features

Generate each FP32 value directly from `(seed, vertex, feature)` through a
specified 64-bit mixing function and map it into `[-1,1]`. Stateless generation
avoids dependence on thread order and does not require a second random array.
Synthetic datasets contain no fabricated labels or splits.

## 6. Version 1 Binary Format

Files use little-endian encoding and the `.gnni` extension. Implementations
read and write individual fields; they never dump a padded C++ struct. Float
payloads use IEEE-754 binary32.

### 6.1 Header

The fixed header contains, in order:

| Field | Type | Meaning |
|---|---|---|
| magic | 8 bytes | ASCII `GNNIDAT1` |
| version | `uint32` | `1` |
| header_bytes | `uint32` | `64` for version 1 |
| flags | `uint32` | directed, self-loops, labels, splits |
| reserved | `uint32` | must be zero |
| num_vertices | `uint64` | `N` |
| num_csr_entries | `uint64` | `E` |
| feature_dim | `uint64` | `F` |
| num_classes | `uint64` | zero when labels are absent |
| payload_bytes | `uint64` | exact following byte count |

Unknown flag bits, versions, nonzero reserved fields, and inconsistent sizes
are rejected.

Flag assignments are fixed:

- bit 0: directed source graph;
- bit 1: exactly one explicit self-loop is stored for every vertex;
- bit 2: labels are present;
- bit 3: split values are present.

Bits 4 through 31 must be zero. The split flag requires the labels flag. A
cleared directed bit means the canonical CSR is symmetric.

### 6.2 Payload

The payload is contiguous and ordered as follows:

1. `uint64 row_offsets[N + 1]`;
2. `uint32 column_indices[E]`;
3. `float32 features[N * F]` in row-major order;
4. optional `int32 labels[N]`;
5. optional `uint8 splits[N]` where `0=unassigned`, `1=train`, `2=valid`, and
   `3=test`.

The reader verifies the exact file length and rejects trailing bytes. Disk
offsets are 64-bit, but the current CUDA `CSRGraph` uses signed `int`; loading
a dataset with `N` or `E` above `INT_MAX` fails with a descriptive error rather
than truncating.

### 6.3 Provenance sidecar

Each prepared dataset also receives `<dataset>.json`, written by the generating
tool and not required by the C++ loader. It records:

- logical dataset name and source URL or generator model;
- source archive SHA-256 when downloaded;
- generator/converter parameters and seed;
- original and stored graph counts;
- direction and self-loop transformations;
- binary format version and output byte count;
- output SHA-256 when written by the Python public-data converter.

The sidecar is provenance, not duplicated graph data.

## 7. Public Dataset Conversion

### 7.1 Cora

The converter maps paper IDs from `cora.content` to contiguous IDs, parses the
1,433 binary word features as FP32, maps the seven textual classes in sorted
order to integer labels, and parses citations from `cora.cites`.

The raw LINQS archive does not contain the Planetoid public split, so the split
field is absent. The converter must not invent a split and describe it as
standard. This does not affect the inference-performance objective.

### 7.2 ogbn-arxiv

The converter supports OGB `ogbn-arxiv` release v1 and validates its official
169,343 vertices, 1,166,243 directed source edges, 128-dimensional FP32 node
features, and 40 classes before applying graph transformations. It preserves
the official time-based train/validation/test split. If the archive layout,
release marker, or source counts differ from v1, conversion fails with an
actionable message instead of guessing.

## 8. Error Handling and Safety

- All count arithmetic is overflow-checked before allocation or byte-size
  calculation.
- Dataset loading validates CSR monotonicity, terminal offset, column ranges,
  strictly increasing columns within every row, feature shape, finite feature
  values, label ranges, split values, and flags.
- Writers use a sibling temporary file and rename it only after close and size
  verification; failed writes remove only their own temporary file.
- Archive extraction rejects absolute paths, `..` traversal, links, and members
  outside the temporary extraction root.
- Existing outputs are not overwritten unless `--force` is supplied.
- Download errors retain no partial final archive; cache entries are renamed
  into place only after a complete response and hash calculation.
- CLI errors go to stderr and return nonzero; successful machine-readable
  output goes to stdout.

## 9. Tests

Implementation follows test-driven development.

### 9.1 Binary I/O tests

- round-trip all optional field combinations;
- zero-edge graph;
- deterministic byte layout for a tiny golden dataset;
- Python-writer to C++-reader compatibility;
- bad magic, version, flags, reserved field, truncation, trailing data;
- overflowing counts, malformed CSR, invalid columns, NaN features, labels,
  and split values;
- refusal to overwrite without `--force`.

### 9.2 Generator tests

- exact determinism for a fixed seed;
- expected vertex counts and model invariants;
- no unintended self-loops, duplicates, or asymmetric stored edges;
- BA edge count and minimum attachment behavior;
- WS degree/rewiring constraints;
- ER empty and nonempty boundary probabilities;
- feature range and deterministic values.

### 9.3 Converter tests

Network-independent miniature Cora `.tgz` and OGB-style `.zip` fixtures verify
parsing, ID remapping, edge orientation, labels, splits, transformations, safe
extraction, hashes, cleanup, and error messages. Tests never download external
data.

### 9.4 Integration tests

- generate a small dataset with each synthetic model;
- load each generated file through the C++ reader;
- run the CUDA inference runner on at least one prepared synthetic fixture;
- convert miniature public fixtures and run their outputs;
- retain all existing GCN numerical correctness tests.

Full external end-to-end verification separately downloads and converts Cora
and `ogbn-arxiv`, then runs the CUDA runner. It is documented and executed for
delivery but is not part of ordinary offline CTest.

## 10. Project Layout and Hygiene

Planned tracked files:

```text
cuda/include/gnni/dataset.hpp
cuda/include/gnni/dataset_io.hpp
cuda/include/gnni/synthetic_graph.hpp
cuda/src/dataset_io.cpp
cuda/src/synthetic_graph.cpp
cuda/apps/dataset_generator.cpp
cuda/apps/gcn_dataset_runner.cu
cuda/tools/prepare_public_dataset.py
cuda/tests/dataset_io_test.cpp
cuda/tests/synthetic_graph_test.cpp
cuda/tests/test_public_converter.py
cuda/docs/dataset-format.md
```

The existing in-memory `gcn_cuda_benchmark` remains as a kernel-focused
microbenchmark; the dataset runner has the distinct role of measuring file
loading plus real prepared data. Shared option parsing and metric formatting
will be factored only where duplication actually appears.

Ignored paths include `cuda/data/`, download caches, temporary outputs, Python
bytecode, build directories, and generated result files. No downloaded archive
or generated dataset is committed. Small converter fixtures are created inside
test temporary directories and removed automatically.

## 11. End-to-End Commands

Representative commands after implementation:

```bash
cmake --fresh -S cuda -B cuda/build-release \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda-13.1/bin/nvcc \
  -DCMAKE_CUDA_ARCHITECTURES=120
cmake --build cuda/build-release --parallel
ctest --test-dir cuda/build-release --output-on-failure

./cuda/build-release/gnni_dataset_generator ba \
  --vertices 100000 --edges-per-new-vertex 8 \
  --features 64 --seed 42 --self-loops \
  --output cuda/data/ba-100k-f64.gnni

python3 cuda/tools/prepare_public_dataset.py cora \
  --make-undirected --self-loops \
  --output cuda/data/cora.gnni

python3 cuda/tools/prepare_public_dataset.py ogbn-arxiv \
  --make-undirected --self-loops \
  --output cuda/data/ogbn-arxiv.gnni

./cuda/build-release/gcn_dataset_runner \
  --dataset cuda/data/ogbn-arxiv.gnni \
  --layer-widths 64,32 --seed 42 \
  --warmup 5 --iterations 20
```

## 12. Delivery Verification and Acceptance

Completion requires evidence for every item below:

1. A clean fresh Release configure and build succeeds.
2. All old and new CTest tests pass.
3. The three synthetic generators produce deterministic, valid datasets.
4. At least one generated synthetic family is exercised at 10,000, 100,000,
   and 1,000,000 vertices; all three families are exercised end to end.
5. Full Cora and `ogbn-arxiv` archives are downloaded from recorded sources,
   converted, reloaded, and checked against expected metadata.
6. The CUDA runner completes on generated synthetic data, Cora, and
   `ogbn-arxiv`, producing nonnegative timings, finite output, positive
   throughput, and stable checksums for identical seeds.
7. Compute Sanitizer reports no memory errors for a representative imported
   dataset run when the installed tool supports the target GPU.
8. README and dataset-format documentation reproduce every required command,
   graph semantic choice, dependency, and data location.
9. `git status --short` contains only intentional source and documentation
   changes; generated datasets, caches, bytecode, temporary files, and build
   products remain ignored or are removed.
