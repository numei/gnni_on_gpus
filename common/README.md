# Common graph dataset module

This directory contains the CPU-only code shared by the C++ and CUDA
implementations: graph containers, dataset validation, the versioned `.gnni`
binary format, synthetic graph generators, and public-dataset converters.

## Build and test independently

From the repository root:

```bash
cmake -S common -B common/build-release -DCMAKE_BUILD_TYPE=Release
cmake --build common/build-release --parallel
ctest --test-dir common/build-release --output-on-failure
```

## Generate synthetic datasets

```bash
./common/build-release/gnni_dataset_generator er \
  --vertices 10000 --average-degree 16 --features 64 --seed 20260815 \
  --self-loops --output common/data/er-10k.gnni
./common/build-release/gnni_dataset_generator ba \
  --vertices 100000 --edges-per-new-vertex 8 --features 64 --seed 20260815 \
  --self-loops --output common/data/ba-100k.gnni
./common/build-release/gnni_dataset_generator ws \
  --vertices 1000000 --neighbors 16 --rewire-probability 0.1 \
  --features 64 --seed 20260815 --self-loops \
  --output common/data/ws-1m.gnni
```

## Convert public datasets

The converter uses only Python's standard library. Cora has labels but no
official split in the LINQS archive; ogbn-arxiv keeps its official time split.
Source edge directions are preserved unless `--make-undirected` is specified.

```bash
python3 common/tools/prepare_public_dataset.py cora \
  --make-undirected --self-loops --output common/data/cora.gnni
python3 common/tools/prepare_public_dataset.py ogbn-arxiv \
  --make-undirected --self-loops --keep-cache \
  --cache-dir common/data/cache --output common/data/ogbn-arxiv.gnni
```

Use `--archive FILE` for an existing official archive and `--force` to replace
an output. Every command writes a JSON provenance sidecar.

## `.gnni` version 1

The 64-byte little-endian header stores the magic `GNNIDAT1`, version, header
length, flags, a reserved word, vertex count, CSR entry count, feature width,
class count, and payload size. The payload contains `uint64` row offsets,
`uint32` column indices, row-major FP32 features, optional `int32` labels, and
optional one-byte split values (`0=none`, `1=train`, `2=valid`, `3=test`).

The loader validates flags, sizes, CSR structure, indices, finite features,
labels, splits, truncation, and trailing bytes.
