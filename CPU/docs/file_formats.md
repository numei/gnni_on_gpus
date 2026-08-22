# Role A File Formats and Loader Contract

This document fixes version 1 of the binary graph, dense-matrix, and model files. They load into `CSRGraph`, `DenseMatrix`, and `Model` respectively.

## 1. General binary rules

- Integer fields use unsigned fixed-width types and little-endian encoding.
- Float payloads are IEEE-754 binary32 (`float32`).
- Dense data is row-major.
- Fields are encoded individually; C++ structs, `size_t`, and `std::vector` internals are never serialized.
- Reserved fields must be zero in version 1.
- Readers check arithmetic and exact expected file size before allocating payloads, and reject trailing bytes.

The suggested extensions are `.gnnigraph`, `.gnnimatrix`, and `.gnnimodel`. They are for user convenience only; file contents determine the format.

## 2. Graph: `.gnnigraph`

Magic bytes: `GNNIGRF\0`.

The fixed header is 48 bytes:

| Offset | Size | Type | Field |
|---:|---:|---|---|
| 0 | 8 | bytes | magic |
| 8 | 4 | `uint32` | version = 1 |
| 12 | 4 | `uint32` | header_bytes = 48 |
| 16 | 4 | `uint32` | flags |
| 20 | 4 | `uint32` | reserved = 0 |
| 24 | 8 | `uint64` | num_nodes |
| 32 | 8 | `uint64` | num_edges |
| 40 | 8 | `uint64` | edge_feature_dim |

Flags: bit 0 means directed; bit 1 means edge features are present; all other bits are reserved and zero.

Payload:

```text
row_ptr       (num_nodes + 1) * uint64
col_idx       num_edges * uint32
edge_features num_edges * edge_feature_dim * float32, only when enabled
```

CSR row `v` contains incoming neighbours. `col_idx[p] = u` in row `v` represents `(u -> v)`. `row_ptr` must be monotonic, start at zero, end at `num_edges`, and each `col_idx[p]` must be a valid node ID. Version 1 rejects explicit self-loops, duplicate neighbours, and non-increasing neighbour IDs within a row.

If edge features are present, `edge_features[p, ef]` belongs to the same CSR entry as `col_idx[p]`. If they are absent, `edge_feature_dim` must be zero.

## 3. Dense matrix: `.gnnimatrix`

Magic bytes: `GNNIMAT\0`.

| Offset | Size | Type | Field |
|---:|---:|---|---|
| 0 | 8 | bytes | magic |
| 8 | 4 | `uint32` | version = 1 |
| 12 | 4 | `uint32` | header_bytes = 48 |
| 16 | 4 | `uint32` | scalar_type = 1 (`float32`) |
| 20 | 4 | `uint32` | layout = 1 (row-major) |
| 24 | 8 | `uint64` | rows |
| 32 | 8 | `uint64` | cols |
| 40 | 8 | `uint64` | reserved = 0 |

Payload is `rows * cols` `float32` values, stored at `values[row * cols + col]`. Version 1 requires positive dimensions and finite payload values.

## 4. Model: `.gnnimodel`

Magic bytes: `GNNIMDL\0`.

| Offset | Size | Type | Field |
|---:|---:|---|---|
| 0 | 8 | bytes | magic |
| 8 | 4 | `uint32` | version = 1 |
| 12 | 4 | `uint32` | header_bytes = 48 |
| 16 | 4 | `uint32` | scalar_type = 1 (`float32`) |
| 20 | 4 | `uint32` | layout = 1 (row-major) |
| 24 | 4 | `uint32` | num_layers |
| 28 | 4 | `uint32` | flags = 0 |
| 32 | 4 | `uint32` | layer_descriptor_bytes = 24 |
| 36 | 4 | `uint32` | reserved_0 = 0 |
| 40 | 8 | `uint64` | reserved_1 = 0 |

The header is followed by `num_layers` descriptors:

| Relative offset | Size | Type | Field |
|---:|---:|---|---|
| 0 | 8 | `uint64` | input_dim |
| 8 | 8 | `uint64` | output_dim |
| 16 | 4 | `uint32` | activation: 0 Identity, 1 ReLU |
| 20 | 4 | `uint32` | reserved = 0 |

Then each layer's `input_dim * output_dim` finite `float32` weights appear consecutively. Weights have shape `F_in x F_out` in row-major order: `W[f, out] = values[f * F_out + out]`.

The model requires at least one layer, positive dimensions, ReLU on hidden layers, Identity on the final layer, and matching adjacent dimensions. Formal benchmark files must also satisfy:

```text
features.rows == graph.num_nodes
features.cols == model.layers[0].weights.rows
```

Version 1 has no bias, attention parameters, or other layer parameters. Adding them requires a new format version.

## 5. Loader API and errors

```cpp
CSRGraph load_graph(const std::filesystem::path& path,
                    InputFileFormat format = InputFileFormat::Auto);
DenseMatrix load_dense_matrix(const std::filesystem::path& path,
                              InputFileFormat format = InputFileFormat::Auto);
Model load_model(const std::filesystem::path& path,
                 InputFileFormat format = InputFileFormat::Auto);
```

The implemented binary loaders validate only their own file: magic, version, flags, shapes, exact file size, numeric finiteness, and the graph/model invariants described above. Cross-object shape consistency is an explicit requirement for official benchmark input, not a separate validation module in the current baseline.

Malformed or unreadable files raise `std::runtime_error` with file and field context. Binary input is implemented. Text fixture formats and full text/binary automatic detection are optional future work.
