# Role A File Formats and Loader Contract

本文档冻结 version 1 的 graph、dense matrix 和 model 磁盘格式。它描述文件如何
转换为 `CSRGraph`、`DenseMatrix` 和 `Model`，不规定 CUDA 内部的 device layout。

## 1. 为什么使用三个文件

一次运行接收：

```text
--graph    graph file
--features dense feature matrix file
--model    model metadata and weights file
```

三者分开的原因：

- 相同 graph 可以重复测试不同 feature dimensions；
- 相同 graph/features 可以重复测试不同 model depths；
- graph topology 不必复制到每个 model 文件；
- B/C 只接收加载后的公共 C++ 对象，不需要解析这些文件；
- 文件局部错误和跨文件维度错误可以分别报告。

Version 1 不保存 labels。以后需要 node-classification accuracy 时，应定义独立
label/split 格式，而不是改变现有三个文件的语义。

## 2. 通用二进制规则

正式 benchmark 使用 versioned binary files。

- 所有整数使用 unsigned fixed-width types。
- 所有多字节值使用 little-endian。
- 浮点 payload 使用 IEEE-754 binary32。
- 稠密数据使用 row-major。
- Boolean information 使用 flags，不直接序列化 C++ `bool`。
- 文件字段逐个编码和解码，不直接读写 C++ struct。
- 不序列化 `size_t`、`std::vector` 或对象内存。
- Version 1 的 reserved fields 必须写零；reader 遇到非零值时拒绝文件。
- Reader 必须在 allocation 前进行 overflow 和 expected-file-size 检查。
- Version 1 不允许未定义 trailing bytes。

推荐扩展名只帮助用户识别文件，不用于判断文件内容：

```text
*.gnnigraph
*.gnnimatrix
*.gnnimodel
```

`InputFileFormat::Auto` 通过 magic/token 检测格式，不能只依赖扩展名。

## 3. Binary graph format

Magic bytes：

```text
G N N I G R F 0x00
```

固定 header 为 48 bytes：

| Offset | Size | Type | Field |
|---:|---:|---|---|
| 0 | 8 | bytes | magic = `GNNIGRF\0` |
| 8 | 4 | `uint32` | version = 1 |
| 12 | 4 | `uint32` | header_bytes = 48 |
| 16 | 4 | `uint32` | flags |
| 20 | 4 | `uint32` | reserved = 0 |
| 24 | 8 | `uint64` | num_nodes |
| 32 | 8 | `uint64` | num_edges |
| 40 | 8 | `uint64` | edge_feature_dim |

Flags：

```text
bit 0: directed
bit 1: has_edge_features
bits 2..31: reserved, must be zero
```

Payload 紧跟 header：

```text
row_ptr      (num_nodes + 1) * uint64
col_idx      num_edges * uint32
edge_features, only when bit 1 is set:
              num_edges * edge_feature_dim * float32
```

`edge_features[p, ef]` 描述的边与 `col_idx[p]` 是同一个 CSR entry。

一致性要求：

- `num_nodes > 0`；
- `num_edges` 可以为零；
- `row_ptr[0] == 0`；
- `row_ptr[num_nodes] == num_edges`；
- `row_ptr` 单调不下降；
- `col_idx[p] < num_nodes`；
- 不允许 self-loop 和 duplicate edge；
- 每一行 `col_idx` 严格递增；
- `has_edge_features == false` 时 `edge_feature_dim == 0`；
- `has_edge_features == true` 时 `edge_feature_dim > 0`。

无向图设置 `directed` bit 为 0，但 payload 中每条无向边仍存储两个有向 CSR
entries。

## 4. Binary dense-matrix format

Magic bytes：

```text
G N N I M A T 0x00
```

固定 header 为 48 bytes：

| Offset | Size | Type | Field |
|---:|---:|---|---|
| 0 | 8 | bytes | magic = `GNNIMAT\0` |
| 8 | 4 | `uint32` | version = 1 |
| 12 | 4 | `uint32` | header_bytes = 48 |
| 16 | 4 | `uint32` | scalar_type = 1 (`float32`) |
| 20 | 4 | `uint32` | layout = 1 (`row-major`) |
| 24 | 8 | `uint64` | rows |
| 32 | 8 | `uint64` | cols |
| 40 | 8 | `uint64` | reserved = 0 |

Payload：

```text
rows * cols * float32
```

位置 `(row, col)` 存在：

```text
payload[row * cols + col]
```

Version 1 要求 `rows > 0`、`cols > 0`，并拒绝 NaN 和 infinity。

Node features 直接使用这个格式。加载完 graph 和 features 后还必须执行跨对象检查：

```text
features.rows == graph.num_nodes
```

## 5. Binary model format

Magic bytes：

```text
G N N I M D L 0x00
```

固定 header 为 48 bytes：

| Offset | Size | Type | Field |
|---:|---:|---|---|
| 0 | 8 | bytes | magic = `GNNIMDL\0` |
| 8 | 4 | `uint32` | version = 1 |
| 12 | 4 | `uint32` | header_bytes = 48 |
| 16 | 4 | `uint32` | scalar_type = 1 (`float32`) |
| 20 | 4 | `uint32` | layout = 1 (`row-major`) |
| 24 | 4 | `uint32` | num_layers |
| 28 | 4 | `uint32` | flags = 0 |
| 32 | 4 | `uint32` | layer_descriptor_bytes = 24 |
| 36 | 4 | `uint32` | reserved_0 = 0 |
| 40 | 8 | `uint64` | reserved_1 = 0 |

Header 后先存储 `num_layers` 个 layer descriptors。每个 descriptor 为 24 bytes：

| Relative offset | Size | Type | Field |
|---:|---:|---|---|
| 0 | 8 | `uint64` | input_dim |
| 8 | 8 | `uint64` | output_dim |
| 16 | 4 | `uint32` | activation |
| 20 | 4 | `uint32` | reserved = 0 |

Activation 编码：

```text
0 = Identity
1 = ReLU
```

全部 descriptors 后，按 layer index 顺序连续存储权重：

```text
layer 0: input_dim[0] * output_dim[0] * float32
layer 1: input_dim[1] * output_dim[1] * float32
...
```

权重是 `F_in x F_out` row-major：

```text
W[f, out] -> W[f * F_out + out]
```

一致性要求：

- `num_layers > 0`；
- 每层 input/output dimension 都大于零；
- 相邻层 `output_dim[l] == input_dim[l + 1]`；
- 隐藏层 activation 为 ReLU；
- 最后一层 activation 为 Identity；
- 所有权重有限，不允许 NaN 和 infinity；
- 第一层 input dimension 在跨对象 validation 中与 `features.cols` 比较。

Version 1 没有 bias、attention parameters 或其他 layer parameters。增加这些内容
需要新的 model format version，而不能重新解释 version 1 的 reserved fields。

## 6. Text fixture formats

Text formats 只用于小型、可手算测试。Parser：

- 以 whitespace 分隔 token；
- 允许空行；
- `#` 到行末是 comment；
- key 顺序固定；
- 数量必须与 header 声明完全一致；
- 最后一个预期 token 后不允许额外非 comment token。

### 6.1 Graph text

```text
GNNI_GRAPH_TEXT 1
directed 1
num_nodes 3
num_edges 3
edge_feature_dim 0
row_ptr 0 0 2 3
col_idx 0 2 1
```

如果 `edge_feature_dim > 0`，在 `col_idx` 后增加：

```text
edge_features
<num_edges * edge_feature_dim float values>
```

### 6.2 Dense-matrix text

```text
GNNI_MATRIX_TEXT 1
rows 3
cols 2
data
1.0 2.0
3.0 4.0
5.0 6.0
```

换行只用于可读性；parser 按 token 数读取 `rows * cols` 个值。

### 6.3 Model text

```text
GNNI_MODEL_TEXT 1
num_layers 2

layer 0
input_dim 2
output_dim 3
activation relu
weights
1.0 0.0 0.5
0.0 1.0 0.5

layer 1
input_dim 3
output_dim 2
activation identity
weights
1.0 0.0
0.0 1.0
0.5 0.5
```

Layer index 必须从零连续增加。每个 `weights` 后必须恰好有
`input_dim * output_dim` 个值。

## 7. Loader API

公共声明位于 `include/gnni/io.hpp`：

```cpp
CSRGraph load_graph(
    const std::filesystem::path& path,
    InputFileFormat format = InputFileFormat::Auto);

DenseMatrix load_dense_matrix(
    const std::filesystem::path& path,
    InputFileFormat format = InputFileFormat::Auto);

Model load_model(
    const std::filesystem::path& path,
    InputFileFormat format = InputFileFormat::Auto);
```

Loader 执行 file-local validation：

- magic/token、version、header、flags；
- 文件长度和 checked arithmetic；
- graph 内部 CSR invariant；
- matrix/model 内部 shape；
- finite float values。

加载三个对象后，调用已有 validation API 做 cross-object validation：

```text
features.rows == graph.num_nodes
features.cols == model.layers[0].weights.rows
all adjacent model dimensions match
```

## 8. Error contract

以下情况抛出 `std::runtime_error`：

- 文件打不开或读取失败；
- magic/token 不匹配；
- 不支持的 version/dtype/layout；
- truncated file 或 unexpected trailing data；
- 文件声明的大小发生 arithmetic overflow；
- 文件内部结构或 payload 非法。

`validate_graph`、`validate_features`、`validate_model` 对已经存在的 C++ 对象发现
不变量错误时抛出 `std::invalid_argument`。

错误信息必须包含：

- 文件路径；
- 文件类型；
- 出错字段或数组；
- 可能时包含 layer/row/index；
- expected 与 actual value。

## 9. Loader implementation status and order

1. 已完成：little-endian fixed-width read helpers。
2. 已完成：checked addition/multiplication 和 file-size check。
3. 已完成：binary dense matrix loader 和 binary auto detection。
4. 下一步：binary graph loader。
5. 然后：binary model loader。
6. 然后：text token/comment reader。
7. 然后：三个 text loaders。
8. 最后：完整 auto detection。
9. 持续增加 valid、truncated、overflow 和 malformed-input tests。

在 binary loader 正确之前，不开始公共数据集的大规模转换。
