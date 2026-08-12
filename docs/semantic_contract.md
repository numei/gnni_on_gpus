# Role A Work Guide: CPU, Data, Correctness, CLI

本文档说明角色 A 需要完成什么、为什么这样设计，以及这些选择如何影响角色 B/C。
它既是角色 A 的实现路线，也是三个人对接时需要遵守的 host-side contract。

项目原始要求以 `problem.pdf` 为准。本文档记录的是我们在题目允许范围内做出的具体选择。

## 0. 当前准备状态

已经确定：

- 公共 host API 使用 C++17。
- 只实现一种必做 GNN 架构：mean-aggregation GCN。
- 只进行 full-batch inference，不实现训练和反向传播。
- 图使用 CSR，CSR 每一行存储目标节点的入邻居。
- mean aggregation 与 PDF 一致，只平均显式邻居，不添加 self-loop。
- 孤立节点的聚合结果定义为全零向量。
- 稠密特征、权重和输出使用 `float32`、row-major。
- 隐藏层使用 ReLU，最后一层使用 identity 并输出 logits/embeddings。
- 顺序 CPU 是全组唯一的 correctness reference。
- OpenMP 采用 node-parallel 路径。
- B/C 通过相同的公共 C++ API 接入，不在公共接口暴露 CUDA 类型。

已经创建公共接口和完整 Role A 模块骨架：

```text
include/gnni/types.hpp  公共数据结构和枚举
include/gnni/api.hpp    validation、comparison、backend 和 metrics 接口
include/gnni/io.hpp     graph、dense matrix 和 model loader 接口
```

具体待实现顺序和验收条件见 `docs/role_a_implementation_checklist.md`。除 binary
dense-matrix loader、底层 binary reader、公共 dispatch 框架和 CLI help 文本外，
带 `TODO(Role A)` 的函数仍是会明确抛出异常的占位实现。

尚未完成：

- 已冻结磁盘数据格式；binary dense-matrix loader 已完成，graph/model/text
  loaders 尚未实现。
- 输入 validation。
- 顺序 CPU inference。
- OpenMP inference。
- correctness comparison。
- CLI、backend dispatch 和 benchmark runner。
- 构建系统、自动化测试和公共数据集转换工具。

## 1. 我的责任范围

角色 A 负责项目的 host-side foundation 和数值正确性：

- 定义公共数据结构、内存布局和 backend API。
- 定义并实现图、特征、模型权重和可选 edge features 的加载方式。
- 验证输入数据的结构和维度。
- 实现顺序 CPU GCN inference。
- 实现 OpenMP 多核 CPU GCN inference。
- 生成 CPU reference output。
- 实现 CPU/OpenMP/CUDA 通用的误差比较。
- 实现统一 CLI 和 backend dispatch。
- 实现统一 benchmark 重复运行、统计和结果报告。
- 编写角色 A 相关的 README、运行参数和数据格式说明。

角色 A 不负责：

- CUDA device memory 的分配和释放。
- Host-to-device/device-to-host copy 的具体实现。
- Vertex-centric、edge-centric 或其他 CUDA kernel。
- Shared memory、atomic、warp reduction 等 GPU 优化。
- Nsight profiling 和 GPU bottleneck 分析。

角色 A 虽然声明 CUDA backend 的公共入口，但函数体由 B/C 实现。

## 2. Required baseline：我选择的 GNN

### 2.1 Full-batch inference

输入是一张静态图和所有节点的特征。每一层都处理整张图：

```text
graph + all node features + preloaded weights -> all node outputs
```

这里不包含：

- 训练；
- gradient；
- optimizer；
- 跨多张小图的 mini-batch；
- 动态修改图结构。

### 2.2 每层数据的含义

对于第 `l` 层：

- `N`：节点数量；
- `F_in`：当前层输入特征维度；
- `F_out`：当前层输出特征维度；
- `H(l)`：`N x F_in`，当前所有节点的表示；
- `M(l)`：`N x F_in`，邻居 mean aggregation 的结果；
- `W(l)`：`F_in x F_out`，预加载且在推理期间固定的权重；
- `Z(l)`：`N x F_out`，线性变换结果；
- `H(l+1)`：`N x F_out`，该层最终输出。

模型输入为：

```text
H(0) = X
```

其中 `X` 是加载的节点特征矩阵。

### 2.3 Mean aggregation

我们严格采用 PDF 展示的 mean aggregation，不自动添加 self-loop。

对节点 `v` 和特征 `f`：

```text
if in_degree(v) > 0:
    M[v, f] =
        sum(H[u, f] for u in incoming_neighbors(v))
        / in_degree(v)
else:
    M[v, f] = 0
```

这里：

- `incoming_neighbors(v)` 是所有存在边 `(u -> v)` 的节点 `u`；
- `in_degree(v)` 是入邻居数量；
- 节点自己的 `H[v,f]` 不会自动加入求和；
- 输入 CSR 不允许显式 self-loop；
- 零入度节点单独定义为零，避免除零。

这个零入度规则是 PDF 没有明确规定的边界选择。因为 baseline 没有 bias，
孤立节点从第一层开始会保持零输出。

### 2.4 Linear update

公共数据布局把节点表示看成 row vector，因此计算写成：

```text
Z[v, out] =
    sum(M[v, f] * W[f, out] for f in [0, F_in))
```

也就是矩阵形式：

```text
Z = M * W
```

PDF 使用列向量记法时可能写成 `W * m`。两者只是向量方向和权重存储约定不同，
并不是不同的神经网络运算。

### 2.5 Activation

对于隐藏层：

```text
H(l+1)[v, out] = max(0, Z(l)[v, out])
```

对于最后一层 `l = L - 1`：

```text
H(L) = Z(L-1)
```

最后一层不执行 ReLU，保留正负 logits/embeddings。Softmax 是可选的最终
post-processing，不属于必做 GCN layer。

### 2.6 为什么选择这条 baseline 路径

- 与 PDF 给出的 mean aggregation 直接一致。
- 算法简单，适合作为可信 CPU reference。
- CSR 上可以直接按目标节点遍历入邻居。
- OpenMP 可以按节点并行且不需要 atomic。
- B 的 vertex-centric CUDA 可以直接让一个线程或线程组处理一个节点。
- C 的 edge-centric CUDA 也可以明确识别 source 和 destination。
- Mean 相比 sum 减少不同度数造成的数值尺度差异。

## 3. Graph semantics：图到底怎样存

### 3.1 CSR 的含义

公共图结构由以下数据组成：

```text
num_nodes
num_edges
row_ptr[num_nodes + 1]
col_idx[num_edges]
```

CSR 第 `v` 行的边范围是：

```text
p in [row_ptr[v], row_ptr[v + 1])
```

真正的邻居 ID 存在：

```text
u = col_idx[p]
```

本项目规定该 entry 表示：

```text
edge (u -> v)
```

因此第 `v` 行存储的是节点 `v` 的入邻居，不是出邻居。

### 3.2 示例

假设有三条有向边：

```text
0 -> 1
2 -> 1
1 -> 2
```

对应 CSR：

```text
row_ptr = [0, 0, 2, 3]
col_idx = [0, 2, 1]
```

解释：

- row 0 是空范围，节点 0 没有入邻居；
- row 1 是 `col_idx[0:2] = [0, 2]`；
- row 2 是 `col_idx[2:3] = [1]`。

### 3.3 图必须满足的 invariant

- 节点 ID 连续，范围为 `[0, num_nodes)`。
- `row_ptr.size() == num_nodes + 1`。
- `row_ptr[0] == 0`。
- `row_ptr` 单调不下降。
- `row_ptr[num_nodes] == num_edges`。
- `col_idx.size() == num_edges`。
- 每个 `col_idx[p]` 都小于 `num_nodes`。
- 不允许显式 self-loop。
- 不允许重复边。
- 每一行的邻居 ID 应排序，保证确定性并方便查重。

对于无向图，每条边 `{u,v}` 存为两个 CSR entry：

```text
(u -> v)
(v -> u)
```

`CSRGraph::directed` 是图的元数据，但无向图进入 kernel 前已经展开为双向 entry。

### 3.4 对 B/C 的影响

- B 处理 row `v` 时，输出目标一定是 `v`。
- B 从 `col_idx[p]` 读取 source node `u` 的特征。
- C 按 edge 处理时，`source = col_idx[p]`，`destination = row containing p`。
- B/C 都不能把相同 CSR 解释成出邻居结构。
- B/C 不应在 GPU 端自行添加 self-loop。

## 4. Dense data and model layout

### 4.1 Numeric type

Baseline 的以下数据全部使用 `float32`：

- node features；
- layer weights；
- intermediate representations；
- final output。

CSR offset 使用 `uint64_t`，节点 ID 使用 `uint32_t`。

使用 `float32` 的原因：

- 与 CUDA 常见高吞吐路径一致；
- 内存占用低于 double；
- CPU/GPU 可以使用相同的数据；
- 允许通过 tolerance 处理并行归约造成的小误差。

### 4.2 Row-major

所有稠密矩阵使用 row-major：

```text
H[v, f]      -> H.values[v * H.cols + f]
W[f, out]    -> W.values[f * W.cols + out]
output[v,o]  -> output.values[v * output.cols + o]
```

每层必须满足：

```text
H.cols == W.rows
output.rows == H.rows
output.cols == W.cols
```

相邻层必须满足：

```text
layer[l].W.cols == layer[l+1].W.rows
```

### 4.3 对 B/C 的影响

- Host-to-device copy 按公共 vector 的连续存储执行。
- Kernel 必须按照上述索引读取输入和写回输出。
- B/C 可以在 device 内部转置或重排数据，但必须恢复为公共 row-major 输出。
- GPU 内部布局优化不能泄漏到 `RunResult` 的公共边界。

## 5. 我的 CPU 实现路径

### 5.1 Sequential CPU：正确性优先

每层明确分为三个阶段：

```text
1. mean aggregation: H -> M
2. linear update:    M * W -> Z
3. activation:       Z -> H_next
```

第一版不融合这些阶段，也不使用代数重排。原因是：

- 容易逐阶段检查；
- 容易生成可信中间值；
- CUDA 出错时容易判断错误发生在哪一步；
- 避免优化后的 CPU 代码与 GPU 同时犯同一种错误。

每层结束后交换 `H_current` 和 `H_next`。不同 GNN layers 之间存在依赖，
因此层本身按顺序执行。

一层的大致计算量是：

```text
aggregation: O(E * F_in)
linear:      O(N * F_in * F_out)
```

### 5.2 OpenMP：按节点并行

Aggregation 的选择路径：

```text
parallel for each destination node v
    compute all M[v, :]
```

每个线程只写一个目标节点对应的输出行，因此：

- 不需要 atomic；
- 不会发生多个线程写同一个 `M[v,f]`；
- CSR row 可以直接分配给线程。

Linear update 可以按节点和输出特征并行：

```text
parallel for each (v, out)
    compute Z[v, out]
```

第一版不使用 nested OpenMP parallelism。`static`、`dynamic` 和 `guided`
可以作为 benchmark 配置比较：

- `static` 开销低；
- `dynamic/guided` 更可能缓解高度数节点造成的 aggregation 负载不均；
- linear 阶段每个节点工作量相同，更适合 `static`。

OpenMP backend 必须先通过 sequential reference，再用于性能比较。

### 5.3 哪些 CPU 优化可以以后做

在 reference 正确之后，可以选择：

- 复用 layer buffers；
- 减少临时矩阵分配；
- 融合 aggregation 和 linear；
- 调整循环顺序；
- vectorization；
- 不同 OpenMP schedule。

这些属于角色 A 的可选 CPU 优化，不得改变公共语义和输出布局。

## 6. Data loading and validation path

### 6.1 已决定的 in-memory contract

Loader 最终必须生成：

```cpp
CSRGraph graph;
DenseMatrix features;
Model model;
```

B/C 只接收这些已加载的对象，不分别实现自己的 dataset parser。

### 6.2 已冻结的磁盘路径

具体字节布局和 text grammar 见 `docs/file_formats.md`。采用：

- versioned binary format：用于正式 benchmark 和大图；
- small text fixtures：用于可手算测试和调试。

二进制格式必须使用固定宽度类型并携带 shape/version 信息，不能直接序列化
`size_t` 或 `std::vector` 的内部表示。Graph、features 和 model 分成三个文件，
以便复用同一个 graph 测试不同 feature sizes 和 model depths。

### 6.3 Validation 必须检查

Graph：

- CSR 长度、起点、终点和单调性；
- vertex ID 范围；
- self-loop；
- duplicate edge；
- 可选 edge features 的 shape。

Features：

- 行数等于 `num_nodes`；
- storage size 与 shape 一致；
- 维度乘法没有整数溢出。

Model：

- 至少一层；
- 第一层 `F_in` 等于 features 的列数；
- 相邻层维度相容；
- 隐藏层 activation 是 ReLU；
- 最后一层 activation 是 Identity；
- weights storage size 与 shape 一致。

Validation 不静默修改输入。错误数据应抛出清楚的异常，而不是自动增加边、
删除节点或切换 backend。

### 6.4 对 B/C 的影响

- B/C 可以假设进入 backend 的 host objects 已通过公共 validation。
- B/C 仍负责检查 device memory、CUDA grid/index 和 device allocation。
- B/C 不得要求另一套 GPU-only 磁盘格式。
- 如果 B/C 需要内部 device format，应在 backend 内部完成转换。

## 7. Correctness path

### 7.1 Reference

顺序 CPU 是唯一 reference：

```text
CPU sequential output = expected result
```

OpenMP、B 的 CUDA、C 的 CUDA 都与它比较。

### 7.2 Comparison rule

浮点加法不满足结合律，不同线程归约顺序可能产生微小差异。因此不要求 bitwise
equality，而使用：

```text
abs(candidate - reference)
    <= atol + rtol * abs(reference)
```

初始默认值：

```text
atol = 1e-5
rtol = 1e-4
```

Comparison report 至少包含：

- shape 是否一致；
- maximum absolute error；
- maximum relative error；
- RMSE；
- mismatch count；
- first mismatch index。

### 7.3 测试图

角色 A 至少准备：

- 可手算的小图；
- 有向图，用于检查 CSR 方向；
- 无向图，用于检查双向 entry；
- 零入度节点；
- 不规则高度数节点；
- 特征维度不是常见线程块倍数的情况；
- 多层且每层 feature dimension 不同的模型；
- 固定 seed 的随机图、特征和权重。

### 7.4 对 B/C 的影响

- B/C 使用相同的 `compare_outputs`，不能各自定义不同真值。
- CUDA reduction 顺序可以不同，但必须通过 tolerance。
- 如果需要逐层排错，可以以后增加 intermediate-layer tracing API。
- CUDA comparison 失败时，A 提供 reference 和错误位置，B/C 修复自己的 backend。

## 8. CLI and backend integration

角色 A 维护唯一的程序入口。B/C 不需要各自编写独立 `main()`。

计划中的公共参数：

```text
--graph <path>
--features <path>
--model <path>
--backend cpu|omp|cuda-vertex|cuda-edge
--threads <n>
--omp-schedule default|static|dynamic|guided
--omp-chunk-size <n>
--warmup <n>
--repeat <n>
--verify
--atol <value>
--rtol <value>
--output logits|probabilities
```

公共 API 中的 backend 入口：

```text
run_cpu_sequential  -> Role A
run_cpu_openmp      -> Role A
run_cuda_vertex     -> Role B
run_cuda_edge       -> Role C or the agreed alternative CUDA backend
```

Dispatcher 必须：

- 检查 backend 是否在当前 build 中可用；
- 检查 requested optional capability；
- 不允许请求 CUDA 后静默改跑 CPU；
- 不允许请求 attention 后静默改跑 mean；
- 使用统一 `RunResult` 返回输出和 metrics。

## 9. Benchmark path

角色 A 控制：

- warm-up；
- repeated runs；
- min、mean、median、standard deviation；
- throughput 计算；
- human-readable/CSV 等统一报告；
- correctness 是否在计时区间外执行。

对 `L` 层和 compute time `t` 秒：

```text
nodes_per_second = num_nodes * L / t
edges_per_second = num_edges * L / t
```

这里统计的是 node-layer/edge-layer work，不是图中不同节点或边的数量。

统一时间含义：

- `compute_ms`：实际 GNN 计算；
- `inference_ms`：输入已加载后的完整 backend invocation；
- 文件加载、CLI parsing 和 correctness comparison 不计入 inference；
- GPU 的 H2D、kernel、D2H 由 B/C 分别测量并写入公共 metrics；
- A 的 benchmark runner 负责重复运行和最终汇总。

## 10. 我的选择对 B/C 的直接影响

| 我的选择 | 对 Role B 的影响 | 对 Role C 的影响 |
|---|---|---|
| CSR row 存入邻居 | Vertex kernel 的 row `v` 输出目标是 `v` | Edge entry 的 source 是 `col_idx[p]`，destination 是所属 row |
| 不使用 self-loop | Kernel 不得加入 `H[v,:]` | Edge preprocessing/kernel 不得添加 `(v,v)` |
| Mean aggregation | 每行聚合后除以入度 | Edge contribution 最终必须按 destination 入度归一化 |
| 零入度输出零 | 空 CSR row 写零 | 没有入边的 destination 必须初始化为零 |
| Row-major float32 | 按公共索引 copy/read/write | 内部可重排，但公共输出必须恢复 row-major |
| Sequential CPU reference | CUDA vertex 输出与它比较 | Alternative CUDA 输出与它比较 |
| 公共 loader | 不写独立 dataset parser | 不写独立 dataset parser |
| 公共 CLI/dispatch | 只提供 backend 函数、capabilities 和 timing | 只提供 backend 函数、capabilities 和 timing |
| 统一 metrics | 提供 H2D/kernel/D2H/device bytes | 提供 H2D/kernel/D2H/device bytes |

## 11. Role A module boundary

角色 A 计划实现：

```text
include/gnni/types.hpp
    公共 graph/matrix/model/enums

include/gnni/api.hpp
    validation/comparison/backend/metrics declarations

include/gnni/io.hpp
    graph/dense matrix/model loader declarations

src/io/graph_loader.cpp
    CSR 和可选 edge features 加载

src/io/matrix_loader.cpp
    node features 和 dense matrix 加载

src/io/model_loader.cpp
    layer metadata 和 preloaded weights 加载

src/core/validation.cpp
    graph/features/model/request validation

src/core/compare.cpp
    tolerance-based correctness comparison

src/backends/cpu_sequential.cpp
    顺序 correctness reference

src/backends/cpu_kernels.cpp
    顺序 aggregation、linear update 和 activation 三阶段

src/backends/cpu_openmp.cpp
    OpenMP implementation

src/backends/dispatch.cpp
    backend selection 和 capability checks

src/backends/cuda_unavailable.cpp
    B/C 尚未接入时的可替换 unavailable stubs

src/cli/main.cpp
    唯一 executable entry point

src/cli/cli_options.cpp
    CLI parsing、help 和 workflow orchestration

src/benchmark/runner.cpp
    warm-up、repeat、timing 和统计

src/benchmark/report.cpp
    统一 terminal/CSV reporting
```

所有 backend：

- 通过 const reference 接收 graph/features/model；
- 不修改 caller input；
- 返回自己拥有的 `RunResult::output`；
- 返回后不保留 caller object 的引用；
- validation error 使用 `std::invalid_argument`；
- unavailable backend/capability 使用 `std::runtime_error`。

## 12. Optional extensions

下面功能保留，但不属于 baseline 完成条件：

### 12.1 Softmax

对最终 logits 的每个节点执行 row-wise stable softmax。建议作为公共 post-processing，
而不是混入每层 GCN kernel。

### 12.2 Edge features

如果存在，shape 为：

```text
num_edges x edge_feature_dim
```

第 `p` 行 edge feature 对应同一个 CSR entry `col_idx[p]`。

### 12.3 Attention

当前 API 只保留 `AggregationKind::Attention` capability。实现前必须另行冻结：

- attention score；
- 参数 shape；
- single-head/multi-head；
- 是否使用 edge features；
- 零入度处理；
- CPU reference；
- CUDA mapping。

请求尚未实现的 optional capability 必须明确报错，不能静默退回 baseline。

## 13. A 与 B/C 的交付关系

角色 A 先提供给 B/C：

1. 本文档中的数学和布局语义；
2. `types.hpp`；
3. `api.hpp`；
4. 数据格式说明和 loader；
5. 小型测试数据；
6. sequential CPU reference output；
7. `compare_outputs`；
8. CLI/benchmark integration。

Role B 交付给 A：

- `run_cuda_vertex`；
- vertex backend capabilities；
- CUDA allocation/transfer/kernel errors；
- H2D、kernel、D2H 和 device-memory metrics。

Role C 交付给 A：

- `run_cuda_edge` 或约定的 alternative backend；
- alternative backend capabilities；
- 对应 timing/memory metrics；
- profiling 结果使用的 configuration metadata。

A 负责把这些 backend 接入同一 executable 和报告格式，但不负责修改 B/C 的 kernel。

## 14. Role A implementation order

推荐严格按以下顺序执行：

1. 冻结本文档和公共 API。
2. 定义 versioned binary format 和 small text fixture format。（已完成设计）
3. 实现 storage-size、CSR 和 shape validation。
4. 创建可手算的小型数据。
5. 实现顺序 CPU 单层 mean aggregation。
6. 实现 linear update、activation 和多层 buffer swap。
7. 实现 correctness comparison。
8. 实现 OpenMP。
9. 实现 backend dispatch 和 CLI。
10. 实现 benchmark runner/report。
11. 接入 B/C backend。
12. 执行 CPU/OpenMP/CUDA correctness 和性能实验。

在第 7 步之前不应把 CUDA 输出视为正确；在顺序 CPU 通过可手算测试之前，
不应开始以它作为全组 reference。
