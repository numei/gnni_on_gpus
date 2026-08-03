你的想法是合理的：**两个人 GPU/CUDA，一个人 CPU/C++**。但我建议不要让两个 CUDA 同学都“随便写 GPU”，而是把 GPU 再拆成两个清晰方向，否则很容易互相踩接口、重复做 kernel。

推荐分工如下。

**A：CPU / C++ 基准负责人**
这个人负责整个项目的“正确性地基”。

职责：

- 定义核心数据结构：`CSRGraph`、特征矩阵、权重矩阵、输出矩阵。
- 实现图读取和数据加载：CSR 格式、节点特征、模型权重。
- 实现顺序 CPU GCN inference。
- 实现 OpenMP 多核版本。
- 提供 correctness reference：GPU 的输出要和 CPU baseline 对齐。
- 负责通用 CLI：例如 `--graph`、`--features`、`--weights`、`--layers`、`--backend cpu|omp|cuda`。
- 负责基础 benchmark 计时框架。

这个角色最重要，因为 CPU baseline 是全组的“真值”。没有它，CUDA 跑得再快也不知道对不对。

**B：CUDA 基础 kernel / vertex-centric 负责人**
这个人负责第一版可工作的 GPU inference。

职责：

- 设计 CUDA 端内存布局：CSR 拷贝到 device、features/weights/output 分配。
- 实现最基础的 GCN CUDA kernel。
- 优先采用 vertex-centric：一个 node 或一个 node-feature 组合对应线程/线程组。
- 处理 CPU-GPU 数据传输。
- 和 A 对接统一接口，例如：

```cpp
run_gcn_cuda_vertex_centric(graph, features, weights, config)
```

- 做 CPU vs CUDA 的数值误差检查。
- 输出第一批 GPU 加速比。

这个人目标是“先跑通”，不要一开始就过度优化。

**C：CUDA 优化 / edge-centric / profiling 负责人**
这个人负责性能探索和实验深度。

职责：

- 实现第二套 CUDA 策略，比如 edge-centric 或 feature-parallel。
- 尝试 shared memory、coalesced access、atomic/reduction 优化。
- 分析高度数节点带来的负载不均。
- 使用 Nsight Compute / Nsight Systems 做 profiling。
- 比较不同 kernel：
  - vertex-centric
  - edge-centric
  - feature-centric
  - 是否使用 shared memory
- 负责性能实验表格和图表的一部分。

这个角色更偏“研究和优化”，适合作为报告亮点。

**三个人之间的接口**
最好一开始就约定公共接口，避免后面合并痛苦：

```text
input:
  CSR graph:
    num_nodes
    num_edges
    row_ptr
    col_idx

  dense node features:
    N x F

  weights:
    F_in x F_out for each layer

output:
  dense node embeddings/logits:
    N x F_out

metrics:
  runtime_ms
  nodes_per_second
  edges_per_second
  memory_bytes
```

**时间安排建议**
第 1 阶段：A 先做数据结构和 CPU sequential，B 同时准备 CUDA memory wrapper，C 准备 benchmark/profiling 模板。

第 2 阶段：A 做 OpenMP，B 做 vertex-centric CUDA，C 做 edge-centric CUDA。

第 3 阶段：三人一起做 correctness、benchmark、图表和文档。A 负责 README/运行说明，B/C 负责 GPU 实验分析，最后统一风格。

**我会这样定最终分工**
如果你们三个人水平差不多：

- 人 1：CPU + 数据格式 + correctness + CLI
- 人 2：CUDA baseline + device memory + vertex-centric kernel
- 人 3：CUDA optimization + alternative kernels + profiling + plots

如果其中一个人 C++ 基础最稳，就让他做 CPU/架构；CUDA 强的两个人分别做“可运行 GPU”和“性能优化 GPU”。

这个分工和 PDF 要求非常贴合：顺序 CPU、多核 CPU、一个或多个 CUDA 版本、不同并行策略、性能评估，都能自然覆盖。
