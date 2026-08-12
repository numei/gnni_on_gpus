# Role A Implementation Checklist

这份清单按依赖顺序列出代码骨架中的待实现项。每完成一项，应先增加对应测试，再把
源码中的 `TODO(Role A)` 删除。不要仅删除异常占位符而不验证行为。

## Stage 1: Data foundations

- [x] 冻结 `types.hpp`、`api.hpp` 和 `io.hpp`。
- [x] 实现 checked arithmetic 和 little-endian `BinaryReader`。
- [x] 实现并测试 binary dense-matrix loader。
- [ ] 实现 binary graph loader：header、CSR、flags、optional edge features。
- [ ] 实现 binary model loader：descriptors、activations、weight payloads。
- [ ] 实现 `TextTokenReader`。
- [ ] 实现 graph/matrix/model text fixture loaders。
- [ ] 完成三类 loader 的 auto detection。

验收条件：valid fixture 正确加载；bad magic/version、truncation、trailing bytes、
overflow、NaN/Inf 和非法局部结构均被拒绝。

## Stage 2: Cross-object validation

- [ ] `validate_graph`。
- [ ] `validate_features`。
- [ ] `validate_model`。
- [ ] `validate_request` 和 backend capability checks。

验收条件：错误消息指出具体 object/field/index；validation 不修改输入。

## Stage 3: Sequential correctness reference

- [ ] `aggregate_mean_sequential`。
- [ ] `linear_transform_sequential`。
- [ ] `apply_activation_in_place`。
- [ ] `run_cpu_sequential` 的多层 orchestration 和 metrics。
- [ ] 为有向、无向、孤立节点、多层/变维模型建立手算 fixtures。

验收条件：每个阶段先单测，再测完整多层输出；空 CSR row 必须输出零聚合。

## Stage 4: Correctness tooling

- [ ] `compare_outputs`。
- [ ] Shape mismatch、absolute/relative tolerance、RMSE 和 non-finite tests。

验收条件：OpenMP 和 B/C 都使用同一 comparison API。

## Stage 5: OpenMP

- [ ] Node-parallel aggregation。
- [ ] Node/output-feature-parallel linear transform。
- [ ] Thread count、schedule 和 chunk-size handling。
- [ ] 与 sequential reference 比较。

验收条件：不同线程数和 schedule 均通过 correctness；计时不包含 validation。

## Stage 6: Integration and reporting

- [ ] `parse_cli`。
- [ ] `run_cli` workflow。
- [ ] `run_backend` 依赖的 validation 完成。
- [ ] `summarize_samples` 和 `benchmark_backend`。
- [ ] Text/CSV reports。
- [ ] 更新最终 README build/run/data-format instructions。

验收条件：未安装 CUDA 时 CPU build 正常；选择不可用 CUDA backend 明确失败；
接入 B/C 源码不需要修改公共 CLI、data types 或 correctness API。

