# gnni_on_gpus

Graph Neural Network Inference on GPUs(prof.quer 1st)

Role A 的设计、数学语义和实现顺序见
[`docs/semantic_contract.md`](docs/semantic_contract.md)，磁盘格式见
[`docs/file_formats.md`](docs/file_formats.md)。当前仓库已搭好 Role A 的 C++17
模块骨架；除 binary dense-matrix loader 外，带 `TODO(Role A)` 的函数仍需实现。
推荐实现顺序见
[`docs/role_a_implementation_checklist.md`](docs/role_a_implementation_checklist.md)。

## Build the framework

```text
cmake -S . -B build
cmake --build build
ctest --test-dir build --output-on-failure
```

未接入 B/C 的 CUDA 源文件时，CUDA backend 会明确报告 unavailable。B/C 接入方式
记录在根目录 `CMakeLists.txt` 的 `GNNI_CUDA_VERTEX_SOURCES` 和
`GNNI_CUDA_EDGE_SOURCES` cache variables 中。
