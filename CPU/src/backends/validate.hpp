#pragma once
#include "gnni/api.hpp"
#include <cmath>
#include <limits>
#include <stdexcept>
namespace gnni_cpu::cpu_detail {
// 在真正访问数组之前，检查图、特征和模型能否配合完成推理。
// g 是图，x 是节点特征矩阵，m 是由多层权重组成的模型。
inline void validate_inputs(const CSRGraph& g, const DenseMatrix& x, const Model& m) {
    // 把“检查一个稠密矩阵”的操作写成可复用的小函数：
    // 既用来检查输入特征 x，也用来检查每一层的权重矩阵。
    auto matrix = [](const DenseMatrix& a) {
        // 行、列都必须大于 0，且 values 中恰好有 rows * cols 个数。
        // 先用除法判断乘法是否会超过 size_t 的范围，避免尺寸乘法溢出。
        // || 会从左向右短路求值：若 cols 为 0，后面的除法不会执行。
        if (!a.rows || !a.cols || a.rows > std::numeric_limits<std::size_t>::max() / a.cols ||
            a.values.size() != a.rows * a.cols)
            throw std::invalid_argument("invalid matrix shape or payload");
        // 特征和权重都必须是有限数；NaN、正无穷、负无穷不能作为合法输入。
        for (float v : a.values) if (!std::isfinite(v))
            throw std::invalid_argument("non-finite matrix value");
    };
    matrix(x);

    // 检查图与特征的基本对应关系：
    // 1. N 个节点需要 N + 1 个 CSR 行偏移
    // 2. 邻居数组从位置 0 开始，最后一个行偏移等于实际边条目数 E。
    // 3. num_edges 与 col_idx 长度一致，特征矩阵也必须恰好有 N 行。
    // 前面的长度检查不通过时会短路，因此不会对空 row_ptr 调用 front/back。
    if (!g.num_nodes || g.num_nodes == std::numeric_limits<Dimension>::max() ||
        g.row_ptr.size() != g.num_nodes + 1 ||
        g.row_ptr.front() != 0 || g.row_ptr.back() != g.col_idx.size() ||
        g.num_edges != g.col_idx.size() || x.rows != g.num_nodes)
        throw std::invalid_argument("graph/features shape mismatch");

    // 每行的邻居范围是 [row_ptr[v], row_ptr[v+1])，所以偏移不能递减。
    // 配合上面的首尾检查，可保证各行范围都位于 col_idx 数组之内。
    for (std::size_t i = 1; i < g.row_ptr.size(); ++i)
        if (g.row_ptr[i] < g.row_ptr[i-1]) throw std::invalid_argument("decreasing CSR offsets");
    // col_idx 存的是邻居节点编号；有效范围为 0 到 N - 1。
    for (auto u : g.col_idx) if (u >= g.num_nodes)
        throw std::invalid_argument("CSR column out of range");

    // 模型至少包含一层。width 跟踪“当前这一层应该接收多少个特征”。
    if (m.layers.empty()) throw std::invalid_argument("empty model");
    auto width = x.cols;
    for (const auto& layer : m.layers) {
        matrix(layer.weights);
        // 输入为 N × width，权重必须为 width × F_out，二者才能相乘。
        if (layer.weights.rows != width) throw std::invalid_argument("model feature width mismatch");
        // 本层会建立 N × F_out 的输出矩阵，提前检查元素数量是否溢出。
        // matrix(weights) 已经保证 weights.cols 大于 0，因此这里可以安全相除。
        if (g.num_nodes > std::numeric_limits<std::size_t>::max() / layer.weights.cols)
            throw std::invalid_argument("output matrix size overflow");
        // 内存接口只接受这两种激活；此处不强制隐藏层/最终层的排列规则。
        if (layer.activation != ActivationKind::Identity && layer.activation != ActivationKind::ReLU)
            throw std::invalid_argument("invalid activation");
        // 这一层输出 F_out 个特征，下一层就必须接收 F_out 个特征。
        width = layer.weights.cols;
    }
}
}
