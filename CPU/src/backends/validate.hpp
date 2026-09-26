#pragma once
#include "gnni/api.hpp"
#include <cmath>
#include <limits>
#include <stdexcept>
namespace gnni_cpu::cpu_detail {
inline void validate_inputs(const CSRGraph& g, const DenseMatrix& x, const Model& m) {
    auto matrix = [](const DenseMatrix& a) {
        if (!a.rows || !a.cols || a.rows > std::numeric_limits<std::size_t>::max() / a.cols ||
            a.values.size() != a.rows * a.cols)
            throw std::invalid_argument("invalid matrix shape or payload");
        for (float v : a.values) if (!std::isfinite(v))
            throw std::invalid_argument("non-finite matrix value");
    };
    matrix(x);
    if (!g.num_nodes || g.num_nodes == std::numeric_limits<Dimension>::max() ||
        g.row_ptr.size() != g.num_nodes + 1 ||
        g.row_ptr.front() != 0 || g.row_ptr.back() != g.col_idx.size() ||
        g.num_edges != g.col_idx.size() || x.rows != g.num_nodes)
        throw std::invalid_argument("graph/features shape mismatch");
    for (std::size_t i = 1; i < g.row_ptr.size(); ++i)
        if (g.row_ptr[i] < g.row_ptr[i-1]) throw std::invalid_argument("decreasing CSR offsets");
    for (auto u : g.col_idx) if (u >= g.num_nodes)
        throw std::invalid_argument("CSR column out of range");
    if (m.layers.empty()) throw std::invalid_argument("empty model");
    auto width = x.cols;
    for (const auto& layer : m.layers) {
        matrix(layer.weights);
        if (layer.weights.rows != width) throw std::invalid_argument("model feature width mismatch");
        if (g.num_nodes > std::numeric_limits<std::size_t>::max() / layer.weights.cols)
            throw std::invalid_argument("output matrix size overflow");
        if (layer.activation != ActivationKind::Identity && layer.activation != ActivationKind::ReLU)
            throw std::invalid_argument("invalid activation");
        width = layer.weights.cols;
    }
}
}
