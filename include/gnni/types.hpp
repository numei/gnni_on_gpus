#pragma once

#include <cstdint>
#include <optional>
#include <vector>

namespace gnni {

using VertexId = std::uint32_t;
using EdgeOffset = std::uint64_t;
using Dimension = std::uint64_t;

struct DenseMatrix {
    // rows 指节点数量 N
    // 每个节点的特征数量 F
    Dimension rows = 0;
    Dimension cols = 0;
    std::vector<float> values;
};

/*
    例如 rows = 3， cols = 2；
    node 0: [1, 2]
    node 1: [3, 4]
    node 2: [5, 6];

    对应matrix的是：
    matrix.rows=3; matrix.cols =2;
    values = {
        1, 2,
        3, 4,
        5, 6
    }
*/

struct CSRGraph {
    Dimension num_nodes = 0;
    EdgeOffset num_edges = 0;
    bool directed = true;

    std::vector<EdgeOffset> row_ptr;
    std::vector<VertexId> col_idx;

    // When present, shape is num_edges x edge_feature_dim. Row p describes
    // the same edge as col_idx[p].
    std::optional<DenseMatrix> edge_features;
};

enum class ActivationKind {
    Identity,
    ReLU,
};

struct Layer {
    // Shape: F_in x F_out, row-major.
    DenseMatrix weights;
    ActivationKind activation = ActivationKind::ReLU;
};

struct Model {
    std::vector<Layer> layers;
};

enum class AggregationKind {
    Mean,
    Attention,
};

enum class OutputTransform {
    Logits,
    Softmax,
};

enum class BackendKind {
    /// Role A's deterministic CPU correctness reference.
    CpuSequential,
    /// Role A's node-parallel OpenMP implementation.
    CpuOpenMP,
};

enum class OpenMPScheduleKind {
    Default,
    Static,
    Dynamic,
    Guided,
};

}  // namespace gnni
