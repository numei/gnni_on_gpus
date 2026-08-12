#pragma once

#include <cstdint>
#include <optional>
#include <vector>

namespace gnni {

using VertexId = std::uint32_t;
using EdgeOffset = std::uint64_t;
using Dimension = std::uint64_t;

struct DenseMatrix {
    Dimension rows = 0;
    Dimension cols = 0;
    std::vector<float> values;
};

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
    /// Public dispatch identifier for Role B's CUDA implementation.
    CudaVertex,
    /// Public dispatch identifier for Role C's CUDA implementation.
    CudaEdge,
};

enum class OpenMPScheduleKind {
    Default,
    Static,
    Dynamic,
    Guided,
};

}  // namespace gnni
