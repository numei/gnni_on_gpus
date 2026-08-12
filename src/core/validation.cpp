#include "gnni/api.hpp"

#include <stdexcept>

namespace gnni {

void validate_graph(const CSRGraph& graph) {
    static_cast<void>(graph);
    // TODO(Role A): implement every CSR invariant in semantic_contract.md,
    // including sorted rows, duplicate detection, and self-loop rejection.
    throw std::logic_error("validate_graph is not implemented");
}

void validate_features(const CSRGraph& graph, const DenseMatrix& features) {
    static_cast<void>(graph);
    static_cast<void>(features);
    // TODO(Role A): validate storage size, finite values, non-zero feature
    // dimension, and features.rows == graph.num_nodes.
    throw std::logic_error("validate_features is not implemented");
}

void validate_model(const DenseMatrix& features, const Model& model) {
    static_cast<void>(features);
    static_cast<void>(model);
    // TODO(Role A): validate all weight shapes, finite values, activation
    // placement, first-layer input size, and adjacent layer compatibility.
    throw std::logic_error("validate_model is not implemented");
}

void validate_request(
    const CSRGraph& graph,
    const DenseMatrix& features,
    const Model& model,
    const InferenceConfig& config,
    const BackendCapabilities& capabilities) {
    static_cast<void>(graph);
    static_cast<void>(features);
    static_cast<void>(model);
    static_cast<void>(config);
    static_cast<void>(capabilities);
    // TODO(Role A): call the three object validators first, then reject an
    // unavailable backend or unsupported aggregation/output/edge features.
    throw std::logic_error("validate_request is not implemented");
}

}  // namespace gnni

