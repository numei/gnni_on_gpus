#include "gnni/api.hpp"

#include <stdexcept>

namespace gnni {

#ifndef GNNI_WITH_CUDA_VERTEX
RunResult run_cuda_vertex(
    const CSRGraph& graph,
    const DenseMatrix& features,
    const Model& model,
    const InferenceConfig& config) {
    static_cast<void>(graph);
    static_cast<void>(features);
    static_cast<void>(model);
    static_cast<void>(config);
    // Role B replaces this build-time stub with the real CUDA implementation.
    throw std::runtime_error("cuda-vertex backend is not available in this build");
}
#endif

#ifndef GNNI_WITH_CUDA_EDGE
RunResult run_cuda_edge(
    const CSRGraph& graph,
    const DenseMatrix& features,
    const Model& model,
    const InferenceConfig& config) {
    static_cast<void>(graph);
    static_cast<void>(features);
    static_cast<void>(model);
    static_cast<void>(config);
    // Role C replaces this build-time stub with the real CUDA implementation.
    throw std::runtime_error("cuda-edge backend is not available in this build");
}
#endif

}  // namespace gnni

