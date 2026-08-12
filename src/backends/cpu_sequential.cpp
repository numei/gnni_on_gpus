#include "gnni/api.hpp"

#include "cpu_kernels.hpp"

#include <stdexcept>

namespace gnni {

RunResult run_cpu_sequential(
    const CSRGraph& graph,
    const DenseMatrix& features,
    const Model& model,
    const InferenceConfig& config) {
    static_cast<void>(graph);
    static_cast<void>(features);
    static_cast<void>(model);
    static_cast<void>(config);
    // TODO(Role A): validate the request; for each layer call mean aggregation,
    // linear transform, and activation; time only computation; fill output and
    // all common metrics. This is the trusted correctness reference.
    throw std::logic_error("run_cpu_sequential is not implemented");
}

}  // namespace gnni

