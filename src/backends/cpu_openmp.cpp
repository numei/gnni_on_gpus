#include "gnni/api.hpp"

#include <stdexcept>

namespace gnni {

RunResult run_cpu_openmp(
    const CSRGraph& graph,
    const DenseMatrix& features,
    const Model& model,
    const InferenceConfig& config) {
    static_cast<void>(graph);
    static_cast<void>(features);
    static_cast<void>(model);
    static_cast<void>(config);
    // TODO(Role A): validate the request; map config schedule/thread settings
    // to OpenMP; parallelize independent destination rows and (node,out)
    // linear outputs; preserve sequential semantics; populate common metrics.
    throw std::logic_error("run_cpu_openmp is not implemented");
}

}  // namespace gnni

