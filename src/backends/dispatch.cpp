#include "gnni/api.hpp"

#include <stdexcept>

namespace gnni {

const char* backend_name(BackendKind backend) noexcept {
    switch (backend) {
        case BackendKind::CpuSequential:
            return "cpu";
        case BackendKind::CpuOpenMP:
            return "omp";
    }
    return "unknown";
}

BackendCapabilities backend_capabilities(BackendKind backend) noexcept {
    BackendCapabilities capabilities;

    switch (backend) {
        case BackendKind::CpuSequential:
            capabilities.available = true;
            capabilities.mean_aggregation = true;
            // TODO(Role A): set softmax=true only after common stable softmax
            // post-processing has been implemented and tested.
            return capabilities;
        case BackendKind::CpuOpenMP:
#ifdef GNNI_HAS_OPENMP
            capabilities.available = true;
            capabilities.mean_aggregation = true;
#endif
            return capabilities;
    }
    return capabilities;
}

RunResult run_backend(
    BackendKind backend,
    const CSRGraph& graph,
    const DenseMatrix& features,
    const Model& model,
    const InferenceConfig& config) {
    switch (backend) {
        case BackendKind::CpuSequential:
            return run_cpu_sequential(graph, features, model, config);
        case BackendKind::CpuOpenMP:
            return run_cpu_openmp(graph, features, model, config);
    }
    throw std::invalid_argument("invalid BackendKind value");
}

}  // namespace gnni
