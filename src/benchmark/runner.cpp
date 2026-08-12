#include "gnni/benchmark.hpp"

#include <stdexcept>

namespace gnni {

MetricSummary summarize_samples(const std::vector<double>& samples) {
    static_cast<void>(samples);
    // TODO(Role A): reject empty/non-finite input, sort a copy for min/median,
    // compute the arithmetic mean, and use a documented population or sample
    // standard-deviation convention consistently.
    throw std::logic_error("summarize_samples is not implemented");
}

BenchmarkReport benchmark_backend(
    BackendKind backend,
    const CSRGraph& graph,
    const DenseMatrix& features,
    const Model& model,
    const InferenceConfig& config,
    const BenchmarkOptions& options) {
    static_cast<void>(backend);
    static_cast<void>(graph);
    static_cast<void>(features);
    static_cast<void>(model);
    static_cast<void>(config);
    static_cast<void>(options);
    // TODO(Role A): validate run counts, execute discarded warm-ups, retain
    // measured RunMetrics, then summarize compute_ms and inference_ms. Do not
    // perform correctness comparison inside the measured region.
    throw std::logic_error("benchmark_backend is not implemented");
}

}  // namespace gnni

