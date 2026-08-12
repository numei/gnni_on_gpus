#pragma once

#include "gnni/types.hpp"

#include <cstdint>
#include <optional>

namespace gnni {

struct InferenceConfig {
    AggregationKind aggregation = AggregationKind::Mean;
    OutputTransform output_transform = OutputTransform::Logits;

    // Zero lets the OpenMP backend choose its default.
    std::uint32_t num_threads = 0;
    OpenMPScheduleKind openmp_schedule = OpenMPScheduleKind::Default;

    // Zero uses the OpenMP runtime's default chunk size.
    std::uint32_t openmp_chunk_size = 0;
};

struct BackendCapabilities {
    bool available = false;
    bool mean_aggregation = false;
    bool attention_aggregation = false;
    bool edge_features = false;
    bool softmax = false;
};

struct TimingMetrics {
    // Time spent executing the graph/model computation.
    double compute_ms = 0.0;

    // GPU-specific fields remain empty for CPU backends.
    std::optional<double> host_to_device_ms;
    std::optional<double> device_to_host_ms;

    // Complete backend invocation, excluding file loading and CLI parsing.
    double inference_ms = 0.0;
};

struct MemoryMetrics {
    std::uint64_t host_bytes = 0;
    std::optional<std::uint64_t> device_bytes;
};

struct RunMetrics {
    TimingMetrics timing;
    MemoryMetrics memory;

    // Compute-throughput: N * L / compute_seconds and E * L / compute_seconds, respectively.
    double nodes_per_second = 0.0;
    double edges_per_second = 0.0;
};

struct RunResult {
    DenseMatrix output;
    RunMetrics metrics;
};

struct ComparisonOptions {
    // correctness test 的误差标准。
    double absolute_tolerance = 1.0e-5;
    double relative_tolerance = 1.0e-4;
};

struct ComparisonReport {
    bool shapes_match = false;
    bool within_tolerance = false;
    double max_absolute_error = 0.0;
    double max_relative_error = 0.0;
    double root_mean_square_error = 0.0;
    std::uint64_t mismatch_count = 0;
    std::optional<std::uint64_t> first_mismatch_index;
};

/** Validate invariants that involve only one CSRGraph object. */
void validate_graph(const CSRGraph& graph);

/** Validate a feature matrix and its node-count relationship with the graph. */
void validate_features(const CSRGraph& graph, const DenseMatrix& features);

/** Validate model storage, activations, and layer-to-layer dimensions. */
void validate_model(const DenseMatrix& features, const Model& model);

/**
 * Validate one complete inference request and the selected backend's declared
 * capabilities. Call this before allocating large CPU or GPU buffers.
 */
void validate_request(
    const CSRGraph& graph,
    const DenseMatrix& features,
    const Model& model,
    const InferenceConfig& config,
    const BackendCapabilities& capabilities);

/**
 * Compare two row-major outputs using absolute plus relative tolerance.
 * Shape mismatch is reported rather than treated as element-wise mismatch.
 */
ComparisonReport compare_outputs(
    const DenseMatrix& reference,
    const DenseMatrix& candidate,
    const ComparisonOptions& options = {});

/** Run the readable, deterministic sequential CPU correctness reference. */
RunResult run_cpu_sequential(
    const CSRGraph& graph,
    const DenseMatrix& features,
    const Model& model,
    const InferenceConfig& config);

/** Run the node-parallel OpenMP backend using config's thread settings. */
RunResult run_cpu_openmp(
    const CSRGraph& graph,
    const DenseMatrix& features,
    const Model& model,
    const InferenceConfig& config);

/**
 * Run Role B's vertex-centric CUDA backend. No CUDA type crosses this API.
 */
RunResult run_cuda_vertex(
    const CSRGraph& graph,
    const DenseMatrix& features,
    const Model& model,
    const InferenceConfig& config);

/**
 * Run Role C's edge-centric or agreed alternative CUDA backend.
 */
RunResult run_cuda_edge(
    const CSRGraph& graph,
    const DenseMatrix& features,
    const Model& model,
    const InferenceConfig& config);

/** Dispatch a request to exactly the selected backend without fallback. */
RunResult run_backend(
    BackendKind backend,
    const CSRGraph& graph,
    const DenseMatrix& features,
    const Model& model,
    const InferenceConfig& config);

/** Return the stable CLI/reporting name of a backend enum. */
const char* backend_name(BackendKind backend) noexcept;

/** Return build-time availability and optional-feature support. */
BackendCapabilities backend_capabilities(BackendKind backend) noexcept;

}  // namespace gnni
