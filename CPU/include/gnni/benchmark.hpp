#pragma once

#include "gnni/api.hpp"

#include <cstdint>
#include <iosfwd>
#include <vector>

namespace gnni_cpu {

struct BenchmarkOptions {
    std::uint32_t warmup_runs = 1; // 不记录
    std::uint32_t measured_runs = 5; // 记录
};

struct MetricSummary {
    double minimum = 0.0;
    double mean = 0.0;
    double median = 0.0;
    double standard_deviation = 0.0;
};

struct BenchmarkReport {
    BackendKind backend = BackendKind::CpuSequential;
    InferenceConfig inference;
    BenchmarkOptions options;
    std::vector<RunMetrics> samples;
    MetricSummary compute_ms;
    MetricSummary inference_ms;
};

/**
 * Summarize a non-empty collection of scalar samples.
 *
 * @throws std::invalid_argument when samples is empty or non-finite.
 */
MetricSummary summarize_samples(const std::vector<double>& samples);

/**
 * Execute warm-up and measured runs using the common backend dispatcher.
 * Correctness comparison is intentionally outside this timed function.
 */
BenchmarkReport benchmark_backend(
    BackendKind backend,
    const CSRGraph& graph,
    const DenseMatrix& features,
    const Model& model,
    const InferenceConfig& config,
    const BenchmarkOptions& options);

/** Write a concise human-readable benchmark report to the supplied stream. */
void write_text_report(const BenchmarkReport& report, std::ostream& output);

/** Write one machine-readable CSV header and data row to the supplied stream. */
void write_csv_report(const BenchmarkReport& report, std::ostream& output);

}  // namespace gnni_cpu

