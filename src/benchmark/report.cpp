#include "gnni/benchmark.hpp"

#include <algorithm>
#include <cstdint>
#include <iomanip>
#include <locale>
#include <ostream>
#include <stdexcept>
#include <vector>

namespace gnni {

namespace {

const char* aggregation_name(AggregationKind aggregation) noexcept {
    switch (aggregation) {
        case AggregationKind::Mean:
            return "mean";
        case AggregationKind::Attention:
            return "attention";
    }
    return "unknown";
}

const char* output_transform_name(OutputTransform transform) noexcept {
    switch (transform) {
        case OutputTransform::Logits:
            return "logits";
        case OutputTransform::Softmax:
            return "probabilities";
    }
    return "unknown";
}

const char* schedule_name(OpenMPScheduleKind schedule) noexcept {
    switch (schedule) {
        case OpenMPScheduleKind::Default:
            return "default";
        case OpenMPScheduleKind::Static:
            return "static";
        case OpenMPScheduleKind::Dynamic:
            return "dynamic";
        case OpenMPScheduleKind::Guided:
            return "guided";
    }
    return "unknown";
}

void write_summary(
    std::ostream& output,
    const char* name,
    const MetricSummary& summary) {
    output << name
           << ": min=" << summary.minimum
           << ", mean=" << summary.mean
           << ", median=" << summary.median
           << ", stddev=" << summary.standard_deviation
           << '\n';
}

struct DerivedReportMetrics {
    MetricSummary nodes_per_second;
    MetricSummary edges_per_second;
    std::uint64_t peak_host_bytes = 0;
};

DerivedReportMetrics derive_metrics(const BenchmarkReport& report) {
    if (report.samples.empty()) {
        throw std::invalid_argument("cannot report an empty benchmark");
    }

    std::vector<double> node_samples;
    std::vector<double> edge_samples;
    node_samples.reserve(report.samples.size());
    edge_samples.reserve(report.samples.size());

    DerivedReportMetrics derived;

    for (const RunMetrics& sample : report.samples) {
        node_samples.push_back(sample.nodes_per_second);
        edge_samples.push_back(sample.edges_per_second);

        derived.peak_host_bytes = std::max(
            derived.peak_host_bytes,
            sample.memory.host_bytes);

    }

    derived.nodes_per_second = summarize_samples(node_samples);
    derived.edges_per_second = summarize_samples(edge_samples);

    return derived;
}

}  // namespace

void write_text_report(const BenchmarkReport& report, std::ostream& output) {
    const DerivedReportMetrics derived = derive_metrics(report);

    output.imbue(std::locale::classic());
    output << std::setprecision(10)
           << "backend: " << backend_name(report.backend) << '\n'
           << "aggregation: "
           << aggregation_name(report.inference.aggregation) << '\n'
           << "output: "
           << output_transform_name(report.inference.output_transform) << '\n'
           << "threads: " << report.inference.num_threads << '\n'
           << "omp_schedule: "
           << schedule_name(report.inference.openmp_schedule) << '\n'
           << "omp_chunk_size: "
           << report.inference.openmp_chunk_size << '\n'
           << "warmup_runs: " << report.options.warmup_runs << '\n'
           << "measured_runs: " << report.options.measured_runs << '\n';

    write_summary(output, "compute_ms", report.compute_ms);
    write_summary(output, "inference_ms", report.inference_ms);

    output << "throughput_definition: node-layer/s and edge-layer/s\n";
    write_summary(output, "nodes_per_second", derived.nodes_per_second);
    write_summary(output, "edges_per_second", derived.edges_per_second);
    output << "peak_host_bytes: " << derived.peak_host_bytes << '\n';

}

void write_csv_report(const BenchmarkReport& report, std::ostream& output) {
    const DerivedReportMetrics derived = derive_metrics(report);

    output.imbue(std::locale::classic());
    output << std::setprecision(17)
           << "backend,aggregation,output,threads,"
              "omp_schedule,omp_chunk_size,"
              "warmup_runs,measured_runs,"
              "compute_ms_min,compute_ms_mean,"
              "compute_ms_median,compute_ms_stddev,"
              "inference_ms_min,inference_ms_mean,"
              "inference_ms_median,inference_ms_stddev,"
              "nodes_per_second_mean,edges_per_second_mean,"
              "peak_host_bytes\n";

    output << backend_name(report.backend) << ','
           << aggregation_name(report.inference.aggregation) << ','
           << output_transform_name(report.inference.output_transform) << ','
           << report.inference.num_threads << ','
           << schedule_name(report.inference.openmp_schedule) << ','
           << report.inference.openmp_chunk_size << ','
           << report.options.warmup_runs << ','
           << report.options.measured_runs << ','
           << report.compute_ms.minimum << ','
           << report.compute_ms.mean << ','
           << report.compute_ms.median << ','
           << report.compute_ms.standard_deviation << ','
           << report.inference_ms.minimum << ','
           << report.inference_ms.mean << ','
           << report.inference_ms.median << ','
           << report.inference_ms.standard_deviation << ','
           << derived.nodes_per_second.mean << ','
           << derived.edges_per_second.mean << ','
           << derived.peak_host_bytes << '\n';
}

}  // namespace gnni
