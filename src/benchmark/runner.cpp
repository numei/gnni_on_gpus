#include "gnni/benchmark.hpp"
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <string>
#include <vector>
#include <stdexcept>

namespace gnni {

    /*
        samples是一堆测试结果， 假设inference 跑了5次，10.2ms, 9,8ms, 10.0ms, 9.9ms, 10.1ms
        那么samples = {10.2, 9.8, 10.0, 9.9, 10.1}
        然后函数算minimum， mean， median， standard deviation
        最后包装成metricSummary。

        这里的数据结构去看api和benchmark.hpp
    */
MetricSummary summarize_samples(const std::vector<double>& samples) {
    if (samples.empty()){
        throw std::invalid_argument(
            "cannot summarize an empty sample collection"
        );
    }

    for (const double sample : samples){
        if(!std::isfinite(sample)) {
            throw std::invalid_argument("benchmark samples must be finite");
        }
    }

    std::vector<double> sorted_samples = samples;
    std::sort(sorted_samples.begin(), sorted_samples.end());

    // minimum值
    MetricSummary summary;
    summary.minimum = sorted_samples.front();

    // mean 值
    long double sum = 0.0L;

    for (const double sample : samples){
        sum += static_cast<long double>(sample);
    }

    const long double sample_count = static_cast<long double>(samples.size());
    const long double mean = sum / sample_count;

    summary.mean = static_cast<double>(mean);

    // median
    const std::size_t middle = sorted_samples.size() / 2;

    if (sorted_samples.size() % 2 ==1 ){
        summary.median = sorted_samples[middle];
    }else{
        summary.median = 0.5 * sorted_samples[middle - 1] + 0.5 * sorted_samples[middle];
    }

    // standard deviation
    long double squared_difference_sum = 0.0L;

    for (const double sample : samples){
        const long double difference = static_cast<long double>(sample) - mean;

        squared_difference_sum += difference * difference;
    }

    // 使用 population standard deviation: 方差除以N， 而不是N -1
    const long double variance = squared_difference_sum / sample_count;
    summary.standard_deviation = static_cast<double>(std::sqrt(variance));
    return summary;
}

// options -> benchmark跑多少次， config -> 怎么做inference，一些具体的参数
BenchmarkReport benchmark_backend(
    BackendKind backend,
    const CSRGraph& graph,
    const DenseMatrix& features,
    const Model& model,
    const InferenceConfig& config,
    const BenchmarkOptions& options) {
        if (options.measured_runs == 0){
            throw std::invalid_argument("measured_runs must be greater than zero");
        }

        // 检查指定的backend能不能用
        const BackendCapabilities capabilities = backend_capabilities(backend);

        if (!capabilities.available){
            throw std::runtime_error(
                std::string("backend is not available: ") + backend_name(backend)
            );
        }

        // warm-up 的结果不记录
        for (std::uint32_t run =0; run < options.warmup_runs; ++run){
            static_cast<void>(run_backend(backend, graph, features, model, config));
        }

        BenchmarkReport report;
        report.backend = backend;
        report.inference = config;
        report.options = options;
        report.samples.reserve(options.measured_runs);

        // 分别储存summarize之后的结果
        std::vector<double> compute_samples;
        std::vector<double> inference_samples;

        compute_samples.reserve(options.measured_runs);
        inference_samples.reserve(options.measured_runs);

        // 保存结果
        for (std::uint32_t run = 0; run < options.measured_runs; ++run){
            const RunResult result = run_backend(backend, graph, features, model, config);
            report.samples.push_back(result.metrics);
            compute_samples.push_back(result.metrics.timing.compute_ms);

            inference_samples.push_back(result.metrics.timing.inference_ms);
        }

        report.compute_ms = summarize_samples(compute_samples);
        report.inference_ms = summarize_samples(inference_samples);

        return report;
}

}  // namespace gnni
