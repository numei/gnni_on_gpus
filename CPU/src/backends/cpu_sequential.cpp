#include "gnni/api.hpp"

#include "cpu_kernels.hpp"
#include "validate.hpp"

#include <stdexcept>
#include <utility>
#include <algorithm>
#include <chrono>
#include <cstdint>

namespace gnni_cpu {

namespace {
    std::uint64_t dense_payload_bytes(const DenseMatrix& matrix) noexcept {
        return static_cast<std::uint64_t>(matrix.values.size()) * 
            static_cast<std::uint64_t>(sizeof(float));
    }

    std::uint64_t persistent_input_bytes(const CSRGraph& graph, const DenseMatrix& features, const Model& model) 
        noexcept {
            std::uint64_t bytes = 0;

            bytes += static_cast<std::uint64_t>(graph.row_ptr.size()) * static_cast<std::uint64_t>(sizeof(EdgeOffset));

            bytes += static_cast<std::uint64_t>(graph.col_idx.size()) * static_cast<std::uint64_t>(sizeof(VertexId));

            bytes += dense_payload_bytes(features);

            if(graph.edge_features.has_value()){
                bytes += dense_payload_bytes(graph.edge_features.value());
            }

            for (const Layer& layer : model.layers){
                bytes += dense_payload_bytes(layer.weights);
            }

            return bytes;
        }
}    // namespace


RunResult run_cpu_sequential(
    const CSRGraph& graph,
    const DenseMatrix& features,
    const Model& model,
    const InferenceConfig& config) {
        //开始计时，推理时长
        using Clock = std::chrono::steady_clock;
        const auto inference_start = Clock::now();

        cpu_detail::validate_inputs(graph, features, model);
        if (config.aggregation != AggregationKind::Mean && config.aggregation != AggregationKind::Sum && config.aggregation != AggregationKind::SymmetricGcn) {
            throw std::runtime_error("CPU sequential backend currently supports only mean aggregation");
        }

        if (config.output_transform != OutputTransform::Logits) {
            throw std::runtime_error(
                "CPU sequential backend currently supports only logits output"
            );
        }

        DenseMatrix current = features;
        // inference_ms包含复制features的时间
        std::uint64_t peak_workspace_bytes = dense_payload_bytes(current);
        const auto compute_start = Clock::now();

        for (const Layer& layer : model.layers){

            DenseMatrix aggregated = cpu_detail::aggregate_mean_sequential(graph, current, config.aggregation);
            DenseMatrix transformed = cpu_detail::linear_transform_sequential(aggregated, layer.weights);

            cpu_detail::apply_activation_in_place(transformed, layer.activation);

            /*
                统计峰值工作内存：
                    在一层计算完成，但局部对象尚未释放时，三分矩阵同时存在
                    current = N * F_in; aggregated = N * F_in; transformed = N * F_out;
                    所以该层工作空间大约是： sizeof(float) * N * (2F_in + F_out) bytes
            */ 
            const std::uint64_t layer_workspcae_bytes = dense_payload_bytes(current) +
                                                        dense_payload_bytes(aggregated) +
                                                        dense_payload_bytes(transformed);

            peak_workspace_bytes = std::max(peak_workspace_bytes, layer_workspcae_bytes);
            // tranformed 成为下一层的输入
            current = std::move(transformed);
        }
        
        const auto compute_end = Clock::now();

        RunResult result;
        result.output = std::move(current);

        const auto inference_end = Clock::now();
        result.metrics.timing.compute_ms = std::chrono::duration<double, std::milli>(
            compute_end - compute_start
        ).count();

        result.metrics.timing.inference_ms = std::chrono::duration<double, std::milli>(
            inference_end - inference_start
        ).count();

        // 长期输入内存 + 推理过程中最大的临时矩阵内存
        result.metrics.memory.host_bytes =
            persistent_input_bytes(graph, features, model) + peak_workspace_bytes;

        // throughput， 先将毫秒转为秒
        const double compute_seconds =
            result.metrics.timing.compute_ms / 1000.0;

        if (compute_seconds > 0.0) {
            const double layer_count =
                static_cast<double>(model.layers.size());

            result.metrics.nodes_per_second =
                static_cast<double>(graph.num_nodes) *
                layer_count /
                compute_seconds;

            result.metrics.edges_per_second =
                static_cast<double>(graph.num_edges) *
                layer_count /
                compute_seconds;
        }

        return result;
}

}  // namespace gnni_cpu

