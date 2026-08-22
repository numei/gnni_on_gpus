#include "gnni/api.hpp"
#include "cpu_kernels.hpp"
#include <stdexcept>
#include <cstddef>
#include <utility>
#include <cstdint>
#include <limits>
#include <algorithm>
#include <chrono>

#ifdef GNNI_HAS_OPENMP
#include <omp.h>
#endif

namespace gnni {

namespace 
{
    #ifdef GNNI_HAS_OPENMP

        void configure_openmp_schedule(OpenMPScheduleKind schedule, std::uint32_t chunk_size){
            if ( chunk_size > static_cast<std::uint32_t>(std::numeric_limits<int>::max()) ){
                throw std::invalid_argument(
                    "OpenMP chunk size exceeds int range"
                );
            }

            const int openmp_chunk = static_cast<int>(chunk_size);

            switch ((schedule))
            {
                case OpenMPScheduleKind::Default:
                    return;
                
                case OpenMPScheduleKind::Static:
                    omp_set_schedule(
                        omp_sched_static, openmp_chunk
                    );
                    return;
                
                case OpenMPScheduleKind::Dynamic:
                    omp_set_schedule(
                        omp_sched_dynamic, openmp_chunk
                    );
                    return;
            
                case OpenMPScheduleKind::Guided:
                    omp_set_schedule(
                        omp_sched_guided, openmp_chunk
                    );
                    return;
            }
            throw std::invalid_argument(
                "invalid OpenMPScheduleKind value"
            );
        }

        void apply_activation_openmp(DenseMatrix& matrix, ActivationKind activation, int thread_count){
            switch (activation)
            {
                case ActivationKind::Identity:
                    return;
                
                case ActivationKind::ReLU:
                    const std::int64_t element_count = static_cast<std::int64_t>(matrix.values.size());
        #pragma omp parallel for schedule(runtime) num_threads(thread_count)
                    for (std::int64_t index = 0; index < element_count; ++index){
                        float& value = matrix.values[static_cast<std::size_t>(index)];
                        
                        if (value < 0.0F){
                            value = 0.0F;
                        }
                    }
                    return;
            }
            throw std::invalid_argument(
                "invalid ActivationKind value"
            );
            // 每个线程修改不同的matrix.values[index]，所以没有data race
        }

        std::uint64_t dense_payload_bytes(
            const DenseMatrix& matrix
        ) noexcept {
            return
                static_cast<std::uint64_t>(
                    matrix.values.size()
                ) *
                static_cast<std::uint64_t>(sizeof(float));
        }

        std::uint64_t persistent_input_bytes(
            const CSRGraph& graph,
            const DenseMatrix& features,
            const Model& model
        ) noexcept {
            std::uint64_t bytes = 0;

            bytes +=
                static_cast<std::uint64_t>(
                    graph.row_ptr.size()
                ) *
                static_cast<std::uint64_t>(
                    sizeof(EdgeOffset)
                );

            bytes +=
                static_cast<std::uint64_t>(
                    graph.col_idx.size()
                ) *
                static_cast<std::uint64_t>(
                    sizeof(VertexId)
                );

            bytes += dense_payload_bytes(features);

            if (graph.edge_features.has_value()) {
                bytes += dense_payload_bytes(
                    graph.edge_features.value()
                );
            }

            for (const Layer& layer : model.layers) {
                bytes += dense_payload_bytes(
                    layer.weights
                );
            }

            return bytes;
        }

        DenseMatrix aggregate_mean_openmp(const CSRGraph& graph, const DenseMatrix& input, int thread_count){
            const std::size_t node_count = static_cast<std::size_t>(graph.num_nodes);
            const std::size_t feature_count = static_cast<std::size_t>(input.cols);

            DenseMatrix output;
            output.rows = graph.num_nodes;
            output.cols = input.cols;

            output.values.assign(
                node_count * feature_count, 0.0F
            );

            const std::int64_t signed_node_count = static_cast<std::int64_t>(node_count);

    #pragma omp parallel for schedule(runtime) num_threads(thread_count)
        for (std::int64_t destination = 0; destination < signed_node_count; ++destination){
            const std::size_t destination_index = static_cast<std::size_t>(destination);
            const EdgeOffset begin = graph.row_ptr[destination_index];
            const EdgeOffset end = graph.row_ptr[destination_index+1];
            
            const EdgeOffset degree = end - begin;

            if (degree==0)
                continue;
            
            for (EdgeOffset edge_position = begin; edge_position < end; ++edge_position){
                const std::size_t edge_index = static_cast<std::size_t>(edge_position);
                const VertexId source = graph.col_idx[edge_index];
            
                for (std::size_t feature =0; feature < feature_count; ++feature){
                    const std::size_t output_index = destination_index * feature_count + feature;
                    const std::size_t input_index = static_cast<std::size_t>(source) * feature_count + feature;

                    output.values[output_index] += input.values[input_index];
                }
            }
            const float degree_as_float = static_cast<float>(degree);
            for (std::size_t feature = 0; feature < feature_count; ++feature){
                const std::size_t output_index = destination_index * feature_count + feature;
                
                output.values[output_index] /= degree_as_float;
            }
        }
        return output;
/*
    不同线程不会写同一个output_index,所以不需要atomic，critical，mutex
*/
    }

    DenseMatrix linear_transform_openmp(
        const DenseMatrix& input, const DenseMatrix& weights, int thread_count
    ){
        const std::size_t node_count = static_cast<std::size_t>(input.rows);
        const std::size_t input_feature_count = static_cast<std::size_t>(input.cols);
        const std::size_t output_feature_count = static_cast<std::size_t>(weights.cols);

        DenseMatrix output;
        output.rows = input.rows;
        output.cols = weights.cols;

        output.values.assign(
            node_count * output_feature_count, 0.0F
        );

        const std::int64_t signed_node_count = static_cast<std::int64_t>(node_count);
        const std::int64_t signed_output_feature_count = static_cast<std::int64_t>(output_feature_count);

#pragma omp parallel for collapse(2) schedule(runtime) num_threads(thread_count)
        for (std::int64_t node =0; node < signed_node_count; ++node){
            for (std::int64_t output_feature = 0; output_feature < signed_output_feature_count; ++output_feature ){
                const std::size_t node_index = static_cast<std::size_t>(node);
                const std::size_t output_feature_index = static_cast<std::size_t>(output_feature);

                float sum = 0.0F;
                for (std::size_t input_feature = 0; input_feature < input_feature_count; ++input_feature){
                    const std::size_t input_index = node_index * input_feature_count + input_feature;
                    const std::size_t weight_index = input_feature * output_feature_count + output_feature_index;

                    sum += input.values[input_index] * weights.values[weight_index];
                }
                const std::size_t output_index = node_index * output_feature_count + output_feature_index;
                output.values[output_index] = sum;
            }
        }
        return output;
    }

    #endif

} // namespace 


RunResult run_cpu_openmp(
    const CSRGraph& graph,
    const DenseMatrix& features,
    const Model& model,
    const InferenceConfig& config) {
        #ifndef GNNI_HAS_OPENMP
            static_cast<void>(graph);
            static_cast<void>(features);
            static_cast<void>(model);
            static_cast<void>(config);

            throw std::runtime_error(
                "OpenMP support is not available in this build"
            );
        #else

            using Clock = std::chrono::steady_clock;
            const auto inference_start = Clock::now();

            if (config.aggregation != AggregationKind::Mean) {
                throw std::runtime_error(
                    "OpenMP backend currently supports only mean aggregation"
                );
            }

            if (config.output_transform != OutputTransform::Logits) {
                throw std::runtime_error(
                    "OpenMP backend currently supports only logits output"
                );
            }

            if (config.num_threads > static_cast<std::uint32_t>(
                std::numeric_limits<int>::max()
            )){
                throw std::invalid_argument(
                    "OpenMP thread count exceeds int range"
                );
            }

            const int thread_count =
                config.num_threads == 0 ? omp_get_max_threads() : static_cast<int>(config.num_threads);

            DenseMatrix current = features;

            std::uint64_t peak_workspace_bytes = dense_payload_bytes(current);

            omp_sched_t previous_schedule;
            int previous_chunk_size = 0;

            omp_get_schedule(
                &previous_schedule,
                &previous_chunk_size
            );

            configure_openmp_schedule(
                config.openmp_schedule,
                config.openmp_chunk_size
            );

            const auto compute_start = Clock::now();

            for (const Layer& layer : model.layers) {
                DenseMatrix aggregated = aggregate_mean_openmp(
                        graph, current, thread_count
                    );

                DenseMatrix transformed = linear_transform_openmp(
                        aggregated, layer.weights, thread_count
                    );

                apply_activation_openmp(
                    transformed, layer.activation, thread_count
                );

                const std::uint64_t layer_workspace_bytes = dense_payload_bytes(current) +
                                                            dense_payload_bytes(aggregated) + 
                                                            dense_payload_bytes(transformed);
                
                peak_workspace_bytes =std::max(peak_workspace_bytes, layer_workspace_bytes);

                current = std::move(transformed);
            }

            const auto compute_end = Clock::now();

            RunResult result;
            result.output = std::move(current);

            // 恢复调用前的OpenMP runtime schedule.
            omp_set_schedule(
                previous_schedule, previous_chunk_size
            );

            const auto inference_end = Clock::now();

            result.metrics.timing.compute_ms =
                std::chrono::duration<double, std::milli>(
                    compute_end - compute_start
                ).count();

            result.metrics.timing.inference_ms =
                std::chrono::duration<double, std::milli>(
                    inference_end - inference_start
                ).count();


            result.metrics.memory.host_bytes =
                persistent_input_bytes(
                    graph,
                    features,
                    model
                ) + peak_workspace_bytes;

            const double compute_seconds =
                result.metrics.timing.compute_ms / 1000.0;

            if (compute_seconds > 0.0) {
                const double layer_count =
                    static_cast<double>(
                        model.layers.size()
                    );

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
        #endif
}

}  // namespace gnni

