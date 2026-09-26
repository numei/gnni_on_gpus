#include "gnni/api.hpp"
#include "validate.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <climits>
#include <limits>
#include <stdexcept>
#include <utility>
#ifdef GNNI_HAS_OPENMP
#include <omp.h>
#endif

namespace gnni_cpu {
namespace {
#ifdef GNNI_HAS_OPENMP
// Explicit schedules work with the MSVC OpenMP 2 runtime, without changing
// runtime scheduling state (including on exceptional exits).
template<class Function>
void parallel_rows(std::int64_t count, const InferenceConfig& config, Function fn) {
    const int threads = config.num_threads ? static_cast<int>(config.num_threads) : omp_get_max_threads();
    const int chunk = config.openmp_chunk_size ? static_cast<int>(config.openmp_chunk_size) : 1;
    switch (config.openmp_schedule) {
    case OpenMPScheduleKind::Dynamic:
        #pragma omp parallel for schedule(dynamic, chunk) num_threads(threads)
        for (std::int64_t i=0; i<count; ++i) fn(i);
        break;
    case OpenMPScheduleKind::Guided:
        #pragma omp parallel for schedule(guided, chunk) num_threads(threads)
        for (std::int64_t i=0; i<count; ++i) fn(i);
        break;
    case OpenMPScheduleKind::Default:
    case OpenMPScheduleKind::Static:
        if (config.openmp_chunk_size) {
            #pragma omp parallel for schedule(static, chunk) num_threads(threads)
            for (std::int64_t i=0; i<count; ++i) fn(i);
        } else {
            #pragma omp parallel for schedule(static) num_threads(threads)
            for (std::int64_t i=0; i<count; ++i) fn(i);
        }
        break;
    default: throw std::invalid_argument("invalid OpenMP schedule");
    }
}
#endif
}

RunResult run_cpu_openmp(const CSRGraph& graph, const DenseMatrix& features,
                        const Model& model, const InferenceConfig& config) {
#ifndef GNNI_HAS_OPENMP
    (void)graph; (void)features; (void)model; (void)config;
    throw std::runtime_error("OpenMP support is not available in this build");
#else
    using Clock = std::chrono::steady_clock;
    const auto start = Clock::now();
    cpu_detail::validate_inputs(graph, features, model);
    if (config.aggregation != AggregationKind::Mean && config.aggregation != AggregationKind::Sum &&
        config.aggregation != AggregationKind::SymmetricGcn)
        throw std::invalid_argument("unsupported aggregation");
    if (config.output_transform != OutputTransform::Logits)
        throw std::invalid_argument("OpenMP backend supports logits output only");
    if (config.num_threads > INT_MAX || config.openmp_chunk_size > INT_MAX)
        throw std::invalid_argument("OpenMP configuration exceeds int range");
    DenseMatrix current = features;
    std::uint64_t persistent = graph.row_ptr.size()*sizeof(EdgeOffset) + graph.col_idx.size()*sizeof(VertexId) + features.values.size()*sizeof(float);
    for (const auto& layer : model.layers) persistent += layer.weights.values.size()*sizeof(float);
    std::uint64_t peak = current.values.size()*sizeof(float);
    const auto compute_start = Clock::now();
    for (const auto& layer : model.layers) {
        DenseMatrix aggregated{graph.num_nodes, current.cols, std::vector<float>(current.values.size(),0.0F)};
        parallel_rows(static_cast<std::int64_t>(graph.num_nodes),config,[&](std::int64_t row){
            const auto degree=graph.row_ptr[row+1]-graph.row_ptr[row];
            for(auto e=graph.row_ptr[row]; e<graph.row_ptr[row+1]; ++e){
                const auto u=graph.col_idx[e];
                float scale=1.0F;
                if(config.aggregation==AggregationKind::SymmetricGcn){
                    const auto du=graph.row_ptr[u+1]-graph.row_ptr[u];
                    scale=du?1.0F/std::sqrt(float(degree)*float(du)):0.0F;
                }
                for(std::size_t f=0;f<current.cols;++f)
                    aggregated.values[row*current.cols+f]+=scale*current.values[u*current.cols+f];
            }
            if(degree && config.aggregation==AggregationKind::Mean)
                for(std::size_t f=0;f<current.cols;++f) aggregated.values[row*current.cols+f]/=float(degree);
        });
        const auto width=layer.weights.cols;
        if(graph.num_nodes > std::numeric_limits<std::size_t>::max()/width)
            throw std::invalid_argument("output shape overflow");
        DenseMatrix next{graph.num_nodes,width,std::vector<float>(graph.num_nodes*width,0.0F)};
        parallel_rows(static_cast<std::int64_t>(graph.num_nodes),config,[&](std::int64_t row){
            for(std::size_t k=0;k<current.cols;++k){
                const float x=aggregated.values[row*current.cols+k];
                for(std::size_t f=0;f<width;++f) next.values[row*width+f]+=x*layer.weights.values[k*width+f];
            }
            if(layer.activation==ActivationKind::ReLU)
                for(std::size_t f=0;f<width;++f) next.values[row*width+f]=std::max(0.0F,next.values[row*width+f]);
        });
        peak=std::max(peak,static_cast<std::uint64_t>((current.values.size()+aggregated.values.size()+next.values.size())*sizeof(float)));
        current=std::move(next);
    }
    const auto end=Clock::now();
    RunResult result;
    result.output=std::move(current);
    result.metrics.timing.compute_ms=std::chrono::duration<double,std::milli>(end-compute_start).count();
    result.metrics.timing.inference_ms=std::chrono::duration<double,std::milli>(end-start).count();
    result.metrics.memory.host_bytes=persistent+peak;
    const double seconds=result.metrics.timing.compute_ms/1000.0;
    if(seconds>0){
        result.metrics.nodes_per_second=double(graph.num_nodes)*model.layers.size()/seconds;
        result.metrics.edges_per_second=double(graph.num_edges)*model.layers.size()/seconds;
    }
    return result;
#endif
}
}
