#include "engine.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <utility>
#ifdef Q1_HAS_CUDA
#include "gnni/gcn_cuda.hpp"
#endif

namespace q1 {
using Clock=std::chrono::steady_clock;
gnni_cpu::Model fixed_model(std::size_t input_width,const std::vector<std::size_t>& widths,unsigned seed){
    if(!input_width || widths.empty()) throw std::invalid_argument("model needs input width and layers");
    gnni_cpu::Model model;
    std::uint32_t state=seed;
    for(std::size_t l=0;l<widths.size();++l){
        const auto out=widths[l];
        if(!out || input_width>std::numeric_limits<std::size_t>::max()/out) throw std::invalid_argument("invalid layer shape");
        gnni_cpu::Layer layer;
        layer.weights={input_width,out,std::vector<float>(input_width*out)};
        const float scale=std::sqrt(6.0F/float(input_width+out));
        for(float& w:layer.weights.values){
            state=1664525U*state+1013904223U;
            w=(float(state>>8)/16777216.0F*2.0F-1.0F)*scale;
        }
        layer.activation=l+1==widths.size()?gnni_cpu::ActivationKind::Identity:gnni_cpu::ActivationKind::ReLU;
        model.layers.push_back(std::move(layer));
        input_width=out;
    }
    return model;
}
std::vector<std::string> available_backends(){
    std::vector<std::string> result{"cpu"};
    if(gnni_cpu::backend_capabilities(gnni_cpu::BackendKind::CpuOpenMP).available) result.push_back("omp");
#ifdef Q1_HAS_CUDA
    result.insert(result.end(),{"cuda-vertex","cuda-tiled","cuda-edge","cuda-edge-tiled"});
#endif
    return result;
}
struct Engine::Impl {
    gnni_cpu::CSRGraph graph;
    gnni_cpu::DenseMatrix input;
    gnni_cpu::Model model;
    gnni_cpu::InferenceConfig config;
    Options options;
    double setup=0, graph_upload=0;
    std::uint64_t extra_host=0;
#ifdef Q1_HAS_CUDA
    gnni::DenseMatrix gpu_input;
    std::vector<gnni::DenseMatrix> weights;
    std::vector<gnni::GcnLayerConfig> configs;
    std::unique_ptr<gnni::CudaGcnContext> gpu;
#endif
    Impl(const gnni::Dataset& data,const gnni_cpu::Model& m,Options o):options(std::move(o)){
        const auto start=Clock::now();
        gnni::validate_dataset(data);
        if(data.graph.num_vertices<=0) throw std::invalid_argument("inference requires nonempty graph");
        auto supported=available_backends();
        if(std::find(supported.begin(),supported.end(),options.backend)==supported.end()) throw std::invalid_argument("backend unavailable: "+options.backend);
        if(options.aggregation=="mean") config.aggregation=gnni_cpu::AggregationKind::Mean;
        else if(options.aggregation=="sum") config.aggregation=gnni_cpu::AggregationKind::Sum;
        else if(options.aggregation=="symmetric") config.aggregation=gnni_cpu::AggregationKind::SymmetricGcn;
        else throw std::invalid_argument("aggregation must be mean, sum, or symmetric");
        if(options.schedule=="static") config.openmp_schedule=gnni_cpu::OpenMPScheduleKind::Static;
        else if(options.schedule=="dynamic") config.openmp_schedule=gnni_cpu::OpenMPScheduleKind::Dynamic;
        else if(options.schedule=="guided") config.openmp_schedule=gnni_cpu::OpenMPScheduleKind::Guided;
        else throw std::invalid_argument("invalid schedule");
        config.num_threads=options.threads;
        if(m.layers.empty()) throw std::invalid_argument("empty model");
        std::size_t width=data.features.cols;
        for(const auto& layer:m.layers){
            const auto& w=layer.weights;
            if(w.rows!=width || !w.cols || w.rows>std::numeric_limits<std::size_t>::max()/w.cols || w.values.size()!=w.rows*w.cols)
                throw std::invalid_argument("model/dataset shape mismatch");
            if(layer.activation!=gnni_cpu::ActivationKind::Identity && layer.activation!=gnni_cpu::ActivationKind::ReLU)
                throw std::invalid_argument("invalid activation");
            for(float value:w.values) if(!std::isfinite(value)) throw std::invalid_argument("non-finite weight");
            width=static_cast<std::size_t>(w.cols);
        }
        if(options.backend=="cpu" || options.backend=="omp"){
            graph.num_nodes=data.graph.num_vertices;
            graph.num_edges=data.graph.column_indices.size();
            graph.directed=data.metadata.directed;
            graph.row_ptr.assign(data.graph.row_offsets.begin(),data.graph.row_offsets.end());
            graph.col_idx.assign(data.graph.column_indices.begin(),data.graph.column_indices.end());
            input={data.features.rows,data.features.cols,data.features.values};
            model=m;
        }
#ifdef Q1_HAS_CUDA
        else {
            gnni::CudaKernelOptions kernels;
            kernels.edge_centric=options.backend=="cuda-edge" || options.backend=="cuda-edge-tiled";
            kernels.tiled_dense=options.backend=="cuda-tiled" || options.backend=="cuda-edge-tiled";
            gpu=std::make_unique<gnni::CudaGcnContext>(data.graph,kernels);
            graph_upload=gpu->graph_upload_ms();
            gpu_input=data.features;
            extra_host=data.features.values.size()*sizeof(float);
            for(const auto& layer:m.layers){
                weights.emplace_back(layer.weights.rows,layer.weights.cols,layer.weights.values);
                extra_host+=layer.weights.values.size()*sizeof(float);
                configs.push_back({options.aggregation=="mean"?gnni::Aggregation::Mean:options.aggregation=="sum"?gnni::Aggregation::Sum:gnni::Aggregation::SymmetricGcn,
                    layer.activation==gnni_cpu::ActivationKind::ReLU?gnni::Activation::Relu:gnni::Activation::None});
            }
        }
#endif
        setup=std::chrono::duration<double,std::milli>(Clock::now()-start).count();
    }
    Result run(){
        Result out;
        if(options.backend=="cpu" || options.backend=="omp"){
            auto r=gnni_cpu::run_backend(options.backend=="cpu"?gnni_cpu::BackendKind::CpuSequential:gnni_cpu::BackendKind::CpuOpenMP,graph,input,model,config);
            out.output=std::move(r.output);
            out.compute_ms=r.metrics.timing.compute_ms;
            out.end_to_end_ms=r.metrics.timing.inference_ms;
            out.host_payload_bytes=r.metrics.memory.host_bytes;
        }
#ifdef Q1_HAS_CUDA
        else {
            auto r=gpu->run_network(gpu_input,weights,configs);
            out.output={r.output.rows,r.output.cols,std::move(r.output.values)};
            out.compute_ms=r.timings.kernel_ms;
            out.end_to_end_ms=r.timings.end_to_end_ms;
            out.h2d_ms=r.timings.h2d_ms;out.d2h_ms=r.timings.d2h_ms;
            out.device_payload_bytes=gpu->device_bytes();
            // Retained input/weights + packed weight staging + returned output.
            out.host_payload_bytes=extra_host+out.output.values.size()*sizeof(float);
            for(const auto& w:weights) out.host_payload_bytes+=w.values.size()*sizeof(float);
        }
#endif
        return out;
    }
};
Engine::Engine(const gnni::Dataset& d,const gnni_cpu::Model& m,Options o):impl_(std::make_unique<Impl>(d,m,std::move(o))){}
Engine::~Engine()=default;
Result Engine::run(){return impl_->run();}
double Engine::setup_ms()const{return impl_->setup;}
double Engine::graph_h2d_ms()const{return impl_->graph_upload;}
}
