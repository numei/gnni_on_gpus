#include "engine.hpp"
#include "gnni/dataset_io.hpp"
#include "gnni/io.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <set>
#include <sstream>
#include <stdexcept>

namespace {
std::size_t number(const std::string& s,bool allow_zero=false){
    if(s.empty() || s.find_first_not_of("0123456789")!=std::string::npos) throw std::invalid_argument("expected nonnegative integer: "+s);
    std::size_t end=0;auto n=std::stoull(s,&end);
    if(end!=s.size() || (!n && !allow_zero) || n>2147483647ULL) throw std::invalid_argument("integer out of range: "+s);
    return static_cast<std::size_t>(n);
}
std::vector<std::size_t> widths(const std::string& s){
    if(s.empty() || s.back()==',') throw std::invalid_argument("invalid layer widths");
    std::vector<std::size_t> result;std::istringstream in(s);std::string part;
    while(std::getline(in,part,',')) result.push_back(number(part));
    return result;
}
std::string quote(const std::string& s){std::string r="\"";for(char c:s){if(c=='"')r+='"';r+=c;}return r+'"';}
}
int main(int argc,char** argv){
try{
    std::map<std::string,std::string> args;
    bool verify=false;
    const std::set<std::string> valid{"--dataset","--model","--layers","--backend","--aggregation","--threads","--schedule","--warmup","--repeat","--seed","--csv"};
    for(int i=1;i<argc;++i){
        std::string key=argv[i];
        if(key=="--help"){
            std::cout<<"gnni_q1 --dataset FILE [--model FILE | --layers 64,32] [--backend cpu|omp|cuda-vertex|cuda-tiled|cuda-edge|cuda-edge-tiled]\n"
              "  [--aggregation mean|sum|symmetric] [--threads N] [--schedule static|dynamic|guided]\n"
              "  [--warmup N] [--repeat N] [--seed N] [--verify] [--csv FILE]\n"
              "--list-backends lists compiled backends. CSV contains one row per measured inference.\n";
            return 0;
        }
        if(key=="--list-backends"){for(const auto& b:q1::available_backends()) std::cout<<b<<'\n';return 0;}
        if(key=="--verify"){if(verify) throw std::invalid_argument("duplicate --verify");verify=true;continue;}
        if(!valid.count(key) || i+1==argc) throw std::invalid_argument("unknown option or missing value: "+key);
        if(!args.emplace(key,argv[++i]).second) throw std::invalid_argument("duplicate option: "+key);
    }
    auto get=[&](const std::string& key,const std::string& fallback){return args.count(key)?args.at(key):fallback;};
    if(!args.count("--dataset")) throw std::invalid_argument("--dataset is required");
    if(args.count("--model") && args.count("--layers")) throw std::invalid_argument("--model and --layers are mutually exclusive");
    const auto load_start=std::chrono::steady_clock::now();
    auto data=gnni::load_dataset(args.at("--dataset"));
    const auto seed=static_cast<unsigned>(number(get("--seed","42"),true));
    const auto model=args.count("--model")?gnni_cpu::load_model(args.at("--model")):q1::fixed_model(data.features.cols,widths(get("--layers","64,32")),seed);
    const double load_ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-load_start).count();
    const auto repeat=number(get("--repeat","10")),warmup=number(get("--warmup","3"),true);
    q1::Options options;
    options.backend=get("--backend","cpu");options.aggregation=get("--aggregation","mean");
    options.threads=static_cast<unsigned>(number(get("--threads","0"),true));options.schedule=get("--schedule","static");
    q1::Engine engine(data,model,options);
    double max_error=0;
    if(verify){
        auto cpu_options=options;cpu_options.backend="cpu";
        q1::Engine reference(data,model,cpu_options);
        const auto expected=reference.run();const auto actual=engine.run();
        const auto comparison=gnni_cpu::compare_outputs(expected.output,actual.output);
        if(!comparison.shapes_match || !comparison.within_tolerance) throw std::runtime_error("correctness verification failed; max absolute error="+std::to_string(comparison.max_absolute_error));
        max_error=comparison.max_absolute_error;
    }
    for(std::size_t i=0;i<warmup;++i) engine.run();
    std::ofstream file;
    if(args.count("--csv")){
        if(std::filesystem::exists(args.at("--csv"))) throw std::invalid_argument("CSV output exists; choose a new path");
        file.open(args.at("--csv"));if(!file) throw std::runtime_error("cannot open CSV");
    }
    std::ostream& out=file.is_open()?file:std::cout;
    std::ostringstream layer_list;
    std::uint64_t model_bytes=0;
    for(std::size_t l=0;l<model.layers.size();++l){if(l)layer_list<<';';layer_list<<model.layers[l].weights.cols;model_bytes+=model.layers[l].weights.values.size()*sizeof(float);}
    int max_degree=0;double square=0;
    for(int v=0;v<data.graph.num_vertices;++v){int d=data.graph.row_offsets[v+1]-data.graph.row_offsets[v];max_degree=std::max(max_degree,d);square+=double(d)*d;}
    const double mean=double(data.graph.column_indices.size())/data.graph.num_vertices;
    const double cv=mean?std::sqrt(std::max(0.0,square/data.graph.num_vertices-mean*mean))/mean:0;
    out<<"dataset,backend,aggregation,vertices,edges,features,layer_widths,layers,seed,threads,schedule,sample,warmup,verified,max_abs_error,load_ms,setup_ms,graph_h2d_ms,compute_ms,end_to_end_ms,h2d_ms,d2h_ms,nodes_per_second,edges_per_second,host_payload_bytes,device_payload_bytes,max_degree,degree_cv,checksum\n";
    out<<std::setprecision(12);
    for(std::size_t i=0;i<repeat;++i){
        auto r=engine.run();double checksum=0;
        for(float v:r.output.values){if(!std::isfinite(v))throw std::runtime_error("non-finite inference output");checksum+=v;}
        const double seconds=r.end_to_end_ms/1000.0;
        if(seconds<=0)throw std::runtime_error("non-positive measurement");
        out<<quote(args.at("--dataset"))<<','<<options.backend<<','<<options.aggregation<<','<<data.graph.num_vertices<<','<<data.graph.column_indices.size()<<','<<data.features.cols<<','<<layer_list.str()<<','<<model.layers.size()<<','<<seed<<','<<options.threads<<','<<options.schedule<<','<<i<<','<<warmup<<','<<(verify?"true":"false")<<','<<max_error<<','<<load_ms<<','<<engine.setup_ms()<<','<<engine.graph_h2d_ms()<<','<<r.compute_ms<<','<<r.end_to_end_ms<<','<<r.h2d_ms<<','<<r.d2h_ms<<','<<data.graph.num_vertices/seconds<<','<<data.graph.column_indices.size()/seconds<<','<<(gnni::dataset_host_bytes(data)+model_bytes+r.host_payload_bytes)<<','<<r.device_payload_bytes<<','<<max_degree<<','<<cv<<','<<checksum<<'\n';
    }
    out.flush();if(!out)throw std::runtime_error("failed to write CSV");
    return 0;
}catch(const std::exception& e){std::cerr<<"gnni_q1: "<<e.what()<<'\n';return 1;}
}
