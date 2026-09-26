#include "engine.hpp"
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>

void require(bool v,const char* message){if(!v)throw std::runtime_error(message);}
int main(){try{
    // Directed asymmetric graph: incoming rows {1,2}, {2}, {}.
    gnni::Dataset d;
    d.metadata.directed=true;
    d.graph={3,{0,2,3,3},{1,2,2}};
    d.features={3,1,{2,4,8}};
    gnni_cpu::Model m{{{{1,1,{2}},gnni_cpu::ActivationKind::Identity}}};
    for(const auto& mode:{"mean","sum","symmetric"}){
        q1::Options o;o.aggregation=mode;
        auto reference=q1::Engine(d,m,o).run();
        float first=std::string(mode)=="mean"?12.0F:std::string(mode)=="sum"?24.0F:8/std::sqrt(2.0F);
        require(std::abs(reference.output.values[0]-first)<1e-5,"literal directed result mismatch");
        require(reference.output.values[2]==0,"isolated node must be zero");
        for(const auto& backend:q1::available_backends()){
            o.backend=backend;o.threads=2;
            for(const auto& schedule:{"static","dynamic","guided"}){
                o.schedule=schedule;
                auto actual=q1::Engine(d,m,o).run();
                require(gnni_cpu::compare_outputs(reference.output,actual.output).within_tolerance,"cross-backend mismatch");
            }
        }
    }
    // Partial tiles, skewed degrees, self loops and context reuse with 3 layers.
    d={};d.metadata.directed=true;d.metadata.self_loops=true;
    d.graph.num_vertices=37;d.graph.row_offsets={0};
    for(int v=0;v<37;++v){
        for(int u=0;u<37;++u)if(v==0 || u==v) d.graph.column_indices.push_back(u);
        d.graph.row_offsets.push_back(static_cast<int>(d.graph.column_indices.size()));
    }
    d.features=gnni::DenseMatrix(37,19);
    for(std::size_t i=0;i<d.features.values.size();++i)d.features.values[i]=float(int(i%17)-8)/9;
    m=q1::fixed_model(19,{33,17,7},42);
    for(const auto& mode:{"mean","sum","symmetric"}){
        q1::Options o;o.aggregation=mode;
        auto ref=q1::Engine(d,m,o).run();
        for(const auto& backend:q1::available_backends()){
            o.backend=backend;o.threads=2;q1::Engine e(d,m,o);
            for(int i=0;i<3;++i)require(gnni_cpu::compare_outputs(ref.output,e.run().output).within_tolerance,"multilayer/reuse mismatch");
        }
    }
    auto malformed=m;malformed.layers[0].weights.rows=20;
    bool rejected=false;try{q1::Engine e(d,malformed,{});}catch(const std::invalid_argument&){rejected=true;}
    require(rejected,"shape mismatch not rejected");
    malformed=m;malformed.layers[0].weights.values[0]=std::numeric_limits<float>::quiet_NaN();
    rejected=false;try{q1::Engine e(d,malformed,{});}catch(const std::invalid_argument&){rejected=true;}
    require(rejected,"NaN weight not rejected");
    gnni_cpu::CSRGraph impossible;
    impossible.num_nodes=std::numeric_limits<gnni_cpu::Dimension>::max();
    rejected=false;
    try{gnni_cpu::run_cpu_sequential(impossible,{1,1,{1.0F}},{{{{1,1,{1.0F}},gnni_cpu::ActivationKind::Identity}}},{});}
    catch(const std::invalid_argument&){rejected=true;}
    require(rejected,"overflowed CSR row count not rejected");
    d.metadata.self_loops=false;d.graph.column_indices.clear();d.graph.row_offsets.assign(38,0);
    for(const auto& backend:q1::available_backends()){
        q1::Options o;o.backend=backend;o.threads=2;
        auto r=q1::Engine(d,m,o).run();for(float x:r.output.values)require(x==0,"zero-edge output not zero");
    }
    std::cout<<"Integration: literal, all backends/modes/schedules, partial tiles, skew, reuse, invalid input and zero edges passed\n";
    return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
