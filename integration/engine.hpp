#pragma once
#include "gnni/dataset.hpp"
#include "gnni/api.hpp"
#include <memory>
#include <string>
#include <vector>

namespace q1 {
struct Options {
    std::string backend = "cpu";
    std::string aggregation = "mean";
    unsigned threads = 0;
    std::string schedule = "static";
};
struct Result {
    gnni_cpu::DenseMatrix output;
    double compute_ms=0, end_to_end_ms=0, h2d_ms=0, d2h_ms=0;
    std::uint64_t host_payload_bytes=0, device_payload_bytes=0;
};
gnni_cpu::Model fixed_model(std::size_t input_width, const std::vector<std::size_t>& widths, unsigned seed);
class Engine {
public:
    Engine(const gnni::Dataset&, const gnni_cpu::Model&, Options);
    ~Engine();
    Engine(const Engine&)=delete;
    Engine& operator=(const Engine&)=delete;
    Result run();
    double setup_ms() const;
    double graph_h2d_ms() const;
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
std::vector<std::string> available_backends();
}
