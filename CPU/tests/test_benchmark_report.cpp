#include "gnni/benchmark.hpp"

#include <iostream>
#include <sstream>
#include <string>
#include <vector>

int main() {
    gnni_cpu::BenchmarkReport report;
    report.backend = gnni_cpu::BackendKind::CpuSequential;
    report.options.warmup_runs = 1;
    report.options.measured_runs = 2;
    report.compute_ms = gnni_cpu::summarize_samples({1.0, 3.0});
    report.inference_ms = gnni_cpu::summarize_samples({2.0, 4.0});

    gnni_cpu::RunMetrics first;
    first.timing.compute_ms = 1.0;
    first.timing.inference_ms = 2.0;
    first.memory.host_bytes = 100;
    first.nodes_per_second = 10.0;
    first.edges_per_second = 20.0;

    gnni_cpu::RunMetrics second;
    second.timing.compute_ms = 3.0;
    second.timing.inference_ms = 4.0;
    second.memory.host_bytes = 120;
    second.nodes_per_second = 30.0;
    second.edges_per_second = 40.0;

    report.samples = {first, second};

    std::ostringstream text_output;
    gnni_cpu::write_text_report(report, text_output);
    const std::string text = text_output.str();

    if (text.find("backend: cpu") == std::string::npos ||
        text.find("compute_ms: min=1, mean=2") == std::string::npos ||
        text.find("peak_host_bytes: 120") == std::string::npos) {
        std::cerr << "text benchmark report is incorrect\n" << text;
        return 1;
    }

    std::ostringstream csv_output;
    gnni_cpu::write_csv_report(report, csv_output);
    const std::string csv = csv_output.str();

    if (csv.find("backend,aggregation,output,threads") != 0 ||
        csv.find("cpu,mean,logits,0,default,0,1,2,") == std::string::npos ||
        csv.find(",20,30,120") == std::string::npos) {
        std::cerr << "CSV benchmark report is incorrect\n" << csv;
        return 1;
    }

    std::cout << "PASS: benchmark text and CSV reports\n";
    return 0;
}
