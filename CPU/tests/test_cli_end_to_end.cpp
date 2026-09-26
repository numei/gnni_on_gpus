#include "cli_options.hpp"

#include <array>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <sstream>
#include <stdexcept>
#include <string>

namespace {

class TemporaryDirectory {
public:
    TemporaryDirectory() {
        const auto unique = std::chrono::high_resolution_clock::now()
                                .time_since_epoch()
                                .count();
        path_ = std::filesystem::temp_directory_path() /
                ("gnni_cli_end_to_end_" + std::to_string(unique));
        std::filesystem::create_directories(path_);
    }

    ~TemporaryDirectory() {
        std::error_code error;
        std::filesystem::remove_all(path_, error);
    }

    const std::filesystem::path& path() const noexcept {
        return path_;
    }

private:
    std::filesystem::path path_;
};

class CoutCapture {
public:
    CoutCapture()
        : previous_buffer_(std::cout.rdbuf(buffer_.rdbuf())) {}

    ~CoutCapture() {
        std::cout.rdbuf(previous_buffer_);
    }

    std::string text() const {
        return buffer_.str();
    }

private:
    std::ostringstream buffer_;
    std::streambuf* previous_buffer_;
};

void write_u32(std::ofstream& output, std::uint32_t value) {
    for (unsigned int shift = 0; shift < 32; shift += 8) {
        output.put(static_cast<char>((value >> shift) & 0xFFU));
    }
}

void write_u64(std::ofstream& output, std::uint64_t value) {
    for (unsigned int shift = 0; shift < 64; shift += 8) {
        output.put(static_cast<char>((value >> shift) & 0xFFU));
    }
}

void write_f32(std::ofstream& output, float value) {
    std::uint32_t bits = 0;
    static_assert(sizeof(bits) == sizeof(value), "float32 is required");
    std::memcpy(&bits, &value, sizeof(bits));
    write_u32(output, bits);
}

void require_open_file(const std::ofstream& output, const char* file_kind) {
    if (!output) {
        throw std::runtime_error(
            std::string("failed to create ") + file_kind + " fixture");
    }
}

void write_graph_file(const std::filesystem::path& path) {
    std::ofstream output(path, std::ios::binary);
    require_open_file(output, "graph");

    constexpr std::array<char, 8> magic{
        'G', 'N', 'N', 'I', 'G', 'R', 'F', 0};
    output.write(magic.data(), static_cast<std::streamsize>(magic.size()));

    write_u32(output, 1);
    write_u32(output, 48);
    write_u32(output, 1);
    write_u32(output, 0);
    write_u64(output, 3);
    write_u64(output, 3);
    write_u64(output, 0);

    for (const std::uint64_t value : {0, 0, 2, 3}) {
        write_u64(output, value);
    }
    for (const std::uint32_t value : {0, 2, 1}) {
        write_u32(output, value);
    }
}

void write_feature_file(const std::filesystem::path& path) {
    std::ofstream output(path, std::ios::binary);
    require_open_file(output, "feature");

    constexpr std::array<char, 8> magic{
        'G', 'N', 'N', 'I', 'M', 'A', 'T', 0};
    output.write(magic.data(), static_cast<std::streamsize>(magic.size()));

    write_u32(output, 1);
    write_u32(output, 48);
    write_u32(output, 1);
    write_u32(output, 1);
    write_u64(output, 3);
    write_u64(output, 2);
    write_u64(output, 0);

    for (const float value : {
             1.0F, 2.0F,
             3.0F, 4.0F,
             5.0F, 6.0F}) {
        write_f32(output, value);
    }
}

void write_model_file(const std::filesystem::path& path) {
    std::ofstream output(path, std::ios::binary);
    require_open_file(output, "model");

    constexpr std::array<char, 8> magic{
        'G', 'N', 'N', 'I', 'M', 'D', 'L', 0};
    output.write(magic.data(), static_cast<std::streamsize>(magic.size()));

    write_u32(output, 1);
    write_u32(output, 48);
    write_u32(output, 1);
    write_u32(output, 1);
    write_u32(output, 2);
    write_u32(output, 0);
    write_u32(output, 24);
    write_u32(output, 0);
    write_u64(output, 0);

    write_u64(output, 2);
    write_u64(output, 2);
    write_u32(output, 1);
    write_u32(output, 0);

    write_u64(output, 2);
    write_u64(output, 1);
    write_u32(output, 0);
    write_u32(output, 0);

    for (const float value : {
             1.0F, -2.0F,
             2.0F,  1.0F,
             2.0F,  3.0F}) {
        write_f32(output, value);
    }
}

std::string read_text_file(const std::filesystem::path& path) {
    std::ifstream input(path, std::ios::binary);
    if (!input) {
        throw std::runtime_error("failed to read generated CSV report");
    }

    return std::string(
        std::istreambuf_iterator<char>(input),
        std::istreambuf_iterator<char>());
}

void require(bool condition, const char* message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

void test_openmp_cli_end_to_end() {
    TemporaryDirectory directory;
    const std::filesystem::path graph_path = directory.path() / "graph.gnnigraph";
    const std::filesystem::path features_path = directory.path() / "features.gnnimatrix";
    const std::filesystem::path model_path = directory.path() / "model.gnnimodel";
    const std::filesystem::path csv_path = directory.path() / "benchmark.csv";

    write_graph_file(graph_path);
    write_feature_file(features_path);
    write_model_file(model_path);

    const std::string graph_argument = graph_path.string();
    const std::string features_argument = features_path.string();
    const std::string model_argument = model_path.string();
    const std::string csv_argument = csv_path.string();

    const char* const argv[] = {
        "gnni",
        "--graph", graph_argument.c_str(),
        "--features", features_argument.c_str(),
        "--model", model_argument.c_str(),
        "--backend", "omp",
        "--threads", "2",
        "--omp-schedule", "static",
        "--omp-chunk-size", "1",
        "--warmup", "0",
        "--repeat", "2",
        "--verify",
        "--csv", csv_argument.c_str(),
    };

    CoutCapture capture;
    const int exit_code = gnni_cpu::cli::run_cli(
        static_cast<int>(sizeof(argv) / sizeof(argv[0])),
        argv);

    const std::string cli_output = capture.text();
    require(exit_code == 0, "CLI returned a non-zero exit code");
    require(
        cli_output.find("backend: omp") != std::string::npos,
        "text report does not identify the OpenMP backend");
    require(
        cli_output.find("within_tolerance: true") != std::string::npos,
        "OpenMP verification did not pass");

    const std::string csv = read_text_file(csv_path);
    require(
        csv.find("backend,aggregation,output") == 0,
        "CSV header is missing");
    require(
        csv.find("omp,mean,logits,2,static,1,0,2,") != std::string::npos,
        "CSV data row has incorrect CLI configuration");
}

}  // namespace

int main() {
    try {
        test_openmp_cli_end_to_end();
        std::cout << "PASS: CLI end-to-end workflow\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "CLI end-to-end test failed: "
                  << error.what() << '\n';
        return 1;
    }
}
