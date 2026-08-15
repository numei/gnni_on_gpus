#include "gnni/dataset_io.hpp"
#include "gnni/gcn_cuda.hpp"
#include "gnni/synthetic_graph.hpp"

#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

struct Options {
  std::filesystem::path dataset;
  std::vector<std::size_t> layer_widths;
  int warmup = 5;
  int iterations = 20;
};

int parse_count(const std::string& value, const std::string& name,
                int minimum) {
  std::size_t consumed = 0;
  long long parsed = 0;
  try {
    parsed = std::stoll(value, &consumed);
  } catch (const std::exception&) {
    throw std::invalid_argument(name + " requires an integer");
  }
  if (consumed != value.size() || parsed < minimum ||
      parsed > std::numeric_limits<int>::max()) {
    throw std::invalid_argument(name + " is outside its allowed range");
  }
  return static_cast<int>(parsed);
}

std::vector<std::size_t> parse_widths(const std::string& text) {
  std::vector<std::size_t> widths;
  std::size_t begin = 0;
  while (begin <= text.size()) {
    const std::size_t end = text.find(',', begin);
    const std::string token = text.substr(begin, end - begin);
    if (token.empty()) {
      throw std::invalid_argument("--layer-widths contains an empty value");
    }
    widths.push_back(static_cast<std::size_t>(
        parse_count(token, "--layer-widths", 1)));
    if (end == std::string::npos) {
      break;
    }
    begin = end + 1U;
  }
  return widths;
}

Options parse_options(int argc, char** argv) {
  Options options;
  bool has_dataset = false;
  bool has_widths = false;
  for (int index = 1; index < argc; index += 2) {
    const std::string name = argv[index];
    if (name == "--help") {
      std::cout << "Usage: gcn_dataset_runner --dataset FILE "
                   "--layer-widths 64,32 [--warmup N] [--iterations N]\n";
      std::exit(0);
    }
    if (index + 1 >= argc) {
      throw std::invalid_argument(name + " requires a value");
    }
    const std::string value = argv[index + 1];
    if (name == "--dataset") {
      if (has_dataset) {
        throw std::invalid_argument("duplicate --dataset");
      }
      options.dataset = value;
      has_dataset = true;
    } else if (name == "--layer-widths") {
      if (has_widths) {
        throw std::invalid_argument("duplicate --layer-widths");
      }
      options.layer_widths = parse_widths(value);
      has_widths = true;
    } else if (name == "--warmup") {
      options.warmup = parse_count(value, name, 0);
    } else if (name == "--iterations") {
      options.iterations = parse_count(value, name, 1);
    } else {
      throw std::invalid_argument("unknown option: " + name);
    }
  }
  if (!has_dataset || !has_widths) {
    throw std::invalid_argument("--dataset and --layer-widths are required");
  }
  return options;
}

gnni::DenseMatrix make_weights(std::size_t rows, std::size_t cols,
                               std::size_t layer) {
  gnni::DenseMatrix weights(rows, cols);
  const std::uint64_t seed =
      UINT64_C(0x726f777765696768) ^ static_cast<std::uint64_t>(layer);
  const float scale = 1.0F / std::sqrt(static_cast<float>(rows));
  for (std::size_t row = 0; row < rows; ++row) {
    for (std::size_t column = 0; column < cols; ++column) {
      weights.values[row * cols + column] =
          gnni::synthetic_feature(seed, row, column) * scale;
    }
  }
  return weights;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const Options options = parse_options(argc, argv);
    const auto load_start = std::chrono::steady_clock::now();
    const gnni::Dataset dataset = gnni::load_dataset(options.dataset);
    const auto load_stop = std::chrono::steady_clock::now();
    const double load_ms = std::chrono::duration<double, std::milli>(
                               load_stop - load_start)
                               .count();

    std::vector<gnni::DenseMatrix> weights;
    std::vector<gnni::GcnLayerConfig> configs;
    weights.reserve(options.layer_widths.size());
    configs.reserve(options.layer_widths.size());
    std::size_t input_width = dataset.features.cols;
    for (std::size_t layer = 0; layer < options.layer_widths.size(); ++layer) {
      const std::size_t output_width = options.layer_widths[layer];
      weights.push_back(make_weights(input_width, output_width, layer));
      configs.push_back(
          {gnni::Aggregation::SymmetricGcn,
           layer + 1U == options.layer_widths.size()
               ? gnni::Activation::None
               : gnni::Activation::Relu});
      input_width = output_width;
    }

    gnni::CudaGcnContext context(dataset.graph);
    for (int iteration = 0; iteration < options.warmup; ++iteration) {
      context.run_network(dataset.features, weights, configs);
    }
    double h2d_ms = 0.0;
    double kernel_ms = 0.0;
    double d2h_ms = 0.0;
    double end_to_end_ms = 0.0;
    gnni::GcnRunResult result;
    for (int iteration = 0; iteration < options.iterations; ++iteration) {
      result = context.run_network(dataset.features, weights, configs);
      h2d_ms += result.timings.h2d_ms;
      kernel_ms += result.timings.kernel_ms;
      d2h_ms += result.timings.d2h_ms;
      end_to_end_ms += result.timings.end_to_end_ms;
    }
    const double count = static_cast<double>(options.iterations);
    h2d_ms /= count;
    kernel_ms /= count;
    d2h_ms /= count;
    end_to_end_ms /= count;
    const double pipeline_ms = h2d_ms + kernel_ms + d2h_ms;
    const double seconds = end_to_end_ms / 1000.0;
    double checksum = 0.0;
    for (float value : result.output.values) {
      if (!std::isfinite(value)) {
        throw std::runtime_error("CUDA output contains a non-finite value");
      }
      checksum += static_cast<double>(value);
    }

    std::cout << std::fixed << std::setprecision(6)
              << "dataset=" << options.dataset.string() << '\n'
              << "directed=" << (dataset.metadata.directed ? 1 : 0) << '\n'
              << "self_loops=" << (dataset.metadata.self_loops ? 1 : 0)
              << '\n'
              << "vertices=" << dataset.graph.num_vertices << '\n'
              << "csr_entries=" << dataset.graph.column_indices.size() << '\n'
              << "feature_dim=" << dataset.features.cols << '\n'
              << "num_classes=" << dataset.metadata.num_classes << '\n'
              << "labels_present=" << (!dataset.labels.empty() ? 1 : 0) << '\n'
              << "splits_present=" << (!dataset.splits.empty() ? 1 : 0) << '\n'
              << "warmup_iterations=" << options.warmup << '\n'
              << "measured_iterations=" << options.iterations << '\n'
              << "load_ms=" << load_ms << '\n'
              << "host_bytes=" << gnni::dataset_host_bytes(dataset) << '\n'
              << "graph_h2d_ms=" << context.graph_upload_ms() << '\n'
              << "avg_h2d_ms=" << h2d_ms << '\n'
              << "avg_kernel_ms=" << kernel_ms << '\n'
              << "avg_d2h_ms=" << d2h_ms << '\n'
              << "avg_cuda_pipeline_ms=" << pipeline_ms << '\n'
              << "avg_end_to_end_ms=" << end_to_end_ms << '\n'
              << "nodes_per_second="
              << static_cast<double>(dataset.graph.num_vertices) / seconds
              << '\n'
              << "csr_entries_per_second="
              << static_cast<double>(dataset.graph.column_indices.size()) /
                     seconds
              << '\n'
              << "output_rows=" << result.output.rows << '\n'
              << "output_cols=" << result.output.cols << '\n'
              << "output_checksum=" << checksum << '\n';
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "dataset runner failed: " << error.what() << '\n';
    return 1;
  }
}
