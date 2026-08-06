#include "gnni/gcn_cuda.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>

namespace {

struct Options {
  int vertices = 100000;
  int degree = 16;
  int input_features = 64;
  int output_features = 32;
  int warmup = 5;
  int iterations = 20;
};

int parse_integer(const char* text, const char* option, int minimum) {
  std::size_t consumed = 0;
  long long value = 0;
  try {
    value = std::stoll(text, &consumed);
  } catch (const std::exception&) {
    throw std::invalid_argument(std::string(option) + " requires an integer");
  }
  if (consumed != std::string(text).size() || value < minimum ||
      value > std::numeric_limits<int>::max()) {
    throw std::invalid_argument(std::string(option) + " is outside its range");
  }
  return static_cast<int>(value);
}

Options parse_options(int argc, char** argv) {
  Options options;
  for (int i = 1; i < argc; i += 2) {
    if (std::string(argv[i]) == "--help") {
      std::cout
          << "Usage: gcn_cuda_benchmark [--vertices N] [--degree D] "
             "[--features F] [--outputs F] [--warmup N] [--iterations N]\n";
      std::exit(0);
    }
    if (i + 1 >= argc) {
      throw std::invalid_argument(std::string(argv[i]) + " requires a value");
    }
    const std::string option = argv[i];
    if (option == "--vertices") {
      options.vertices = parse_integer(argv[i + 1], argv[i], 1);
    } else if (option == "--degree") {
      options.degree = parse_integer(argv[i + 1], argv[i], 0);
    } else if (option == "--features") {
      options.input_features = parse_integer(argv[i + 1], argv[i], 1);
    } else if (option == "--outputs") {
      options.output_features = parse_integer(argv[i + 1], argv[i], 1);
    } else if (option == "--warmup") {
      options.warmup = parse_integer(argv[i + 1], argv[i], 1);
    } else if (option == "--iterations") {
      options.iterations = parse_integer(argv[i + 1], argv[i], 1);
    } else {
      throw std::invalid_argument("unknown option: " + option);
    }
  }
  if (options.degree >= options.vertices && options.degree != 0) {
    throw std::invalid_argument("degree must be smaller than vertex count");
  }
  const long long edge_count =
      static_cast<long long>(options.vertices) * options.degree;
  if (edge_count > std::numeric_limits<int>::max()) {
    throw std::invalid_argument("edge count exceeds the CSR int range");
  }
  return options;
}

gnni::CSRGraph make_regular_graph(int vertices, int degree) {
  std::vector<int> offsets(static_cast<std::size_t>(vertices) + 1U);
  std::vector<int> columns;
  columns.reserve(static_cast<std::size_t>(vertices) * degree);
  for (int vertex = 0; vertex < vertices; ++vertex) {
    for (int edge = 0; edge < degree; ++edge) {
      columns.push_back((vertex + edge + 1) % vertices);
    }
    offsets[static_cast<std::size_t>(vertex) + 1U] =
        static_cast<int>(columns.size());
  }
  return {vertices, std::move(offsets), std::move(columns)};
}

gnni::DenseMatrix make_matrix(std::size_t rows, std::size_t cols,
                              float scale) {
  std::vector<float> values(rows * cols);
  for (std::size_t i = 0; i < values.size(); ++i) {
    const int centered = static_cast<int>((i * 48271U + 17U) % 257U) - 128;
    values[i] = scale * static_cast<float>(centered);
  }
  return {rows, cols, std::move(values)};
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const Options options = parse_options(argc, argv);
    const auto graph = make_regular_graph(options.vertices, options.degree);
    const auto features =
        make_matrix(options.vertices, options.input_features, 1.0F / 128.0F);
    const auto weights = make_matrix(options.input_features,
                                     options.output_features, 1.0F / 256.0F);
    const gnni::GcnLayerConfig config{gnni::Aggregation::SymmetricGcn,
                                      gnni::Activation::Relu};

    gnni::CudaGcnContext context(graph);
    for (int iteration = 0; iteration < options.warmup; ++iteration) {
      context.run_layer(features, weights, config);
    }

    double h2d_ms = 0.0;
    double kernel_ms = 0.0;
    double d2h_ms = 0.0;
    double cuda_pipeline_ms = 0.0;
    double end_to_end_ms = 0.0;
    gnni::GcnRunResult result;
    for (int iteration = 0; iteration < options.iterations; ++iteration) {
      result = context.run_layer(features, weights, config);
      h2d_ms += result.timings.h2d_ms;
      kernel_ms += result.timings.kernel_ms;
      d2h_ms += result.timings.d2h_ms;
      cuda_pipeline_ms += result.timings.h2d_ms + result.timings.kernel_ms +
                          result.timings.d2h_ms;
      end_to_end_ms += result.timings.end_to_end_ms;
    }

    const double repetitions = static_cast<double>(options.iterations);
    h2d_ms /= repetitions;
    kernel_ms /= repetitions;
    d2h_ms /= repetitions;
    cuda_pipeline_ms /= repetitions;
    end_to_end_ms /= repetitions;
    const double seconds = end_to_end_ms / 1000.0;
    const double nodes_per_second = options.vertices / seconds;
    const double edges_per_second = graph.column_indices.size() / seconds;
    double checksum = 0.0;
    for (float value : result.output.values) {
      checksum += value;
    }

    std::cout << std::fixed << std::setprecision(6)
              << "vertices=" << options.vertices << '\n'
              << "edges=" << graph.column_indices.size() << '\n'
              << "input_features=" << options.input_features << '\n'
              << "output_features=" << options.output_features << '\n'
              << "warmup_iterations=" << options.warmup << '\n'
              << "measured_iterations=" << options.iterations << '\n'
              << "graph_h2d_ms=" << context.graph_upload_ms() << '\n'
              << "avg_h2d_ms=" << h2d_ms << '\n'
              << "avg_kernel_ms=" << kernel_ms << '\n'
              << "avg_d2h_ms=" << d2h_ms << '\n'
              << "avg_cuda_pipeline_ms=" << cuda_pipeline_ms << '\n'
              << "avg_end_to_end_ms=" << end_to_end_ms << '\n'
              << "nodes_per_second=" << nodes_per_second << '\n'
              << "edges_per_second=" << edges_per_second << '\n'
              << "output_checksum=" << checksum << '\n';
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "benchmark failed: " << error.what() << '\n';
    return 1;
  }
}
