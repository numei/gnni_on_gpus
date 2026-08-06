#include "gnni/device_buffer.hpp"
#include "gnni/gcn_cuda.hpp"

#include <algorithm>
#include <cmath>
#include <exception>
#include <functional>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

using gnni::Activation;
using gnni::Aggregation;
using gnni::CSRGraph;
using gnni::DeviceBuffer;
using gnni::CudaGcnContext;
using gnni::DenseMatrix;
using gnni::GcnLayerConfig;

constexpr float kAtol = 1.0e-5F;
constexpr float kRtol = 1.0e-4F;

DenseMatrix cpu_layer(const CSRGraph& graph, const DenseMatrix& input,
                      const DenseMatrix& weights,
                      const GcnLayerConfig& config) {
  DenseMatrix aggregate(graph.num_vertices, input.cols);
  for (int vertex = 0; vertex < graph.num_vertices; ++vertex) {
    const int begin = graph.row_offsets[vertex];
    const int end = graph.row_offsets[vertex + 1];
    const int degree = end - begin;
    for (int edge = begin; edge < end; ++edge) {
      const int neighbor = graph.column_indices[edge];
      float scale = 1.0F;
      if (config.aggregation == Aggregation::Mean) {
        scale = 1.0F / static_cast<float>(degree);
      } else if (config.aggregation == Aggregation::SymmetricGcn) {
        const int neighbor_degree =
            graph.row_offsets[neighbor + 1] - graph.row_offsets[neighbor];
        scale = neighbor_degree == 0
                    ? 0.0F
                    : 1.0F / std::sqrt(static_cast<float>(degree) *
                                       static_cast<float>(neighbor_degree));
      }
      for (std::size_t feature = 0; feature < input.cols; ++feature) {
        aggregate.values[static_cast<std::size_t>(vertex) * input.cols +
                         feature] +=
            scale * input.values[static_cast<std::size_t>(neighbor) *
                                     input.cols +
                                 feature];
      }
    }
  }

  DenseMatrix output(graph.num_vertices, weights.cols);
  for (int vertex = 0; vertex < graph.num_vertices; ++vertex) {
    for (std::size_t out_feature = 0; out_feature < weights.cols;
         ++out_feature) {
      float value = 0.0F;
      for (std::size_t in_feature = 0; in_feature < input.cols; ++in_feature) {
        value += aggregate.values[static_cast<std::size_t>(vertex) *
                                      input.cols +
                                  in_feature] *
                 weights.values[in_feature * weights.cols + out_feature];
      }
      if (config.activation == Activation::Relu) {
        value = std::max(0.0F, value);
      }
      output.values[static_cast<std::size_t>(vertex) * weights.cols +
                    out_feature] = value;
    }
  }
  return output;
}

DenseMatrix cpu_network(const CSRGraph& graph, DenseMatrix input,
                        const std::vector<DenseMatrix>& weights,
                        const std::vector<GcnLayerConfig>& configs) {
  for (std::size_t layer = 0; layer < weights.size(); ++layer) {
    input = cpu_layer(graph, input, weights[layer], configs[layer]);
  }
  return input;
}

void expect_close(const DenseMatrix& actual, const DenseMatrix& expected,
                  const std::string& label) {
  if (actual.rows != expected.rows || actual.cols != expected.cols) {
    throw std::runtime_error(label + ": shape mismatch");
  }
  float max_error = 0.0F;
  for (std::size_t i = 0; i < actual.values.size(); ++i) {
    if (!std::isfinite(actual.values[i]) ||
        !std::isfinite(expected.values[i])) {
      throw std::runtime_error(label + ": non-finite value at " +
                               std::to_string(i));
    }
    const float error = std::abs(actual.values[i] - expected.values[i]);
    const float tolerance = kAtol + kRtol * std::abs(expected.values[i]);
    max_error = std::max(max_error, error);
    if (error > tolerance) {
      std::ostringstream message;
      message << label << ": mismatch at " << i << ", expected "
              << expected.values[i] << ", got " << actual.values[i]
              << ", tolerance " << tolerance;
      throw std::runtime_error(message.str());
    }
  }
  std::cout << "  " << label << " max_abs_error=" << max_error << '\n';
}

void test_comparator_rejects_non_finite() {
  const DenseMatrix actual(
      1, 1, {std::numeric_limits<float>::quiet_NaN()});
  const DenseMatrix expected(1, 1, {0.0F});
  try {
    expect_close(actual, expected, "nan_comparator_guard");
  } catch (const std::runtime_error&) {
    return;
  }
  throw std::runtime_error(
      "correctness comparator must reject non-finite CUDA output");
}

void test_device_buffer_rejects_byte_overflow() {
  DeviceBuffer<float> buffer;
  const std::size_t overflowing_count =
      std::numeric_limits<std::size_t>::max() / sizeof(float) + 1U;
  try {
    buffer.resize(overflowing_count);
  } catch (const std::invalid_argument&) {
    return;
  }
  throw std::runtime_error("DeviceBuffer must reject byte-count overflow");
}

DenseMatrix sequence_matrix(std::size_t rows, std::size_t cols, float scale) {
  std::vector<float> values(rows * cols);
  for (std::size_t i = 0; i < values.size(); ++i) {
    const int centered = static_cast<int>((i * 17U + 3U) % 23U) - 11;
    values[i] = scale * static_cast<float>(centered);
  }
  return DenseMatrix(rows, cols, std::move(values));
}

CSRGraph bidirectional_chain(int vertices) {
  std::vector<int> offsets(static_cast<std::size_t>(vertices) + 1U, 0);
  std::vector<int> columns;
  for (int vertex = 0; vertex < vertices; ++vertex) {
    if (vertex > 0) {
      columns.push_back(vertex - 1);
    }
    if (vertex + 1 < vertices) {
      columns.push_back(vertex + 1);
    }
    offsets[static_cast<std::size_t>(vertex) + 1U] =
        static_cast<int>(columns.size());
  }
  return CSRGraph(vertices, std::move(offsets), std::move(columns));
}

void check_timing(const gnni::CudaTimings& timing) {
  if (timing.graph_h2d_ms < 0.0F || timing.h2d_ms < 0.0F ||
      timing.kernel_ms < 0.0F || timing.d2h_ms < 0.0F ||
      timing.end_to_end_ms < 0.0F) {
    throw std::runtime_error("timings must be non-negative");
  }
}

void test_single_vertex_literal_result() {
  const CSRGraph graph(1, {0, 1}, {0});
  const DenseMatrix input(1, 3, {2.0F, -1.0F, 3.0F});
  const DenseMatrix weights(3, 2,
                            {1.0F, 2.0F, -1.0F, 0.5F, 0.25F, -1.0F});
  const DenseMatrix expected(1, 2, {3.75F, 0.5F});

  const auto result = gnni::run_gcn_cuda_vertex_centric(
      graph, input, weights, {Aggregation::Sum, Activation::None});
  expect_close(result.output, expected, "single_vertex_literal");
  check_timing(result.timings);
}

void test_chain_mean_relu() {
  const auto graph = bidirectional_chain(5);
  const auto input = sequence_matrix(5, 3, 0.25F);
  const auto weights = sequence_matrix(3, 4, 0.125F);
  const GcnLayerConfig config{Aggregation::Mean, Activation::Relu};
  const auto result =
      gnni::run_gcn_cuda_vertex_centric(graph, input, weights, config);
  expect_close(result.output, cpu_layer(graph, input, weights, config),
               "chain_mean_relu");
}

void test_star_sum() {
  const CSRGraph graph(6, {0, 5, 6, 7, 8, 9, 10},
                       {1, 2, 3, 4, 5, 0, 0, 0, 0, 0});
  const auto input = sequence_matrix(6, 7, 0.0625F);
  const auto weights = sequence_matrix(7, 5, 0.03125F);
  const GcnLayerConfig config{Aggregation::Sum, Activation::None};
  const auto result =
      gnni::run_gcn_cuda_vertex_centric(graph, input, weights, config);
  expect_close(result.output, cpu_layer(graph, input, weights, config),
               "star_sum");
}

void test_isolated_vertices_symmetric() {
  const CSRGraph graph(4, {0, 1, 2, 2, 3}, {1, 0, 3});
  const auto input = sequence_matrix(4, 5, 0.2F);
  const auto weights = sequence_matrix(5, 3, 0.1F);
  const GcnLayerConfig config{Aggregation::SymmetricGcn, Activation::None};
  const auto expected = cpu_layer(graph, input, weights, config);
  const auto result =
      gnni::run_gcn_cuda_vertex_centric(graph, input, weights, config);
  expect_close(result.output, expected, "isolated_symmetric");
  for (std::size_t feature = 0; feature < result.output.cols; ++feature) {
    if (result.output.values[2U * result.output.cols + feature] != 0.0F) {
      throw std::runtime_error("isolated vertex must produce zero output");
    }
  }
}

void test_zero_edge_graph() {
  const CSRGraph graph(3, {0, 0, 0, 0}, {});
  const auto input = sequence_matrix(3, 4, 0.5F);
  const auto weights = sequence_matrix(4, 3, 0.25F);
  const GcnLayerConfig config{Aggregation::Mean, Activation::Relu};
  const auto result =
      gnni::run_gcn_cuda_vertex_centric(graph, input, weights, config);
  expect_close(result.output, DenseMatrix(3, 3), "zero_edge_graph");
}

void test_non_block_multiple_dimensions() {
  const auto graph = bidirectional_chain(37);
  const auto input = sequence_matrix(37, 13, 0.015625F);
  const auto weights = sequence_matrix(13, 11, 0.0078125F);
  const GcnLayerConfig config{Aggregation::SymmetricGcn, Activation::Relu};
  const auto result =
      gnni::run_gcn_cuda_vertex_centric(graph, input, weights, config);
  expect_close(result.output, cpu_layer(graph, input, weights, config),
               "non_block_multiple");
}

void test_multilayer_and_context_reuse() {
  const auto graph = bidirectional_chain(7);
  const auto first_input = sequence_matrix(7, 3, 0.125F);
  const std::vector<DenseMatrix> first_weights = {
      sequence_matrix(3, 5, 0.0625F), sequence_matrix(5, 2, 0.03125F)};
  const std::vector<GcnLayerConfig> first_configs = {
      {Aggregation::Mean, Activation::Relu},
      {Aggregation::Sum, Activation::None}};

  CudaGcnContext context(graph);
  const auto first =
      context.run_network(first_input, first_weights, first_configs);
  expect_close(first.output,
               cpu_network(graph, first_input, first_weights, first_configs),
               "multilayer_first");

  const auto changed_input = sequence_matrix(7, 3, 0.2F);
  const std::vector<DenseMatrix> narrower_weights = {
      sequence_matrix(3, 4, 0.04F), sequence_matrix(4, 3, 0.03F)};
  const std::vector<GcnLayerConfig> changed_configs = {
      {Aggregation::SymmetricGcn, Activation::Relu},
      {Aggregation::Mean, Activation::None}};
  const auto reused =
      context.run_network(changed_input, narrower_weights, changed_configs);
  expect_close(reused.output,
               cpu_network(graph, changed_input, narrower_weights,
                           changed_configs),
               "multilayer_reused_changed_data");

  const std::vector<DenseMatrix> grown_weights = {
      sequence_matrix(3, 9, 0.02F), sequence_matrix(9, 4, 0.01F)};
  const auto grown =
      context.run_network(changed_input, grown_weights, first_configs);
  expect_close(grown.output,
               cpu_network(graph, changed_input, grown_weights, first_configs),
               "multilayer_reused_grown_workspace");
  check_timing(grown.timings);
  if (grown.timings.graph_h2d_ms != context.graph_upload_ms()) {
    throw std::runtime_error("result must report persistent graph upload time");
  }
}

template <typename Function>
void expect_invalid_argument(Function&& function, const std::string& label) {
  try {
    function();
  } catch (const std::invalid_argument&) {
    return;
  }
  throw std::runtime_error(label + ": expected std::invalid_argument");
}

void test_invalid_inputs() {
  expect_invalid_argument(
      [] { CudaGcnContext context(CSRGraph(2, {0, 1}, {1})); },
      "row offset length");
  expect_invalid_argument(
      [] { CudaGcnContext context(CSRGraph(2, {0, 1, 1}, {2})); },
      "column index range");

  const CSRGraph graph(2, {0, 1, 2}, {1, 0});
  CudaGcnContext context(graph);
  expect_invalid_argument(
      [&] {
        context.run_layer(DenseMatrix(1, 2, {1.0F, 2.0F}),
                          DenseMatrix(2, 1, {1.0F, 1.0F}), {});
      },
      "feature rows");
  expect_invalid_argument(
      [&] {
        context.run_network(DenseMatrix(2, 2, {1, 2, 3, 4}),
                            {DenseMatrix(2, 1, {1, 1})}, {});
      },
      "layer config count");
}

}  // namespace

int main() {
  const std::vector<std::pair<std::string, std::function<void()>>> tests = {
      {"comparator rejects non-finite", test_comparator_rejects_non_finite},
      {"device buffer rejects byte overflow",
       test_device_buffer_rejects_byte_overflow},
      {"single vertex literal result", test_single_vertex_literal_result},
      {"chain mean relu", test_chain_mean_relu},
      {"star sum", test_star_sum},
      {"isolated vertices symmetric", test_isolated_vertices_symmetric},
      {"zero edge graph", test_zero_edge_graph},
      {"non-block-multiple dimensions", test_non_block_multiple_dimensions},
      {"multilayer and context reuse", test_multilayer_and_context_reuse},
      {"invalid inputs", test_invalid_inputs},
  };

  int failures = 0;
  for (const auto& [name, test] : tests) {
    try {
      test();
      std::cout << "[PASS] " << name << '\n';
    } catch (const std::exception& error) {
      ++failures;
      std::cerr << "[FAIL] " << name << ": " << error.what() << '\n';
    }
  }
  if (failures != 0) {
    std::cerr << failures << " test(s) failed\n";
    return 1;
  }
  std::cout << tests.size() << " test(s) passed\n";
  return 0;
}
