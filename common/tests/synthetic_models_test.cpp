#include "gnni/synthetic_graph.hpp"

#include <algorithm>
#include <cmath>
#include <exception>
#include <functional>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>

namespace {

bool same_graph(const gnni::CSRGraph& left, const gnni::CSRGraph& right) {
  return left.num_vertices == right.num_vertices &&
         left.row_offsets == right.row_offsets &&
         left.column_indices == right.column_indices;
}

bool has_edge(const gnni::CSRGraph& graph, int row, int column) {
  const std::size_t index = static_cast<std::size_t>(row);
  return std::binary_search(
      graph.column_indices.begin() + graph.row_offsets[index],
      graph.column_indices.begin() + graph.row_offsets[index + 1U], column);
}

void validate(const gnni::GeneratedGraph& generated, bool self_loops) {
  gnni::Dataset dataset;
  dataset.graph = generated.graph;
  dataset.features = gnni::DenseMatrix(
      static_cast<std::size_t>(generated.graph.num_vertices), 1);
  dataset.metadata.self_loops = self_loops;
  gnni::validate_dataset(dataset);
  const std::uint64_t expected_entries =
      generated.logical_edges * 2U +
      (self_loops ? static_cast<std::uint64_t>(generated.graph.num_vertices)
                  : 0U);
  if (generated.graph.column_indices.size() != expected_entries) {
    throw std::runtime_error("logical edge count disagrees with CSR");
  }
}

template <typename Function>
void expect_invalid(Function&& function) {
  try {
    function();
  } catch (const std::invalid_argument&) {
    return;
  }
  throw std::runtime_error("expected invalid_argument");
}

void test_ba() {
  const auto first = gnni::generate_barabasi_albert(20, 3, 42, false);
  const auto again = gnni::generate_barabasi_albert(20, 3, 42, false);
  const auto changed = gnni::generate_barabasi_albert(20, 3, 43, false);
  if (first.logical_edges != 54 || !same_graph(first.graph, again.graph) ||
      same_graph(first.graph, changed.graph)) {
    throw std::runtime_error("BA count or determinism is wrong");
  }
  for (int row = 0; row < 4; ++row) {
    for (int column = 0; column < 4; ++column) {
      if (row != column && !has_edge(first.graph, row, column)) {
        throw std::runtime_error("BA initial clique is incomplete");
      }
    }
  }
  validate(first, false);
  validate(gnni::generate_barabasi_albert(20, 3, 42, true), true);
  expect_invalid([] { gnni::generate_barabasi_albert(1, 1, 1, false); });
  expect_invalid([] { gnni::generate_barabasi_albert(10, 0, 1, false); });
  expect_invalid([] { gnni::generate_barabasi_albert(10, 10, 1, false); });
}

void test_ws() {
  const auto lattice = gnni::generate_watts_strogatz(20, 4, 0.0, 42, false);
  if (lattice.logical_edges != 40) {
    throw std::runtime_error("WS lattice edge count is wrong");
  }
  for (int vertex = 0; vertex < 20; ++vertex) {
    for (int distance = 1; distance <= 2; ++distance) {
      if (!has_edge(lattice.graph, vertex, (vertex + distance) % 20) ||
          !has_edge(lattice.graph, vertex, (vertex + 20 - distance) % 20)) {
        throw std::runtime_error("WS ring lattice is wrong");
      }
    }
  }
  const auto first = gnni::generate_watts_strogatz(40, 6, 1.0, 42, false);
  const auto again = gnni::generate_watts_strogatz(40, 6, 1.0, 42, false);
  const auto changed = gnni::generate_watts_strogatz(40, 6, 1.0, 43, false);
  if (!same_graph(first.graph, again.graph) ||
      same_graph(first.graph, changed.graph)) {
    throw std::runtime_error("WS determinism is wrong");
  }
  validate(lattice, false);
  validate(first, false);
  validate(gnni::generate_watts_strogatz(20, 4, 0.5, 42, true), true);
  expect_invalid([] { gnni::generate_watts_strogatz(1, 2, 0.5, 1, false); });
  expect_invalid([] { gnni::generate_watts_strogatz(10, 0, 0.5, 1, false); });
  expect_invalid([] { gnni::generate_watts_strogatz(10, 3, 0.5, 1, false); });
  expect_invalid([] { gnni::generate_watts_strogatz(10, 10, 0.5, 1, false); });
  expect_invalid([] { gnni::generate_watts_strogatz(10, 4, -0.1, 1, false); });
  expect_invalid([] {
    gnni::generate_watts_strogatz(
        10, 4, std::numeric_limits<double>::quiet_NaN(), 1, false);
  });
}

}  // namespace

int main() {
  try {
    test_ba();
    test_ws();
    std::cout << "synthetic model tests passed\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "synthetic model test failed: " << error.what() << '\n';
    return 1;
  }
}
