#include "gnni/synthetic_graph.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <exception>
#include <functional>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

using Test = std::pair<std::string, std::function<void()>>;

template <typename Function>
void expect_invalid_argument(Function&& function, const std::string& label) {
  try {
    function();
  } catch (const std::invalid_argument&) {
    return;
  }
  throw std::runtime_error(label + ": expected std::invalid_argument");
}

bool same_graph(const gnni::CSRGraph& left, const gnni::CSRGraph& right) {
  return left.num_vertices == right.num_vertices &&
         left.row_offsets == right.row_offsets &&
         left.column_indices == right.column_indices;
}

bool has_edge(const gnni::CSRGraph& graph, int row, int column) {
  const auto row_index = static_cast<std::size_t>(row);
  const auto begin =
      graph.column_indices.begin() + graph.row_offsets[row_index];
  const auto end =
      graph.column_indices.begin() + graph.row_offsets[row_index + 1U];
  return std::binary_search(begin, end, column);
}

void expect_graph_invariants(const gnni::GeneratedGraph& generated,
                             bool self_loops) {
  gnni::Dataset dataset;
  dataset.graph = generated.graph;
  dataset.features =
      gnni::DenseMatrix(static_cast<std::size_t>(generated.graph.num_vertices),
                        1);
  dataset.metadata.self_loops = self_loops;
  gnni::validate_dataset(dataset);

  std::uint64_t off_diagonal_entries = 0;
  std::uint64_t diagonal_entries = 0;
  for (std::size_t row = 0;
       row < static_cast<std::size_t>(generated.graph.num_vertices); ++row) {
    const auto begin =
        static_cast<std::size_t>(generated.graph.row_offsets[row]);
    const auto end =
        static_cast<std::size_t>(generated.graph.row_offsets[row + 1U]);
    for (std::size_t entry = begin; entry < end; ++entry) {
      if (generated.graph.column_indices[entry] == static_cast<int>(row)) {
        ++diagonal_entries;
      } else {
        ++off_diagonal_entries;
      }
    }
  }
  const std::uint64_t expected_diagonal =
      self_loops ? static_cast<std::uint64_t>(generated.graph.num_vertices) : 0;
  if (diagonal_entries != expected_diagonal) {
    throw std::runtime_error("self-loop count does not match option");
  }
  if ((off_diagonal_entries % 2U) != 0 ||
      generated.logical_edges != off_diagonal_entries / 2U) {
    throw std::runtime_error("logical edge count does not match symmetric CSR");
  }
}

void test_feature_golden_values() {
  const std::vector<float> expected = {
      -0x1.ef00d8p-1F, 0x1.9939b8p-2F, -0x1.5426a8p-1F,
      -0x1.b7fb8p-2F,  -0x1.0befacp-1F, 0x1.f15ee8p-2F,
      -0x1.930994p-1F, 0x1.f6c3cp-4F,
  };
  const gnni::DenseMatrix features =
      gnni::generate_synthetic_features(2, 4, 42);
  if (features.rows != 2 || features.cols != 4 ||
      features.values != expected) {
    throw std::runtime_error("seed 42 feature golden values changed");
  }
  for (std::size_t index = 0; index < expected.size(); ++index) {
    const std::size_t vertex = index / 4;
    const std::size_t feature = index % 4;
    if (gnni::synthetic_feature(42, vertex, feature) != expected[index]) {
      throw std::runtime_error("synthetic_feature disagrees with matrix output");
    }
  }
}

void test_features_are_repeatable_bounded_and_coordinate_sensitive() {
  const gnni::DenseMatrix first =
      gnni::generate_synthetic_features(17, 9, 123456789);
  const gnni::DenseMatrix repeated =
      gnni::generate_synthetic_features(17, 9, 123456789);
  const gnni::DenseMatrix changed_seed =
      gnni::generate_synthetic_features(17, 9, 123456790);
  if (first.values != repeated.values) {
    throw std::runtime_error("same feature options are not repeatable");
  }
  if (first.values == changed_seed.values) {
    throw std::runtime_error("changing feature seed did not change the sequence");
  }
  for (float value : first.values) {
    if (!std::isfinite(value) || value < -1.0F || value > 1.0F) {
      throw std::runtime_error("feature lies outside finite [-1,1]");
    }
  }

  const float origin = gnni::synthetic_feature(123456789, 0, 0);
  if (origin == gnni::synthetic_feature(123456790, 0, 0) ||
      origin == gnni::synthetic_feature(123456789, 1, 0) ||
      origin == gnni::synthetic_feature(123456789, 0, 1)) {
    throw std::runtime_error("seed or coordinate change did not alter feature");
  }
}

void test_rng_primitives_have_fixed_mappings() {
  std::mt19937_64 bounded_generator(42);
  const std::vector<std::uint64_t> expected_mod_ten = {6, 4, 0, 2,
                                                       1, 8, 6, 4};
  std::vector<std::uint64_t> actual_mod_ten;
  for (std::size_t index = 0; index < expected_mod_ten.size(); ++index) {
    actual_mod_ten.push_back(
        gnni::detail::sample_bounded(bounded_generator, 10));
  }
  if (actual_mod_ten != expected_mod_ten) {
    throw std::runtime_error("bounded sampler golden sequence changed");
  }

  std::mt19937_64 rejection_generator(42);
  constexpr std::uint64_t large_bound = UINT64_C(9223372036854775809);
  const std::vector<std::uint64_t> expected_after_rejection = {
      UINT64_C(4706788815403344597), UINT64_C(2564676540648719015),
      UINT64_C(4651257987612965641), UINT64_C(7438999416573663572),
  };
  std::vector<std::uint64_t> actual_after_rejection;
  for (std::size_t index = 0; index < expected_after_rejection.size(); ++index) {
    actual_after_rejection.push_back(
        gnni::detail::sample_bounded(rejection_generator, large_bound));
  }
  if (actual_after_rejection != expected_after_rejection) {
    throw std::runtime_error("bounded sampler rejection sequence changed");
  }

  std::mt19937_64 unit_generator(42);
  if (gnni::detail::sample_unit_double(unit_generator) !=
      0x1.82a3befaddcbcp-1) {
    throw std::runtime_error("unit-double mapping changed");
  }
  expect_invalid_argument(
      [&] { gnni::detail::sample_bounded(bounded_generator, 0); },
      "zero bound");
}

void test_feature_generation_rejects_invalid_shapes() {
  expect_invalid_argument(
      [] { gnni::generate_synthetic_features(0, 1, 42); },
      "zero vertices");
  expect_invalid_argument(
      [] { gnni::generate_synthetic_features(-1, 1, 42); },
      "negative vertices");
  expect_invalid_argument(
      [] { gnni::generate_synthetic_features(1, 0, 42); },
      "zero feature dimension");
  expect_invalid_argument(
      [] {
        gnni::generate_synthetic_features(
            2, std::numeric_limits<std::size_t>::max(), 42);
      },
      "feature element count overflow");
}

void test_probability_extremes_and_self_loops() {
  const gnni::GeneratedGraph empty =
      gnni::generate_erdos_renyi(4, 0.0, 99, false);
  if (empty.logical_edges != 0 ||
      empty.graph.row_offsets != std::vector<int>({0, 0, 0, 0, 0}) ||
      !empty.graph.column_indices.empty()) {
    throw std::runtime_error("p=0 did not produce the empty graph");
  }
  expect_graph_invariants(empty, false);

  const gnni::GeneratedGraph empty_with_loops =
      gnni::generate_erdos_renyi(4, 0.0, 99, true);
  if (empty_with_loops.logical_edges != 0 ||
      empty_with_loops.graph.row_offsets !=
          std::vector<int>({0, 1, 2, 3, 4}) ||
      empty_with_loops.graph.column_indices !=
          std::vector<int>({0, 1, 2, 3})) {
    throw std::runtime_error("p=0 self-loop graph is not exact");
  }
  expect_graph_invariants(empty_with_loops, true);

  const gnni::GeneratedGraph complete =
      gnni::generate_erdos_renyi(4, 1.0, 99, false);
  if (complete.logical_edges != 6 ||
      complete.graph.row_offsets != std::vector<int>({0, 3, 6, 9, 12}) ||
      complete.graph.column_indices !=
          std::vector<int>({1, 2, 3, 0, 2, 3, 0, 1, 3, 0, 1, 2})) {
    throw std::runtime_error("p=1 did not produce the canonical complete graph");
  }
  expect_graph_invariants(complete, false);

  const gnni::GeneratedGraph complete_with_loops =
      gnni::generate_erdos_renyi(4, 1.0, 99, true);
  if (complete_with_loops.logical_edges != 6 ||
      complete_with_loops.graph.row_offsets !=
          std::vector<int>({0, 4, 8, 12, 16}) ||
      complete_with_loops.graph.column_indices !=
          std::vector<int>({0, 1, 2, 3, 0, 1, 2, 3,
                            0, 1, 2, 3, 0, 1, 2, 3})) {
    throw std::runtime_error("p=1 self-loop graph is not exact");
  }
  expect_graph_invariants(complete_with_loops, true);
}

void test_average_degree_probability_conversion() {
  if (gnni::er_probability_from_average_degree(5, 0.0) != 0.0 ||
      gnni::er_probability_from_average_degree(5, 2.0) != 0.5 ||
      gnni::er_probability_from_average_degree(5, 4.0) != 1.0) {
    throw std::runtime_error("average degree did not map to d/(n-1)");
  }
}

void test_random_graphs_are_repeatable_canonical_and_seed_sensitive() {
  const gnni::GeneratedGraph first =
      gnni::generate_erdos_renyi(64, 0.17, 987654321, false);
  const gnni::GeneratedGraph repeated =
      gnni::generate_erdos_renyi(64, 0.17, 987654321, false);
  const gnni::GeneratedGraph changed_seed =
      gnni::generate_erdos_renyi(64, 0.17, 987654322, false);
  if (first.logical_edges == 0 || first.logical_edges >= 2016) {
    throw std::runtime_error("fixed random graph is trivial");
  }
  if (first.logical_edges != repeated.logical_edges ||
      !same_graph(first.graph, repeated.graph)) {
    throw std::runtime_error("same ER options are not repeatable");
  }
  if (same_graph(first.graph, changed_seed.graph)) {
    throw std::runtime_error("changing graph seed did not alter CSR");
  }
  expect_graph_invariants(first, false);
  expect_graph_invariants(changed_seed, false);

  const gnni::GeneratedGraph with_loops =
      gnni::generate_erdos_renyi(64, 0.17, 987654321, true);
  expect_graph_invariants(with_loops, true);
  if (with_loops.logical_edges != first.logical_edges) {
    throw std::runtime_error("self-loop option changed logical ER edges");
  }
  for (int row = 0; row < first.graph.num_vertices; ++row) {
    if (!has_edge(with_loops.graph, row, row)) {
      throw std::runtime_error("requested self-loop is missing");
    }
    const auto row_index = static_cast<std::size_t>(row);
    const auto begin =
        static_cast<std::size_t>(with_loops.graph.row_offsets[row_index]);
    const auto end =
        static_cast<std::size_t>(with_loops.graph.row_offsets[row_index + 1U]);
    for (std::size_t entry = begin; entry < end; ++entry) {
      const int column = with_loops.graph.column_indices[entry];
      if (column != row && !has_edge(first.graph, row, column)) {
        throw std::runtime_error("self-loop option changed an off-diagonal edge");
      }
    }
  }
}

void test_sparse_large_graph_uses_skip_path() {
  const gnni::GeneratedGraph sparse =
      gnni::generate_erdos_renyi(100000, 0.000002, 314159, false);
  if (sparse.logical_edges == 0 || sparse.logical_edges > 100000) {
    throw std::runtime_error("large sparse graph has implausible edge count");
  }
  expect_graph_invariants(sparse, false);
}

void test_make_synthetic_dataset() {
  gnni::GeneratedGraph generated =
      gnni::generate_erdos_renyi(12, 0.25, 8675309, true);
  const gnni::CSRGraph expected_graph = generated.graph;
  const gnni::DenseMatrix expected_features =
      gnni::generate_synthetic_features(12, 3, 4242);

  const gnni::Dataset dataset =
      gnni::make_synthetic_dataset(std::move(generated), 3, 4242);
  if (!same_graph(dataset.graph, expected_graph) ||
      dataset.features.rows != expected_features.rows ||
      dataset.features.cols != expected_features.cols ||
      dataset.features.values != expected_features.values ||
      dataset.metadata.directed || !dataset.metadata.self_loops ||
      dataset.metadata.num_classes != 0 || !dataset.labels.empty() ||
      !dataset.splits.empty()) {
    throw std::runtime_error("synthetic dataset fields are inconsistent");
  }
  gnni::validate_dataset(dataset);
}

void test_er_generation_rejects_invalid_inputs() {
  expect_invalid_argument(
      [] { gnni::generate_erdos_renyi(0, 0.5, 1, false); },
      "zero ER vertices");
  expect_invalid_argument(
      [] { gnni::generate_erdos_renyi(-1, 0.5, 1, false); },
      "negative ER vertices");
  expect_invalid_argument(
      [] { gnni::generate_erdos_renyi(4, -0.01, 1, false); },
      "negative probability");
  expect_invalid_argument(
      [] { gnni::generate_erdos_renyi(4, 1.01, 1, false); },
      "probability above one");
  expect_invalid_argument(
      [] {
        gnni::generate_erdos_renyi(
            4, std::numeric_limits<double>::quiet_NaN(), 1, false);
      },
      "NaN probability");
  expect_invalid_argument(
      [] {
        gnni::generate_erdos_renyi(
            4, std::numeric_limits<double>::infinity(), 1, false);
      },
      "infinite probability");
  expect_invalid_argument(
      [] {
        gnni::generate_erdos_renyi(std::numeric_limits<int>::max(), 1.0,
                                   1, false);
      },
      "complete graph CSR size overflow");

  expect_invalid_argument(
      [] { gnni::er_probability_from_average_degree(1, 0.0); },
      "average degree vertex count");
  expect_invalid_argument(
      [] { gnni::er_probability_from_average_degree(5, -0.1); },
      "negative average degree");
  expect_invalid_argument(
      [] { gnni::er_probability_from_average_degree(5, 4.1); },
      "average degree above maximum");
  expect_invalid_argument(
      [] {
        gnni::er_probability_from_average_degree(
            5, std::numeric_limits<double>::quiet_NaN());
      },
      "NaN average degree");
}

std::vector<Test> make_tests() {
  return {
      {"feature golden values", test_feature_golden_values},
      {"features are repeatable bounded and coordinate sensitive",
       test_features_are_repeatable_bounded_and_coordinate_sensitive},
      {"RNG primitives have fixed mappings",
       test_rng_primitives_have_fixed_mappings},
      {"feature generation rejects invalid shapes",
       test_feature_generation_rejects_invalid_shapes},
      {"probability extremes and self loops",
       test_probability_extremes_and_self_loops},
      {"average degree probability conversion",
       test_average_degree_probability_conversion},
      {"random graphs are repeatable canonical and seed sensitive",
       test_random_graphs_are_repeatable_canonical_and_seed_sensitive},
      {"sparse large graph uses skip path",
       test_sparse_large_graph_uses_skip_path},
      {"make synthetic dataset", test_make_synthetic_dataset},
      {"ER generation rejects invalid inputs",
       test_er_generation_rejects_invalid_inputs},
  };
}

}  // namespace

int main() {
  const std::vector<Test> tests = make_tests();
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
