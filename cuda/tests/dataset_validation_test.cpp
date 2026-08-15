#include "gnni/dataset.hpp"

#include <cstdint>
#include <exception>
#include <functional>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

using Dataset = gnni::Dataset;
using Test = std::pair<std::string, std::function<void()>>;

Dataset directed_dataset() {
  Dataset dataset;
  dataset.graph = gnni::CSRGraph(3, {0, 1, 2, 3}, {1, 2, 0});
  dataset.features =
      gnni::DenseMatrix(3, 2, {1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F});
  dataset.labels = {0, 1, 0};
  dataset.splits = {gnni::NodeSplit::Train, gnni::NodeSplit::Validation,
                    gnni::NodeSplit::Test};
  dataset.metadata.directed = true;
  dataset.metadata.num_classes = 2;
  return dataset;
}

Dataset symmetric_dataset() {
  Dataset dataset;
  dataset.graph = gnni::CSRGraph(3, {0, 1, 3, 4}, {1, 0, 2, 1});
  dataset.features = gnni::DenseMatrix(3, 1, {1.0F, 2.0F, 3.0F});
  return dataset;
}

template <typename Mutator>
std::function<void()> invalid_directed(Mutator mutator,
                                       std::string expected_context) {
  return [mutator = std::move(mutator),
          expected_context = std::move(expected_context)] {
    Dataset dataset = directed_dataset();
    mutator(dataset);
    try {
      gnni::validate_dataset(dataset);
    } catch (const std::invalid_argument& error) {
      if (std::string(error.what()).find(expected_context) == std::string::npos) {
        throw std::runtime_error("error lacks context '" + expected_context +
                                 "': " + error.what());
      }
      return;
    }
    throw std::runtime_error("expected std::invalid_argument containing '" +
                             expected_context + "'");
  };
}

std::vector<Test> make_tests() {
  std::vector<Test> tests;
  tests.push_back({"accepts valid directed dataset",
                   [] { gnni::validate_dataset(directed_dataset()); }});
  tests.push_back({"accepts valid symmetric dataset",
                   [] { gnni::validate_dataset(symmetric_dataset()); }});
  tests.push_back({"accepts exact reported self-loops", [] {
                     Dataset dataset;
                     dataset.graph =
                         gnni::CSRGraph(2, {0, 2, 4}, {0, 1, 0, 1});
                     dataset.features =
                         gnni::DenseMatrix(2, 1, {1.0F, 2.0F});
                     dataset.metadata.self_loops = true;
                     gnni::validate_dataset(dataset);
                   }});
  tests.push_back({"accepts zero-edge CSR", [] {
                     Dataset dataset;
                     dataset.graph = gnni::CSRGraph(3, {0, 0, 0, 0}, {});
                     dataset.features =
                         gnni::DenseMatrix(3, 1, {0.0F, 1.0F, 2.0F});
                     gnni::validate_dataset(dataset);
                   }});

  tests.push_back({"rejects malformed offset count", invalid_directed(
      [](Dataset& d) { d.graph.row_offsets.pop_back(); }, "row offset count")});
  tests.push_back({"rejects nonzero first offset", invalid_directed(
      [](Dataset& d) { d.graph.row_offsets.front() = 1; }, "first offset")});
  tests.push_back({"rejects decreasing offsets", invalid_directed(
      [](Dataset& d) { d.graph.row_offsets = {0, 2, 1, 3}; }, "decreases")});
  tests.push_back({"rejects final offset mismatch", invalid_directed(
      [](Dataset& d) { d.graph.row_offsets.back() = 2; }, "final offset")});
  tests.push_back({"rejects unsorted columns", invalid_directed(
      [](Dataset& d) {
        d.graph = gnni::CSRGraph(3, {0, 2, 2, 3}, {2, 1, 0});
      }, "CSR row 0")});
  tests.push_back({"rejects duplicate columns", invalid_directed(
      [](Dataset& d) {
        d.graph = gnni::CSRGraph(3, {0, 2, 2, 3}, {1, 1, 0});
      }, "strictly increasing")});
  tests.push_back({"rejects out-of-range column", invalid_directed(
      [](Dataset& d) { d.graph.column_indices.front() = 3; }, "column 3")});
  tests.push_back({"rejects negative column", invalid_directed(
      [](Dataset& d) { d.graph.column_indices.front() = -1; }, "column -1")});
  tests.push_back({"rejects asymmetric undirected input", invalid_directed(
      [](Dataset& d) {
        d.graph = gnni::CSRGraph(2, {0, 1, 1}, {1});
        d.features = gnni::DenseMatrix(2, 1, {1.0F, 2.0F});
        d.labels.clear();
        d.splits.clear();
        d.metadata.directed = false;
        d.metadata.num_classes = 0;
      }, "reverse entry (0,1)")});
  tests.push_back({"rejects wrong feature rows", invalid_directed(
      [](Dataset& d) {
        d.features = gnni::DenseMatrix(2, 2, {1.0F, 2.0F, 3.0F, 4.0F});
      }, "feature row count")});
  tests.push_back({"rejects zero feature width", invalid_directed(
      [](Dataset& d) { d.features = gnni::DenseMatrix(3, 0); },
      "feature width")});
  tests.push_back({"rejects feature value count mismatch", invalid_directed(
      [](Dataset& d) { d.features.values.pop_back(); }, "feature value count")});
  tests.push_back({"rejects feature shape overflow", invalid_directed(
      [](Dataset& d) {
        d.features.cols = std::numeric_limits<std::size_t>::max();
        d.features.values.clear();
      }, "feature shape")});
  tests.push_back({"rejects NaN feature", invalid_directed(
      [](Dataset& d) {
        d.features.values[2] = std::numeric_limits<float>::quiet_NaN();
      }, "feature value 2")});
  tests.push_back({"rejects infinite feature", invalid_directed(
      [](Dataset& d) {
        d.features.values[4] = std::numeric_limits<float>::infinity();
      }, "feature value 4")});
  tests.push_back({"rejects wrong label count", invalid_directed(
      [](Dataset& d) { d.labels.pop_back(); }, "label count")});
  tests.push_back({"rejects negative label", invalid_directed(
      [](Dataset& d) { d.labels[1] = -1; }, "label 1")});
  tests.push_back({"rejects label at class count", invalid_directed(
      [](Dataset& d) { d.labels[1] = 2; }, "label 1")});
  tests.push_back({"rejects split without labels", invalid_directed(
      [](Dataset& d) {
        d.labels.clear();
        d.metadata.num_classes = 0;
      }, "splits require labels")});
  tests.push_back({"rejects wrong split count", invalid_directed(
      [](Dataset& d) { d.splits.pop_back(); }, "split count")});
  tests.push_back({"rejects invalid split byte", invalid_directed(
      [](Dataset& d) { d.splits[1] = static_cast<gnni::NodeSplit>(4); },
      "split 1")});
  tests.push_back({"rejects classes without labels", invalid_directed(
      [](Dataset& d) {
        d.labels.clear();
        d.splits.clear();
      }, "num_classes")});
  tests.push_back({"rejects labels without classes", invalid_directed(
      [](Dataset& d) { d.metadata.num_classes = 0; }, "num_classes")});
  tests.push_back({"rejects unreported self-loop", invalid_directed(
      [](Dataset& d) {
        d.graph = gnni::CSRGraph(2, {0, 1, 2}, {0, 1});
        d.features = gnni::DenseMatrix(2, 1, {1.0F, 2.0F});
        d.labels.clear();
        d.splits.clear();
        d.metadata.num_classes = 0;
      }, "self-loop flag is false")});
  tests.push_back({"rejects missing reported self-loop", invalid_directed(
      [](Dataset& d) {
        d.graph = gnni::CSRGraph(2, {0, 1, 2}, {0, 0});
        d.features = gnni::DenseMatrix(2, 1, {1.0F, 2.0F});
        d.labels.clear();
        d.splits.clear();
        d.metadata.num_classes = 0;
        d.metadata.self_loops = true;
      }, "missing self-loop at vertex 1")});
  tests.push_back({"counts owned host payload bytes", [] {
                     constexpr std::uint64_t expected = 67;
                     if (gnni::dataset_host_bytes(directed_dataset()) != expected) {
                       throw std::runtime_error("wrong host payload byte count");
                     }
                   }});
  return tests;
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
