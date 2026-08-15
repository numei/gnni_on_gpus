#pragma once

#include "gnni/graph_types.hpp"

#include <cstdint>
#include <vector>

namespace gnni {

enum class NodeSplit : std::uint8_t {
  Unassigned = 0,
  Train = 1,
  Validation = 2,
  Test = 3
};

struct DatasetMetadata {
  bool directed = false;
  bool self_loops = false;
  std::uint64_t num_classes = 0;
};

struct Dataset {
  CSRGraph graph;
  DenseMatrix features;
  std::vector<std::int32_t> labels;
  std::vector<NodeSplit> splits;
  DatasetMetadata metadata;
};

void validate_dataset(const Dataset& dataset);
std::uint64_t dataset_host_bytes(const Dataset& dataset);

}  // namespace gnni
