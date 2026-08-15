#include "gnni/dataset.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>

namespace gnni {
namespace {

std::size_t checked_feature_element_count(std::size_t rows, std::size_t cols) {
  if (rows != 0 && cols > std::numeric_limits<std::size_t>::max() / rows) {
    throw std::invalid_argument("feature shape overflows size_t");
  }
  return rows * cols;
}

std::uint64_t checked_vector_bytes(std::size_t count, std::uint64_t element_bytes,
                                   const char* field) {
  if constexpr (std::numeric_limits<std::size_t>::digits >
                std::numeric_limits<std::uint64_t>::digits) {
    if (count > static_cast<std::size_t>(
                    std::numeric_limits<std::uint64_t>::max())) {
      throw std::overflow_error(std::string(field) +
                                " element count overflows uint64_t");
    }
  }
  const auto count64 = static_cast<std::uint64_t>(count);
  if (element_bytes != 0 &&
      count64 > std::numeric_limits<std::uint64_t>::max() / element_bytes) {
    throw std::overflow_error(std::string(field) + " byte count overflows uint64_t");
  }
  return count64 * element_bytes;
}

void checked_add(std::uint64_t& total, std::uint64_t value, const char* field) {
  if (value > std::numeric_limits<std::uint64_t>::max() - total) {
    throw std::overflow_error(std::string("dataset host bytes overflow at ") +
                              field);
  }
  total += value;
}

}  // namespace

void validate_dataset(const Dataset& dataset) {
  const CSRGraph& graph = dataset.graph;
  if (graph.num_vertices < 0) {
    throw std::invalid_argument("CSR vertex count is negative: " +
                                std::to_string(graph.num_vertices));
  }
  const auto vertex_count = static_cast<std::size_t>(graph.num_vertices);

  if (graph.row_offsets.size() != vertex_count + 1) {
    throw std::invalid_argument(
        "CSR row offset count " + std::to_string(graph.row_offsets.size()) +
        " does not equal vertex count + 1 (" +
        std::to_string(vertex_count + 1) + ")");
  }
  if (graph.row_offsets.front() != 0) {
    throw std::invalid_argument("CSR first offset must be zero, got " +
                                std::to_string(graph.row_offsets.front()));
  }
  for (std::size_t index = 1; index < graph.row_offsets.size(); ++index) {
    if (graph.row_offsets[index] < graph.row_offsets[index - 1]) {
      throw std::invalid_argument(
          "CSR offset " + std::to_string(index) + " decreases from " +
          std::to_string(graph.row_offsets[index - 1]) + " to " +
          std::to_string(graph.row_offsets[index]));
    }
  }
  if (static_cast<std::size_t>(graph.row_offsets.back()) !=
      graph.column_indices.size()) {
    throw std::invalid_argument(
        "CSR final offset " + std::to_string(graph.row_offsets.back()) +
        " does not match column count " +
        std::to_string(graph.column_indices.size()));
  }

  for (std::size_t row = 0; row < vertex_count; ++row) {
    const auto begin = static_cast<std::size_t>(graph.row_offsets[row]);
    const auto end = static_cast<std::size_t>(graph.row_offsets[row + 1]);
    int previous_column = -1;
    bool has_self_loop = false;
    for (std::size_t entry = begin; entry < end; ++entry) {
      const int column = graph.column_indices[entry];
      if (column < 0 || column >= graph.num_vertices) {
        throw std::invalid_argument(
            "CSR row " + std::to_string(row) + " contains out-of-range column " +
            std::to_string(column) + " at entry " + std::to_string(entry));
      }
      if (entry != begin && column <= previous_column) {
        throw std::invalid_argument(
            "CSR row " + std::to_string(row) +
            " is not strictly increasing at column " + std::to_string(column));
      }
      has_self_loop = has_self_loop || column == static_cast<int>(row);
      previous_column = column;
    }
    if (dataset.metadata.self_loops && !has_self_loop) {
      throw std::invalid_argument("dataset declares self-loops but CSR is missing "
                                  "self-loop at vertex " +
                                  std::to_string(row));
    }
    if (!dataset.metadata.self_loops && has_self_loop) {
      throw std::invalid_argument(
          "CSR contains self-loop at vertex " + std::to_string(row) +
          " but self-loop flag is false");
    }
  }

  if (!dataset.metadata.directed) {
    for (std::size_t row = 0; row < vertex_count; ++row) {
      const auto begin = static_cast<std::size_t>(graph.row_offsets[row]);
      const auto end = static_cast<std::size_t>(graph.row_offsets[row + 1]);
      for (std::size_t entry = begin; entry < end; ++entry) {
        const int column = graph.column_indices[entry];
        const auto reverse_begin =
            graph.column_indices.begin() + graph.row_offsets[column];
        const auto reverse_end =
            graph.column_indices.begin() + graph.row_offsets[column + 1];
        if (!std::binary_search(reverse_begin, reverse_end,
                                static_cast<int>(row))) {
          throw std::invalid_argument(
              "undirected CSR is missing reverse entry (" +
              std::to_string(row) + "," + std::to_string(column) + ")");
        }
      }
    }
  }

  if (dataset.features.rows != vertex_count) {
    throw std::invalid_argument(
        "feature row count " + std::to_string(dataset.features.rows) +
        " does not match vertex count " + std::to_string(vertex_count));
  }
  if (dataset.features.cols == 0) {
    throw std::invalid_argument("feature width must be greater than zero");
  }
  const std::size_t expected_feature_values =
      checked_feature_element_count(dataset.features.rows, dataset.features.cols);
  if (dataset.features.values.size() != expected_feature_values) {
    throw std::invalid_argument(
        "feature value count " +
        std::to_string(dataset.features.values.size()) +
        " does not match matrix shape element count " +
        std::to_string(expected_feature_values));
  }
  for (std::size_t index = 0; index < dataset.features.values.size(); ++index) {
    if (!std::isfinite(dataset.features.values[index])) {
      throw std::invalid_argument("feature value " + std::to_string(index) +
                                  " is not finite");
    }
  }

  if (dataset.labels.empty()) {
    if (dataset.metadata.num_classes != 0) {
      throw std::invalid_argument(
          "num_classes must be zero when labels are absent, got " +
          std::to_string(dataset.metadata.num_classes));
    }
    if (!dataset.splits.empty()) {
      throw std::invalid_argument("node splits require labels");
    }
  } else {
    if (dataset.labels.size() != vertex_count) {
      throw std::invalid_argument(
          "label count " + std::to_string(dataset.labels.size()) +
          " does not match vertex count " + std::to_string(vertex_count));
    }
    if (dataset.metadata.num_classes == 0) {
      throw std::invalid_argument(
          "num_classes must be greater than zero when labels are present");
    }
    for (std::size_t index = 0; index < dataset.labels.size(); ++index) {
      const std::int32_t label = dataset.labels[index];
      if (label < 0 ||
          static_cast<std::uint64_t>(label) >= dataset.metadata.num_classes) {
        throw std::invalid_argument(
            "label " + std::to_string(index) + " value " +
            std::to_string(label) + " is outside [0, num_classes)");
      }
    }
  }

  if (!dataset.splits.empty()) {
    if (dataset.splits.size() != vertex_count) {
      throw std::invalid_argument(
          "split count " + std::to_string(dataset.splits.size()) +
          " does not match vertex count " + std::to_string(vertex_count));
    }
    for (std::size_t index = 0; index < dataset.splits.size(); ++index) {
      const auto value = static_cast<std::uint8_t>(dataset.splits[index]);
      if (value > static_cast<std::uint8_t>(NodeSplit::Test)) {
        throw std::invalid_argument(
            "split " + std::to_string(index) + " has invalid byte value " +
            std::to_string(value));
      }
    }
  }
}

std::uint64_t dataset_host_bytes(const Dataset& dataset) {
  std::uint64_t total = 0;
  checked_add(total,
              checked_vector_bytes(dataset.graph.row_offsets.size(),
                                   sizeof(decltype(dataset.graph.row_offsets)::
                                              value_type),
                                   "CSR row offsets"),
              "CSR row offsets");
  checked_add(total,
              checked_vector_bytes(dataset.graph.column_indices.size(),
                                   sizeof(decltype(dataset.graph.column_indices)::
                                              value_type),
                                   "CSR columns"),
              "CSR columns");
  checked_add(total,
              checked_vector_bytes(dataset.features.values.size(),
                                   sizeof(decltype(dataset.features.values)::
                                              value_type),
                                   "features"),
              "features");
  checked_add(total,
              checked_vector_bytes(dataset.labels.size(),
                                   sizeof(decltype(dataset.labels)::value_type),
                                   "labels"),
              "labels");
  checked_add(total,
              checked_vector_bytes(dataset.splits.size(),
                                   sizeof(decltype(dataset.splits)::value_type),
                                   "splits"),
              "splits");
  return total;
}

}  // namespace gnni
