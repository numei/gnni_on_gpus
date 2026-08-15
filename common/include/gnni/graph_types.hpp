#pragma once

#include <cstddef>
#include <utility>
#include <vector>

namespace gnni {

struct CSRGraph {
  int num_vertices = 0;
  std::vector<int> row_offsets;
  std::vector<int> column_indices;

  CSRGraph() = default;
  CSRGraph(int vertices, std::vector<int> offsets, std::vector<int> columns)
      : num_vertices(vertices),
        row_offsets(std::move(offsets)),
        column_indices(std::move(columns)) {}
};

struct DenseMatrix {
  std::size_t rows = 0;
  std::size_t cols = 0;
  std::vector<float> values;
  DenseMatrix() = default;
  DenseMatrix(std::size_t rows, std::size_t cols);
  DenseMatrix(std::size_t rows, std::size_t cols, std::vector<float> values);
};

}  // namespace gnni
