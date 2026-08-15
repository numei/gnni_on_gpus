#include "gnni/graph_types.hpp"

#include <limits>
#include <stdexcept>
#include <string>
#include <utility>

namespace gnni {
namespace {

std::size_t checked_element_count(std::size_t rows, std::size_t cols) {
  if (rows != 0 && cols > std::numeric_limits<std::size_t>::max() / rows) {
    throw std::invalid_argument("matrix shape overflows size_t");
  }
  return rows * cols;
}

}  // namespace

DenseMatrix::DenseMatrix(std::size_t row_count, std::size_t column_count)
    : rows(row_count),
      cols(column_count),
      values(checked_element_count(row_count, column_count), 0.0F) {}

DenseMatrix::DenseMatrix(std::size_t row_count, std::size_t column_count,
                         std::vector<float> data)
    : rows(row_count), cols(column_count), values(std::move(data)) {
  if (values.size() != checked_element_count(rows, cols)) {
    throw std::invalid_argument("matrix value count does not match its shape");
  }
}

}  // namespace gnni
