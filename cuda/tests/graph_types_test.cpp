#include "gnni/graph_types.hpp"

#include <exception>
#include <functional>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

template <typename Function>
void expect_invalid_argument(Function&& function, const std::string& label) {
  try {
    function();
  } catch (const std::invalid_argument&) {
    return;
  }
  throw std::runtime_error(label + ": expected std::invalid_argument");
}

void test_zero_filled_construction() {
  const gnni::DenseMatrix zeros(2, 3);
  if (zeros.rows != 2 || zeros.cols != 3 ||
      zeros.values != std::vector<float>(6, 0.0F)) {
    throw std::runtime_error("zero-filled construction has the wrong shape or values");
  }
}

void test_construction_from_supplied_values() {
  const gnni::DenseMatrix matrix(2, 2, {1.0F, -2.0F, 3.5F, 0.0F});
  if (matrix.rows != 2 || matrix.cols != 2 ||
      matrix.values != std::vector<float>({1.0F, -2.0F, 3.5F, 0.0F})) {
    throw std::runtime_error("supplied values are not preserved");
  }
}

void test_rejects_value_count_mismatch() {
  expect_invalid_argument([] { gnni::DenseMatrix(2, 2, {1.0F}); },
                          "value count mismatch");
}

void test_rejects_element_count_overflow() {
  expect_invalid_argument(
      [] {
        gnni::DenseMatrix(std::numeric_limits<std::size_t>::max(), 2);
      },
      "element count overflow");
}

}  // namespace

int main() {
  const std::vector<std::pair<std::string, std::function<void()>>> tests = {
      {"zero-filled construction", test_zero_filled_construction},
      {"construction from supplied values", test_construction_from_supplied_values},
      {"rejects value count mismatch", test_rejects_value_count_mismatch},
      {"rejects element count overflow", test_rejects_element_count_overflow},
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
