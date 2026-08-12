#include "gnni/api.hpp"

#include <stdexcept>

namespace gnni {

ComparisonReport compare_outputs(
    const DenseMatrix& reference,
    const DenseMatrix& candidate,
    const ComparisonOptions& options) {
    static_cast<void>(reference);
    static_cast<void>(candidate);
    static_cast<void>(options);
    // TODO(Role A): validate tolerances, handle shape mismatch, and compute
    // max absolute/relative error, RMSE, mismatch count, and first mismatch.
    throw std::logic_error("compare_outputs is not implemented");
}

}  // namespace gnni

