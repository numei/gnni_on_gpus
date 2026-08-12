#include "cpu_kernels.hpp"

#include <stdexcept>

namespace gnni::cpu_detail {

DenseMatrix aggregate_mean_sequential(
    const CSRGraph& graph,
    const DenseMatrix& input) {
    static_cast<void>(graph);
    static_cast<void>(input);
    // TODO(Role A): allocate N x F, iterate destination rows, sum source-node
    // features, divide non-empty rows by degree, and leave empty rows zero.
    throw std::logic_error("aggregate_mean_sequential is not implemented");
}

DenseMatrix linear_transform_sequential(
    const DenseMatrix& input,
    const DenseMatrix& weights) {
    static_cast<void>(input);
    static_cast<void>(weights);
    // TODO(Role A): implement the clear triple-loop reference Z = M * W.
    // Keep this unfused until all hand-computed fixtures pass.
    throw std::logic_error("linear_transform_sequential is not implemented");
}

void apply_activation_in_place(DenseMatrix& matrix, ActivationKind activation) {
    static_cast<void>(matrix);
    static_cast<void>(activation);
    // TODO(Role A): Identity does nothing; ReLU replaces negative values with
    // zero. Reject unknown enum values rather than guessing.
    throw std::logic_error("apply_activation_in_place is not implemented");
}

}  // namespace gnni::cpu_detail

