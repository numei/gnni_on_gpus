#pragma once

#include "gnni/types.hpp"

namespace gnni_cpu::cpu_detail {

/**
 * Compute PDF-style mean aggregation for every destination node.
 *
 * Empty CSR rows produce zero rows. No self-message is added. This function is
 * the readable sequential reference for the aggregation stage.
 */
DenseMatrix aggregate_mean_sequential(
    const CSRGraph& graph,
    const DenseMatrix& input,
    AggregationKind aggregation = AggregationKind::Mean);

/**
 * Compute output = input * weights using row-major matrices.
 *
 * The caller must validate input.cols == weights.rows before calling.
 */
DenseMatrix linear_transform_sequential(
    const DenseMatrix& input,
    const DenseMatrix& weights);

/** Apply Identity or ReLU to every matrix element in place. */
void apply_activation_in_place(DenseMatrix& matrix, ActivationKind activation);

}  // namespace gnni_cpu::cpu_detail

