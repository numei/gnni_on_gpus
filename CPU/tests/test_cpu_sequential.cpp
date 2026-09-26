#include "gnni/api.hpp"

#include <iostream>

int main() {
    // Incoming-edge CSR for: 0 -> 1, 2 -> 1, 1 -> 2.
    gnni_cpu::CSRGraph graph;
    graph.num_nodes = 3;
    graph.num_edges = 3;
    graph.directed = true;
    graph.row_ptr = {0, 0, 2, 3};
    graph.col_idx = {0, 2, 1};

    // One row per node: [1,2], [3,4], [5,6].
    gnni_cpu::DenseMatrix features;
    features.rows = 3;
    features.cols = 2;
    features.values = {1.0F, 2.0F,
                       3.0F, 4.0F,
                       5.0F, 6.0F};

    gnni_cpu::Model model;

    // Layer 0: mean aggregation, a 2x2 transform, then ReLU.
    gnni_cpu::Layer first_layer;
    first_layer.weights.rows = 2;
    first_layer.weights.cols = 2;
    first_layer.weights.values = {1.0F, -2.0F,
                                  2.0F,  1.0F};
    first_layer.activation = gnni_cpu::ActivationKind::ReLU;
    model.layers.push_back(first_layer);

    // Layer 1: aggregate again, transform 2 features to 1 logit, no ReLU.
    gnni_cpu::Layer final_layer;
    final_layer.weights.rows = 2;
    final_layer.weights.cols = 1;
    final_layer.weights.values = {2.0F, 3.0F};
    final_layer.activation = gnni_cpu::ActivationKind::Identity;
    model.layers.push_back(final_layer);

    const gnni_cpu::InferenceConfig config;
    const gnni_cpu::RunResult result =
        gnni_cpu::run_cpu_sequential(graph, features, model, config);

    // Hand calculation gives one final logit per node: [0, 11, 22].
    gnni_cpu::DenseMatrix expected;
    expected.rows = 3;
    expected.cols = 1;
    expected.values = {0.0F, 11.0F, 22.0F};

    const gnni_cpu::ComparisonReport comparison =
        gnni_cpu::compare_outputs(expected, result.output);

    std::cout << "CPU sequential output ("
              << result.output.rows << "x" << result.output.cols << "):";
    for (const float value : result.output.values) {
        std::cout << ' ' << value;
    }
    std::cout << '\n'
              << "expected: 0 11 22\n"
              << "compute_ms: " << result.metrics.timing.compute_ms << '\n'
              << "inference_ms: " << result.metrics.timing.inference_ms << '\n'
              << "host_bytes: " << result.metrics.memory.host_bytes << '\n'
              << "nodes_per_second: " << result.metrics.nodes_per_second << '\n'
              << "edges_per_second: " << result.metrics.edges_per_second << '\n';

    if (!comparison.shapes_match || !comparison.within_tolerance) {
        std::cerr << "CPU sequential output did not match the hand-calculated result\n"
                  << "mismatch_count: " << comparison.mismatch_count << '\n'
                  << "max_absolute_error: " << comparison.max_absolute_error << '\n';
        return 1;
    }

    std::cout << "PASS: output matches the hand-calculated result\n";
    return 0;
}
