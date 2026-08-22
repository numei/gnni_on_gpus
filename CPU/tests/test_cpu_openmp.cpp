#include "gnni/api.hpp"

#include <array>
#include <cstddef>
#include <iostream>

int main() {
    // Incoming-edge CSR for: 0 -> 1, 2 -> 1, 1 -> 2.
    gnni::CSRGraph graph;
    graph.num_nodes = 3;
    graph.num_edges = 3;
    graph.directed = true;
    graph.row_ptr = {0, 0, 2, 3};
    graph.col_idx = {0, 2, 1};

    // One row per node: [1,2], [3,4], [5,6].
    gnni::DenseMatrix features;
    features.rows = 3;
    features.cols = 2;
    features.values = {1.0F, 2.0F,
                       3.0F, 4.0F,
                       5.0F, 6.0F};

    gnni::Model model;

    // Layer 0: mean aggregation, a 2x2 transform, then ReLU.
    gnni::Layer first_layer;
    first_layer.weights.rows = 2;
    first_layer.weights.cols = 2;
    first_layer.weights.values = {1.0F, -2.0F,
                                  2.0F,  1.0F};
    first_layer.activation = gnni::ActivationKind::ReLU;
    model.layers.push_back(first_layer);

    // Layer 1: aggregate again, transform 2 features to 1 logit, no ReLU.
    gnni::Layer final_layer;
    final_layer.weights.rows = 2;
    final_layer.weights.cols = 1;
    final_layer.weights.values = {2.0F, 3.0F};
    final_layer.activation = gnni::ActivationKind::Identity;
    model.layers.push_back(final_layer);

    const gnni::InferenceConfig sequential_config;
    const gnni::RunResult reference =
        gnni::run_cpu_sequential(graph, features, model, sequential_config);

    const std::array<gnni::OpenMPScheduleKind, 4> schedules{
        gnni::OpenMPScheduleKind::Default,
        gnni::OpenMPScheduleKind::Static,
        gnni::OpenMPScheduleKind::Dynamic,
        gnni::OpenMPScheduleKind::Guided,
    };

    const std::array<const char*, 4> schedule_names{
        "default",
        "static",
        "dynamic",
        "guided",
    };

    for (std::size_t schedule_index = 0;
         schedule_index < schedules.size();
         ++schedule_index) {
        gnni::InferenceConfig openmp_config;
        openmp_config.num_threads = 4;
        openmp_config.openmp_schedule = schedules[schedule_index];
        openmp_config.openmp_chunk_size =
            schedules[schedule_index] == gnni::OpenMPScheduleKind::Default
                ? 0
                : 1;

        const gnni::RunResult candidate =
            gnni::run_cpu_openmp(graph, features, model, openmp_config);

        const gnni::ComparisonReport comparison =
            gnni::compare_outputs(reference.output, candidate.output);

        std::cout << "schedule: " << schedule_names[schedule_index] << '\n'
                  << "compute_ms: " << candidate.metrics.timing.compute_ms << '\n'
                  << "inference_ms: " << candidate.metrics.timing.inference_ms << '\n'
                  << "nodes_per_second: " << candidate.metrics.nodes_per_second << '\n'
                  << "edges_per_second: " << candidate.metrics.edges_per_second << '\n';

        if (!comparison.shapes_match || !comparison.within_tolerance) {
            std::cerr << "OpenMP output differs from sequential reference\n"
                      << "schedule: " << schedule_names[schedule_index] << '\n'
                      << "mismatch_count: " << comparison.mismatch_count << '\n'
                      << "max_absolute_error: "
                      << comparison.max_absolute_error << '\n'
                      << "max_relative_error: "
                      << comparison.max_relative_error << '\n'
                      << "RMSE: " << comparison.root_mean_square_error << '\n';

            if (comparison.first_mismatch_index.has_value()) {
                std::cerr << "first_mismatch_index: "
                          << comparison.first_mismatch_index.value() << '\n';
            }

            return 1;
        }
    }

    std::cout << "PASS: all OpenMP schedules match "
                 "the sequential CPU reference\n";
    return 0;
}
