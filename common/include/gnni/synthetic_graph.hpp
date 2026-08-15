#pragma once

#include "gnni/dataset.hpp"

#include <cstddef>
#include <cstdint>
#include <random>

namespace gnni {

struct GeneratedGraph {
  CSRGraph graph;
  std::uint64_t logical_edges = 0;
};

float synthetic_feature(std::uint64_t seed, std::size_t vertex,
                        std::size_t feature);

DenseMatrix generate_synthetic_features(int vertices, std::size_t feature_dim,
                                        std::uint64_t seed);

double er_probability_from_average_degree(int vertices,
                                          double average_degree);

GeneratedGraph generate_erdos_renyi(int vertices, double probability,
                                    std::uint64_t seed, bool self_loops);

GeneratedGraph generate_barabasi_albert(int vertices,
                                        int edges_per_new_vertex,
                                        std::uint64_t seed,
                                        bool self_loops);

GeneratedGraph generate_watts_strogatz(int vertices, int neighbors,
                                       double rewire_probability,
                                       std::uint64_t seed,
                                       bool self_loops);

Dataset make_synthetic_dataset(GeneratedGraph graph, std::size_t feature_dim,
                               std::uint64_t feature_seed);

namespace detail {

std::uint64_t sample_bounded(std::mt19937_64& generator,
                             std::uint64_t exclusive_upper_bound);

double sample_unit_double(std::mt19937_64& generator);

}  // namespace detail
}  // namespace gnni
