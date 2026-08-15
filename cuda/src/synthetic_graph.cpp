#include "gnni/synthetic_graph.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

namespace gnni {
namespace {

std::uint64_t mix64(std::uint64_t value) {
  value = (value ^ (value >> 30U)) * UINT64_C(0xbf58476d1ce4e5b9);
  value = (value ^ (value >> 27U)) * UINT64_C(0x94d049bb133111eb);
  return value ^ (value >> 31U);
}

}  // namespace

float synthetic_feature(std::uint64_t seed, std::size_t vertex,
                        std::size_t feature) {
  std::uint64_t key = seed + UINT64_C(0x9e3779b97f4a7c15);
  key ^= (static_cast<std::uint64_t>(vertex) + 1U) *
         UINT64_C(0xbf58476d1ce4e5b9);
  key ^= (static_cast<std::uint64_t>(feature) + 1U) *
         UINT64_C(0x94d049bb133111eb);
  const std::uint32_t high_bits =
      static_cast<std::uint32_t>(mix64(key) >> 40U);
  return static_cast<float>(high_bits) * 0x1p-23F - 1.0F;
}

DenseMatrix generate_synthetic_features(int vertices, std::size_t feature_dim,
                                        std::uint64_t seed) {
  if (vertices <= 0) {
    throw std::invalid_argument("feature vertex count must be positive");
  }
  if (feature_dim == 0) {
    throw std::invalid_argument("feature dimension must be positive");
  }
  const auto vertex_count = static_cast<std::size_t>(vertices);
  if (feature_dim > std::numeric_limits<std::size_t>::max() / vertex_count) {
    throw std::invalid_argument("feature shape overflows size_t");
  }

  DenseMatrix features(vertex_count, feature_dim);
  for (std::size_t vertex = 0; vertex < vertex_count; ++vertex) {
    for (std::size_t feature = 0; feature < feature_dim; ++feature) {
      features.values[vertex * feature_dim + feature] =
          synthetic_feature(seed, vertex, feature);
    }
  }
  return features;
}

namespace {

using Edge = std::pair<int, int>;

std::uint64_t complete_edge_count(int vertices) {
  const auto count = static_cast<std::uint64_t>(vertices);
  return count * (count - 1U) / 2U;
}

std::uint64_t maximum_logical_edges(int vertices, bool self_loops) {
  const std::uint64_t self_entries =
      self_loops ? static_cast<std::uint64_t>(vertices) : 0U;
  return (static_cast<std::uint64_t>(std::numeric_limits<int>::max()) -
          self_entries) /
         2U;
}

void append_checked_edge(std::vector<Edge>& edges, int first, int second,
                         std::uint64_t maximum_edges) {
  if (static_cast<std::uint64_t>(edges.size()) >= maximum_edges) {
    throw std::invalid_argument("generated CSR exceeds int offset capacity");
  }
  edges.emplace_back(first, second);
}

CSRGraph build_canonical_graph(int vertices, const std::vector<Edge>& edges,
                               bool self_loops) {
  const std::uint64_t entry_count =
      static_cast<std::uint64_t>(edges.size()) * 2U +
      (self_loops ? static_cast<std::uint64_t>(vertices) : 0U);
  if (entry_count >
      static_cast<std::uint64_t>(std::numeric_limits<int>::max())) {
    throw std::invalid_argument("generated CSR exceeds int offset capacity");
  }

  const auto vertex_count = static_cast<std::size_t>(vertices);
  std::vector<int> lower_degrees(vertex_count, 0);
  std::vector<int> upper_degrees(vertex_count, 0);
  for (const auto& [first, second] : edges) {
    ++lower_degrees[static_cast<std::size_t>(first)];
    ++upper_degrees[static_cast<std::size_t>(second)];
  }

  std::vector<int> row_offsets(vertex_count + 1U, 0);
  for (std::size_t vertex = 0; vertex < vertex_count; ++vertex) {
    row_offsets[vertex + 1U] =
        row_offsets[vertex] + lower_degrees[vertex] +
        (self_loops ? 1 : 0) + upper_degrees[vertex];
  }
  std::vector<int> columns(static_cast<std::size_t>(row_offsets.back()));
  // ER generation emits lexicographic (higher, lower) pairs. Splitting
  // each row around its diagonal preserves canonical order in linear time.
  std::vector<int> lower_cursor = row_offsets;
  std::vector<int> upper_cursor(vertex_count, 0);
  for (std::size_t vertex = 0; vertex < vertex_count; ++vertex) {
    const int diagonal = row_offsets[vertex] + lower_degrees[vertex];
    if (self_loops) {
      columns[static_cast<std::size_t>(diagonal)] =
          static_cast<int>(vertex);
    }
    upper_cursor[vertex] = diagonal + (self_loops ? 1 : 0);
  }

  for (const auto& [first, second] : edges) {
    columns[static_cast<std::size_t>(
        lower_cursor[static_cast<std::size_t>(first)]++)] = second;
    columns[static_cast<std::size_t>(
        upper_cursor[static_cast<std::size_t>(second)]++)] = first;
  }
  return CSRGraph(vertices, std::move(row_offsets), std::move(columns));
}

std::uint64_t remaining_pair_slots(std::int64_t vertex,
                                   std::int64_t neighbor, int vertices) {
  const auto current_row =
      static_cast<std::uint64_t>(vertex - neighbor - 1);
  const auto later_rows =
      static_cast<std::uint64_t>(vertices - vertex - 1);
  const auto first_plus_last =
      static_cast<std::uint64_t>(vertices + vertex);
  return current_row + later_rows * first_plus_last / 2U;
}

bool infer_exact_self_loops(const CSRGraph& graph) {
  if (graph.num_vertices <= 0) {
    throw std::invalid_argument("synthetic graph vertex count must be positive");
  }
  const auto vertex_count = static_cast<std::size_t>(graph.num_vertices);
  if (graph.row_offsets.size() != vertex_count + 1U ||
      graph.row_offsets.front() != 0) {
    throw std::invalid_argument("synthetic graph has invalid CSR offsets");
  }
  for (std::size_t index = 1; index < graph.row_offsets.size(); ++index) {
    if (graph.row_offsets[index] < graph.row_offsets[index - 1] ||
        graph.row_offsets[index] < 0) {
      throw std::invalid_argument("synthetic graph has invalid CSR offsets");
    }
  }
  if (static_cast<std::size_t>(graph.row_offsets.back()) !=
      graph.column_indices.size()) {
    throw std::invalid_argument("synthetic graph CSR size is inconsistent");
  }

  bool exact_self_loops = false;
  for (std::size_t vertex = 0; vertex < vertex_count; ++vertex) {
    const auto begin =
        graph.column_indices.begin() + graph.row_offsets[vertex];
    const auto end =
        graph.column_indices.begin() + graph.row_offsets[vertex + 1U];
    const bool has_loop =
        std::find(begin, end, static_cast<int>(vertex)) != end;
    if (vertex == 0) {
      exact_self_loops = has_loop;
    } else if (has_loop != exact_self_loops) {
      throw std::invalid_argument(
          "synthetic graph must have self-loops at every vertex or none");
    }
  }
  return exact_self_loops;
}

}  // namespace

double er_probability_from_average_degree(int vertices,
                                          double average_degree) {
  if (vertices < 2) {
    throw std::invalid_argument(
        "average-degree conversion requires at least two vertices");
  }
  const double maximum_degree = static_cast<double>(vertices - 1);
  if (!std::isfinite(average_degree) || average_degree < 0.0 ||
      average_degree > maximum_degree) {
    throw std::invalid_argument(
        "average degree must be finite and in [0, vertices-1]");
  }
  if (average_degree == 0.0) {
    return 0.0;
  }
  return average_degree / maximum_degree;
}

GeneratedGraph generate_erdos_renyi(int vertices, double probability,
                                    std::uint64_t seed, bool self_loops) {
  if (vertices <= 0) {
    throw std::invalid_argument("ER vertex count must be positive");
  }
  if (!std::isfinite(probability) || probability < 0.0 ||
      probability > 1.0) {
    throw std::invalid_argument("ER probability must be finite and in [0,1]");
  }

  const std::uint64_t maximum_edges =
      maximum_logical_edges(vertices, self_loops);
  std::vector<Edge> edges;
  if (probability == 1.0) {
    const std::uint64_t logical_edges = complete_edge_count(vertices);
    if (logical_edges > maximum_edges) {
      throw std::invalid_argument("complete graph exceeds int offset capacity");
    }
    edges.reserve(static_cast<std::size_t>(logical_edges));
    for (int vertex = 1; vertex < vertices; ++vertex) {
      for (int neighbor = 0; neighbor < vertex; ++neighbor) {
        edges.emplace_back(vertex, neighbor);
      }
    }
  } else if (probability > 0.0) {
    std::mt19937_64 generator(seed);
    const double log_one_minus_probability = std::log1p(-probability);
    std::int64_t vertex = 1;
    std::int64_t neighbor = -1;
    while (vertex < vertices) {
      const double unit = detail::sample_unit_double(generator);
      const double skip_value =
          std::log1p(-unit) / log_one_minus_probability;
      if (!std::isfinite(skip_value) || skip_value < 0.0 ||
          skip_value >= 0x1p63) {
        break;
      }
      const auto skip =
          static_cast<std::uint64_t>(std::floor(skip_value));
      const std::uint64_t remaining =
          remaining_pair_slots(vertex, neighbor, vertices);
      if (skip >= remaining) {
        break;
      }
      neighbor += 1 + static_cast<std::int64_t>(skip);
      while (neighbor >= vertex && vertex < vertices) {
        neighbor -= vertex;
        ++vertex;
      }
      if (vertex < vertices) {
        append_checked_edge(edges, static_cast<int>(vertex),
                            static_cast<int>(neighbor), maximum_edges);
      }
    }
  }

  GeneratedGraph generated;
  generated.logical_edges = static_cast<std::uint64_t>(edges.size());
  generated.graph = build_canonical_graph(vertices, edges, self_loops);
  return generated;
}

Dataset make_synthetic_dataset(GeneratedGraph generated,
                               std::size_t feature_dim,
                               std::uint64_t feature_seed) {
  const int vertices = generated.graph.num_vertices;
  const bool self_loops = infer_exact_self_loops(generated.graph);

  Dataset dataset;
  dataset.graph = std::move(generated.graph);
  dataset.features =
      generate_synthetic_features(vertices, feature_dim, feature_seed);
  dataset.metadata.directed = false;
  dataset.metadata.self_loops = self_loops;
  validate_dataset(dataset);
  return dataset;
}

namespace detail {

std::uint64_t sample_bounded(std::mt19937_64& generator,
                             std::uint64_t exclusive_upper_bound) {
  if (exclusive_upper_bound == 0) {
    throw std::invalid_argument("bounded sampler requires a nonzero bound");
  }
  const std::uint64_t threshold =
      (UINT64_C(0) - exclusive_upper_bound) % exclusive_upper_bound;
  std::uint64_t sample = 0;
  do {
    sample = generator();
  } while (sample < threshold);
  return sample % exclusive_upper_bound;
}

double sample_unit_double(std::mt19937_64& generator) {
  return static_cast<double>(generator() >> 11U) * 0x1p-53;
}

}  // namespace detail
}  // namespace gnni
