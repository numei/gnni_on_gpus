#include "gnni/gcn_cuda.hpp"

#include "gnni/cuda_error.hpp"
#include "gnni/device_buffer.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace gnni {
namespace {

constexpr unsigned int kThreadsPerBlock = 256;
constexpr unsigned int kMaxBaselineBlocks = 65535;

std::size_t checked_element_count(std::size_t rows, std::size_t cols,
                                  const char* label) {
  if (rows != 0 && cols > std::numeric_limits<std::size_t>::max() / rows) {
    throw std::invalid_argument(std::string(label) + " shape overflows size_t");
  }
  return rows * cols;
}

void validate_graph(const CSRGraph& graph) {
  if (graph.num_vertices <= 0) {
    throw std::invalid_argument("CSR graph must contain at least one vertex");
  }
  const auto expected_offsets =
      static_cast<std::size_t>(graph.num_vertices) + 1U;
  if (graph.row_offsets.size() != expected_offsets) {
    throw std::invalid_argument("CSR row offset length must be N + 1");
  }
  if (graph.row_offsets.front() != 0) {
    throw std::invalid_argument("CSR row offsets must start at zero");
  }
  for (std::size_t i = 1; i < graph.row_offsets.size(); ++i) {
    if (graph.row_offsets[i] < graph.row_offsets[i - 1]) {
      throw std::invalid_argument("CSR row offsets must be nondecreasing");
    }
  }
  if (graph.row_offsets.back() < 0 ||
      static_cast<std::size_t>(graph.row_offsets.back()) !=
          graph.column_indices.size()) {
    throw std::invalid_argument(
        "last CSR row offset must equal the column index count");
  }
  for (int neighbor : graph.column_indices) {
    if (neighbor < 0 || neighbor >= graph.num_vertices) {
      throw std::invalid_argument("CSR column index is outside [0, N)");
    }
  }
}

void validate_matrix(const DenseMatrix& matrix, const char* label) {
  if (matrix.cols == 0) {
    throw std::invalid_argument(std::string(label) +
                                " must have at least one column");
  }
  const auto expected = checked_element_count(matrix.rows, matrix.cols, label);
  if (matrix.values.size() != expected) {
    throw std::invalid_argument(std::string(label) +
                                " value count does not match its shape");
  }
}

void validate_config(const GcnLayerConfig& config) {
  switch (config.aggregation) {
    case Aggregation::Sum:
    case Aggregation::Mean:
    case Aggregation::SymmetricGcn:
      break;
    default:
      throw std::invalid_argument("unsupported aggregation mode");
  }
  switch (config.activation) {
    case Activation::None:
    case Activation::Relu:
      break;
    default:
      throw std::invalid_argument("unsupported activation mode");
  }
}

class CudaStream {
 public:
  CudaStream() {
    GNNI_CUDA_CHECK(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking));
  }

  ~CudaStream() {
    if (stream_ != nullptr) {
      check_cuda_noexcept(cudaStreamDestroy(stream_), "cudaStreamDestroy");
    }
  }

  CudaStream(const CudaStream&) = delete;
  CudaStream& operator=(const CudaStream&) = delete;

  cudaStream_t get() const noexcept { return stream_; }

 private:
  cudaStream_t stream_ = nullptr;
};

class CudaEvent {
 public:
  CudaEvent() { GNNI_CUDA_CHECK(cudaEventCreate(&event_)); }

  ~CudaEvent() {
    if (event_ != nullptr) {
      check_cuda_noexcept(cudaEventDestroy(event_), "cudaEventDestroy");
    }
  }

  CudaEvent(const CudaEvent&) = delete;
  CudaEvent& operator=(const CudaEvent&) = delete;

  void record(cudaStream_t stream) {
    GNNI_CUDA_CHECK(cudaEventRecord(event_, stream));
  }

  void synchronize() { GNNI_CUDA_CHECK(cudaEventSynchronize(event_)); }

  cudaEvent_t get() const noexcept { return event_; }

 private:
  cudaEvent_t event_ = nullptr;
};

float elapsed_ms(const CudaEvent& start, const CudaEvent& stop) {
  float milliseconds = 0.0F;
  GNNI_CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start.get(), stop.get()));
  return milliseconds;
}

__global__ void aggregate_vertex_feature_kernel(
    const int* row_offsets, const int* column_indices, const float* input,
    float* aggregate, std::size_t vertices, std::size_t features,
    int aggregation_mode) {
  const std::size_t first = static_cast<std::size_t>(blockIdx.x) * blockDim.x +
                            threadIdx.x;
  const std::size_t stride =
      static_cast<std::size_t>(blockDim.x) * gridDim.x;
  const std::size_t elements = vertices * features;
  for (std::size_t index = first; index < elements; index += stride) {
    const std::size_t vertex = index / features;
    const std::size_t feature = index - vertex * features;
    const int begin = row_offsets[vertex];
    const int end = row_offsets[vertex + 1U];
    const int degree = end - begin;
    float sum = 0.0F;
    for (int edge = begin; edge < end; ++edge) {
      const int neighbor = column_indices[edge];
      float scale = 1.0F;
      if (aggregation_mode == static_cast<int>(Aggregation::Mean)) {
        scale = 1.0F / static_cast<float>(degree);
      } else if (aggregation_mode ==
                 static_cast<int>(Aggregation::SymmetricGcn)) {
        const int neighbor_degree =
            row_offsets[neighbor + 1] - row_offsets[neighbor];
        scale = neighbor_degree == 0
                    ? 0.0F
                    : rsqrtf(static_cast<float>(degree) *
                             static_cast<float>(neighbor_degree));
      }
      sum += scale * input[static_cast<std::size_t>(neighbor) * features +
                           feature];
    }
    aggregate[index] = sum;
  }
}

__global__ void dense_transform_kernel(const float* aggregate,
                                       const float* weights, float* output,
                                       std::size_t vertices,
                                       std::size_t input_features,
                                       std::size_t output_features,
                                       int activation_mode) {
  const std::size_t first = static_cast<std::size_t>(blockIdx.x) * blockDim.x +
                            threadIdx.x;
  const std::size_t stride =
      static_cast<std::size_t>(blockDim.x) * gridDim.x;
  const std::size_t elements = vertices * output_features;
  for (std::size_t index = first; index < elements; index += stride) {
    const std::size_t vertex = index / output_features;
    const std::size_t output_feature = index - vertex * output_features;
    float value = 0.0F;
    for (std::size_t input_feature = 0; input_feature < input_features;
         ++input_feature) {
      value += aggregate[vertex * input_features + input_feature] *
               weights[input_feature * output_features + output_feature];
    }
    if (activation_mode == static_cast<int>(Activation::Relu)) {
      value = fmaxf(value, 0.0F);
    }
    output[index] = value;
  }
}

unsigned int baseline_block_count(std::size_t elements) {
  const std::size_t required =
      (elements + kThreadsPerBlock - 1U) / kThreadsPerBlock;
  return static_cast<unsigned int>(
      std::min<std::size_t>(required, kMaxBaselineBlocks));
}

void launch_aggregate(const int* row_offsets, const int* column_indices,
                      const float* input, float* aggregate,
                      std::size_t vertices, std::size_t features,
                      Aggregation aggregation, cudaStream_t stream) {
  const auto elements = checked_element_count(vertices, features, "aggregate");
  const unsigned int blocks = baseline_block_count(elements);
  aggregate_vertex_feature_kernel<<<blocks, kThreadsPerBlock, 0, stream>>>(
      row_offsets, column_indices, input, aggregate, vertices, features,
      static_cast<int>(aggregation));
  GNNI_CUDA_CHECK_LAST_KERNEL();
}

void launch_dense(const float* aggregate, const float* weights, float* output,
                  std::size_t vertices, std::size_t input_features,
                  std::size_t output_features, Activation activation,
                  cudaStream_t stream) {
  const auto elements =
      checked_element_count(vertices, output_features, "output");
  const unsigned int blocks = baseline_block_count(elements);
  dense_transform_kernel<<<blocks, kThreadsPerBlock, 0, stream>>>(
      aggregate, weights, output, vertices, input_features, output_features,
      static_cast<int>(activation));
  GNNI_CUDA_CHECK_LAST_KERNEL();
}

}  // namespace

DenseMatrix::DenseMatrix(std::size_t row_count, std::size_t column_count)
    : rows(row_count),
      cols(column_count),
      values(checked_element_count(row_count, column_count, "matrix"), 0.0F) {}

DenseMatrix::DenseMatrix(std::size_t row_count, std::size_t column_count,
                         std::vector<float> data)
    : rows(row_count), cols(column_count), values(std::move(data)) {
  validate_matrix(*this, "matrix");
}

class CudaGcnContext::Impl {
 public:
  explicit Impl(const CSRGraph& graph) : vertices_(graph.num_vertices) {
    validate_graph(graph);

    CudaEvent start;
    CudaEvent stop;
    start.record(stream_.get());
    row_offsets_.copy_from_host(graph.row_offsets.data(),
                                graph.row_offsets.size(), stream_.get());
    column_indices_.copy_from_host(graph.column_indices.data(),
                                  graph.column_indices.size(), stream_.get());
    stop.record(stream_.get());
    stop.synchronize();
    graph_h2d_ms_ = elapsed_ms(start, stop);
  }

  GcnRunResult run_network(
      const DenseMatrix& features,
      const std::vector<DenseMatrix>& layer_weights,
      const std::vector<GcnLayerConfig>& layer_configs) {
    const auto wall_start = std::chrono::steady_clock::now();
    validate_matrix(features, "features");
    if (features.rows != static_cast<std::size_t>(vertices_)) {
      throw std::invalid_argument("feature row count must equal graph vertices");
    }
    if (layer_weights.empty()) {
      throw std::invalid_argument("network must contain at least one layer");
    }
    if (layer_configs.size() != layer_weights.size()) {
      throw std::invalid_argument(
          "layer configuration count must equal weight matrix count");
    }

    std::size_t current_features = features.cols;
    std::size_t max_feature_width = current_features;
    std::size_t total_weight_values = 0;
    std::vector<std::size_t> weight_offsets;
    weight_offsets.reserve(layer_weights.size());
    for (std::size_t layer = 0; layer < layer_weights.size(); ++layer) {
      const auto& weights = layer_weights[layer];
      validate_matrix(weights, "weights");
      validate_config(layer_configs[layer]);
      if (weights.rows != current_features) {
        throw std::invalid_argument(
            "weight rows must equal the preceding feature width");
      }
      weight_offsets.push_back(total_weight_values);
      if (weights.values.size() >
          std::numeric_limits<std::size_t>::max() - total_weight_values) {
        throw std::invalid_argument("combined weight size overflows size_t");
      }
      total_weight_values += weights.values.size();
      max_feature_width =
          std::max({max_feature_width, weights.rows, weights.cols});
      current_features = weights.cols;
    }

    std::vector<float> packed_weights;
    packed_weights.reserve(total_weight_values);
    for (const auto& weights : layer_weights) {
      packed_weights.insert(packed_weights.end(), weights.values.begin(),
                            weights.values.end());
    }

    const std::size_t workspace_elements = checked_element_count(
        static_cast<std::size_t>(vertices_), max_feature_width, "workspace");
    feature_a_.resize(workspace_elements);
    feature_b_.resize(workspace_elements);
    aggregate_.resize(workspace_elements);
    weights_.resize(total_weight_values);

    CudaEvent h2d_start;
    CudaEvent h2d_stop;
    CudaEvent kernel_start;
    CudaEvent kernel_stop;
    CudaEvent d2h_start;
    CudaEvent d2h_stop;

    h2d_start.record(stream_.get());
    GNNI_CUDA_CHECK(cudaMemcpyAsync(
        feature_a_.data(), features.values.data(),
        features.values.size() * sizeof(float), cudaMemcpyHostToDevice,
        stream_.get()));
    GNNI_CUDA_CHECK(cudaMemcpyAsync(
        weights_.data(), packed_weights.data(),
        packed_weights.size() * sizeof(float), cudaMemcpyHostToDevice,
        stream_.get()));
    h2d_stop.record(stream_.get());

    kernel_start.record(stream_.get());
    const float* current = feature_a_.data();
    float* next = feature_b_.data();
    current_features = features.cols;
    for (std::size_t layer = 0; layer < layer_weights.size(); ++layer) {
      const auto& layer_weight = layer_weights[layer];
      launch_aggregate(row_offsets_.data(), column_indices_.data(), current,
                       aggregate_.data(), static_cast<std::size_t>(vertices_),
                       current_features, layer_configs[layer].aggregation,
                       stream_.get());
      launch_dense(aggregate_.data(),
                   weights_.data() + weight_offsets[layer], next,
                   static_cast<std::size_t>(vertices_), current_features,
                   layer_weight.cols, layer_configs[layer].activation,
                   stream_.get());
      current = next;
      next = next == feature_a_.data() ? feature_b_.data() : feature_a_.data();
      current_features = layer_weight.cols;
    }
    kernel_stop.record(stream_.get());

    DenseMatrix output(static_cast<std::size_t>(vertices_), current_features);
    d2h_start.record(stream_.get());
    GNNI_CUDA_CHECK(cudaMemcpyAsync(
        output.values.data(), current, output.values.size() * sizeof(float),
        cudaMemcpyDeviceToHost, stream_.get()));
    d2h_stop.record(stream_.get());
    d2h_stop.synchronize();

    CudaTimings timings;
    timings.graph_h2d_ms = graph_h2d_ms_;
    timings.h2d_ms = elapsed_ms(h2d_start, h2d_stop);
    timings.kernel_ms = elapsed_ms(kernel_start, kernel_stop);
    timings.d2h_ms = elapsed_ms(d2h_start, d2h_stop);
    timings.end_to_end_ms =
        std::chrono::duration<float, std::milli>(
            std::chrono::steady_clock::now() - wall_start)
            .count();
    return {std::move(output), timings};
  }

  float graph_upload_ms() const noexcept { return graph_h2d_ms_; }

 private:
  int vertices_;
  CudaStream stream_;
  DeviceBuffer<int> row_offsets_;
  DeviceBuffer<int> column_indices_;
  DeviceBuffer<float> feature_a_;
  DeviceBuffer<float> feature_b_;
  DeviceBuffer<float> aggregate_;
  DeviceBuffer<float> weights_;
  float graph_h2d_ms_ = 0.0F;
};

CudaGcnContext::CudaGcnContext(const CSRGraph& graph)
    : impl_(std::make_unique<Impl>(graph)) {}

CudaGcnContext::~CudaGcnContext() = default;
CudaGcnContext::CudaGcnContext(CudaGcnContext&&) noexcept = default;
CudaGcnContext& CudaGcnContext::operator=(CudaGcnContext&&) noexcept = default;

GcnRunResult CudaGcnContext::run_layer(const DenseMatrix& features,
                                       const DenseMatrix& weights,
                                       const GcnLayerConfig& config) {
  const auto wall_start = std::chrono::steady_clock::now();
  auto result = run_network(features, {weights}, {config});
  result.timings.end_to_end_ms =
      std::chrono::duration<float, std::milli>(
          std::chrono::steady_clock::now() - wall_start)
          .count();
  return result;
}

GcnRunResult CudaGcnContext::run_network(
    const DenseMatrix& features,
    const std::vector<DenseMatrix>& layer_weights,
    const std::vector<GcnLayerConfig>& layer_configs) {
  if (!impl_) {
    throw std::logic_error("cannot run a moved-from CUDA GCN context");
  }
  return impl_->run_network(features, layer_weights, layer_configs);
}

float CudaGcnContext::graph_upload_ms() const noexcept {
  return impl_ ? impl_->graph_upload_ms() : 0.0F;
}

GcnRunResult run_gcn_cuda_vertex_centric(const CSRGraph& graph,
                                         const DenseMatrix& features,
                                         const DenseMatrix& weights,
                                         const GcnLayerConfig& config) {
  CudaGcnContext context(graph);
  return context.run_layer(features, weights, config);
}

}  // namespace gnni

