#pragma once

#include "gnni/graph_types.hpp"

#include <memory>
#include <vector>

namespace gnni {

enum class Aggregation : int {
  Sum = 0,
  Mean = 1,
  SymmetricGcn = 2,
};

enum class Activation : int {
  None = 0,
  Relu = 1,
};

struct GcnLayerConfig {
  Aggregation aggregation = Aggregation::Sum;
  Activation activation = Activation::None;
};

struct CudaTimings {
  float graph_h2d_ms = 0.0F;
  float h2d_ms = 0.0F;
  float kernel_ms = 0.0F;
  float d2h_ms = 0.0F;
  float end_to_end_ms = 0.0F;
};

struct GcnRunResult {
  DenseMatrix output;
  CudaTimings timings;
};

class CudaGcnContext {
 public:
  explicit CudaGcnContext(const CSRGraph& graph);
  ~CudaGcnContext();

  CudaGcnContext(const CudaGcnContext&) = delete;
  CudaGcnContext& operator=(const CudaGcnContext&) = delete;
  CudaGcnContext(CudaGcnContext&&) noexcept;
  CudaGcnContext& operator=(CudaGcnContext&&) noexcept;

  GcnRunResult run_layer(const DenseMatrix& features,
                         const DenseMatrix& weights,
                         const GcnLayerConfig& config = {});

  GcnRunResult run_network(
      const DenseMatrix& features,
      const std::vector<DenseMatrix>& layer_weights,
      const std::vector<GcnLayerConfig>& layer_configs);

  float graph_upload_ms() const noexcept;

 private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

GcnRunResult run_gcn_cuda_vertex_centric(
    const CSRGraph& graph, const DenseMatrix& features,
    const DenseMatrix& weights, const GcnLayerConfig& config = {});

}  // namespace gnni

