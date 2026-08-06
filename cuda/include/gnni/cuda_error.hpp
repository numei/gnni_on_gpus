#pragma once

#include <cuda_runtime_api.h>

#include <cstdio>
#include <stdexcept>
#include <string>

namespace gnni {

class CudaError : public std::runtime_error {
 public:
  CudaError(cudaError_t code, const std::string& message)
      : std::runtime_error(message), code_(code) {}

  cudaError_t code() const noexcept { return code_; }

 private:
  cudaError_t code_;
};

inline void check_cuda(cudaError_t status, const char* expression,
                       const char* file, int line) {
  if (status == cudaSuccess) {
    return;
  }
  throw CudaError(status,
                  std::string(expression) + " failed at " + file + ":" +
                      std::to_string(line) + ": " +
                      cudaGetErrorString(status));
}

inline void check_cuda_noexcept(cudaError_t status,
                                const char* expression) noexcept {
  if (status != cudaSuccess) {
    std::fprintf(stderr, "%s failed during cleanup: %s\n", expression,
                 cudaGetErrorString(status));
  }
}

}  // namespace gnni

#define GNNI_CUDA_CHECK(expression)                                      \
  ::gnni::check_cuda((expression), #expression, __FILE__, __LINE__)

#define GNNI_CUDA_CHECK_LAST_KERNEL()                                    \
  ::gnni::check_cuda(cudaGetLastError(), "CUDA kernel launch", __FILE__, \
                     __LINE__)

