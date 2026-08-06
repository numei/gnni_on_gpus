#pragma once

#include "gnni/cuda_error.hpp"

#include <cuda_runtime_api.h>

#include <cstddef>
#include <limits>
#include <stdexcept>
#include <utility>

namespace gnni {

template <typename T>
class DeviceBuffer {
 public:
  DeviceBuffer() = default;
  explicit DeviceBuffer(std::size_t count) { resize(count); }

  ~DeviceBuffer() { release_noexcept(); }

  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;

  DeviceBuffer(DeviceBuffer&& other) noexcept { swap(other); }

  DeviceBuffer& operator=(DeviceBuffer&& other) noexcept {
    if (this != &other) {
      release_noexcept();
      swap(other);
    }
    return *this;
  }

  void resize(std::size_t count) {
    const std::size_t bytes = checked_byte_count(count);
    if (count <= capacity_) {
      size_ = count;
      return;
    }

    T* replacement = nullptr;
    GNNI_CUDA_CHECK(
        cudaMalloc(reinterpret_cast<void**>(&replacement), bytes));
    if (data_ != nullptr) {
      try {
        GNNI_CUDA_CHECK(cudaFree(data_));
      } catch (...) {
        check_cuda_noexcept(cudaFree(replacement),
                            "cudaFree(replacement)");
        throw;
      }
    }
    data_ = replacement;
    size_ = count;
    capacity_ = count;
  }

  void copy_from_host(const T* source, std::size_t count,
                      cudaStream_t stream = nullptr) {
    resize(count);
    if (count != 0) {
      GNNI_CUDA_CHECK(
          cudaMemcpyAsync(data_, source, checked_byte_count(count),
                          cudaMemcpyHostToDevice, stream));
    }
  }

  void copy_to_host(T* destination, std::size_t count,
                    cudaStream_t stream = nullptr) const {
    if (count > size_) {
      throw std::invalid_argument("device-to-host copy exceeds buffer size");
    }
    if (count != 0) {
      GNNI_CUDA_CHECK(
          cudaMemcpyAsync(destination, data_, checked_byte_count(count),
                          cudaMemcpyDeviceToHost, stream));
    }
  }

  T* data() noexcept { return data_; }
  const T* data() const noexcept { return data_; }
  std::size_t size() const noexcept { return size_; }
  std::size_t capacity() const noexcept { return capacity_; }

 private:
  static std::size_t checked_byte_count(std::size_t count) {
    if (count > std::numeric_limits<std::size_t>::max() / sizeof(T)) {
      throw std::invalid_argument(
          "device buffer element count overflows its byte size");
    }
    return count * sizeof(T);
  }

  void release_noexcept() noexcept {
    if (data_ != nullptr) {
      check_cuda_noexcept(cudaFree(data_), "cudaFree");
    }
    data_ = nullptr;
    size_ = 0;
    capacity_ = 0;
  }

  void swap(DeviceBuffer& other) noexcept {
    std::swap(data_, other.data_);
    std::swap(size_, other.size_);
    std::swap(capacity_, other.capacity_);
  }

  T* data_ = nullptr;
  std::size_t size_ = 0;
  std::size_t capacity_ = 0;
};

}  // namespace gnni

