#include "gnni/dataset_io.hpp"

#include <array>
#include <cerrno>
#include <cstdio>
#include <cstring>
#include <fcntl.h>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>
#include <unistd.h>
#include <vector>

namespace gnni {
namespace {

constexpr std::array<std::uint8_t, 8> kMagic = {
    'G', 'N', 'N', 'I', 'D', 'A', 'T', '1'};
constexpr std::uint32_t kKnownFlags =
    kDirectedFlag | kSelfLoopsFlag | kLabelsFlag | kSplitsFlag;

std::runtime_error file_error(const std::filesystem::path& path,
                              const std::string& message) {
  return std::runtime_error("dataset file '" + path.string() + "': " + message);
}

void require_binary32(const std::filesystem::path& path) {
  if (sizeof(float) != 4 || !std::numeric_limits<float>::is_iec559 ||
      std::numeric_limits<float>::radix != 2 ||
      std::numeric_limits<float>::digits != 24 ||
      std::numeric_limits<float>::max_exponent != 128) {
    throw file_error(path,
                     "host float is not IEEE-754 binary32 and cannot encode "
                     "version 1 features");
  }
}

std::uint64_t checked_add(std::uint64_t left, std::uint64_t right,
                          const std::filesystem::path& path,
                          const char* field) {
  if (right > std::numeric_limits<std::uint64_t>::max() - left) {
    throw file_error(path, std::string("payload size overflow while adding ") +
                               field);
  }
  return left + right;
}

std::uint64_t checked_multiply(std::uint64_t left, std::uint64_t right,
                               const std::filesystem::path& path,
                               const char* field) {
  if (left != 0 && right > std::numeric_limits<std::uint64_t>::max() / left) {
    throw file_error(path,
                     std::string("payload size overflow while multiplying ") +
                         field);
  }
  return left * right;
}

std::uint64_t payload_size(std::uint64_t vertices, std::uint64_t entries,
                           std::uint64_t feature_dim, std::uint32_t flags,
                           const std::filesystem::path& path) {
  const std::uint64_t row_offset_count =
      checked_add(vertices, 1, path, "CSR row offset count");
  std::uint64_t total =
      checked_multiply(row_offset_count, 8, path, "CSR row offsets");
  total = checked_add(total,
                      checked_multiply(entries, 4, path, "CSR columns"), path,
                      "CSR columns");
  const std::uint64_t feature_count =
      checked_multiply(vertices, feature_dim, path, "feature element count");
  total = checked_add(
      total, checked_multiply(feature_count, 4, path, "features"), path,
      "features");
  if ((flags & kLabelsFlag) != 0) {
    total = checked_add(
        total, checked_multiply(vertices, 4, path, "labels"), path, "labels");
  }
  if ((flags & kSplitsFlag) != 0) {
    total = checked_add(total, vertices, path, "splits");
  }
  return total;
}

std::size_t checked_size_t(std::uint64_t value,
                           const std::filesystem::path& path,
                           const char* field) {
  if constexpr (std::numeric_limits<std::size_t>::digits <
                std::numeric_limits<std::uint64_t>::digits) {
    if (value > std::numeric_limits<std::size_t>::max()) {
      throw file_error(path, std::string(field) + " exceeds SIZE_MAX");
    }
  }
  return static_cast<std::size_t>(value);
}

template <typename Element>
std::size_t checked_vector_count(std::uint64_t value,
                                 const std::filesystem::path& path,
                                 const char* field) {
  const std::size_t count = checked_size_t(value, path, field);
  if (count > std::vector<Element>().max_size()) {
    throw file_error(path, std::string(field) + " exceeds vector max_size");
  }
  return count;
}

void read_exact(std::ifstream& input, std::uint8_t* bytes, std::size_t count,
                const std::filesystem::path& path, const char* field) {
  input.read(reinterpret_cast<char*>(bytes),
             static_cast<std::streamsize>(count));
  if (input.gcount() != static_cast<std::streamsize>(count)) {
    throw file_error(path, std::string("truncated ") + field);
  }
}

std::uint32_t read_u32(std::ifstream& input,
                       const std::filesystem::path& path, const char* field) {
  std::array<std::uint8_t, 4> bytes{};
  read_exact(input, bytes.data(), bytes.size(), path, field);
  std::uint32_t value = 0;
  for (unsigned byte = 0; byte != bytes.size(); ++byte) {
    value |= static_cast<std::uint32_t>(bytes[byte]) << (byte * 8);
  }
  return value;
}

std::uint64_t read_u64(std::ifstream& input,
                       const std::filesystem::path& path, const char* field) {
  std::array<std::uint8_t, 8> bytes{};
  read_exact(input, bytes.data(), bytes.size(), path, field);
  std::uint64_t value = 0;
  for (unsigned byte = 0; byte != bytes.size(); ++byte) {
    value |= static_cast<std::uint64_t>(bytes[byte]) << (byte * 8);
  }
  return value;
}

float read_f32(std::ifstream& input, const std::filesystem::path& path,
               const char* field) {
  const std::uint32_t bits = read_u32(input, path, field);
  float value = 0.0F;
  std::memcpy(&value, &bits, sizeof(bits));
  return value;
}

void write_exact(std::FILE* output, const std::uint8_t* bytes,
                 std::size_t count, const std::filesystem::path& path,
                 const char* field) {
  if (std::fwrite(bytes, 1, count, output) != count) {
    throw file_error(path, std::string("failed to write ") + field + ": " +
                               std::strerror(errno));
  }
}

void write_u32(std::FILE* output, std::uint32_t value,
               const std::filesystem::path& path, const char* field) {
  std::array<std::uint8_t, 4> bytes{};
  for (unsigned byte = 0; byte != bytes.size(); ++byte) {
    bytes[byte] = static_cast<std::uint8_t>(value >> (byte * 8));
  }
  write_exact(output, bytes.data(), bytes.size(), path, field);
}

void write_u64(std::FILE* output, std::uint64_t value,
               const std::filesystem::path& path, const char* field) {
  std::array<std::uint8_t, 8> bytes{};
  for (unsigned byte = 0; byte != bytes.size(); ++byte) {
    bytes[byte] = static_cast<std::uint8_t>(value >> (byte * 8));
  }
  write_exact(output, bytes.data(), bytes.size(), path, field);
}

void write_f32(std::FILE* output, float value,
               const std::filesystem::path& path, const char* field) {
  std::uint32_t bits = 0;
  std::memcpy(&bits, &value, sizeof(bits));
  write_u32(output, bits, path, field);
}

std::uint64_t size_to_u64(std::size_t value,
                          const std::filesystem::path& path,
                          const char* field) {
  if constexpr (std::numeric_limits<std::size_t>::digits >
                std::numeric_limits<std::uint64_t>::digits) {
    if (value >
        static_cast<std::size_t>(std::numeric_limits<std::uint64_t>::max())) {
      throw file_error(path, std::string(field) + " exceeds UINT64_MAX");
    }
  }
  return static_cast<std::uint64_t>(value);
}

std::filesystem::path temporary_path(const std::filesystem::path& path) {
  std::filesystem::path temporary = path;
  temporary += ".tmp-" + std::to_string(static_cast<long long>(::getpid()));
  return temporary;
}

}  // namespace

Dataset load_dataset(const std::filesystem::path& path) {
  require_binary32(path);

  std::error_code size_error;
  const std::uintmax_t file_length = std::filesystem::file_size(path, size_error);
  if (size_error) {
    throw file_error(path,
                     "cannot determine file length: " + size_error.message());
  }
  if (file_length < kDatasetHeaderBytes) {
    throw file_error(path, "file length is smaller than the 64-byte header");
  }

  std::ifstream input(path, std::ios::binary);
  if (!input) {
    throw file_error(path, "cannot open for reading");
  }

  std::array<std::uint8_t, 8> magic{};
  read_exact(input, magic.data(), magic.size(), path, "magic");
  if (magic != kMagic) {
    throw file_error(path, "invalid magic; expected GNNIDAT1");
  }

  const std::uint32_t version = read_u32(input, path, "version");
  if (version != kDatasetFormatVersion) {
    throw file_error(path, "unsupported version " + std::to_string(version));
  }
  const std::uint32_t header_bytes = read_u32(input, path, "header length");
  if (header_bytes != kDatasetHeaderBytes) {
    throw file_error(path,
                     "invalid header length " + std::to_string(header_bytes));
  }
  const std::uint32_t flags = read_u32(input, path, "flags");
  if ((flags & ~kKnownFlags) != 0) {
    throw file_error(path, "unknown flag bits " +
                               std::to_string(flags & ~kKnownFlags));
  }
  const std::uint32_t reserved = read_u32(input, path, "reserved word");
  if (reserved != 0) {
    throw file_error(path, "reserved word must be zero");
  }
  if ((flags & kSplitsFlag) != 0 && (flags & kLabelsFlag) == 0) {
    throw file_error(path, "split flag requires labels flag");
  }

  const std::uint64_t vertices = read_u64(input, path, "num_vertices");
  const std::uint64_t entries = read_u64(input, path, "num_csr_entries");
  const std::uint64_t feature_dim = read_u64(input, path, "feature_dim");
  const std::uint64_t num_classes = read_u64(input, path, "num_classes");
  const std::uint64_t declared_payload =
      read_u64(input, path, "payload_bytes");

  if (vertices > static_cast<std::uint64_t>(std::numeric_limits<int>::max())) {
    throw file_error(path, "num_vertices exceeds INT_MAX");
  }
  if (entries > static_cast<std::uint64_t>(std::numeric_limits<int>::max())) {
    throw file_error(path, "num_csr_entries exceeds INT_MAX");
  }

  const std::uint64_t expected_payload =
      payload_size(vertices, entries, feature_dim, flags, path);
  if (declared_payload != expected_payload) {
    throw file_error(path, "payload_bytes " + std::to_string(declared_payload) +
                               " does not match computed payload length " +
                               std::to_string(expected_payload));
  }
  const std::uint64_t expected_file_length =
      checked_add(kDatasetHeaderBytes, expected_payload, path, "file length");
  if (file_length != static_cast<std::uintmax_t>(expected_file_length)) {
    throw file_error(path, "file length " + std::to_string(file_length) +
                               " does not equal expected length " +
                               std::to_string(expected_file_length));
  }

  const std::uint64_t row_count_u64 =
      checked_add(vertices, 1, path, "CSR row offset count");
  const std::uint64_t feature_count_u64 =
      checked_multiply(vertices, feature_dim, path, "feature element count");
  const std::size_t row_count =
      checked_vector_count<int>(row_count_u64, path, "CSR row offset count");
  const std::size_t column_count =
      checked_vector_count<int>(entries, path, "CSR column count");
  const std::size_t feature_count = checked_vector_count<float>(
      feature_count_u64, path, "feature element count");
  const std::size_t vertex_count =
      checked_vector_count<std::int32_t>(vertices, path, "vertex count");
  const std::size_t feature_width =
      checked_size_t(feature_dim, path, "feature_dim");

  Dataset dataset;
  dataset.graph.num_vertices = static_cast<int>(vertices);
  dataset.graph.row_offsets.resize(row_count);
  dataset.graph.column_indices.resize(column_count);
  dataset.features.rows = vertex_count;
  dataset.features.cols = feature_width;
  dataset.features.values.resize(feature_count);
  if ((flags & kLabelsFlag) != 0) {
    dataset.labels.resize(vertex_count);
  }
  if ((flags & kSplitsFlag) != 0) {
    checked_vector_count<NodeSplit>(vertices, path, "split count");
    dataset.splits.resize(vertex_count);
  }
  dataset.metadata.directed = (flags & kDirectedFlag) != 0;
  dataset.metadata.self_loops = (flags & kSelfLoopsFlag) != 0;
  dataset.metadata.num_classes = num_classes;

  for (std::size_t index = 0; index < row_count; ++index) {
    const std::uint64_t offset = read_u64(input, path, "CSR row offset");
    if (offset >
        static_cast<std::uint64_t>(std::numeric_limits<int>::max())) {
      throw file_error(path, "CSR row offset " + std::to_string(index) +
                                 " exceeds INT_MAX");
    }
    dataset.graph.row_offsets[index] = static_cast<int>(offset);
  }
  for (std::size_t index = 0; index < column_count; ++index) {
    const std::uint32_t column = read_u32(input, path, "CSR column");
    if (column >
        static_cast<std::uint32_t>(std::numeric_limits<int>::max())) {
      throw file_error(path, "CSR column " + std::to_string(index) +
                                 " exceeds INT_MAX");
    }
    dataset.graph.column_indices[index] = static_cast<int>(column);
  }
  for (float& feature : dataset.features.values) {
    feature = read_f32(input, path, "feature");
  }
  for (std::size_t index = 0; index < dataset.labels.size(); ++index) {
    const std::uint32_t label = read_u32(input, path, "label");
    if (label >
        static_cast<std::uint32_t>(std::numeric_limits<std::int32_t>::max())) {
      throw file_error(path, "label " + std::to_string(index) +
                                 " is negative or exceeds INT32_MAX");
    }
    dataset.labels[index] = static_cast<std::int32_t>(label);
  }
  for (NodeSplit& split : dataset.splits) {
    std::uint8_t value = 0;
    read_exact(input, &value, 1, path, "split");
    split = static_cast<NodeSplit>(value);
  }

  char trailing = 0;
  if (input.get(trailing)) {
    throw file_error(path, "trailing byte after declared payload");
  }
  if (!input.eof()) {
    throw file_error(path, "I/O failure while checking exact file length");
  }

  validate_dataset(dataset);
  return dataset;
}

void save_dataset(const std::filesystem::path& path, const Dataset& dataset,
                  bool overwrite) {
  validate_dataset(dataset);
  require_binary32(path);

  const std::uint64_t vertices =
      static_cast<std::uint64_t>(dataset.graph.num_vertices);
  const std::uint64_t entries = size_to_u64(
      dataset.graph.column_indices.size(), path, "num_csr_entries");
  const std::uint64_t feature_dim =
      size_to_u64(dataset.features.cols, path, "feature_dim");
  if (entries > static_cast<std::uint64_t>(std::numeric_limits<int>::max())) {
    throw file_error(path, "num_csr_entries exceeds INT_MAX");
  }

  std::uint32_t flags = 0;
  if (dataset.metadata.directed) {
    flags |= kDirectedFlag;
  }
  if (dataset.metadata.self_loops) {
    flags |= kSelfLoopsFlag;
  }
  if (!dataset.labels.empty()) {
    flags |= kLabelsFlag;
  }
  if (!dataset.splits.empty()) {
    flags |= kSplitsFlag;
  }
  const std::uint64_t payload_bytes =
      payload_size(vertices, entries, feature_dim, flags, path);

  std::error_code exists_error;
  const bool target_exists = std::filesystem::exists(path, exists_error);
  if (exists_error) {
    throw file_error(path,
                     "cannot inspect output path: " + exists_error.message());
  }
  if (target_exists && !overwrite) {
    throw file_error(path, "output already exists and overwrite is false");
  }

  const std::filesystem::path temporary = temporary_path(path);
  const int descriptor = ::open(temporary.c_str(), O_WRONLY | O_CREAT | O_EXCL,
                                0666);
  if (descriptor < 0) {
    throw file_error(path, "cannot create unique temporary file '" +
                               temporary.string() + "': " +
                               std::strerror(errno));
  }

  bool owns_temporary = true;
  std::FILE* output = ::fdopen(descriptor, "wb");
  if (output == nullptr) {
    const int saved_errno = errno;
    ::close(descriptor);
    std::error_code ignored;
    std::filesystem::remove(temporary, ignored);
    throw file_error(path, "cannot open temporary stream: " +
                               std::string(std::strerror(saved_errno)));
  }

  try {
    write_exact(output, kMagic.data(), kMagic.size(), path, "magic");
    write_u32(output, kDatasetFormatVersion, path, "version");
    write_u32(output, kDatasetHeaderBytes, path, "header length");
    write_u32(output, flags, path, "flags");
    write_u32(output, 0, path, "reserved word");
    write_u64(output, vertices, path, "num_vertices");
    write_u64(output, entries, path, "num_csr_entries");
    write_u64(output, feature_dim, path, "feature_dim");
    write_u64(output, dataset.metadata.num_classes, path, "num_classes");
    write_u64(output, payload_bytes, path, "payload_bytes");

    for (int offset : dataset.graph.row_offsets) {
      write_u64(output, static_cast<std::uint64_t>(offset), path,
                "CSR row offset");
    }
    for (int column : dataset.graph.column_indices) {
      write_u32(output, static_cast<std::uint32_t>(column), path,
                "CSR column");
    }
    for (float feature : dataset.features.values) {
      write_f32(output, feature, path, "feature");
    }
    for (std::int32_t label : dataset.labels) {
      write_u32(output, static_cast<std::uint32_t>(label), path, "label");
    }
    for (NodeSplit split : dataset.splits) {
      const std::uint8_t value = static_cast<std::uint8_t>(split);
      write_exact(output, &value, 1, path, "split");
    }

    if (std::fflush(output) != 0) {
      throw file_error(path, "failed to flush temporary file: " +
                                 std::string(std::strerror(errno)));
    }
    const int close_result = std::fclose(output);
    output = nullptr;
    if (close_result != 0) {
      throw file_error(path, "failed to close temporary file: " +
                                 std::string(std::strerror(errno)));
    }

    if (!overwrite) {
      exists_error.clear();
      if (std::filesystem::exists(path, exists_error)) {
        throw file_error(path, "output already exists and overwrite is false");
      }
      if (exists_error) {
        throw file_error(path,
                         "cannot recheck output path: " + exists_error.message());
      }
    }

    std::error_code rename_error;
    std::filesystem::rename(temporary, path, rename_error);
    if (rename_error) {
      throw file_error(path, "cannot rename temporary file into place: " +
                                 rename_error.message());
    }
    owns_temporary = false;
  } catch (...) {
    if (output != nullptr) {
      std::fclose(output);
    }
    if (owns_temporary) {
      std::error_code ignored;
      std::filesystem::remove(temporary, ignored);
    }
    throw;
  }
}

}  // namespace gnni
