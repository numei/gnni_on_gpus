#include "gnni/dataset_io.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <exception>
#include <filesystem>
#include <fcntl.h>
#include <fstream>
#include <functional>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#ifdef __linux__
#include <sys/syscall.h>
#include <unistd.h>
#endif
#include <thread>
#include <utility>
#include <vector>

namespace {

std::atomic<bool> install_hook_enabled{false};
std::atomic<bool> install_hook_reached{false};
std::atomic<bool> competitor_finished{false};
std::atomic<bool> competitor_cancelled{false};
std::string install_hook_target;

void wait_for_competing_create(const char* new_path) noexcept {
  if (!install_hook_enabled.load(std::memory_order_acquire) ||
      new_path == nullptr ||
      std::strcmp(new_path, install_hook_target.c_str()) != 0) {
    return;
  }
  install_hook_reached.store(true, std::memory_order_release);
  while (!competitor_finished.load(std::memory_order_acquire)) {
    std::this_thread::yield();
  }
}

}  // namespace

#ifdef __linux__
extern "C" int rename(const char* old_path, const char* new_path) noexcept {
  wait_for_competing_create(new_path);
  return static_cast<int>(::syscall(SYS_renameat, AT_FDCWD, old_path, AT_FDCWD,
                                    new_path));
}

extern "C" int renameat2(int old_directory, const char* old_path,
                          int new_directory, const char* new_path,
                          unsigned int flags) noexcept {
  wait_for_competing_create(new_path);
#ifdef SYS_renameat2
  return static_cast<int>(::syscall(SYS_renameat2, old_directory, old_path,
                                    new_directory, new_path, flags));
#else
  static_cast<void>(old_directory);
  static_cast<void>(old_path);
  static_cast<void>(new_directory);
  static_cast<void>(flags);
  errno = ENOSYS;
  return -1;
#endif
}

#endif
namespace {

using Bytes = std::vector<std::uint8_t>;
using Dataset = gnni::Dataset;
using Test = std::pair<std::string, std::function<void()>>;

class TestDirectory {
 public:
  TestDirectory() {
    static std::atomic<unsigned long long> next_id{0};
    const auto timestamp =
        std::chrono::steady_clock::now().time_since_epoch().count();
    path_ = std::filesystem::temp_directory_path() /
            ("gnni-dataset-io-test-" + std::to_string(timestamp) + "-" +
             std::to_string(next_id.fetch_add(1)));
    std::filesystem::create_directory(path_);
  }

  ~TestDirectory() {
    std::error_code ignored;
    std::filesystem::remove_all(path_, ignored);
  }

  std::filesystem::path file(const std::string& name) const {
    return path_ / name;
  }

 private:
  std::filesystem::path path_;
};

void append_u32(Bytes& bytes, std::uint32_t value) {
  for (unsigned shift = 0; shift != 32; shift += 8) {
    bytes.push_back(static_cast<std::uint8_t>(value >> shift));
  }
}

void append_u64(Bytes& bytes, std::uint64_t value) {
  for (unsigned shift = 0; shift != 64; shift += 8) {
    bytes.push_back(static_cast<std::uint8_t>(value >> shift));
  }
}

void patch_u32(Bytes& bytes, std::size_t offset, std::uint32_t value) {
  for (unsigned byte = 0; byte != 4; ++byte) {
    bytes.at(offset + byte) =
        static_cast<std::uint8_t>(value >> (byte * 8));
  }
}

void patch_u64(Bytes& bytes, std::size_t offset, std::uint64_t value) {
  for (unsigned byte = 0; byte != 8; ++byte) {
    bytes.at(offset + byte) =
        static_cast<std::uint8_t>(value >> (byte * 8));
  }
}

void write_bytes(const std::filesystem::path& path, const Bytes& bytes) {
  std::ofstream output(path, std::ios::binary | std::ios::trunc);
  if (!output) {
    throw std::runtime_error("cannot create test file");
  }
  output.write(reinterpret_cast<const char*>(bytes.data()),
               static_cast<std::streamsize>(bytes.size()));
  if (!output) {
    throw std::runtime_error("cannot write test file");
  }
}

Bytes read_bytes(const std::filesystem::path& path) {
  std::ifstream input(path, std::ios::binary | std::ios::ate);
  if (!input) {
    throw std::runtime_error("cannot open test output");
  }
  const auto length = input.tellg();
  if (length < 0) {
    throw std::runtime_error("cannot determine test output length");
  }
  Bytes bytes(static_cast<std::size_t>(length));
  input.seekg(0);
  input.read(reinterpret_cast<char*>(bytes.data()), length);
  if (!input) {
    throw std::runtime_error("cannot read test output");
  }
  return bytes;
}

Dataset directed_dataset() {
  Dataset dataset;
  dataset.graph = gnni::CSRGraph(3, {0, 1, 2, 3}, {1, 2, 0});
  dataset.features =
      gnni::DenseMatrix(3, 2, {1.0F, -2.0F, 3.5F, 4.0F, 5.25F, 6.0F});
  dataset.labels = {0, 1, 0};
  dataset.splits = {gnni::NodeSplit::Train, gnni::NodeSplit::Validation,
                    gnni::NodeSplit::Test};
  dataset.metadata.directed = true;
  dataset.metadata.num_classes = 2;
  return dataset;
}

Dataset undirected_dataset() {
  Dataset dataset;
  dataset.graph = gnni::CSRGraph(3, {0, 1, 3, 4}, {1, 0, 2, 1});
  dataset.features = gnni::DenseMatrix(3, 1, {1.0F, 2.0F, 3.0F});
  return dataset;
}

Dataset self_loop_dataset() {
  Dataset dataset;
  dataset.graph = gnni::CSRGraph(2, {0, 2, 4}, {0, 1, 0, 1});
  dataset.features = gnni::DenseMatrix(2, 1, {7.0F, 8.0F});
  dataset.metadata.self_loops = true;
  return dataset;
}

Dataset zero_edge_dataset() {
  Dataset dataset;
  dataset.graph = gnni::CSRGraph(2, {0, 0, 0}, {});
  dataset.features = gnni::DenseMatrix(2, 2, {0.0F, -0.0F, 2.5F, -3.0F});
  return dataset;
}

void expect_equal(const Dataset& actual, const Dataset& expected) {
  if (actual.graph.num_vertices != expected.graph.num_vertices ||
      actual.graph.row_offsets != expected.graph.row_offsets ||
      actual.graph.column_indices != expected.graph.column_indices ||
      actual.features.rows != expected.features.rows ||
      actual.features.cols != expected.features.cols ||
      actual.features.values != expected.features.values ||
      actual.labels != expected.labels || actual.splits != expected.splits ||
      actual.metadata.directed != expected.metadata.directed ||
      actual.metadata.self_loops != expected.metadata.self_loops ||
      actual.metadata.num_classes != expected.metadata.num_classes) {
    throw std::runtime_error("loaded dataset differs from saved dataset");
  }
}

void round_trip(const Dataset& dataset) {
  TestDirectory directory;
  const auto path = directory.file("round-trip.gnni");
  gnni::save_dataset(path, dataset);
  expect_equal(gnni::load_dataset(path), dataset);
}

Bytes valid_unlabeled_bytes() {
  Bytes bytes{'G', 'N', 'N', 'I', 'D', 'A', 'T', '1'};
  append_u32(bytes, 1);   // version
  append_u32(bytes, 64);  // header bytes
  append_u32(bytes, gnni::kDirectedFlag);
  append_u32(bytes, 0);   // reserved
  append_u64(bytes, 1);   // N
  append_u64(bytes, 0);   // E
  append_u64(bytes, 1);   // F
  append_u64(bytes, 0);   // num_classes
  append_u64(bytes, 20);  // payload: two u64 offsets and one f32
  append_u64(bytes, 0);
  append_u64(bytes, 0);
  append_u32(bytes, UINT32_C(0x3fc00000));  // 1.5F
  return bytes;
}

Bytes valid_labeled_split_bytes() {
  Bytes bytes{'G', 'N', 'N', 'I', 'D', 'A', 'T', '1'};
  append_u32(bytes, 1);
  append_u32(bytes, 64);
  append_u32(bytes, gnni::kDirectedFlag | gnni::kLabelsFlag |
                        gnni::kSplitsFlag);
  append_u32(bytes, 0);
  append_u64(bytes, 1);   // N
  append_u64(bytes, 0);   // E
  append_u64(bytes, 1);   // F
  append_u64(bytes, 2);   // num_classes
  append_u64(bytes, 25);  // offsets + feature + label + split
  append_u64(bytes, 0);
  append_u64(bytes, 0);
  append_u32(bytes, UINT32_C(0x3fc00000));
  append_u32(bytes, 1);  // label
  bytes.push_back(3);    // test split
  return bytes;
}

template <typename Function>
void expect_failure(Function&& function, const std::string& context) {
  try {
    function();
  } catch (const std::exception& error) {
    if (std::string(error.what()).find(context) == std::string::npos) {
      throw std::runtime_error("error lacks context '" + context + "': " +
                               error.what());
    }
    return;
  }
  throw std::runtime_error("expected failure containing '" + context + "'");
}

void expect_load_failure(Bytes bytes, const std::string& context) {
  TestDirectory directory;
  const auto path = directory.file("corrupt.gnni");
  write_bytes(path, bytes);
  expect_failure([&] { static_cast<void>(gnni::load_dataset(path)); }, context);
}

std::vector<Test> make_tests() {
  std::vector<Test> tests;
  tests.push_back({"round-trips directed dataset with labels and splits",
                   [] { round_trip(directed_dataset()); }});
  tests.push_back({"round-trips labels without splits", [] {
                     Dataset dataset = directed_dataset();
                     dataset.splits.clear();
                     round_trip(dataset);
                   }});
  tests.push_back({"round-trips directed dataset without optional sections", [] {
                     Dataset dataset = directed_dataset();
                     dataset.labels.clear();
                     dataset.splits.clear();
                     dataset.metadata.num_classes = 0;
                     round_trip(dataset);
                   }});
  tests.push_back({"round-trips undirected dataset",
                   [] { round_trip(undirected_dataset()); }});
  tests.push_back({"round-trips exact self-loop flag",
                   [] { round_trip(self_loop_dataset()); }});
  tests.push_back({"round-trips zero-edge graph",
                   [] { round_trip(zero_edge_dataset()); }});

  tests.push_back({"writes literal version 1 golden header", [] {
                     Dataset dataset;
                     dataset.graph = gnni::CSRGraph(1, {0, 0}, {});
                     dataset.features = gnni::DenseMatrix(1, 2, {1.0F, -2.0F});
                     dataset.metadata.directed = true;
                     TestDirectory directory;
                     const auto path = directory.file("golden.gnni");
                     gnni::save_dataset(path, dataset);
                     const Bytes actual = read_bytes(path);
                     const std::array<std::uint8_t, 64> expected = {
                         0x47, 0x4e, 0x4e, 0x49, 0x44, 0x41, 0x54, 0x31,
                         0x01, 0x00, 0x00, 0x00, 0x40, 0x00, 0x00, 0x00,
                         0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
                         0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
                         0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
                         0x02, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
                         0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
                         0x18, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
                     };
                     if (actual.size() < expected.size() ||
                         !std::equal(expected.begin(), expected.end(),
                                     actual.begin())) {
                       throw std::runtime_error(
                           "first 64 bytes differ from golden header");
                     }
                   }});
  tests.push_back({"writes uint32 columns and IEEE-754 float payload", [] {
                     Dataset dataset;
                     dataset.graph = gnni::CSRGraph(1, {0, 1}, {0});
                     dataset.features = gnni::DenseMatrix(1, 1, {-2.5F});
                     dataset.metadata.directed = true;
                     dataset.metadata.self_loops = true;
                     TestDirectory directory;
                     const auto path = directory.file("payload.gnni");
                     gnni::save_dataset(path, dataset);
                     const Bytes actual = read_bytes(path);
                     const std::array<std::uint8_t, 24> expected = {
                         0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
                         0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
                         0x00, 0x00, 0x00, 0x00,
                         0x00, 0x00, 0x20, 0xc0,
                     };
                     if (actual.size() != 64 + expected.size() ||
                         !std::equal(expected.begin(), expected.end(),
                                     actual.begin() + 64)) {
                       throw std::runtime_error(
                           "payload differs from literal v1 byte layout");
                     }
                   }});

  tests.push_back({"rejects bad magic", [] {
                     Bytes bytes = valid_unlabeled_bytes();
                     bytes[0] = 'X';
                     expect_load_failure(std::move(bytes), "magic");
                   }});
  tests.push_back({"rejects unsupported version", [] {
                     Bytes bytes = valid_unlabeled_bytes();
                     patch_u32(bytes, 8, 2);
                     expect_load_failure(std::move(bytes), "version");
                   }});
  tests.push_back({"rejects wrong header length", [] {
                     Bytes bytes = valid_unlabeled_bytes();
                     patch_u32(bytes, 12, 63);
                     expect_load_failure(std::move(bytes), "header");
                   }});
  tests.push_back({"rejects unknown flag", [] {
                     Bytes bytes = valid_unlabeled_bytes();
                     patch_u32(bytes, 16, gnni::kDirectedFlag | (1U << 4U));
                     expect_load_failure(std::move(bytes), "flag");
                   }});
  tests.push_back({"rejects nonzero reserved word", [] {
                     Bytes bytes = valid_unlabeled_bytes();
                     patch_u32(bytes, 20, 1);
                     expect_load_failure(std::move(bytes), "reserved");
                   }});
  tests.push_back({"rejects split flag without label flag", [] {
                     Bytes bytes = valid_unlabeled_bytes();
                     patch_u32(bytes, 16,
                               gnni::kDirectedFlag | gnni::kSplitsFlag);
                     expect_load_failure(std::move(bytes), "labels");
                   }});
  tests.push_back({"rejects vertex count above INT_MAX", [] {
                     Bytes bytes = valid_unlabeled_bytes();
                     patch_u64(bytes, 24,
                               static_cast<std::uint64_t>(
                                   std::numeric_limits<int>::max()) +
                                   1U);
                     expect_load_failure(std::move(bytes), "num_vertices");
                   }});
  tests.push_back({"rejects CSR entry count above INT_MAX", [] {
                     Bytes bytes = valid_unlabeled_bytes();
                     patch_u64(bytes, 32,
                               static_cast<std::uint64_t>(
                                   std::numeric_limits<int>::max()) +
                                   1U);
                     expect_load_failure(std::move(bytes), "num_csr_entries");
                   }});
  tests.push_back({"rejects payload size overflow before allocation", [] {
                     Bytes bytes = valid_unlabeled_bytes();
                     patch_u64(bytes, 40,
                               std::numeric_limits<std::uint64_t>::max());
                     expect_load_failure(std::move(bytes), "overflow");
                   }});
  tests.push_back({"rejects truncated payload section", [] {
                     Bytes bytes = valid_unlabeled_bytes();
                     bytes.pop_back();
                     expect_load_failure(std::move(bytes), "length");
                   }});
  tests.push_back({"rejects trailing byte", [] {
                     Bytes bytes = valid_unlabeled_bytes();
                     bytes.push_back(0);
                     expect_load_failure(std::move(bytes), "length");
                   }});
  tests.push_back({"rejects malformed disk CSR", [] {
                     Bytes bytes = valid_unlabeled_bytes();
                     patch_u64(bytes, 72, 1);
                     expect_load_failure(std::move(bytes), "final offset");
                   }});
  tests.push_back({"rejects invalid disk label", [] {
                     Bytes bytes = valid_labeled_split_bytes();
                     patch_u32(bytes, 84, 2);
                     expect_load_failure(std::move(bytes), "label 0");
                   }});
  tests.push_back({"rejects invalid disk split", [] {
                     Bytes bytes = valid_labeled_split_bytes();
                     bytes.at(88) = 4;
                     expect_load_failure(std::move(bytes), "split 0");
                   }});

  tests.push_back({"validates before creating output", [] {
                     Dataset invalid = directed_dataset();
                     invalid.labels[0] = -1;
                     TestDirectory directory;
                     const auto path = directory.file("invalid.gnni");
                     expect_failure([&] { gnni::save_dataset(path, invalid); },
                                    "label 0");
                     if (std::filesystem::exists(path)) {
                       throw std::runtime_error("invalid dataset created output");
                     }
                   }});
  tests.push_back({"does not overwrite existing output by default", [] {
                     TestDirectory directory;
                     const auto path = directory.file("existing.gnni");
                     const Bytes sentinel{0xde, 0xad, 0xbe, 0xef};
                     write_bytes(path, sentinel);
                     expect_failure(
                         [&] { gnni::save_dataset(path, zero_edge_dataset()); },
                         "exists");
                     if (read_bytes(path) != sentinel) {
                       throw std::runtime_error("existing output was modified");
                     }
                   }});
#ifdef __linux__
  tests.push_back({"does not overwrite a dangling symlink by default", [] {
                     TestDirectory directory;
                     const auto path = directory.file("dangling.gnni");
                     const std::filesystem::path link_value("missing-target");
                     std::filesystem::create_symlink(link_value, path);
                     if (std::filesystem::exists(path) ||
                         !std::filesystem::is_symlink(
                             std::filesystem::symlink_status(path))) {
                       throw std::runtime_error(
                           "dangling-symlink fixture was not created");
                     }
                     expect_failure(
                         [&] { gnni::save_dataset(path, zero_edge_dataset()); },
                         "exists");
                     if (!std::filesystem::is_symlink(
                             std::filesystem::symlink_status(path)) ||
                         std::filesystem::read_symlink(path) != link_value) {
                       throw std::runtime_error(
                           "dangling symlink was replaced or modified");
                     }
                   }});
  tests.push_back({"atomically rejects a concurrently created target", [] {
                     TestDirectory directory;
                     const auto path = directory.file("raced.gnni");
                     const Bytes sentinel{0xca, 0xfe, 0xba, 0xbe};

                     install_hook_target = path.string();
                     install_hook_reached.store(false,
                                                std::memory_order_relaxed);
                     competitor_finished.store(false,
                                               std::memory_order_relaxed);
                     competitor_cancelled.store(false,
                                                std::memory_order_relaxed);
                     std::exception_ptr competitor_error;
                     std::thread competitor([&] {
                       while (!install_hook_reached.load(
                                  std::memory_order_acquire) &&
                              !competitor_cancelled.load(
                                  std::memory_order_acquire)) {
                         std::this_thread::yield();
                       }
                       if (install_hook_reached.load(
                               std::memory_order_acquire)) {
                         try {
                           write_bytes(path, sentinel);
                         } catch (...) {
                           competitor_error = std::current_exception();
                         }
                       }
                       competitor_finished.store(true,
                                                 std::memory_order_release);
                     });
                     install_hook_enabled.store(true,
                                                std::memory_order_release);

                     std::exception_ptr save_error;
                     try {
                       gnni::save_dataset(path, zero_edge_dataset());
                     } catch (...) {
                       save_error = std::current_exception();
                     }
                     competitor_cancelled.store(true,
                                                std::memory_order_release);
                     competitor.join();
                     install_hook_enabled.store(false,
                                                std::memory_order_release);

                     if (competitor_error != nullptr) {
                       std::rethrow_exception(competitor_error);
                     }
                     if (!install_hook_reached.load(
                             std::memory_order_acquire)) {
                       throw std::runtime_error(
                           "save never reached the atomic install boundary");
                     }
                     expect_failure(
                         [&] {
                           if (save_error != nullptr) {
                             std::rethrow_exception(save_error);
                           }
                         },
                         "exists");
                     if (read_bytes(path) != sentinel) {
                       throw std::runtime_error(
                           "concurrently created target was overwritten");
                     }
                   }});
#endif
  tests.push_back({"overwrites existing output when requested", [] {
                     TestDirectory directory;
                     const auto path = directory.file("existing.gnni");
                     write_bytes(path, Bytes{0xde, 0xad, 0xbe, 0xef});
                     const Dataset dataset = zero_edge_dataset();
                     gnni::save_dataset(path, dataset, true);
                     expect_equal(gnni::load_dataset(path), dataset);
                   }});
  return tests;
}

}  // namespace

int main() {
  const std::vector<Test> tests = make_tests();
  int failures = 0;
  for (const auto& [name, test] : tests) {
    try {
      test();
      std::cout << "[PASS] " << name << '\n';
    } catch (const std::exception& error) {
      ++failures;
      std::cerr << "[FAIL] " << name << ": " << error.what() << '\n';
    }
  }
  if (failures != 0) {
    std::cerr << failures << " test(s) failed\n";
    return 1;
  }
  std::cout << tests.size() << " test(s) passed\n";
  return 0;
}
