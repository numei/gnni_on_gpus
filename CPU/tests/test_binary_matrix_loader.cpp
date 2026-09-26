#include "gnni/io.hpp"

#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

class TemporaryFile {
public:
    /** Create a unique temporary path for one test case. */
    explicit TemporaryFile(const std::string& case_name) {
        const auto unique = std::chrono::high_resolution_clock::now()
                                .time_since_epoch()
                                .count();
        path_ = std::filesystem::temp_directory_path() /
                ("gnni_" + case_name + "_" + std::to_string(unique) + ".gnnimatrix");
    }

    /** Remove the test file without throwing during stack unwinding. */
    ~TemporaryFile() {
        std::error_code error;
        std::filesystem::remove(path_, error);
    }

    /** Return the path used by the writer and loader under test. */
    const std::filesystem::path& path() const noexcept {
        return path_;
    }

private:
    std::filesystem::path path_;
};

/** Encode one little-endian uint32 value into a test fixture. */
void write_u32(std::ofstream& output, std::uint32_t value) {
    for (unsigned int shift = 0; shift < 32; shift += 8) {
        output.put(static_cast<char>((value >> shift) & 0xFFU));
    }
}

/** Encode one little-endian uint64 value into a test fixture. */
void write_u64(std::ofstream& output, std::uint64_t value) {
    for (unsigned int shift = 0; shift < 64; shift += 8) {
        output.put(static_cast<char>((value >> shift) & 0xFFU));
    }
}

/** Encode one IEEE-754 binary32 value into a test fixture. */
void write_f32(std::ofstream& output, float value) {
    std::uint32_t bits = 0;
    static_assert(sizeof(bits) == sizeof(value), "float32 is required");
    std::memcpy(&bits, &value, sizeof(bits));
    write_u32(output, bits);
}

/** Write one configurable version-1-like matrix file for positive/negative tests. */
void write_matrix_file(
    const std::filesystem::path& path,
    std::uint32_t version,
    std::uint64_t rows,
    std::uint64_t cols,
    const std::vector<float>& values,
    bool add_trailing_byte = false) {
    std::ofstream output(path, std::ios::binary);
    if (!output) {
        throw std::runtime_error("failed to create test file");
    }

    constexpr std::array<char, 8> magic{'G', 'N', 'N', 'I', 'M', 'A', 'T', 0};
    output.write(magic.data(), static_cast<std::streamsize>(magic.size()));
    write_u32(output, version);
    write_u32(output, 48);
    write_u32(output, 1);
    write_u32(output, 1);
    write_u64(output, rows);
    write_u64(output, cols);
    write_u64(output, 0);
    for (float value : values) {
        write_f32(output, value);
    }
    if (add_trailing_byte) {
        output.put('\x7f');
    }
}

/** Fail the current test with a readable message when a condition is false. */
void require(bool condition, const char* message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

/** Require an operation to reject malformed input with std::runtime_error. */
void expect_runtime_error(const std::function<void()>& operation, const char* case_name) {
    try {
        operation();
    } catch (const std::runtime_error&) {
        return;
    }
    throw std::runtime_error(std::string(case_name) + " did not throw std::runtime_error");
}

/** Verify valid payload decoding and both binary-selection paths. */
void test_valid_binary_and_auto_detection() {
    TemporaryFile file("valid_matrix");
    const std::vector<float> expected{1.0F, -2.0F, 3.5F, 4.0F, 5.0F, 6.0F};
    write_matrix_file(file.path(), 1, 2, 3, expected);

    const gnni_cpu::DenseMatrix matrix = gnni_cpu::load_dense_matrix(file.path());
    require(matrix.rows == 2, "valid matrix rows mismatch");
    require(matrix.cols == 3, "valid matrix cols mismatch");
    require(matrix.values == expected, "valid matrix values mismatch");

    const gnni_cpu::DenseMatrix explicit_binary =
        gnni_cpu::load_dense_matrix(file.path(), gnni_cpu::InputFileFormat::Binary);
    require(explicit_binary.values == expected, "explicit binary load mismatch");
}

/** Verify that a shorter-than-declared payload is rejected. */
void test_truncated_payload() {
    TemporaryFile file("truncated_matrix");
    write_matrix_file(file.path(), 1, 2, 3, {1.0F, 2.0F, 3.0F, 4.0F, 5.0F});
    expect_runtime_error(
        [&] { static_cast<void>(gnni_cpu::load_dense_matrix(file.path())); },
        "truncated payload");
}

/** Verify that undefined trailing bytes are rejected. */
void test_trailing_payload() {
    TemporaryFile file("trailing_matrix");
    write_matrix_file(
        file.path(), 1, 1, 1, {1.0F}, true);
    expect_runtime_error(
        [&] { static_cast<void>(gnni_cpu::load_dense_matrix(file.path())); },
        "trailing payload");
}

/** Verify that a future/unknown file version is rejected. */
void test_unsupported_version() {
    TemporaryFile file("version_matrix");
    write_matrix_file(file.path(), 2, 1, 1, {1.0F});
    expect_runtime_error(
        [&] { static_cast<void>(gnni_cpu::load_dense_matrix(file.path())); },
        "unsupported version");
}

/** Verify that NaN payload data is rejected. */
void test_non_finite_payload() {
    TemporaryFile file("nan_matrix");
    write_matrix_file(
        file.path(), 1, 1, 1, {std::numeric_limits<float>::quiet_NaN()});
    expect_runtime_error(
        [&] { static_cast<void>(gnni_cpu::load_dense_matrix(file.path())); },
        "non-finite payload");
}

/** Verify that hostile dimensions are rejected before allocation. */
void test_dimension_overflow() {
    TemporaryFile file("overflow_matrix");
    write_matrix_file(
        file.path(),
        1,
        std::numeric_limits<std::uint64_t>::max(),
        2,
        {});
    expect_runtime_error(
        [&] { static_cast<void>(gnni_cpu::load_dense_matrix(file.path())); },
        "dimension overflow");
}

}  // namespace

/** Execute all loader test cases without requiring a third-party test framework. */
int main() {
    try {
        test_valid_binary_and_auto_detection();
        test_truncated_payload();
        test_trailing_payload();
        test_unsupported_version();
        test_non_finite_payload();
        test_dimension_overflow();
        std::cout << "binary matrix loader tests passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "binary matrix loader test failed: " << error.what() << '\n';
        return 1;
    }
}
