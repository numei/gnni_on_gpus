#include "gnni/io.hpp"

#include "binary_reader.hpp"

#include <array>
#include <cmath>
#include <cstdint>
#include <string>

namespace gnni_cpu {
namespace {

constexpr std::array<std::uint8_t, 8> matrix_magic{
    'G', 'N', 'N', 'I', 'M', 'A', 'T', 0};
constexpr std::uint32_t format_version = 1;
constexpr std::uint32_t matrix_header_bytes = 48;
constexpr std::uint32_t scalar_float32 = 1;
constexpr std::uint32_t layout_row_major = 1;

/** Parse and fully validate a version-1 binary dense-matrix file. */
DenseMatrix load_binary_dense_matrix(const std::filesystem::path& path) {
    constexpr const char* file_kind = "dense-matrix";
    io_detail::BinaryReader reader(path, file_kind);

    const auto actual_magic = reader.read_magic();
    if (actual_magic != matrix_magic) {
        io_detail::throw_file_error(path, file_kind, "invalid magic");
    }

    const std::uint32_t version = reader.read_u32("version");
    if (version != format_version) {
        io_detail::throw_file_error(
            path,
            file_kind,
            "unsupported version " + std::to_string(version) + ", expected 1");
    }

    const std::uint32_t header_bytes = reader.read_u32("header_bytes");
    if (header_bytes != matrix_header_bytes) {
        io_detail::throw_file_error(
            path,
            file_kind,
            "header_bytes is " + std::to_string(header_bytes) + ", expected 48");
    }

    const std::uint32_t scalar_type = reader.read_u32("scalar_type");
    if (scalar_type != scalar_float32) {
        io_detail::throw_file_error(
            path, file_kind, "unsupported scalar_type " + std::to_string(scalar_type));
    }

    const std::uint32_t layout = reader.read_u32("layout");
    if (layout != layout_row_major) {
        io_detail::throw_file_error(
            path, file_kind, "unsupported layout " + std::to_string(layout));
    }

    const std::uint64_t rows = reader.read_u64("rows");
    const std::uint64_t cols = reader.read_u64("cols");
    const std::uint64_t reserved = reader.read_u64("reserved");

    if (reserved != 0) {
        io_detail::throw_file_error(path, file_kind, "reserved field must be zero");
    }
    if (rows == 0) {
        io_detail::throw_file_error(path, file_kind, "rows must be greater than zero");
    }
    if (cols == 0) {
        io_detail::throw_file_error(path, file_kind, "cols must be greater than zero");
    }

    const std::uint64_t value_count = io_detail::checked_multiply(
        rows, cols, path, file_kind, "matrix element count");
    const std::uint64_t payload_bytes = io_detail::checked_multiply(
        value_count, sizeof(float), path, file_kind, "matrix payload");
    const std::uint64_t expected_bytes = io_detail::checked_add(
        matrix_header_bytes, payload_bytes, path, file_kind, "matrix file");
    reader.require_exact_size(expected_bytes);

    DenseMatrix matrix;
    matrix.rows = rows;
    matrix.cols = cols;
    matrix.values = reader.read_f32_vector(value_count, "matrix payload");
    reader.require_end();

    for (std::size_t index = 0; index < matrix.values.size(); ++index) {
        if (!std::isfinite(matrix.values[index])) {
            io_detail::throw_file_error(
                path,
                file_kind,
                "non-finite value at element " + std::to_string(index));
        }
    }

    return matrix;
}

/** Probe whether a file starts with the binary dense-matrix magic. */
bool has_binary_matrix_magic(const std::filesystem::path& path) {
    io_detail::BinaryReader reader(path, "dense-matrix");
    return reader.read_magic() == matrix_magic;
}

}  // namespace

/** Route dense-matrix loading according to the explicit or detected format. */
DenseMatrix load_dense_matrix(
    const std::filesystem::path& path,
    InputFileFormat format) {
    switch (format) {
        case InputFileFormat::Binary:
            return load_binary_dense_matrix(path);
        case InputFileFormat::Text:
            io_detail::throw_file_error(
                path, "dense-matrix", "text format loader is not implemented yet");
        case InputFileFormat::Auto:
            if (has_binary_matrix_magic(path)) {
                return load_binary_dense_matrix(path);
            }
            io_detail::throw_file_error(
                path, "dense-matrix", "unrecognized format or text loader not implemented yet");
    }

    io_detail::throw_file_error(path, "dense-matrix", "invalid InputFileFormat value");
}

}  // namespace gnni_cpu
