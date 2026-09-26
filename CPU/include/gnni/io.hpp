#pragma once

#include "gnni/types.hpp"

#include <filesystem>

namespace gnni_cpu {

enum class InputFileFormat {
    /// Detect the file type from its magic bytes or leading text token.
    Auto,
    /// Parse the versioned little-endian binary representation.
    Binary,
    /// Parse the strict human-readable fixture representation.
    Text,
};

/**
 * Load a CSR graph and its optional edge-feature matrix.
 *
 * The loader performs file-local validation.
 *
 * @throws std::runtime_error if the file cannot be read or is malformed.
 */
CSRGraph load_graph(
    const std::filesystem::path& path,
    InputFileFormat format = InputFileFormat::Auto);

/**
 * Load one row-major float32 dense matrix, normally the node-feature matrix.
 *
 * @throws std::runtime_error if the file cannot be read or is malformed.
 */
DenseMatrix load_dense_matrix(
    const std::filesystem::path& path,
    InputFileFormat format = InputFileFormat::Auto);

/**
 * Load layer metadata and all preloaded row-major weight matrices.
 *
 * @throws std::runtime_error if the file cannot be read or is malformed.
 */
Model load_model(
    const std::filesystem::path& path,
    InputFileFormat format = InputFileFormat::Auto);

}  // namespace gnni_cpu
