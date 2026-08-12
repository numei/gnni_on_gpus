#include "gnni/io.hpp"

#include <stdexcept>

namespace gnni {

/**
 * TODO(Role A): implement the graph loader described in docs/file_formats.md.
 *
 * Suggested implementation order:
 * 1. Detect or validate the requested format.
 * 2. Decode header fields using BinaryReader for binary input.
 * 3. Check all size arithmetic before allocating vectors.
 * 4. Read row_ptr, col_idx, and optional edge features.
 * 5. Enforce file-local CSR invariants before returning.
 */
CSRGraph load_graph(const std::filesystem::path& path, InputFileFormat format) {
    static_cast<void>(path);
    static_cast<void>(format);
    throw std::logic_error("load_graph is not implemented");
}

}  // namespace gnni

