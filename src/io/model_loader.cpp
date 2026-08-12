#include "gnni/io.hpp"

#include <stdexcept>

namespace gnni {

/**
 * TODO(Role A): implement the model loader described in docs/file_formats.md.
 *
 * Parse all layer descriptors before allocating weight payloads. Validate
 * descriptor sizes, activation codes, adjacent dimensions, checked weight
 * counts, finite values, and exact end-of-file.
 */
Model load_model(const std::filesystem::path& path, InputFileFormat format) {
    static_cast<void>(path);
    static_cast<void>(format);
    throw std::logic_error("load_model is not implemented");
}

}  // namespace gnni

