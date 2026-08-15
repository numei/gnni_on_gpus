#pragma once

#include "gnni/dataset.hpp"

#include <cstdint>
#include <filesystem>

namespace gnni {

inline constexpr std::uint32_t kDatasetFormatVersion = 1;
inline constexpr std::uint32_t kDatasetHeaderBytes = 64;
inline constexpr std::uint32_t kDirectedFlag = 1U << 0U;
inline constexpr std::uint32_t kSelfLoopsFlag = 1U << 1U;
inline constexpr std::uint32_t kLabelsFlag = 1U << 2U;
inline constexpr std::uint32_t kSplitsFlag = 1U << 3U;

Dataset load_dataset(const std::filesystem::path& path);
void save_dataset(const std::filesystem::path& path,
                  const Dataset& dataset,
                  bool overwrite = false);

}  // namespace gnni
