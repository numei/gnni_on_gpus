#pragma once

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace gnni_cpu::io_detail {

/** Throw a consistently formatted runtime error for one input file. */
[[noreturn]] inline void throw_file_error(
    const std::filesystem::path& path,
    const char* file_kind,
    const std::string& message) {
    throw std::runtime_error(
        std::string(file_kind) + " file '" + path.string() + "': " + message);
} // 错误处理，throw error info.

/** Add two byte/count values or throw before uint64 overflow. */
inline std::uint64_t checked_add(
    std::uint64_t left,
    std::uint64_t right,
    const std::filesystem::path& path,
    const char* file_kind,
    const char* field) {
    if (left > std::numeric_limits<std::uint64_t>::max() - right) {
        throw_file_error(path, file_kind, std::string(field) + " size overflows uint64");
    }
    return left + right;
}

/** Multiply two byte/count values or throw before uint64 overflow. */
inline std::uint64_t checked_multiply(
    std::uint64_t left,
    std::uint64_t right,
    const std::filesystem::path& path,
    const char* file_kind,
    const char* field) {
    if (right != 0 && left > std::numeric_limits<std::uint64_t>::max() / right) {
        throw_file_error(path, file_kind, std::string(field) + " size overflows uint64");
    }
    return left * right;
}

/** Convert a validated uint64 count to the host allocation size type. */
inline std::size_t checked_size_t(
    std::uint64_t value,
    const std::filesystem::path& path,
    const char* file_kind,
    const char* field) {
    if (value > static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max())) {
        throw_file_error(path, file_kind, std::string(field) + " does not fit size_t");
    }
    return static_cast<std::size_t>(value);
}

class BinaryReader {
public:
    /** Open a binary file and cache its exact byte length. */
    BinaryReader(const std::filesystem::path& path, const char* file_kind)
        : path_(path), file_kind_(file_kind), stream_(path, std::ios::binary) {
        if (!stream_) {
            throw_file_error(path_, file_kind_, "cannot open file");
        }

        // 获得文件大小
        stream_.seekg(0, std::ios::end);
        const std::streampos end = stream_.tellg();
        if (end < 0) {
            throw_file_error(path_, file_kind_, "cannot determine file size");
        }
        file_size_ = static_cast<std::uint64_t>(end);

        // 再把指针移回来
        stream_.seekg(0, std::ios::beg);
        if (!stream_) {
            throw_file_error(path_, file_kind_, "cannot seek to beginning of file");
        }
    }

    /** Return the path used in diagnostics. */
    const std::filesystem::path& path() const noexcept {
        return path_;
    }

    /** Return the total file size in bytes. */
    std::uint64_t file_size() const noexcept {
        return file_size_;
    }

    /** Return the logical byte offset of the next read. */
    std::uint64_t position() const noexcept {
        return position_;
    }

    /** Read an exact byte count, rejecting truncation and stream failures. */
    void read_bytes(void* destination, std::uint64_t byte_count, const char* field) {
        if (byte_count > file_size_ - position_) {
            throw_file_error(
                path_,
                file_kind_,
                std::string("truncated while reading ") + field + " at byte " +
                    std::to_string(position_));
        }

        auto* output = static_cast<char*>(destination);
        std::uint64_t remaining = byte_count;
        constexpr std::uint64_t max_chunk = 1U << 20;

        while (remaining != 0) {
            const std::uint64_t chunk = std::min(remaining, max_chunk);
            stream_.read(output, static_cast<std::streamsize>(chunk));
            if (!stream_) {
                throw_file_error(
                    path_, file_kind_, std::string("read failure while reading ") + field);
            }
            output += static_cast<std::size_t>(chunk);
            position_ += chunk;
            remaining -= chunk;
        }
    }

    // 读取文件格式表示
    /** Read the common eight-byte file magic. */
    std::array<std::uint8_t, 8> read_magic() {
        std::array<std::uint8_t, 8> value{};
        read_bytes(value.data(), value.size(), "magic");
        return value;
    }

    /** Decode one little-endian uint32 value. */
    std::uint32_t read_u32(const char* field) {
        std::array<std::uint8_t, 4> bytes{};
        read_bytes(bytes.data(), bytes.size(), field);
        return static_cast<std::uint32_t>(bytes[0]) |
               (static_cast<std::uint32_t>(bytes[1]) << 8U) |
               (static_cast<std::uint32_t>(bytes[2]) << 16U) |
               (static_cast<std::uint32_t>(bytes[3]) << 24U);
    }

    /** Decode one little-endian uint64 value. */
    std::uint64_t read_u64(const char* field) {
        std::array<std::uint8_t, 8> bytes{};
        read_bytes(bytes.data(), bytes.size(), field);

        std::uint64_t value = 0;
        for (std::size_t index = 0; index < bytes.size(); ++index) {
            value |= static_cast<std::uint64_t>(bytes[index]) << (8U * index);
        }
        return value;
    }

    /** Decode count little-endian IEEE-754 binary32 values in chunks. */
    std::vector<float> read_f32_vector(std::uint64_t count, const char* field) {
        static_assert(
            std::numeric_limits<float>::is_iec559 && sizeof(float) == 4,
            "IEEE-754 binary32 float is required");
        const std::size_t output_size = checked_size_t(
            count, path_, file_kind_, field);
        std::vector<float> result(output_size);

        constexpr std::size_t values_per_chunk = 1U << 16;
        std::vector<std::uint8_t> bytes(values_per_chunk * sizeof(float));

        std::uint64_t processed = 0;
        while (processed < count) {
            const std::uint64_t remaining = count - processed;
            const std::size_t chunk_values = static_cast<std::size_t>(
                std::min<std::uint64_t>(remaining, values_per_chunk));
            const std::size_t chunk_bytes = chunk_values * sizeof(float);
            read_bytes(bytes.data(), chunk_bytes, field);
            const std::size_t output_offset = checked_size_t(
                processed, path_, file_kind_, field);

            for (std::size_t index = 0; index < chunk_values; ++index) {
                const std::size_t offset = index * sizeof(float);
                const std::uint32_t bits =
                    static_cast<std::uint32_t>(bytes[offset]) |
                    (static_cast<std::uint32_t>(bytes[offset + 1]) << 8U) |
                    (static_cast<std::uint32_t>(bytes[offset + 2]) << 16U) |
                    (static_cast<std::uint32_t>(bytes[offset + 3]) << 24U);
                float value = 0.0F;
                static_assert(sizeof(value) == sizeof(bits), "float32 is required");
                std::memcpy(&value, &bits, sizeof(value));
                result[output_offset + index] = value;
            }
            processed += chunk_values;
        }

        return result;
    }

    // 检查文件实际尺寸是否符合格式
    /** Require the physical file length to match a checked expected length. */
    void require_exact_size(std::uint64_t expected_size) const {
        if (file_size_ != expected_size) {
            throw_file_error(
                path_,
                file_kind_,
                "expected " + std::to_string(expected_size) + " bytes, found " +
                    std::to_string(file_size_));
        }
    }
    
    // 检查解析完成后有没有多余数据
    /** Require all bytes to have been consumed after payload decoding. */
    void require_end() const {
        if (position_ != file_size_) {
            throw_file_error(
                path_,
                file_kind_,
                "unexpected trailing data beginning at byte " + std::to_string(position_));
        }
    }

private:
    std::filesystem::path path_;
    const char* file_kind_;
    std::ifstream stream_;
    std::uint64_t file_size_ = 0;
    std::uint64_t position_ = 0;
};

}  // namespace gnni_cpu::io_detail
