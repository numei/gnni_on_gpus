#include "gnni/io.hpp"

#include <array>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

class TemporaryFile {
public:
    explicit TemporaryFile(const std::string& case_name) {
        const auto unique = std::chrono::high_resolution_clock::now()
                                .time_since_epoch()
                                .count();
        path_ = std::filesystem::temp_directory_path() /
                ("gnni_" + case_name + "_" + std::to_string(unique) +
                 ".gnnigraph");
    }

    ~TemporaryFile() {
        std::error_code error;
        std::filesystem::remove(path_, error);
    }

    const std::filesystem::path& path() const noexcept {
        return path_;
    }

private:
    std::filesystem::path path_;
};

/** Write one unsigned 32-bit integer in the frozen little-endian format. */
void write_u32(std::ofstream& output, std::uint32_t value) {
    for (unsigned int shift = 0; shift < 32; shift += 8) {
        output.put(static_cast<char>((value >> shift) & 0xFFU));
    }
}

/** Write one unsigned 64-bit integer in the frozen little-endian format. */
void write_u64(std::ofstream& output, std::uint64_t value) {
    for (unsigned int shift = 0; shift < 64; shift += 8) {
        output.put(static_cast<char>((value >> shift) & 0xFFU));
    }
}

/**
 * Write a configurable graph file. The caller can intentionally provide an
 * invalid CSR payload, which lets each negative test isolate one invariant.
 */
void write_graph_file(
    const std::filesystem::path& path,
    std::uint32_t version,
    std::uint32_t flags,
    std::uint64_t num_nodes,
    std::uint64_t num_edges,
    std::uint64_t edge_feature_dim,
    const std::vector<std::uint64_t>& row_ptr,
    const std::vector<std::uint32_t>& col_idx,
    bool valid_magic = true,
    bool add_trailing_byte = false
) {
    std::ofstream output(path, std::ios::binary);
    if (!output) {
        throw std::runtime_error("failed to create graph test file");
    }

    constexpr std::array<char, 8> graph_magic{
        'G', 'N', 'N', 'I', 'G', 'R', 'F', 0
    };
    std::array<char, 8> magic = graph_magic;
    if (!valid_magic) {
        magic[0] = 'X';
    }

    output.write(magic.data(), static_cast<std::streamsize>(magic.size()));
    write_u32(output, version);
    write_u32(output, 48);
    write_u32(output, flags);
    write_u32(output, 0);
    write_u64(output, num_nodes);
    write_u64(output, num_edges);
    write_u64(output, edge_feature_dim);

    for (const std::uint64_t value : row_ptr) {
        write_u64(output, value);
    }
    for (const std::uint32_t value : col_idx) {
        write_u32(output, value);
    }
    if (add_trailing_byte) {
        output.put('\x7f');
    }
}

void require(bool condition, const char* message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

/** Require malformed input to fail with the loader's documented exception type. */
void expect_runtime_error(const std::function<void()>& operation, const char* case_name) {
    try {
        operation();
    } catch (const std::runtime_error&) {
        return;
    }
    throw std::runtime_error(std::string(case_name) +
                             " did not throw std::runtime_error");
}

/** Verify decoded fields and both Auto/Binary dispatch paths. */
void test_valid_graph_and_auto_detection() {
    TemporaryFile file("valid_graph");
    const std::vector<std::uint64_t> row_ptr{0, 0, 2, 3};
    const std::vector<std::uint32_t> col_idx{0, 2, 1};
    write_graph_file(file.path(), 1, 1, 3, 3, 0, row_ptr, col_idx);

    const gnni::CSRGraph graph = gnni::load_graph(file.path());
    require(graph.num_nodes == 3, "valid graph num_nodes mismatch");
    require(graph.num_edges == 3, "valid graph num_edges mismatch");
    require(graph.directed, "valid graph directed flag mismatch");
    require(graph.row_ptr == row_ptr, "valid graph row_ptr mismatch");
    require(graph.col_idx == col_idx, "valid graph col_idx mismatch");
    require(!graph.edge_features.has_value(), "unexpected edge features");

    const gnni::CSRGraph explicit_binary =
        gnni::load_graph(file.path(), gnni::InputFileFormat::Binary);
    require(explicit_binary.row_ptr == row_ptr, "explicit binary graph mismatch");
}

void test_header_and_size_errors() {
    TemporaryFile bad_magic("bad_graph_magic");
    write_graph_file(bad_magic.path(), 1, 1, 1, 0, 0, {0, 0}, {}, false);
    expect_runtime_error(
        [&] { static_cast<void>(gnni::load_graph(bad_magic.path())); },
        "bad graph magic"
    );

    TemporaryFile bad_version("bad_graph_version");
    write_graph_file(bad_version.path(), 2, 1, 1, 0, 0, {0, 0}, {});
    expect_runtime_error(
        [&] { static_cast<void>(gnni::load_graph(bad_version.path())); },
        "unsupported graph version"
    );

    TemporaryFile truncated("truncated_graph");
    write_graph_file(truncated.path(), 1, 1, 2, 1, 0, {0, 1, 1}, {});
    expect_runtime_error(
        [&] { static_cast<void>(gnni::load_graph(truncated.path())); },
        "truncated graph payload"
    );

    TemporaryFile trailing("trailing_graph");
    write_graph_file(trailing.path(), 1, 1, 1, 0, 0, {0, 0}, {}, true, true);
    expect_runtime_error(
        [&] { static_cast<void>(gnni::load_graph(trailing.path())); },
        "trailing graph payload"
    );
}

void test_csr_structure_errors() {
    TemporaryFile wrong_first_offset("graph_wrong_first_offset");
    write_graph_file(wrong_first_offset.path(), 1, 1, 2, 1, 0, {1, 1, 1}, {0});
    expect_runtime_error(
        [&] { static_cast<void>(gnni::load_graph(wrong_first_offset.path())); },
        "non-zero first row_ptr"
    );

    TemporaryFile decreasing_offsets("graph_decreasing_offsets");
    write_graph_file(decreasing_offsets.path(), 1, 1, 3, 3, 0, {0, 2, 1, 3}, {1, 2, 0});
    expect_runtime_error(
        [&] { static_cast<void>(gnni::load_graph(decreasing_offsets.path())); },
        "decreasing row_ptr"
    );

    TemporaryFile invalid_vertex("graph_invalid_vertex");
    write_graph_file(invalid_vertex.path(), 1, 1, 3, 1, 0, {0, 0, 1, 1}, {3});
    expect_runtime_error(
        [&] { static_cast<void>(gnni::load_graph(invalid_vertex.path())); },
        "out-of-range neighbour"
    );
}

void test_row_neighbour_errors() {
    TemporaryFile self_loop("graph_self_loop");
    write_graph_file(self_loop.path(), 1, 1, 3, 1, 0, {0, 0, 1, 1}, {1});
    expect_runtime_error(
        [&] { static_cast<void>(gnni::load_graph(self_loop.path())); },
        "self-loop"
    );

    TemporaryFile duplicate("graph_duplicate_neighbour");
    write_graph_file(duplicate.path(), 1, 1, 3, 2, 0, {0, 0, 2, 2}, {0, 0});
    expect_runtime_error(
        [&] { static_cast<void>(gnni::load_graph(duplicate.path())); },
        "duplicate neighbour"
    );

    TemporaryFile unsorted("graph_unsorted_neighbours");
    write_graph_file(unsorted.path(), 1, 1, 3, 2, 0, {0, 0, 2, 2}, {2, 0});
    expect_runtime_error(
        [&] { static_cast<void>(gnni::load_graph(unsorted.path())); },
        "unsorted neighbours"
    );
}

/** Optional edge features currently have a defined format but no loader implementation. */
void test_edge_features_are_explicitly_rejected() {
    TemporaryFile file("graph_edge_features");
    write_graph_file(file.path(), 1, 3, 2, 0, 1, {0, 0, 0}, {});
    expect_runtime_error(
        [&] { static_cast<void>(gnni::load_graph(file.path())); },
        "unimplemented edge features"
    );
}

}  // namespace

int main() {
    try {
        test_valid_graph_and_auto_detection();
        test_header_and_size_errors();
        test_csr_structure_errors();
        test_row_neighbour_errors();
        test_edge_features_are_explicitly_rejected();
        std::cout << "graph loader tests passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "graph loader test failed: " << error.what() << '\n';
        return 1;
    }
}
