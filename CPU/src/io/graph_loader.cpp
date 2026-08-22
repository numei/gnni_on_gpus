#include "gnni/io.hpp"
#include "binary_reader.hpp"
#include "array"
#include "cstdint"
#include "limits"
#include <string>
#include <stdexcept>

namespace gnni {
namespace {
    constexpr std::array<std::uint8_t, 8> graph_magic{
        'G', 'N', 'N', 'I', 'G', 'R', 'F', 0
    };

constexpr std::uint32_t format_version = 1;
constexpr std::uint32_t graph_header_bytes = 48;
constexpr std::uint32_t flag_directed = 1U << 0U;
constexpr std::uint32_t flag_edge_features = 1U << 1U;
constexpr std::uint32_t know_flags = flag_directed | flag_edge_features;

CSRGraph load_binary_graph(const std::filesystem::path& path){
    constexpr const char* file_kind = "graph";

    io_detail::BinaryReader reader(path, file_kind);
    // reader创建时，已经打开文件，检查是否能读取，获得文件大小，将读取位置放回文件开头

    // 读取和检查magic
    const auto actual_magic = reader.read_magic();
    if (actual_magic != graph_magic){
        io_detail::throw_file_error(
            path, file_kind, "Invaild magic"
        );
    }

    // 读取header
    const std::uint32_t version = reader.read_u32("version");
    const std::uint32_t header_bytes = reader.read_u32("header_bytes");
    const std::uint32_t flags = reader.read_u32("flags");
    const std::uint32_t reserved = reader.read_u32("reserved");
    const std::uint64_t num_nodes = reader.read_u64("num_nodes");
    const std::uint64_t num_edges = reader.read_u64("num_edges");
    const std::uint64_t edge_feature_dim = reader.read_u64("edge_feature_dim");

    // 校验header
    if (version != format_version){
        io_detail::throw_file_error(
            path, file_kind, "unsupported version" + std::to_string(version)
        );
    };

    if (header_bytes != graph_header_bytes) {
        io_detail::throw_file_error(
            path, file_kind, 
            "header_bytes is " + std::to_string(header_bytes) +", expected 48"
        );
    }

    if (reserved != 0) {
        io_detail::throw_file_error(
            path, file_kind, "reserved field must be zero"
        );
    }

    if ((flags & ~know_flags) != 0) {
        io_detail::throw_file_error(
            path, file_kind,
        "flags contain unsupported bits"
        );
    }

    // true 就是有向或者有edge features
    const bool directed = (flags & flag_directed) != 0;
    const bool has_edge_features = (flags & flag_edge_features) != 0;

    // 下面都是check 读取的数据对不对
    if (num_nodes == 0) {
        io_detail::throw_file_error(
            path, file_kind, "num_nodes must be greater than zero"
        );
    }

    const std::uint64_t max_vertex_id = std::numeric_limits<VertexId>::max();
    if(num_nodes - 1 > max_vertex_id){
        io_detail::throw_file_error(
            path, file_kind, "vertex identifiers do not fit VertexId"
        );
    }

    if (!has_edge_features && edge_feature_dim != 0) {
        io_detail::throw_file_error(
            path, file_kind, "edge_feature_dim must be zero when edge features are absent"
        );
    }

    if (has_edge_features && edge_feature_dim == 0) {
        io_detail::throw_file_error(
            path, file_kind, "edge_feature_dim must be positive when edge features are present"
        );
    }

    if (has_edge_features) {
        io_detail::throw_file_error(
            path, file_kind, "edge-feature graph loading is not implemented yet"
        );
    }

    const std::uint64_t row_ptr_count = io_detail::checked_add(
        num_nodes, 1, path, file_kind, "row_ptr count"
    );

    const std::uint64_t row_ptr_bytes = io_detail::checked_multiply(
        row_ptr_count, sizeof(EdgeOffset), path, file_kind, "row_ptr payload"
    );

    const std::uint64_t col_idx_bytes = io_detail::checked_multiply(
        num_edges, sizeof(VertexId), path, file_kind, "col_idx payload"
    );

    std::uint64_t expected_size = graph_header_bytes;

    expected_size = io_detail::checked_add(
        expected_size, row_ptr_bytes, path, file_kind, "graph file"
    );

    expected_size = io_detail::checked_add(
        expected_size, col_idx_bytes, path, file_kind, "graph file"
    );

    reader.require_exact_size(expected_size);

    // 创建graph 对象
    CSRGraph graph;

    graph.num_nodes = num_nodes;
    graph.num_edges = num_edges;
    graph.directed = directed;

    // 为vector 分配空间：
    const std::size_t row_ptr_size = io_detail::checked_size_t(
        row_ptr_count, path, file_kind, "row_ptr count"
    );

    const std::size_t col_idx_size = io_detail::checked_size_t(
        num_edges, path, file_kind, "col_idx count"
    );

    graph.row_ptr.resize(row_ptr_size);
    graph.col_idx.resize(col_idx_size);

    for (std::size_t index = 0; index < graph.row_ptr.size(); ++index){
        graph.row_ptr[index] = reader.read_u64("row_ptr");
    }

    for (std::size_t index = 0; index < graph.col_idx.size(); ++index){
        graph.col_idx[index] = reader.read_u32("col_idx");
    }

    // 表示这个optional里面没有denseMatrix
    graph.edge_features = std::nullopt;
    reader.require_end();

    if ( graph.row_ptr.front() != 0){
        io_detail::throw_file_error(
            path, file_kind, "row_ptr[0] must be zero"
        );
    }

    if (graph.row_ptr.back() != graph.num_edges){
        io_detail::throw_file_error(
            path, file_kind, "row_ptr.back() is " + std::to_string(graph.row_ptr.back()) +
            ", expected num_edges " + std::to_string(graph.num_edges)
        );
    }

    for (std::uint64_t v = 0; v < graph.num_nodes; ++v){
        // 把uint64_t 转成索引类型
        const std::size_t row = static_cast<std::size_t>(v);
        const EdgeOffset begin = graph.row_ptr[row];
        const EdgeOffset end = graph.row_ptr[row+1];

        if(begin > end) {
            io_detail::throw_file_error(
                path, file_kind, "row_ptr decreases at row" + std::to_string(v)
            );
        }

        if (end > graph.num_edges){
            io_detail::throw_file_error(
                path, file_kind, 
                "row_ptr exceeds num_edges at row " + std::to_string(v)
            );
        }

        bool has_previous = false;
        VertexId previous = 0;

        for (EdgeOffset p = begin; p < end; ++p){
            const std::size_t edge_index = static_cast<std::size_t>(p);
            const VertexId u = graph.col_idx[edge_index];

            if (static_cast<std::uint64_t>(u) >= graph.num_nodes ){
                io_detail::throw_file_error(
                    path, file_kind, "col_idx[" + std::to_string(p) + "] is outside the vertex range" 
                );
            }

            if (static_cast<std::uint64_t>(u) == v){
                io_detail::throw_file_error(
                    path, file_kind, "self-loop in row" + std::to_string(v)
                );
            }

            if (has_previous && u== previous){
                io_detail::throw_file_error(
                    path, file_kind, "duplicate neighbor " + std::to_string(u) + "in row" + std::to_string(v)
                );
            }

            if (has_previous && u < previous) {
                io_detail::throw_file_error(
                    path, file_kind, "neighbors are not sorted in row " + std::to_string(v)
                );
            }

            previous = u;
            has_previous = true;
        }

    }

    return graph;
}

bool has_binary_graph_magic(
    const std::filesystem::path& path
) {
    io_detail::BinaryReader reader(path, "graph");
    return reader.read_magic() == graph_magic;
}

} // namespace

/** Load a graph using the selected format or binary magic detection. */
CSRGraph load_graph(const std::filesystem::path& path, InputFileFormat format) {
    switch (format)
    {
    case InputFileFormat::Binary:
        return load_binary_graph(path);
    
    case InputFileFormat::Text:
        throw std::logic_error(
            "graph text loader is not implemented"
        );
    case InputFileFormat::Auto:
        if (has_binary_graph_magic(path)){
            return load_binary_graph(path);
        }
        io_detail::throw_file_error(path, "graph", "unrecognized format or text loader not implemented");
    }

    io_detail::throw_file_error(path, "graph", "invalid InputFileFormat");
}

}  // namespace gnni
