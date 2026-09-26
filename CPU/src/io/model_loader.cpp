#include "gnni/io.hpp"

#include <stdexcept>
#include "array"
#include "cmath"
#include "cstdint"
#include "binary_reader.hpp"
#include "string"
#include "utility"
#include "vector"

namespace gnni_cpu {

namespace {
    constexpr std::array<std::uint8_t, 8> model_magic{
        'G', 'N', 'N', 'I', 'M', 'D', 'L', 0
    };

    constexpr std::uint32_t format_version = 1;
    constexpr std::uint32_t model_header_bytes = 48;
    constexpr std::uint32_t descriptor_bytes = 24;
    constexpr std::uint32_t scalar_float32 = 1;
    constexpr std::uint32_t layout_row_major = 1;
    constexpr std::uint32_t activation_identity = 0;
    constexpr std::uint32_t activation_relu =1;

    /*
        48-byte model header 
        所有的 layer descriptors 0-n
        所有 layer weights 0-n
    */

    Model load_binary_model(
        const std::filesystem::path& path
    ){
        constexpr const char* file_kind = "model";

        //读magic
        io_detail::BinaryReader reader(path, file_kind);
        const auto actual_magic = reader.read_magic();
        // check magic
        if (actual_magic != model_magic){
            io_detail::throw_file_error(
                path, file_kind, "invalid magic"
            );
        }

        // 读取model header
        const std::uint32_t version = reader.read_u32("version");
        const std::uint32_t header_bytes = reader.read_u32("header_bytes");
        const std::uint32_t scalar_type = reader.read_u32("scalar_type");
        const std::uint32_t layout = reader.read_u32("layout");
        const std::uint32_t num_layers = reader.read_u32("num_layers");
        const std::uint32_t flags = reader.read_u32("flags");
        const std::uint32_t layer_descriptor_bytes = reader.read_u32("layer_descriptor_bytes");
        const std::uint32_t reserved_0 = reader.read_u32("reserved_0");
        const std::uint64_t reserved_1 = reader.read_u64("reserved_1");
        
        // 验证header
        if (version != format_version){
            io_detail::throw_file_error(
                path, file_kind, "unsupported version " + std::to_string(version)
            );
        }

        if (header_bytes != model_header_bytes){
            io_detail::throw_file_error(
                path, file_kind, "header+bytes"
            );
        }

        if (scalar_type != scalar_float32){
            io_detail::throw_file_error(
                path, file_kind, "scalar_type"
            );
        }

        if (layout != layout_row_major ){
            io_detail::throw_file_error(
                path, file_kind, "unsupported layout" + std::to_string(layout)
            );
        }

        if (num_layers == 0){
            io_detail::throw_file_error(
                path, file_kind, "num_layers must be greater than zero"
            );
        }

         if (flags != 0){
            io_detail::throw_file_error(
                path, file_kind, "flags must be zero"
            );
        }

         if (layer_descriptor_bytes != descriptor_bytes){
            io_detail::throw_file_error(
                path, file_kind, "layer_descriptor_bytes "
            );
        }

        if (reserved_0 != 0 || reserved_1 != 0){
            io_detail::throw_file_error(
                path, file_kind, "reserved fields must be zero"
            );
        }

        const std::uint64_t descriptors_size =
            io_detail::checked_multiply(
                num_layers,
                descriptor_bytes,
                path,
                file_kind,
                "layer descriptors"
        );

        const std::uint64_t descriptors_end =
            io_detail::checked_add(
                model_header_bytes,
                descriptors_size,
                path,
                file_kind,
                "model descriptor region"
        );

        if (reader.file_size() < descriptors_end) {
            io_detail::throw_file_error(
                path,
                file_kind,
                "file is truncated inside layer descriptors"
            );
        }

        // 创建model 和 layer 数组
        const std::size_t layer_count = io_detail::checked_size_t(
            num_layers, path, file_kind, "num_layers"
        );

        Model model;
        model.layers.resize(layer_count);

        // 读取descriptor
        std::uint64_t tot_weight_count = 0;

        for (std::size_t index = 0; index < model.layers.size(); ++index){
            const std::uint64_t input_dim = reader.read_u64("layer input_dim");
            const std::uint64_t output_dim = reader.read_u64("layer output_dim");
            const std::uint32_t activation_code = reader.read_u32("layer activation");
            const std::uint32_t layer_reserved = reader.read_u32("layer reserved");

            if(input_dim == 0 || output_dim ==0){
                io_detail::throw_file_error(
                    path, file_kind, "layer " + std::to_string(index) + 
                        "dimensions must be greater than zero "
                );
            }

            if( layer_reserved != 0){
                io_detail::throw_file_error(
                    path, file_kind, "layer " + std::to_string(index) + 
                        "reserved field must be zero "
                );
            }


            ActivationKind activation;
            if (activation_code == activation_identity){
                activation = ActivationKind::Identity;
            } else if (activation_code == activation_relu){
                activation = ActivationKind::ReLU;
            }else{
                io_detail::throw_file_error(
                    path, file_kind, 
                    "invalid activation code " + std::to_string(activation_code) 
                    + "in layer " + std::to_string(index)
                );
            }

            const bool is_last_layer = index + 1 == model.layers.size();
            if (!is_last_layer && activation != ActivationKind::ReLU){
                io_detail::throw_file_error(
                    path, file_kind, "hidden layer" + std::to_string(index) + " must use ReLU"
                );
            }

            if (is_last_layer && activation != ActivationKind::Identity){
                io_detail::throw_file_error(
                    path, file_kind, "final layer must use Identity"
                );
            }
            
            if (index > 0) {
                const Layer& previous_layer = model.layers[index -1];

                if (previous_layer.weights.cols != input_dim){
                    io_detail::throw_file_error(
                        path,
                        file_kind,
                        "layer " + std::to_string(index) +
                            " input_dim does not match previous output_dim"
                    );
                }
            }

            Layer& layer = model.layers[index];
            layer.weights.rows = input_dim;
            layer.weights.cols = output_dim;
            layer.activation = activation;

            //每层weight 数量
            const std::uint64_t layer_weight_count = io_detail::checked_multiply(
                input_dim, output_dim, path, file_kind, "layer weight count"
            );

            tot_weight_count = io_detail::checked_add(
                tot_weight_count, layer_weight_count, path, file_kind, "total model weight count"
            );
        }

        // 计算weight大小
        const std::uint64_t weight_bytes = io_detail::checked_multiply(
            tot_weight_count, sizeof(float), path, file_kind,
            "model weight payload"
        );

        const std::uint64_t expected_size = io_detail::checked_add(
            descriptors_end, weight_bytes, path, file_kind,
            "model file"
        );

        reader.require_exact_size(expected_size);

        // 逐层读取weights
        for (std::size_t layer_index = 0; layer_index < model.layers.size(); ++layer_index){
            Layer& layer = model.layers[layer_index];
            // 计算该层数量
            const std::uint64_t weight_count = io_detail::checked_multiply(
                layer.weights.rows, layer.weights.cols, path, file_kind, "layer weight count"
            );

            layer.weights.values = reader.read_f32_vector(weight_count, "layer weights");

            for (std::size_t weight_index = 0;
                weight_index < layer.weights.values.size(); ++weight_index) {
                const float value = layer.weights.values[weight_index];

                if (!std::isfinite(value)) {
                    io_detail::throw_file_error(
                        path,
                        file_kind,
                        "non-finite weight at layer " +
                            std::to_string(layer_index) +
                            ", element " +
                            std::to_string(weight_index)
                    );
                }
            }
        }

        reader.require_end();
        return model;
    }

    bool has_binary_model_magic(
        const std::filesystem::path& path
    ){
        io_detail::BinaryReader reader(path, "model");
        return reader.read_magic() == model_magic;
    }
}

/** Load a model using the selected format or binary magic detection. */
Model load_model(const std::filesystem::path& path, InputFileFormat format) {
    switch (format){
        case InputFileFormat::Binary:
            return load_binary_model(path);
        case InputFileFormat::Text:
            throw std::logic_error(
                "model text is not implemented"
            );
        case InputFileFormat::Auto:
            if(has_binary_model_magic(path)){
                return load_binary_model(path);
            }

            io_detail::throw_file_error(
                path, "model", "unrecognized format or text loader not implemented"
            );
    }
    throw std::logic_error("invalid InputFileFormat value");
}

}  // namespace gnni_cpu
