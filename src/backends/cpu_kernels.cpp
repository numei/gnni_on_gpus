#include "cpu_kernels.hpp"

#include <stdexcept>

namespace gnni::cpu_detail {

DenseMatrix aggregate_mean_sequential(
    const CSRGraph& graph,
    const DenseMatrix& input) {
        const std::size_t node_count = static_cast<std::size_t>(graph.num_nodes);
        const std::size_t feature_count = static_cast<std::size_t>(input.cols);

        DenseMatrix output;
        output.rows = graph.num_nodes;
        output.cols = input.cols;

        /*
            相当于建立一个N x F的数组，并把所有位置初始化成0.0F
            row-major索引： row * numberofcol + col
        */
        output.values.assign(
            node_count * feature_count, 0.0F
        );

        for (std::size_t destination = 0; destination < node_count; ++destination){
            const EdgeOffset begin = graph.row_ptr[destination];
            const EdgeOffset end = graph.row_ptr[destination + 1];
            const EdgeOffset degree = end - begin;

            // 没有入邻居时，输出行继续保持全零
            if(degree ==0){
                continue;
            }

            // 将所有入邻居的特征加到destination的输出行
            for(EdgeOffset edge_position = begin; edge_position < end; ++edge_position){
                const std::size_t edge_index = 
                    static_cast<std::size_t>(edge_position);
                const VertexId source = graph.col_idx[edge_index];
                
                for (std::size_t feature = 0; feature < feature_count; ++feature){
                    const std::size_t output_index = destination * feature_count + feature;
                    const std::size_t input_index = static_cast<std::size_t>(source) * feature_count + feature;

                    output.values[output_index] += input.values[input_index];
                }
            }

            // 求和完成后除以总数
            const float degree_as_float = static_cast<float>(degree);

            for (std::size_t feature = 0; feature < feature_count; ++feature){
                const std::size_t output_index = destination * feature_count + feature;
                output.values[output_index] /= degree_as_float;
            }
        }
        return output;
}

DenseMatrix linear_transform_sequential(
    const DenseMatrix& input,
    const DenseMatrix& weights) {
        const std::size_t node_count = static_cast<std::size_t>(input.rows);
        const std::size_t input_feature_count = static_cast<std::size_t>(input.cols);
        const std::size_t output_feature_count = static_cast<std::size_t>(weights.cols);

        DenseMatrix output;
        output.rows = input.rows;
        output.cols = weights.cols;

        output.values.assign(node_count * output_feature_count, 0.0F);
        // 负责计算每个节点
        for(std::size_t node = 0; node < node_count; ++node){
            // 计算该节点输出向量中的每一项
            for (std::size_t output_feature = 0; output_feature < output_feature_count; ++output_feature){
                float sum = 0.0F;
                // 做点积
                for (std::size_t input_feature = 0; input_feature < input_feature_count; ++input_feature){
                    const std::size_t input_index = node * input_feature_count + input_feature;
                    const std::size_t weight_index = input_feature * output_feature_count + output_feature;

                    sum += input.values[input_index] * weights.values[weight_index];
                }

                const std::size_t output_index = node * output_feature_count + output_feature;
                output.values[output_index] = sum;
            }
        }

        return output;
}

void apply_activation_in_place(DenseMatrix& matrix, ActivationKind activation) {
    switch(activation){
        case ActivationKind::Identity:
            return;

        case ActivationKind::ReLU:
        for (float& value : matrix.values){
            if (value < 0.0F){
                value = 0.0F;
            }
        }
        return;
    }
    throw std::invalid_argument("invalid ActivationKind value");
}

}  // namespace gnni::cpu_detail

