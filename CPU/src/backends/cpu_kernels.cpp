#include "cpu_kernels.hpp"

#include <stdexcept>
#include <cmath>

namespace gnni_cpu::cpu_detail {

DenseMatrix aggregate_mean_sequential(
    const CSRGraph& graph,
    const DenseMatrix& input, AggregationKind aggregation) {
        // input指当前这一层所有节点的特征，shape = N * F_in
        const std::size_t node_count = static_cast<std::size_t>(graph.num_nodes);
        const std::size_t feature_count = static_cast<std::size_t>(input.cols);

        // node_count和feature_count表示的是节点和特征数量；
        // size_T 能表达：1.数组/容器长度， 2. 容器索引 3. 元素数量 4.内存对象大小
        // 后面要用它： 访问std::vector; 计算vector的元素数量 3.作为循环索引
        DenseMatrix output;
        // 这里assign的是 dimension(uint64_t)
        output.rows = graph.num_nodes;
        output.cols = input.cols;

        /*
            相当于建立一个N x F的数组，并把所有位置初始化成0.0F
            row-major索引： row * numberofcol + col
        */
        output.values.assign(
            // 这两个在上面已经转化成了size_t
            // 这里计算的是矩阵需要多少个元素，如 100*64=6400 个float = node_count * feature_count
            node_count * feature_count, 0.0F
            // 因为float占 4个字节，so 6400 * 4
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
                const std::size_t edge_index = static_cast<std::size_t>(edge_position);
                // 读取当前入邻居， source -> destination
                const VertexId source = graph.col_idx[edge_index];
                // 累加邻居的每个特征
                for (std::size_t feature = 0; feature < feature_count; ++feature){
                    // output[destination, feature]
                    const std::size_t output_index = destination * feature_count + feature;
                    // input[source, feature];
                    const std::size_t input_index = static_cast<std::size_t>(source) * feature_count + feature;

                    // 把每个source节点的同一个feature，累加到destination对应的同一个feature上
                    float scale = 1.0F;
                    if (aggregation == AggregationKind::SymmetricGcn) {
                        const auto other_degree = graph.row_ptr[source+1] - graph.row_ptr[source];
                        scale = other_degree == 0 ? 0.0F : 1.0F / std::sqrt(float(degree) * float(other_degree));
                    }
                    output.values[output_index] += scale * input.values[input_index];
                }
            }

            // 求和完成后除以总数
            const float degree_as_float = static_cast<float>(degree);

            /*
                上面output得到的数据结构是：
                    output[destination, 0] = 所有邻居的 feature 0 之和
                    output[destination, 1] = 所有邻居的 feature 1 之和
                    output[destination, 2] = 所有邻居的 feature 2 之和
                所以这里需要再用一个for除以degree
            */ 
            for (std::size_t feature = 0; feature < feature_count; ++feature){
                const std::size_t output_index = destination * feature_count + feature;
                if (aggregation == AggregationKind::Mean) output.values[output_index] /= degree_as_float;
            }
        }
        return output;
}

// 这里计算 output = input x weights, 等于z = M * w
DenseMatrix linear_transform_sequential(
    const DenseMatrix& input,
    const DenseMatrix& weights) {
        const std::size_t node_count = static_cast<std::size_t>(input.rows); // N
        const std::size_t input_feature_count = static_cast<std::size_t>(input.cols); // F_in
        const std::size_t output_feature_count = static_cast<std::size_t>(weights.cols); // F_out

        DenseMatrix output;
        output.rows = input.rows;
        output.cols = weights.cols; 

        output.values.assign(node_count * output_feature_count, 0.0F);
        // 负责计算每个节点
        for(std::size_t node = 0; node < node_count; ++node){
            // 计算该节点输出向量中的每一项
            for (std::size_t output_feature = 0; output_feature < output_feature_count; ++output_feature){
                float sum = 0.0F;
                // 做点积在input_feature与weight之间
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

}  // namespace gnni_cpu::cpu_detail

