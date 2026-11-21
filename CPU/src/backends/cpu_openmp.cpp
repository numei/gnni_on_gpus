#include "gnni/api.hpp"
#include "validate.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <climits>
#include <limits>
#include <stdexcept>
#include <utility>
#ifdef GNNI_HAS_OPENMP
#include <omp.h>
#endif

namespace gnni_cpu {
namespace {
#ifdef GNNI_HAS_OPENMP
// 按行分工的公共帮助函数：count 是行数，fn(i) 负责完成第 i 行的工作。
// 使用显式 schedule 子句兼容 MSVC 较旧的 OpenMP 运行时，
// 不必调用 omp_set_schedule 修改运行时调度设置，异常退出时也无需恢复调度状态。
template<class Function>
void parallel_rows(std::int64_t count, const InferenceConfig& config, Function fn) {
    // num_threads 为 0 时，使用 OpenMP 当前的最大线程数设置。
    const int threads = config.num_threads ? static_cast<int>(config.num_threads) : omp_get_max_threads();
    // chunk 表示一次分配多少行；dynamic/guided 在未指定时使用 1。
    // static 的未指定情况在下面单独处理，省略 chunk 子句。
    const int chunk = config.openmp_chunk_size ? static_cast<int>(config.openmp_chunk_size) : 1;
    switch (config.openmp_schedule) {
    case OpenMPScheduleKind::Dynamic:
        // 动态调度：线程完成一块后再领取下一块，有利于分摊不均匀的行工作量，
        // 但频繁领取小块也会增加调度开销。
        #pragma omp parallel for schedule(dynamic, chunk) num_threads(threads)
        for (std::int64_t i=0; i<count; ++i) fn(i);
        break;
    case OpenMPScheduleKind::Guided:
        // 引导调度：开始分配较大的块，剩余工作减少时逐渐缩小；chunk 是最小块大小。
        #pragma omp parallel for schedule(guided, chunk) num_threads(threads)
        for (std::int64_t i=0; i<count; ++i) fn(i);
        break;
    case OpenMPScheduleKind::Default:
    case OpenMPScheduleKind::Static:
        // 本实现的 Default 走 static：预先分配行，通常调度开销较低。
        if (config.openmp_chunk_size) {
            #pragma omp parallel for schedule(static, chunk) num_threads(threads)
            for (std::int64_t i=0; i<count; ++i) fn(i);
        } else {
            // 没有指定块大小时，让 static 自行划分，而不是强制每块只有 1 行。
            #pragma omp parallel for schedule(static) num_threads(threads)
            for (std::int64_t i=0; i<count; ++i) fn(i);
        }
        break;
    default: throw std::invalid_argument("invalid OpenMP schedule");
    }
    // 上面的 parallel for 都带有结束时的隐式同步；返回前所有行均已完成。
}
#endif
}

RunResult run_cpu_openmp(const CSRGraph& graph, const DenseMatrix& features,
                        const Model& model, const InferenceConfig& config) {
#ifndef GNNI_HAS_OPENMP
    // 构建时没有启用 OpenMP 就明确报错，避免把串行执行误认为并行结果。
    (void)graph; (void)features; (void)model; (void)config;
    throw std::runtime_error("OpenMP support is not available in this build");
#else
    // steady_clock 是适合测量经过时间的单调时钟。
    // 完整推理计时从检查输入之前开始，因此包含检查、初始复制和各层计算。
    using Clock = std::chrono::steady_clock;
    const auto start = Clock::now();

    // 先检查 CSR、矩阵形状和有限值，保证后面的数组访问满足输入契约。
    cpu_detail::validate_inputs(graph, features, model);
    // 当前支持 mean、sum 和对称归一化；输出直接返回 logits/特征，不做 softmax。
    if (config.aggregation != AggregationKind::Mean && config.aggregation != AggregationKind::Sum &&
        config.aggregation != AggregationKind::SymmetricGcn)
        throw std::invalid_argument("unsupported aggregation");
    if (config.output_transform != OutputTransform::Logits)
        throw std::invalid_argument("OpenMP backend supports logits output only");
    // OpenMP 的线程数和块大小使用 int，转换前先检查能否表示。
    if (config.num_threads > INT_MAX || config.openmp_chunk_size > INT_MAX)
        throw std::invalid_argument("OpenMP configuration exceeds int range");

    // current 是本层要读取的特征。这里复制输入，后面可替换中间结果，
    // 而调用者的原始 features 仍保留，可供下一次完整推理使用。
    DenseMatrix current = features;
    // persistent 估计长期保留的 CSR、原始特征和全部层权重的数组载荷。
    // 这里按元素数量计字节，不是操作系统的进程总内存，也不含分配器等开销。
    std::uint64_t persistent = graph.row_ptr.size()*sizeof(EdgeOffset) + graph.col_idx.size()*sizeof(VertexId) + features.values.size()*sizeof(float);
    for (const auto& layer : model.layers) persistent += layer.weights.values.size()*sizeof(float);
    // peak 记录临时工作区峰值；此刻只有 current 这一份特征副本。
    std::uint64_t peak = current.values.size()*sizeof(float);
    // compute 计时从初始复制之后开始，但仍包含各层的分配、清零和计算。
    const auto compute_start = Clock::now();

    // 模型层之间依次执行：一层的输出是下一层的输入；层内的行交给线程并行处理。
    for (const auto& layer : model.layers) {
        // 第一步：邻居聚合。结果形状为 N × F_in，先全部置零，供 += 累加。
        // 空邻居行不会产生贡献，所以自然保持零向量。
        DenseMatrix aggregated{graph.num_nodes, current.cols, std::vector<float>(current.values.size(),0.0F)};
        // 每次迭代独占 aggregated 的一行，只读共享的 graph/current，
        // 不同线程不会累加到同一输出位置，因此聚合不需要 atomic 或锁。
        parallel_rows(static_cast<std::int64_t>(graph.num_nodes),config,[&](std::int64_t row){
            // CSR 第 row 行保存指向 row 的入邻居；相邻偏移之差就是入度。
            const auto degree=graph.row_ptr[row+1]-graph.row_ptr[row];
            for(auto e=graph.row_ptr[row]; e<graph.row_ptr[row+1]; ++e){
                // e 是边记录在数组中的位置，u 才是这条边的来源节点编号：u -> row。
                const auto u=graph.col_idx[e];
                // sum/mean 在累加时均先用系数 1；mean 会在整行求和后再除以入度。
                float scale=1.0F;
                if(config.aggregation==AggregationKind::SymmetricGcn){
                    // 对称模式使用 1 / sqrt(degree(row) * degree(u))。
                    // 两端度数都按 CSR 行入度计算；来源入度为 0 时规定贡献为 0。
                    const auto du=graph.row_ptr[u+1]-graph.row_ptr[u];
                    scale=du?1.0F/std::sqrt(float(degree)*float(du)):0.0F;
                }
                // 同一节点的特征按行连续存储，下标是“节点编号 * 特征数 + 特征编号”。
                // 把来源 u 的每项特征，累加到目标 row 的相同特征位置。
                for(std::size_t f=0;f<current.cols;++f)
                    aggregated.values[row*current.cols+f]+=scale*current.values[u*current.cols+f];
            }
            // mean 逐项取平均；sum 和 symmetric 保留上面各自的累加结果。
            // degree 为 0 时跳过除法，既避免除零，也保留初始化时的全零输出。
            if(degree && config.aggregation==AggregationKind::Mean)
                for(std::size_t f=0;f<current.cols;++f) aggregated.values[row*current.cols+f]/=float(degree);
        });

        // 聚合的 parallel_rows 已等待全部行完成，接下来才读取聚合结果。
        // 第二步：线性变换，N × F_in 乘 F_in × F_out，得到 N × F_out。
        const auto width=layer.weights.cols;
        // 分配输出数组前，检查 N * F_out 是否会溢出 size_t。
        if(graph.num_nodes > std::numeric_limits<std::size_t>::max()/width)
            throw std::invalid_argument("output shape overflow");
        // 点积通过 += 累加，所以 next 也必须从全零开始。
        DenseMatrix next{graph.num_nodes,width,std::vector<float>(graph.num_nodes*width,0.0F)};
        parallel_rows(static_cast<std::int64_t>(graph.num_nodes),config,[&](std::int64_t row){
            // k 遍历输入特征，f 遍历输出特征：
            // next[row,f] += aggregated[row,k] * weights[k,f]。
            // f 在最内层，使权重的一行和输出的一行连续访问，并复用读出的 x。
            for(std::size_t k=0;k<current.cols;++k){
                const float x=aggregated.values[row*current.cols+k];
                for(std::size_t f=0;f<width;++f) next.values[row*width+f]+=x*layer.weights.values[k*width+f];
            }
            // 第三步：同一线程在自己的输出行上做 ReLU，把负数改为 0。
            // Identity 无需处理；把激活放在这里可避免再启动单独的并行循环。
            if(layer.activation==ActivationKind::ReLU)
                for(std::size_t f=0;f<width;++f) next.values[row*width+f]=std::max(0.0F,next.values[row*width+f]);
        });

        // 此刻 current、aggregated、next 同时存活，计入本层工作区峰值。
        // 对应约 4 * N * (2 * F_in + F_out) 字节；不同层之间取最大值。
        peak=std::max(peak,static_cast<std::uint64_t>((current.values.size()+aggregated.values.size()+next.values.size())*sizeof(float)));
        // 把本层结果交给下一层，move 通常避免再次复制整块 vector 数据。
        // 前面的并行循环已经同步完成，下一层不会读到尚未写好的结果。
        current=std::move(next);
    }

    // 层循环结束，停止计时并转交最终输出，不再复制整个结果矩阵。
    const auto end=Clock::now();
    RunResult result;
    result.output=std::move(current);
    result.metrics.timing.compute_ms=std::chrono::duration<double,std::milli>(end-compute_start).count();
    result.metrics.timing.inference_ms=std::chrono::duration<double,std::milli>(end-start).count();
    // 内存指标 = 持久输入载荷 + 最大临时工作区，不是各层工作区的简单累加。
    result.metrics.memory.host_bytes=persistent+peak;
    // 下面保留旧 CPU API 的吞吐量口径：节点数/边数都乘模型层数 L，
    // 再除以 compute 秒数，含义是 node-layer/s 和 edge-layer/s。
    // 统一 gnni_q1 的吞吐量另按 N/E 除以完整调用时间计算，不使用这里的值。
    const double seconds=result.metrics.timing.compute_ms/1000.0;
    // 先确认时间大于 0，避免计时过短时出现除零。
    if(seconds>0){
        result.metrics.nodes_per_second=double(graph.num_nodes)*model.layers.size()/seconds;
        result.metrics.edges_per_second=double(graph.num_edges)*model.layers.size()/seconds;
    }
    return result;
#endif
}
}
