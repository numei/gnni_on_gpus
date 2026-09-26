#include "cli_options.hpp"

#include <charconv>
#include <cmath>
#include <cstdint>
#include <limits>
#include <ostream>
#include <stdexcept>
#include <string>
#include <system_error>
#include <unordered_set>
#include <fstream>
#include <iostream>
#include <optional>

namespace gnni_cpu::cli {

namespace {

std::string require_value(
    int& index, int argc, const char* const argv[], const std::string& option
) {
    if (index + 1 >= argc) {
        throw std::invalid_argument("missing value after " + option);
    }

    ++index;
    return argv[index];
}

void reject_duplicate(std::unordered_set<std::string>& seen, const std::string& option) {
    if (!seen.insert(option).second) {
        throw std::invalid_argument("duplicate option: " + option);
    }
}

std::uint32_t parse_u32(const std::string& text, const std::string& option, bool allow_zero) {
    std::uint64_t value = 0;
    const char* const begin = text.data();
    const char* const end = begin + text.size();
    const auto result = std::from_chars(begin, end, value);

    if (result.ec != std::errc{} || result.ptr != end) {
        throw std::invalid_argument(
            "invalid integer for " + option + ": " + text);
    }

    if (value > std::numeric_limits<std::uint32_t>::max()) {
        throw std::invalid_argument("integer is too large for " + option);
    }

    if (!allow_zero && value == 0) {
        throw std::invalid_argument(option + " must be greater than zero");
    }

    return static_cast<std::uint32_t>(value);
}

double parse_non_negative_double(const std::string& text, const std::string& option) {
    std::size_t consumed = 0;
    double value = 0.0;

    try {
        value = std::stod(text, &consumed);
    } catch (const std::exception&) {
        throw std::invalid_argument(
            "invalid number for " + option + ": " + text);
    }

    if (consumed != text.size() || !std::isfinite(value) || value < 0.0) {
        throw std::invalid_argument(
            option + " must be a finite non-negative number");
    }

    return value;
}

BackendKind parse_backend(const std::string& text) {
    if (text == "cpu") {
        return BackendKind::CpuSequential;
    }
    if (text == "omp") {
        return BackendKind::CpuOpenMP;
    }
    throw std::invalid_argument("invalid backend: " + text);
}

InputFileFormat parse_input_format(const std::string& text) {
    if (text == "auto") {
        return InputFileFormat::Auto;
    }
    if (text == "binary") {
        return InputFileFormat::Binary;
    }
    if (text == "text") {
        return InputFileFormat::Text;
    }

    throw std::invalid_argument("invalid input format: " + text);
}

OpenMPScheduleKind parse_openmp_schedule(const std::string& text) {
    if (text == "default") {
        return OpenMPScheduleKind::Default;
    }
    if (text == "static") {
        return OpenMPScheduleKind::Static;
    }
    if (text == "dynamic") {
        return OpenMPScheduleKind::Dynamic;
    }
    if (text == "guided") {
        return OpenMPScheduleKind::Guided;
    }

    throw std::invalid_argument("invalid OpenMP schedule: " + text);
}

OutputTransform parse_output_transform(const std::string& text) {
    if (text == "logits") {
        return OutputTransform::Logits;
    }
    if (text == "probabilities") {
        return OutputTransform::Softmax;
    }

    throw std::invalid_argument("invalid output transform: " + text);
}

void require_backend_support(BackendKind backend, const InferenceConfig& config){
    const BackendCapabilities capabilities = backend_capabilities(backend);
    const std::string name = backend_name(backend);

    if (!capabilities.available) {
        throw std::runtime_error("backend is not available: " + name );
    }

    if (config.aggregation == AggregationKind::Mean && !capabilities.mean_aggregation) {
        throw std::runtime_error("backend does not support mean aggregation: " + name);
    }

    if (config.aggregation == AggregationKind::Attention && !capabilities.attention_aggregation){
        throw std::runtime_error("backend does not support attention aggregation: " + name);
    }

    if (config.output_transform == OutputTransform::Softmax && !capabilities.softmax) {
        throw std::runtime_error(
            "backend does not support softmax output: " + name
        );
    }
}

}  // namespace

CliOptions parse_cli(int argc, const char* const argv[]) {
    CliOptions options;
    std::unordered_set<std::string> seen;

    for (int index = 1; index < argc; ++index) {
        const std::string argument = argv[index];
        reject_duplicate(seen, argument);

        if (argument == "--help") {
            options.show_help = true;
        } else if (argument == "--verify") {
            options.verify = true;
        } else if (argument == "--graph") {
            options.graph_path = require_value(index, argc, argv, argument);
        } else if (argument == "--features") {
            options.features_path = require_value(index, argc, argv, argument);
        } else if (argument == "--model") {
            options.model_path = require_value(index, argc, argv, argument);
        } else if (argument == "--backend") {
            options.backend = parse_backend(
                require_value(index, argc, argv, argument));
        } else if (argument == "--input-format") {
            options.input_format = parse_input_format(
                require_value(index, argc, argv, argument));
        } else if (argument == "--threads") {
            options.inference.num_threads = parse_u32(
                require_value(index, argc, argv, argument), argument, true);
        } else if (argument == "--omp-schedule") {
            options.inference.openmp_schedule = parse_openmp_schedule(
                require_value(index, argc, argv, argument));
        } else if (argument == "--omp-chunk-size") {
            options.inference.openmp_chunk_size = parse_u32(
                require_value(index, argc, argv, argument), argument, true);
        } else if (argument == "--warmup") {
            options.benchmark.warmup_runs = parse_u32(
                require_value(index, argc, argv, argument), argument, true);
        } else if (argument == "--repeat") {
            options.benchmark.measured_runs = parse_u32(
                require_value(index, argc, argv, argument), argument, false);
        } else if (argument == "--atol") {
            options.comparison.absolute_tolerance = parse_non_negative_double(
                require_value(index, argc, argv, argument), argument);
        } else if (argument == "--rtol") {
            options.comparison.relative_tolerance = parse_non_negative_double(
                require_value(index, argc, argv, argument), argument);
        } else if (argument == "--output") {
            options.inference.output_transform = parse_output_transform(
                require_value(index, argc, argv, argument));
        } else if (argument == "--csv") {
            options.csv_output_path =
                require_value(index, argc, argv, argument);
        } else {
            throw std::invalid_argument("unknown option: " + argument);
        }
    }

    if (options.show_help) {
        return options;
    }

    if (options.graph_path.empty()) {
        throw std::invalid_argument("missing required option: --graph");
    }
    if (options.features_path.empty()) {
        throw std::invalid_argument("missing required option: --features");
    }
    if (options.model_path.empty()) {
        throw std::invalid_argument("missing required option: --model");
    }

    return options;
}

void print_help(std::ostream& output) {
    output
        << "Usage: gnni --graph FILE --features FILE --model FILE [options]\n"
        << "\n"
        << "Core options:\n"
        << "  --backend cpu|omp\n"
        << "  --input-format auto|binary|text\n"
        << "  --threads N\n"
        << "  --omp-schedule default|static|dynamic|guided\n"
        << "  --omp-chunk-size N\n"
        << "  --warmup N\n"
        << "  --repeat N\n"
        << "  --verify\n"
        << "  --atol VALUE\n"
        << "  --rtol VALUE\n"
        << "  --output logits|probabilities\n"
        << "  --csv FILE\n"
        << "  --help\n";
}

int run_cli(int argc, const char* const argv[]) {
    const CliOptions options = parse_cli(argc, argv);

    if (options.show_help){
        print_help(std::cout);
        return 0;
    }

    require_backend_support(options.backend, options.inference);
    const CSRGraph graph = load_graph(options.graph_path, options.input_format);

    const DenseMatrix features = load_dense_matrix(options.features_path, options.input_format);

    const Model model = load_model(options.model_path, options.input_format);

    const BenchmarkReport benchmark = benchmark_backend(options.backend, graph, features, model, options.inference, options.benchmark);

    write_text_report(benchmark, std::cout);
    int exit_code = 0;
    if (options.verify){
        // sequential CPU 是全组唯一reference,所以无论选择什么，sequential都要运行一次
        const RunResult reference = run_cpu_sequential(
            graph, features, model, options.inference
        );
        const DenseMatrix* candidate_output = &reference.output;
        std::optional<RunResult> candidate;

        // 如果选择的是sequential cpu, 不需要再运行一次
        if(options.backend != BackendKind::CpuSequential){
            candidate.emplace(run_backend(options.backend, graph, features, model, options.inference));
            candidate_output = &candidate->output;
        }
        const ComparisonReport comparison = compare_outputs(reference.output, *candidate_output, options.comparison);

        std::cout
            << '\n'
            << "verification:\n"
            << "shapes_match: "
            << (comparison.shapes_match
                    ? "true"
                    : "false")
            << '\n'
            << "within_tolerance: "
            << (comparison.within_tolerance
                    ? "true"
                    : "false")
            << '\n'
            << "mismatch_count: "
            << comparison.mismatch_count
            << '\n'
            << "max_absolute_error: "
            << comparison.max_absolute_error
            << '\n'
            << "max_relative_error: "
            << comparison.max_relative_error
            << '\n'
            << "rmse: "
            << comparison.root_mean_square_error
            << '\n';

        if (comparison.first_mismatch_index.has_value()){
            const std::uint64_t index = comparison.first_mismatch_index.value();

            std::cout
                << "first_mismatch_index: "
                << index << '\n';

            if (reference.output.cols != 0) {
                std::cout << "first_mismatch_row: " << index / reference.output.cols << '\n'
                    << "first_mismatch_col: " << index % reference.output.cols << '\n';
            }
        }
        if (!comparison.shapes_match || !comparison.within_tolerance) {
            exit_code = 2;
        }
    }

    if (options.csv_output_path.has_value()) {
        const std::filesystem::path& csv_path =
            options.csv_output_path.value();

        std::ofstream csv_output(
            csv_path,
            std::ios::out | std::ios::trunc
        );

        if (!csv_output) {
            throw std::runtime_error(
                "cannot open CSV output file: " +
                csv_path.string()
            );
        }

        write_csv_report(
            benchmark,
            csv_output
        );

        csv_output.flush();

        if (!csv_output) {
            throw std::runtime_error(
                "failed to write CSV output file: " +
                csv_path.string()
            );
        }
    }

    return exit_code;
}

}  // namespace gnni_cpu::cli
