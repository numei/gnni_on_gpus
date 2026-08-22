#include "cli_options.hpp"

#include <iostream>
#include <stdexcept>

namespace {

bool parse_throws_invalid_argument(
    int argc,
    const char* const argv[]) {
    try {
        static_cast<void>(gnni::cli::parse_cli(argc, argv));
    } catch (const std::invalid_argument&) {
        return true;
    }
    return false;
}

}  // namespace

int main() {
    const char* valid_arguments[] = {
        "gnni",
        "--graph", "graph.bin",
        "--features", "features.bin",
        "--model", "model.bin",
        "--backend", "omp",
        "--input-format", "binary",
        "--threads", "4",
        "--omp-schedule", "guided",
        "--omp-chunk-size", "8",
        "--warmup", "0",
        "--repeat", "3",
        "--verify",
        "--atol", "0.00001",
        "--rtol", "0.0001",
        "--output", "logits",
        "--csv", "report.csv",
    };

    const auto valid_count = static_cast<int>(
        sizeof(valid_arguments) / sizeof(valid_arguments[0]));
    const gnni::cli::CliOptions options =
        gnni::cli::parse_cli(valid_count, valid_arguments);

    if (options.graph_path != "graph.bin" ||
        options.features_path != "features.bin" ||
        options.model_path != "model.bin" ||
        options.backend != gnni::BackendKind::CpuOpenMP ||
        options.input_format != gnni::InputFileFormat::Binary ||
        options.inference.num_threads != 4 ||
        options.inference.openmp_schedule !=
            gnni::OpenMPScheduleKind::Guided ||
        options.inference.openmp_chunk_size != 8 ||
        options.benchmark.warmup_runs != 0 ||
        options.benchmark.measured_runs != 3 ||
        !options.verify ||
        !options.csv_output_path.has_value()) {
        std::cerr << "valid CLI arguments were parsed incorrectly\n";
        return 1;
    }

    const char* duplicate_arguments[] = {
        "gnni",
        "--graph", "a.bin",
        "--graph", "b.bin",
        "--features", "features.bin",
        "--model", "model.bin",
    };
    const auto duplicate_count = static_cast<int>(
        sizeof(duplicate_arguments) / sizeof(duplicate_arguments[0]));
    if (!parse_throws_invalid_argument(
            duplicate_count, duplicate_arguments)) {
        std::cerr << "duplicate CLI option was not rejected\n";
        return 1;
    }

    const char* missing_arguments[] = {
        "gnni",
        "--graph", "graph.bin",
        "--features", "features.bin",
    };
    const auto missing_count = static_cast<int>(
        sizeof(missing_arguments) / sizeof(missing_arguments[0]));
    if (!parse_throws_invalid_argument(missing_count, missing_arguments)) {
        std::cerr << "missing required CLI option was not rejected\n";
        return 1;
    }

    const char* help_arguments[] = {"gnni", "--help"};
    const gnni::cli::CliOptions help =
        gnni::cli::parse_cli(2, help_arguments);
    if (!help.show_help) {
        std::cerr << "--help was not parsed correctly\n";
        return 1;
    }

    std::cout << "PASS: CLI option parsing\n";
    return 0;
}
