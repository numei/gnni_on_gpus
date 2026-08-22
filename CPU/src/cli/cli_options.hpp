#pragma once

#include "gnni/api.hpp"
#include "gnni/benchmark.hpp"
#include "gnni/io.hpp"

#include <filesystem>
#include <iosfwd>
#include <optional>

namespace gnni::cli {

struct CliOptions {
    std::filesystem::path graph_path;
    std::filesystem::path features_path;
    std::filesystem::path model_path;
    InputFileFormat input_format = InputFileFormat::Auto;
    BackendKind backend = BackendKind::CpuSequential;
    InferenceConfig inference;
    BenchmarkOptions benchmark;
    bool verify = false;
    ComparisonOptions comparison;
    std::optional<std::filesystem::path> csv_output_path;
    bool show_help = false;
};

/**
 * Parse command-line arguments without loading data or executing a backend.
 *
 * @throws std::invalid_argument for unknown, missing, duplicate, or malformed
 * arguments.
 */
CliOptions parse_cli(int argc, const char* const argv[]);

/** Print all supported command-line options and their defaults. */
void print_help(std::ostream& output);

/**
 * Run the complete host workflow: parse, load, benchmark, optionally verify,
 * and report. Returns a process exit code and does not call exit().
 */
int run_cli(int argc, const char* const argv[]);

}  // namespace gnni::cli
