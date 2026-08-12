#include "cli_options.hpp"

#include <ostream>
#include <stdexcept>

namespace gnni::cli {

CliOptions parse_cli(int argc, const char* const argv[]) {
    static_cast<void>(argc);
    static_cast<void>(argv);
    // TODO(Role A): parse every option documented in semantic_contract.md,
    // reject duplicate/conflicting options, and validate numeric ranges. Keep
    // parsing free of file I/O so it can be unit tested independently.
    throw std::logic_error("parse_cli is not implemented");
}

void print_help(std::ostream& output) {
    output
        << "Usage: gnni --graph FILE --features FILE --model FILE [options]\n"
        << "\n"
        << "Core options:\n"
        << "  --backend cpu|omp|cuda-vertex|cuda-edge\n"
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
    static_cast<void>(argc);
    static_cast<void>(argv);
    // TODO(Role A): compose existing modules only. Do not duplicate parsing,
    // loader, validation, comparison, or benchmark logic in this function.
    throw std::logic_error("run_cli workflow is not implemented");
}

}  // namespace gnni::cli

