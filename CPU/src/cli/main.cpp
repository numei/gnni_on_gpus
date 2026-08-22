#include "cli_options.hpp"

#include <exception>
#include <iostream>

/** Translate uncaught workflow exceptions into a concise CLI error and code. */
int main(int argc, char* argv[]) {
    try {
        return gnni::cli::run_cli(argc, const_cast<const char* const*>(argv));
    } catch (const std::exception& error) {
        std::cerr << "gnni: " << error.what() << '\n';
        return 1;
    }
}

