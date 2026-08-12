#include "gnni/benchmark.hpp"

#include <ostream>
#include <stdexcept>

namespace gnni {

void write_text_report(const BenchmarkReport& report, std::ostream& output) {
    static_cast<void>(report);
    static_cast<void>(output);
    // TODO(Role A): print backend/configuration, run count, timing summaries,
    // throughput definition, and optional GPU transfer/device-memory fields.
    throw std::logic_error("write_text_report is not implemented");
}

void write_csv_report(const BenchmarkReport& report, std::ostream& output) {
    static_cast<void>(report);
    static_cast<void>(output);
    // TODO(Role A): emit a stable schema suitable for plots. Document units in
    // column names and avoid locale-dependent decimal formatting.
    throw std::logic_error("write_csv_report is not implemented");
}

}  // namespace gnni

