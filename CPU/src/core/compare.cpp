#include "gnni/api.hpp"
#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <stdexcept>

namespace gnni_cpu {

ComparisonReport compare_outputs(
    const DenseMatrix& reference,
    const DenseMatrix& candidate,
    const ComparisonOptions& options) {
        if (!std::isfinite(options.absolute_tolerance) ||
            options.absolute_tolerance < 0.0) {
            throw std::invalid_argument(
                "absolute tolerance must be finite and non-negative"
            );
        }

        if (!std::isfinite(options.relative_tolerance) ||
            options.relative_tolerance < 0.0) {
            throw std::invalid_argument(
                "relative tolerance must be finite and non-negative"
            );
        }

        ComparisonReport report;

        const bool dimensions_match =
            reference.rows == candidate.rows &&
            reference.cols == candidate.cols;

        const bool storage_sizes_match =
            reference.values.size() ==
            candidate.values.size();

        report.shapes_match =
            dimensions_match && storage_sizes_match;

        if (!report.shapes_match) {
            return report;
        }

        // 两个相同的空矩阵可以认为完全一致。
        if (reference.values.empty()) {
            report.within_tolerance = true;
            return report;
        }

        double squared_error_sum = 0.0;

        for (std::size_t index = 0;
            index < reference.values.size();
            ++index) {

            const double reference_value =
                static_cast<double>(
                    reference.values[index]
                );

            const double candidate_value =
                static_cast<double>(
                    candidate.values[index]
                );

            bool element_matches = false;
            double absolute_error = 0.0;
            double relative_error = 0.0;

            // NaN 和 infinity 不能被认为是正确输出。
            if (!std::isfinite(reference_value) ||
                !std::isfinite(candidate_value)) {

                absolute_error =
                    std::numeric_limits<double>::infinity();

                relative_error =
                    std::numeric_limits<double>::infinity();

                squared_error_sum =
                    std::numeric_limits<double>::infinity();
            } else {
                absolute_error = std::abs(
                    candidate_value - reference_value
                );

                if (reference_value == 0.0) {
                    relative_error =
                        absolute_error == 0.0
                            ? 0.0
                            : std::numeric_limits<double>::infinity();
                } else {
                    relative_error =
                        absolute_error /
                        std::abs(reference_value);
                }

                const double allowed_error =
                    options.absolute_tolerance +
                    options.relative_tolerance *
                        std::abs(reference_value);

                element_matches =
                    absolute_error <= allowed_error;

                squared_error_sum +=
                    absolute_error * absolute_error;
            }

            report.max_absolute_error = std::max(
                report.max_absolute_error,
                absolute_error
            );

            report.max_relative_error = std::max(
                report.max_relative_error,
                relative_error
            );

            if (!element_matches) {
                ++report.mismatch_count;

                if (!report.first_mismatch_index.has_value()) {
                    report.first_mismatch_index =
                        static_cast<std::uint64_t>(index);
                }
            }
        }

        report.root_mean_square_error = std::sqrt(
            squared_error_sum /
            static_cast<double>(reference.values.size())
        );

        report.within_tolerance =
            report.mismatch_count == 0;

        return report;
    }

}  // namespace gnni_cpu

