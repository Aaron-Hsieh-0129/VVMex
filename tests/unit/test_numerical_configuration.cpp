#include "utils/NumericalConfigurationValidation.hpp"

#include <mpi.h>

#include <cstdio>
#include <functional>
#include <limits>
#include <stdexcept>

namespace {

using VVM::Utils::NumericalConfigurationValues;
using VVM::Utils::validate_numerical_configuration_values;

int failures = 0;

NumericalConfigurationValues valid_values() {
    return {64, 48, 32, 2, 100.0, 100.0, 50.0, 1.0, 120.0, 60.0};
}

void check(bool condition, const char* message) {
    if (!condition) {
        std::fprintf(stderr, "FAIL: %s\n", message);
        ++failures;
    }
}

void expect_invalid(
    const char* message,
    const std::function<void(NumericalConfigurationValues&)>& mutate,
    int compute_ranks = 4) {
    auto values = valid_values();
    mutate(values);
    try {
        validate_numerical_configuration_values(values, compute_ranks);
        check(false, message);
    } catch (const std::runtime_error&) {
    }
}

} // namespace

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    try {
        validate_numerical_configuration_values(valid_values(), 4);
    } catch (const std::exception& error) {
        std::fprintf(stderr, "FAIL: valid configuration rejected: %s\n", error.what());
        ++failures;
    }

    expect_invalid("zero halo width accepted", [](auto& v) { v.halo_width = 0; });
    expect_invalid("one-cell halo accepted", [](auto& v) { v.halo_width = 1; });
    expect_invalid("zero grid spacing accepted", [](auto& v) { v.dx = 0.0; });
    expect_invalid("non-finite grid spacing accepted", [](auto& v) {
        v.dz = std::numeric_limits<double>::quiet_NaN();
    });
    expect_invalid("zero timestep accepted", [](auto& v) { v.dt = 0.0; });
    expect_invalid("negative timestep accepted", [](auto& v) { v.dt = -1.0; });
    expect_invalid("zero output interval accepted", [](auto& v) { v.output_interval = 0.0; });
    expect_invalid("negative total time accepted", [](auto& v) { v.total_time = -1.0; });
    expect_invalid("zero grid dimension accepted", [](auto& v) { v.nx = 0; });
    expect_invalid("local domain narrower than halo accepted", [](auto& v) {
        v.nx = 3;
        v.ny = 1;
    }, 2);

    auto column = valid_values();
    column.nx = 1;
    column.ny = 1;
    try {
        validate_numerical_configuration_values(column, 2);
        check(false, "multi-rank 1x1 horizontal domain accepted");
    } catch (const std::runtime_error&) {
    }

    if (failures == 0) {
        std::puts("test_numerical_configuration: PASS");
    }
    MPI_Finalize();
    return failures == 0 ? 0 : 1;
}
