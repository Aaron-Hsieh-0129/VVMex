#include "core/Grid.hpp"
#include "core/Parameters.hpp"
#include "core/geometry/GeometryKind.hpp"
#include "utils/ConfigurationManager.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <exception>
#include <limits>
#include <stdexcept>
#include <string>

#include <Kokkos_Core.hpp>
#include <mpi.h>

namespace {

using VVM::Real;
using VVM::real;
using VVM::Core::Grid;
using VVM::Core::Parameters;
using VVM::Core::Geometry::GeometryKind;
using VVM::Utils::ConfigurationManager;

int failures = 0;

void check(const bool condition, const char* message) {
    if (!condition) {
        ++failures;
        std::fprintf(stderr, "FAIL: %s\n", message);
    }
}

void check_value(const Parameters& parameters, const Kokkos::View<Real>& field, const Real expected, const char* name) {
    const Real actual = parameters.get_value_host(field);
    const Real scale = std::max(std::abs(expected), std::numeric_limits<Real>::min());
    const Real tolerance = real(16.0) * std::numeric_limits<Real>::epsilon() * scale;

    if (!std::isfinite(actual) || std::abs(actual - expected) > tolerance) {
        ++failures;
        std::fprintf(stderr, "FAIL: %s actual=%.17e expected=%.17e\n",
            name, static_cast<double>(actual), static_cast<double>(expected));
    }
}

void run_tests(const ConfigurationManager& config, const Grid& grid) {
    if (grid.geometry().kind() != GeometryKind::Cartesian) {
        bool rejected_for_units = false;

        try {
            Parameters parameters(config, grid);
        } catch (const std::runtime_error& error) {
            rejected_for_units = std::string(error.what()).find("Cartesian dx/dy") != std::string::npos;
        }

        check(rejected_for_units, "Non-Cartesian Parameters must reject angular spacing entering Cartesian dx/dy.");
        return;
    }

    const Parameters parameters(config, grid);

    // All three Cartesian fixtures resolve to these values. The mixed
    // fixture deliberately contains different legacy spacing values.
    const Real expected_dx = real(500.0);
    const Real expected_dy = real(750.0);
    const Real expected_dz = real(1000.0);

    check(grid.horizontal_specification().geometry.dq1 == expected_dx, "Grid must resolve dx to 500 m.");
    check(grid.horizontal_specification().geometry.dq2 == expected_dy, "Grid must resolve dy to 750 m.");
    check(grid.vertical_specification().dz == expected_dz, "Grid must resolve reference dz to 1000 m.");

    check_value(parameters, parameters.dx, expected_dx, "dx");
    check_value(parameters, parameters.dy, expected_dy, "dy");
    check_value(parameters, parameters.dz, expected_dz, "dz");

    check_value(parameters, parameters.rdx, real(1.0) / expected_dx, "rdx");
    check_value(parameters, parameters.rdy, real(1.0) / expected_dy, "rdy");
    check_value(parameters, parameters.rdz, real(1.0) / expected_dz, "rdz");

    check_value(parameters, parameters.rdx2, real(1.0) / (expected_dx * expected_dx), "rdx2");
    check_value(parameters, parameters.rdy2, real(1.0) / (expected_dy * expected_dy), "rdy2");
    check_value(parameters, parameters.rdz2, real(1.0) / (expected_dz * expected_dz), "rdz2");

    check_value(parameters, parameters.dt, real(2.0), "dt");
    check_value(parameters, parameters.WRXMU, real(0.25), "WRXMU");
    check(parameters.solver_iteration == 4, "Solver iteration count must remain unchanged.");
}

} // namespace

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);
    Kokkos::initialize(argc, argv);

    try {
        if (argc < 2) {
            throw std::invalid_argument("Usage: test_parameters_resolved_grid <configuration.json>");
        }

        const ConfigurationManager config(argv[1]);
        const Grid grid(config);
        run_tests(config, grid);
    } catch (const std::exception& error) {
        ++failures;
        std::fprintf(stderr, "test_parameters_resolved_grid: %s\n", error.what());
    }

    int global_failures = 0;
    MPI_Allreduce(&failures, &global_failures, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

    if (global_failures == 0) {
        std::puts("test_parameters_resolved_grid: PASS");
    }

    Kokkos::finalize();
    MPI_Finalize();

    return global_failures == 0 ? 0 : 1;
}
