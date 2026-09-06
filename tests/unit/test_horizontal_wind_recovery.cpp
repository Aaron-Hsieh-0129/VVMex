#include "core/Field.hpp"
#include "core/Grid.hpp"
#include "core/geometry/RegularLatLonGeometry.hpp"
#include "core/haloexchange/HaloExchanger.hpp"
#include "dynamics/operators/HorizontalWindReconstruction.hpp"
#include "dynamics/solvers/HorizontalEllipticSolver.hpp"
#include "utils/ConfigurationManager.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <exception>
#include <stdexcept>
#include <string>

#include <Kokkos_Core.hpp>
#include <mpi.h>

#if defined(ENABLE_NCCL)
#include <cuda_runtime.h>
#include <nccl.h>
#endif

namespace {

using VVM::Real;
using VVM::real;
using VVM::Core::Field;
using VVM::Core::Grid;
using VVM::Core::HaloExchanger;
using VVM::Core::Geometry::GeometryKind;
using VVM::Core::Geometry::HorizontalLocation;
using VVM::Core::Geometry::RegularLatLonGeometry;
using VVM::Dynamics::HorizontalEllipticSolver;
using VVM::Dynamics::Operators::make_horizontal_wind_reconstruction_device_view;
using VVM::Utils::ConfigurationManager;

enum class TestMode {
    Rotational,
    Divergent,
    Mixed,
    Zero
};

const char* mode_name(const TestMode mode) {
    switch (mode) {
        case TestMode::Rotational:
            return "rotational";
        case TestMode::Divergent:
            return "divergent";
        case TestMode::Mixed:
            return "mixed";
        case TestMode::Zero:
            return "zero";
    }

    return "unknown";
}

int run_case(const Grid& grid, HorizontalEllipticSolver& paired_solver,
    HorizontalEllipticSolver& reference_solver, const TestMode mode, const int iterations) {

    const int h = grid.get_halo_cells();
    const int ny = grid.get_local_total_points_y();
    const int nx = grid.get_local_total_points_x();
    const int global_start_i = grid.get_local_physical_start_x();
    const int global_start_j = grid.get_local_physical_start_y();
    const int global_nx = grid.get_global_points_x();

    Field<2> rhs_psi("recovery_rhs_psi", {ny, nx});
    Field<2> rhs_chi("recovery_rhs_chi", {ny, nx});
    Field<2> current_psi("recovery_current_psi", {ny, nx});
    Field<2> previous_psi("recovery_previous_psi", {ny, nx});
    Field<2> current_chi("recovery_current_chi", {ny, nx});
    Field<2> previous_chi("recovery_previous_chi", {ny, nx});

    Field<2> solved_psi("recovery_solved_psi", {ny, nx});
    Field<2> solved_chi("recovery_solved_chi", {ny, nx});
    Field<2> reference_psi("recovery_reference_psi", {ny, nx});
    Field<2> reference_chi("recovery_reference_chi", {ny, nx});

    const auto zeta = rhs_psi.get_mutable_device_data();
    const auto divergence_source = rhs_chi.get_mutable_device_data();
    const auto psi_current = current_psi.get_mutable_device_data();
    const auto psi_previous = previous_psi.get_mutable_device_data();
    const auto chi_current = current_chi.get_mutable_device_data();
    const auto chi_previous = previous_chi.get_mutable_device_data();

    const bool use_psi = mode == TestMode::Rotational || mode == TestMode::Mixed;
    const bool use_chi = mode == TestMode::Divergent || mode == TestMode::Mixed;
    const Real two_pi = real(2.0) * static_cast<Real>(std::acos(-1.0));
    const Real amplitude = real(1000.0) * static_cast<Real>(iterations);

    Kokkos::parallel_for("InitializeHorizontalWindRecovery",
        Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {ny, nx}),
        KOKKOS_LAMBDA(const int j, const int i) {
            const Real global_i = static_cast<Real>(global_start_i + i - h);
            const Real global_j = static_cast<Real>(global_start_j + j - h);
            const Real angle_z = two_pi * (global_i + real(1.0)) / static_cast<Real>(global_nx);
            const Real angle_t = two_pi * (global_i + real(0.5)) / static_cast<Real>(global_nx);

            if (use_psi) {
                zeta(j, i) = real(1.0e-10) * Kokkos::sin(angle_z);
                psi_current(j, i) = amplitude * (Kokkos::sin(angle_z) * Kokkos::cos(real(0.11) * global_j) + real(0.3));
                psi_previous(j, i) = amplitude * (real(0.8) * Kokkos::cos(angle_z) * Kokkos::cos(real(0.07) * global_j) - real(0.2));
            } else {
                zeta(j, i) = real(0.0);
                psi_current(j, i) = real(0.0);
                psi_previous(j, i) = real(0.0);
            }

            if (use_chi) {
                divergence_source(j, i) = real(-6.0e-11) * Kokkos::cos(real(2.0) * angle_t);
                chi_current(j, i) = amplitude * (real(-0.6) * Kokkos::cos(angle_t) * Kokkos::sin(real(0.13) * global_j) + real(0.2));
                chi_previous(j, i) = amplitude * (real(0.4) * Kokkos::sin(angle_t) * Kokkos::cos(real(0.09) * global_j) - real(0.1));
            } else {
                divergence_source(j, i) = real(0.0);
                chi_current(j, i) = real(0.0);
                chi_previous(j, i) = real(0.0);
            }
        });

    // Independent snapshots detect input modification on CPU as well as GPU.
    // create_mirror_view could alias host-accessible input storage.
    const std::array<const Field<2>*, 6> input_fields = {
        &rhs_psi, &rhs_chi,
        &current_psi, &previous_psi,
        &current_chi, &previous_chi
    };

    std::array<Field<2>::HostMirrorType, 6> input_snapshots;

    for (std::size_t n = 0; n < input_fields.size(); ++n) {
        input_snapshots[n] = Kokkos::create_mirror(input_fields[n]->get_device_data());
        Kokkos::deep_copy(input_snapshots[n], input_fields[n]->get_device_data());
    }

    HorizontalEllipticSolver::Options options;
    options.iterations = iterations;
    options.diagonal_shift = real(0.25);
    options.refresh_initial_halos = true;

    // Workflow under test: extrapolate both potentials, solve them together,
    // and reconstruct directly from the returned one-ring solution halos.
    paired_solver.make_extrapolated_guess(current_psi, previous_psi, solved_psi);
    paired_solver.make_extrapolated_guess(current_chi, previous_chi, solved_chi);

    // These are unweighted sources. The solver applies the Jacobian internally.
    paired_solver.solve_at_z_and_t(rhs_psi, solved_psi, rhs_chi, solved_chi, options);

    Kokkos::View<Real**> q1("recovery_contravariant_q1", ny, nx);
    Kokkos::View<Real**> q2("recovery_contravariant_q2", ny, nx);
    Kokkos::View<Real**> physical_q1("recovery_physical_q1", ny, nx);
    Kokkos::View<Real**> physical_q2("recovery_physical_q2", ny, nx);

    const auto reconstruction = make_horizontal_wind_reconstruction_device_view(grid.geometry());
    const auto physical_scale_q1 = grid.geometry().device_view(HorizontalLocation::U).contravariant_to_physical.a11;
    const auto physical_scale_q2 = grid.geometry().device_view(HorizontalLocation::V).contravariant_to_physical.a22;

    const auto psi = solved_psi.get_device_data();
    const auto chi = solved_chi.get_device_data();

    // Cartesian and RLL have diagonal physical-component transformations.
    // This test does not assume that for a future cubed-sphere implementation.
    Kokkos::parallel_for("ReconstructRecoveredHorizontalWind",
        Kokkos::MDRangePolicy<Kokkos::Rank<2>>({h, h}, {ny - h, nx - h}),
        KOKKOS_LAMBDA(const int j, const int i) {
            const Real value_q1 = reconstruction.calculate_contravariant_q1_at_u(psi, chi, j, i);
            const Real value_q2 = reconstruction.calculate_contravariant_q2_at_v(psi, chi, j, i);

            q1(j, i) = value_q1;
            q2(j, i) = value_q2;
            physical_q1(j, i) = physical_scale_q1(j, i) * value_q1;
            physical_q2(j, i) = physical_scale_q2(j, i) * value_q2;
        });

    // Reference workflow: use the existing separate T and Z solves.
    // This checks paired-solver integration, not solver convergence.
    reference_solver.make_extrapolated_guess(current_psi, previous_psi, reference_psi);
    reference_solver.make_extrapolated_guess(current_chi, previous_chi, reference_chi);

    reference_solver.solve_at_z(rhs_psi, reference_psi, options);
    reference_solver.solve_at_t(rhs_chi, reference_chi, options);

    const auto reference_psi_host = reference_psi.get_host_data();
    const auto reference_chi_host = reference_chi.get_host_data();
    const auto solved_psi_host = solved_psi.get_host_data();
    const auto solved_chi_host = solved_chi.get_host_data();

    const auto q1_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), q1);
    const auto q2_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), q2);
    const auto physical_q1_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), physical_q1);
    const auto physical_q2_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), physical_q2);

    bool inputs_unchanged = true;

    for (std::size_t n = 0; n < input_fields.size(); ++n) {
        const auto after = input_fields[n]->get_host_data();

        for (int j = 0; j < ny; ++j) {
            for (int i = 0; i < nx; ++i) {
                inputs_unchanged = inputs_unchanged && input_snapshots[n](j, i) == after(j, i);
            }
        }
    }

    const double dq1 = static_cast<double>(grid.geometry().dq1());
    const double dq2 = static_cast<double>(grid.geometry().dq2());
    const bool spherical = grid.geometry().kind() == GeometryKind::RegularLatLon;

    double radius = 1.0;
    double south_edge = 0.0;

    if (spherical) {
        const auto& geometry = static_cast<const RegularLatLonGeometry&>(grid.geometry());
        radius = static_cast<double>(geometry.radius());
        south_edge = static_cast<double>(geometry.latitude_south_edge());
    }

    // Components: psi, chi, contravariant q1/q2, physical q1/q2.
    std::array<double, 6> maximum_errors = {};
    std::array<double, 6> reference_scales = {};
    bool finite = true;

    for (int j = h; j < ny - h; ++j) {
        for (int i = h; i < nx - h; ++i) {
            const double dpsi_dq1 = (static_cast<double>(reference_psi_host(j, i)) - static_cast<double>(reference_psi_host(j, i - 1))) / dq1;
            const double dpsi_dq2 = (static_cast<double>(reference_psi_host(j, i)) - static_cast<double>(reference_psi_host(j - 1, i))) / dq2;
            const double dchi_dq1 = (static_cast<double>(reference_chi_host(j, i + 1)) - static_cast<double>(reference_chi_host(j, i))) / dq1;
            const double dchi_dq2 = (static_cast<double>(reference_chi_host(j + 1, i)) - static_cast<double>(reference_chi_host(j, i))) / dq2;

            double expected_q1 = -dpsi_dq2 + dchi_dq1;
            double expected_q2 = dpsi_dq1 + dchi_dq2;
            double expected_physical_q1 = expected_q1;
            double expected_physical_q2 = expected_q2;

            if (spherical) {
                const double global_j = static_cast<double>(global_start_j + j - h);
                const double latitude_u = south_edge + (global_j + 0.5) * dq2;
                const double latitude_v = south_edge + (global_j + 1.0) * dq2;
                const double cos_u = std::cos(latitude_u);
                const double cos_v = std::cos(latitude_v);
                const double radius_squared = radius * radius;

                expected_q1 = -dpsi_dq2 / (radius_squared * cos_u) + dchi_dq1 / (radius_squared * cos_u * cos_u);
                expected_q2 = dpsi_dq1 / (radius_squared * cos_v) + dchi_dq2 / radius_squared;

                expected_physical_q1 = -dpsi_dq2 / radius + dchi_dq1 / (radius * cos_u);
                expected_physical_q2 = dpsi_dq1 / (radius * cos_v) + dchi_dq2 / radius;
            }

            const std::array<double, 6> expected = {
                static_cast<double>(reference_psi_host(j, i)),
                static_cast<double>(reference_chi_host(j, i)),
                expected_q1,
                expected_q2,
                expected_physical_q1,
                expected_physical_q2
            };

            const std::array<double, 6> actual = {
                static_cast<double>(solved_psi_host(j, i)),
                static_cast<double>(solved_chi_host(j, i)),
                static_cast<double>(q1_host(j, i)),
                static_cast<double>(q2_host(j, i)),
                static_cast<double>(physical_q1_host(j, i)),
                static_cast<double>(physical_q2_host(j, i))
            };

            for (std::size_t component = 0; component < expected.size(); ++component) {
                finite = finite && std::isfinite(actual[component]) && std::isfinite(expected[component]);
                maximum_errors[component] = std::max(maximum_errors[component], std::abs(actual[component] - expected[component]));
                reference_scales[component] = std::max(reference_scales[component], std::abs(expected[component]));
            }
        }
    }

    const double tolerance = sizeof(Real) == sizeof(float) ? 5.0e-4 : 5.0e-11;
    bool passed = finite && inputs_unchanged;

    for (std::size_t component = 0; component < maximum_errors.size(); ++component) {
        maximum_errors[component] /= std::max(1.0e-30, reference_scales[component]);
        passed = passed && maximum_errors[component] <= tolerance;
    }

    std::printf("%s mode=%s iterations=%d psi=%.3e chi=%.3e q1=%.3e q2=%.3e physical_q1=%.3e physical_q2=%.3e inputs_unchanged=%d %s\n",
        grid.geometry().name(), mode_name(mode), iterations,
        maximum_errors[0], maximum_errors[1], maximum_errors[2],
        maximum_errors[3], maximum_errors[4], maximum_errors[5],
        inputs_unchanged ? 1 : 0, passed ? "PASS" : "FAIL");

    return passed ? 0 : 1;
}

int run_tests(const Grid& grid, HaloExchanger& halo_exchanger) {
    HorizontalEllipticSolver paired_solver(grid, halo_exchanger);
    HorizontalEllipticSolver reference_solver(grid, halo_exchanger);

    const std::array<TestMode, 3> modes = {
        TestMode::Rotational,
        TestMode::Divergent,
        TestMode::Mixed
    };

    // Exercise odd/even scratch-buffer paths without a convergence criterion.
    const std::array<int, 2> iteration_counts = {1, 4};

    int failures = 0;

    for (const TestMode mode : modes) {
        for (const int iterations : iteration_counts) {
            failures += run_case(grid, paired_solver, reference_solver, mode, iterations);
        }
    }

    // Reuse solver scratch storage after nonzero cases.
    failures += run_case(grid, paired_solver, reference_solver, TestMode::Zero, 3);

    Kokkos::fence();
    return failures;
}

#if defined(ENABLE_NCCL)
void require_nccl(const ncclResult_t status, const char* operation) {
    if (status != ncclSuccess) {
        throw std::runtime_error(std::string(operation) + ": " + ncclGetErrorString(status));
    }
}
#endif

} // namespace

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);
    Kokkos::initialize(argc, argv);

    int failures = 0;

    try {
        int mpi_size = 0;
        MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);

        if (mpi_size != 1) {
            throw std::invalid_argument("This integration test currently requires one MPI rank.");
        }

        if (argc < 2) {
            throw std::invalid_argument("Usage: test_horizontal_wind_recovery <configuration.json>");
        }

        const ConfigurationManager config(argv[1]);
        Grid grid(config);

#if defined(ENABLE_NCCL)
        ncclUniqueId nccl_id;
        require_nccl(ncclGetUniqueId(&nccl_id), "Create NCCL identifier");

        ncclComm_t nccl_comm = nullptr;
        require_nccl(ncclCommInitRank(&nccl_comm, 1, nccl_id, 0), "Initialize NCCL communicator");

        try {
            const cudaStream_t stream = Kokkos::Cuda().cuda_stream();
            HaloExchanger halo_exchanger(config, grid, nccl_comm, stream);
            failures = run_tests(grid, halo_exchanger);
        } catch (...) {
            ncclCommAbort(nccl_comm);
            throw;
        }

        require_nccl(ncclCommDestroy(nccl_comm), "Destroy NCCL communicator");
#else
        HaloExchanger halo_exchanger(grid);
        failures = run_tests(grid, halo_exchanger);
#endif
    } catch (const std::exception& error) {
        std::fprintf(stderr, "test_horizontal_wind_recovery: %s\n", error.what());
        failures = 1;
    }

    int global_failures = 0;
    MPI_Allreduce(&failures, &global_failures, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

    Kokkos::finalize();
    MPI_Finalize();

    return global_failures == 0 ? 0 : 1;
}
