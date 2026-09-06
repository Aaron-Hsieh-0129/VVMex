#include "core/Field.hpp"
#include "core/Grid.hpp"
#include "core/geometry/RegularLatLonGeometry.hpp"
#include "core/haloexchange/HaloExchanger.hpp"
#include "dynamics/solvers/HorizontalEllipticSolver.hpp"
#include "dynamics/solvers/HorizontalWindColumnRecovery.hpp"
#include "utils/ConfigurationManager.hpp"

#include <Kokkos_Core.hpp>
#include <mpi.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>

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
using VVM::Core::Geometry::RegularLatLonGeometry;
using VVM::Dynamics::HorizontalEllipticSolver;
using VVM::Dynamics::HorizontalWindColumnRecovery;
using VVM::Utils::ConfigurationManager;

template<std::size_t Dim>
auto snapshot(const Field<Dim>& field) {
    // create_mirror always allocates separate storage, including on CPU.
    auto result = Kokkos::create_mirror(field.get_device_data());
    Kokkos::deep_copy(result, field.get_device_data());
    return result;
}

bool same_bits(Real first, Real second) {
    return std::memcmp(&first, &second, sizeof(Real)) == 0;
}

template<std::size_t Dim>
bool unchanged(const Field<Dim>& field, const typename Field<Dim>::HostMirrorType& before) {
    const auto after = field.get_host_data();

    if constexpr (Dim == 1) {
        for (std::size_t i = 0; i < after.extent(0); ++i) {
            if (!same_bits(before(i), after(i))) return false;
        }
    } else if constexpr (Dim == 2) {
        for (std::size_t j = 0; j < after.extent(0); ++j) {
            for (std::size_t i = 0; i < after.extent(1); ++i) {
                if (!same_bits(before(j, i), after(j, i))) return false;
            }
        }
    } else if constexpr (Dim == 3) {
        for (std::size_t k = 0; k < after.extent(0); ++k) {
            for (std::size_t j = 0; j < after.extent(1); ++j) {
                for (std::size_t i = 0; i < after.extent(2); ++i) {
                    if (!same_bits(before(k, j, i), after(k, j, i))) return false;
                }
            }
        }
    } else {
        static_assert(Dim >= 1 && Dim <= 3, "Unsupported field dimension.");
    }

    return true;
}

int run_case(const Grid& grid, HorizontalEllipticSolver& paired_solver,
    HorizontalEllipticSolver& reference_solver, int mode, int iterations, bool stretched) {

    const int h = grid.get_halo_cells();
    const int nx = grid.get_local_total_points_x();
    const int ny = grid.get_local_total_points_y();
    const int start_i = grid.get_local_physical_start_x();
    const int start_j = grid.get_local_physical_start_y();
    const int global_nx = grid.get_global_points_x();
    const int global_ny = grid.get_global_points_y();

    // Synthetic vertical columns isolate recovery from Parameters/Initializer.
    constexpr int nz = 6;
    constexpr int top = 4;
    const Real sentinel = real(-12345.0);

    const bool spherical = grid.geometry().kind() == GeometryKind::RegularLatLon;
    double radius = 1.0;
    double south_edge = 0.0;

    if (spherical) {
        const auto& geometry = static_cast<const RegularLatLonGeometry&>(grid.geometry());
        radius = static_cast<double>(geometry.radius());
        south_edge = static_cast<double>(geometry.latitude_south_edge());
    }

    const double dq1 = static_cast<double>(grid.geometry().dq1());
    const double dq2 = static_cast<double>(grid.geometry().dq2());
    const double two_pi = 2.0 * std::acos(-1.0);

    Field<2> rhs_psi("solver_column_rhs_psi", {ny, nx});
    Field<2> rhs_chi("solver_column_rhs_chi", {ny, nx});
    Field<2> current_psi("solver_column_current_psi", {ny, nx});
    Field<2> current_chi("solver_column_current_chi", {ny, nx});
    Field<2> previous_psi("solver_column_previous_psi", {ny, nx});
    Field<2> previous_chi("solver_column_previous_chi", {ny, nx});

    Field<2> solved_psi("solver_column_solved_psi", {ny, nx});
    Field<2> solved_chi("solver_column_solved_chi", {ny, nx});
    Field<2> reference_psi("solver_column_reference_psi", {ny, nx});
    Field<2> reference_chi("solver_column_reference_chi", {ny, nx});

    Field<3> w("solver_column_w", {nz, ny, nx});
    Field<3> omega1("solver_column_omega1", {nz, ny, nx});
    Field<3> omega2("solver_column_omega2", {nz, ny, nx});
    Field<3> output1("solver_column_output1", {nz, ny, nx});
    Field<3> output2("solver_column_output2", {nz, ny, nx});
    Field<1> spacing("solver_column_spacing", {nz - 1});

    const HorizontalWindColumnRecovery recovery(grid.geometry());

    const std::array<Field<2>*, 6> source_fields = {
        &rhs_psi, &rhs_chi, &current_psi, &current_chi, &previous_psi, &previous_chi
    };

    std::array<Field<2>::HostMirrorType, 6> source_values;

    for (std::size_t n = 0; n < source_fields.size(); ++n) {
        source_values[n] = snapshot(*source_fields[n]);
    }

    auto w_values = snapshot(w);
    auto omega1_values = snapshot(omega1);
    auto omega2_values = snapshot(omega2);
    auto spacing_values = snapshot(spacing);
    auto reference_psi_guess = snapshot(reference_psi);
    auto reference_chi_guess = snapshot(reference_chi);

    const double psi_amplitude = (mode & 1) ? 1000.0 : 0.0;
    const double chi_amplitude = (mode & 2) ? 700.0 : 0.0;
    const double active = mode == 0 ? 0.0 : 1.0;

    for (int j = 0; j < ny; ++j) {
        const double global_j = static_cast<double>(start_j + j - h);
        const double angle_y = two_pi * (global_j + 0.5) / global_ny;

        for (int i = 0; i < nx; ++i) {
            const double global_i = static_cast<double>(start_i + i - h);
            const double angle_z = two_pi * (global_i + 1.0) / global_nx;
            const double angle_t = two_pi * (global_i + 0.5) / global_nx;

            // Longitude modes have zero discrete zonal mean.
            // Sources are physical, not pre-multiplied by the Jacobian.
            source_values[0](j, i) = static_cast<Real>((mode & 1) ? 1e-10 * std::sin(angle_z) : 0.0);
            source_values[1](j, i) = static_cast<Real>((mode & 2) ? -6e-11 * std::cos(2.0 * angle_t) : 0.0);

            source_values[2](j, i) = static_cast<Real>(psi_amplitude * (std::sin(angle_z) * std::cos(angle_y) + 0.3));
            source_values[3](j, i) = static_cast<Real>(chi_amplitude * (std::cos(angle_t) * std::sin(angle_y) - 0.2));
            source_values[4](j, i) = static_cast<Real>(psi_amplitude * (0.8 * std::cos(angle_z) * std::cos(angle_y) - 0.1));
            source_values[5](j, i) = static_cast<Real>(chi_amplitude * (0.6 * std::sin(angle_t) * std::sin(angle_y) + 0.15));

            // Independent reference for the initial extrapolation.
            reference_psi_guess(j, i) = real(2.0) * source_values[2](j, i) - source_values[4](j, i);
            reference_chi_guess(j, i) = real(2.0) * source_values[3](j, i) - source_values[5](j, i);

            for (int k = 0; k < nz; ++k) {
                const double level = static_cast<double>(k);

                w_values(k, j, i) = static_cast<Real>(active * 0.1 * ((1.0 + 0.125 * level) * std::sin(angle_t) + 0.3 * std::cos(angle_y)));
                omega1_values(k, j, i) = static_cast<Real>(active * 2e-4 / radius * (1.0 + 0.125 * level + 0.2 * std::cos(angle_t)));
                omega2_values(k, j, i) = static_cast<Real>(active * -3e-4 / radius * (1.0 + 0.0625 * level + 0.1 * std::sin(angle_y)));
            }
        }
    }

    for (int k = 0; k < nz - 1; ++k) {
        spacing_values(k) = stretched ? real(40.0) * static_cast<Real>(k + 1) : real(100.0);
    }

    for (std::size_t n = 0; n < source_fields.size(); ++n) {
        Kokkos::deep_copy(source_fields[n]->get_mutable_device_data(), source_values[n]);
    }

    Kokkos::deep_copy(w.get_mutable_device_data(), w_values);
    Kokkos::deep_copy(omega1.get_mutable_device_data(), omega1_values);
    Kokkos::deep_copy(omega2.get_mutable_device_data(), omega2_values);
    Kokkos::deep_copy(spacing.get_mutable_device_data(), spacing_values);
    Kokkos::deep_copy(reference_psi.get_mutable_device_data(), reference_psi_guess);
    Kokkos::deep_copy(reference_chi.get_mutable_device_data(), reference_chi_guess);

    HorizontalEllipticSolver::Options options;
    options.iterations = iterations;
    options.diagonal_shift = real(0.25);
    options.refresh_initial_halos = true;

    paired_solver.make_extrapolated_guess(current_psi, previous_psi, solved_psi);
    paired_solver.make_extrapolated_guess(current_chi, previous_chi, solved_chi);
    paired_solver.solve_at_z_and_t(rhs_psi, solved_psi, rhs_chi, solved_chi, options);

    // These existing solves provide the fixed-iteration reference.
    // No requirement to converge to prescribed analytic potentials is imposed.
    reference_solver.solve_at_z(rhs_psi, reference_psi, options);
    reference_solver.solve_at_t(rhs_chi, reference_chi, options);
    Kokkos::fence();

    const auto psi_reference = reference_psi.get_host_data();
    const auto chi_reference = reference_chi.get_host_data();

    // Save the returned potentials, including halos, before column recovery.
    const auto psi_before_recovery = snapshot(solved_psi);
    const auto chi_before_recovery = snapshot(solved_chi);

    int failures = 0;

    for (int bottom : {0, 2, top}) {
        Kokkos::deep_copy(output1.get_mutable_device_data(), sentinel);
        Kokkos::deep_copy(output2.get_mutable_device_data(), sentinel);

        // Use the solver's returned one-ring potential halos directly.
        recovery.recover(solved_psi, solved_chi, w, omega1, omega2, spacing, output1, output2, bottom, top);
        Kokkos::fence();

        const auto actual1 = output1.get_host_data();
        const auto actual2 = output2.get_host_data();

        // Components: psi, chi, physical eastward wind, physical northward wind.
        std::array<double, 4> errors = {};
        std::array<double, 4> scales = {};
        bool finite = true;
        bool untouched = true;
        bool zero_result = true;

        const auto compare = [&](double actual, double expected, std::size_t component) {
            if (!std::isfinite(actual) || !std::isfinite(expected)) {
                finite = false;
                return;
            }

            errors[component] = std::max(errors[component], std::abs(actual - expected));
            scales[component] = std::max(scales[component], std::abs(expected));
        };

        for (int j = h; j < ny - h; ++j) {
            const double global_j = static_cast<double>(start_j + j - h);
            const double phi_u = south_edge + (global_j + 0.5) * dq2;
            const double phi_v = south_edge + (global_j + 1.0) * dq2;
            const double cos_u = spherical ? std::cos(phi_u) : 1.0;
            const double cos_v = spherical ? std::cos(phi_v) : 1.0;

            const double h1_u = radius * cos_u;
            const double h2 = radius;
            const double jacobian_u = radius * radius * cos_u;
            const double jacobian_v = radius * radius * cos_v;

            for (int i = h; i < nx - h; ++i) {
                compare(static_cast<double>(psi_before_recovery(j, i)), static_cast<double>(psi_reference(j, i)), 0);
                compare(static_cast<double>(chi_before_recovery(j, i)), static_cast<double>(chi_reference(j, i)), 1);

                const double dpsi_q2 = (static_cast<double>(psi_reference(j, i)) - static_cast<double>(psi_reference(j - 1, i))) / dq2;
                const double dpsi_q1 = (static_cast<double>(psi_reference(j, i)) - static_cast<double>(psi_reference(j, i - 1))) / dq1;
                const double dchi_q1 = (static_cast<double>(chi_reference(j, i + 1)) - static_cast<double>(chi_reference(j, i))) / dq1;
                const double dchi_q2 = (static_cast<double>(chi_reference(j + 1, i)) - static_cast<double>(chi_reference(j, i))) / dq2;

                // Explicit Cartesian/RLL covariant reconstruction.
                // No production reconstruction or vorticity helper is called here.
                double expected1 = -cos_u * dpsi_q2 + dchi_q1;
                double expected2 = dpsi_q1 / cos_v + dchi_q2;

                compare(static_cast<double>(actual1(top, j, i)) / h1_u, expected1 / h1_u, 2);
                compare(static_cast<double>(actual2(top, j, i)) / h2, expected2 / h2, 3);

                for (int k = top - 1; k >= bottom; --k) {
                    const double dw_q1 = (static_cast<double>(w_values(k, j, i + 1)) - static_cast<double>(w_values(k, j, i))) / dq1;
                    const double dw_q2 = (static_cast<double>(w_values(k, j + 1, i)) - static_cast<double>(w_values(k, j, i))) / dq2;
                    const double dz = static_cast<double>(spacing_values(k));

                    expected1 -= (dw_q1 + jacobian_u * static_cast<double>(omega2_values(k, j, i))) * dz;
                    expected2 -= (dw_q2 - jacobian_v * static_cast<double>(omega1_values(k, j, i))) * dz;

                    compare(static_cast<double>(actual1(k, j, i)) / h1_u, expected1 / h1_u, 2);
                    compare(static_cast<double>(actual2(k, j, i)) / h2, expected2 / h2, 3);
                }
            }
        }

        for (int k = 0; k < nz; ++k) {
            for (int j = 0; j < ny; ++j) {
                for (int i = 0; i < nx; ++i) {
                    const bool written = k >= bottom && k <= top && j >= h && j < ny - h && i >= h && i < nx - h;

                    if (!written) {
                        untouched = untouched && actual1(k, j, i) == sentinel && actual2(k, j, i) == sentinel;
                    } else if (mode == 0) {
                        zero_result = zero_result && actual1(k, j, i) == real(0.0) && actual2(k, j, i) == real(0.0);
                    }
                }
            }
        }

        bool inputs = unchanged(w, w_values) && unchanged(omega1, omega1_values);
        inputs = inputs && unchanged(omega2, omega2_values) && unchanged(spacing, spacing_values);
        inputs = inputs && unchanged(solved_psi, psi_before_recovery) && unchanged(solved_chi, chi_before_recovery);

        for (std::size_t n = 0; n < source_fields.size(); ++n) {
            inputs = inputs && unchanged(*source_fields[n], source_values[n]);
        }

        const double tolerance = sizeof(Real) == sizeof(float) ? 5e-4 : 5e-10;
        bool passed = finite && untouched && inputs && zero_result;

        for (std::size_t component = 0; component < errors.size(); ++component) {
            errors[component] /= std::max(1e-12, scales[component]);
            passed = passed && errors[component] <= tolerance;
        }

        std::printf("%s mode=%d iterations=%d stretched=%d bottom=%d psi=%.3e chi=%.3e wind_e=%.3e wind_n=%.3e inputs=%d untouched=%d zero=%d %s\n",
            grid.geometry().name(), mode, iterations, static_cast<int>(stretched), bottom,
            errors[0], errors[1], errors[2], errors[3], static_cast<int>(inputs),
            static_cast<int>(untouched), static_cast<int>(zero_result), passed ? "PASS" : "FAIL");

        if (!passed) ++failures;
    }

    return failures;
}

int run_tests(const Grid& grid, HaloExchanger& halo_exchanger) {
    HorizontalEllipticSolver paired_solver(grid, halo_exchanger);
    HorizontalEllipticSolver reference_solver(grid, halo_exchanger);

    int failures = 0;

    // Reuse solver scratch storage; finish with zero inputs after nonzero cases.
    for (int mode : {1, 2, 3, 0}) {
        for (int iterations : {1, 4}) {
            for (bool stretched : {false, true}) {
                failures += run_case(grid, paired_solver, reference_solver, mode, iterations, stretched);
            }
        }
    }

    Kokkos::fence();
    return failures;
}

#if defined(ENABLE_NCCL)
void require_nccl(ncclResult_t status, const char* operation) {
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
        int size = 0;
        MPI_Comm_size(MPI_COMM_WORLD, &size);

        if (size != 1) {
            throw std::invalid_argument("This integration test currently requires one MPI rank.");
        }

        if (argc < 2) {
            throw std::invalid_argument("Usage: test_horizontal_wind_solver_column <configuration.json>");
        }

        const ConfigurationManager config(argv[1]);
        Grid grid(config);

#if defined(ENABLE_NCCL)
        ncclUniqueId id;
        require_nccl(ncclGetUniqueId(&id), "Create NCCL identifier");

        ncclComm_t communication = nullptr;
        require_nccl(ncclCommInitRank(&communication, 1, id, 0), "Initialize NCCL communicator");

        try {
            const cudaStream_t stream = Kokkos::Cuda().cuda_stream();
            HaloExchanger halo_exchanger(config, grid, communication, stream);
            failures = run_tests(grid, halo_exchanger);
        } catch (...) {
            ncclCommAbort(communication);
            throw;
        }

        require_nccl(ncclCommDestroy(communication), "Destroy NCCL communicator");
#else
        HaloExchanger halo_exchanger(grid);
        failures = run_tests(grid, halo_exchanger);
#endif
    } catch (const std::exception& error) {
        std::fprintf(stderr, "test_horizontal_wind_solver_column: %s\n", error.what());
        failures = 1;
    }

    int global_failures = 0;
    MPI_Allreduce(&failures, &global_failures, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

    Kokkos::finalize();
    MPI_Finalize();

    return global_failures == 0 ? 0 : 1;
}
