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
#include <stdexcept>
#include <string>
#include <vector>

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

constexpr int nz = 6;
constexpr int bottom = 1;
constexpr int top = 4;
constexpr int wind_levels = top - bottom + 1;
const Real sentinel = real(-12345.0);

template<std::size_t Dim>
bool unchanged(const Field<Dim>& field, const typename Field<Dim>::HostMirrorType& before) {
    const auto after = field.get_host_data();

    if constexpr (Dim == 1) {
        for (std::size_t i = 0; i < after.extent(0); ++i) {
            if (before(i) != after(i)) return false;
        }
    } else if constexpr (Dim == 2) {
        for (std::size_t j = 0; j < after.extent(0); ++j) {
            for (std::size_t i = 0; i < after.extent(1); ++i) {
                if (before(j, i) != after(j, i)) return false;
            }
        }
    } else if constexpr (Dim == 3) {
        for (std::size_t k = 0; k < after.extent(0); ++k) {
            for (std::size_t j = 0; j < after.extent(1); ++j) {
                for (std::size_t i = 0; i < after.extent(2); ++i) {
                    if (before(k, j, i) != after(k, j, i)) return false;
                }
            }
        }
    } else {
        static_assert(Dim >= 1 && Dim <= 3, "Unsupported field dimension.");
    }

    return true;
}

struct Result {
    int nx = 0;
    int ny = 0;
    int start_i = 0;
    int start_j = 0;
    int mode = 0;
    int iterations = 0;
    bool valid = true;

    // psi, chi, physical eastward wind, physical northward wind.
    std::array<std::vector<double>, 4> fields;
};

Result run_case(const Grid& grid, HorizontalEllipticSolver& solver, int mode, int iterations) {
    const int h = grid.get_halo_cells();
    const int nx = grid.get_local_total_points_x();
    const int ny = grid.get_local_total_points_y();
    const int physical_nx = grid.get_local_physical_points_x();
    const int physical_ny = grid.get_local_physical_points_y();
    const int start_i = grid.get_local_physical_start_x();
    const int start_j = grid.get_local_physical_start_y();

    const bool spherical = grid.geometry().kind() == GeometryKind::RegularLatLon;
    double radius = 1.0;
    double south_edge = 0.0;

    if (spherical) {
        const auto& geometry = static_cast<const RegularLatLonGeometry&>(grid.geometry());
        radius = static_cast<double>(geometry.radius());
        south_edge = static_cast<double>(geometry.latitude_south_edge());
    }

    const double dq2 = static_cast<double>(grid.geometry().dq2());
    const double two_pi = 2.0 * std::acos(-1.0);
    const double active = mode == 0 ? 0.0 : 1.0;

    Field<2> rhs_psi("mpi_column_rhs_psi", {ny, nx});
    Field<2> rhs_chi("mpi_column_rhs_chi", {ny, nx});
    Field<2> psi("mpi_column_psi", {ny, nx});
    Field<2> chi("mpi_column_chi", {ny, nx});
    Field<3> w("mpi_column_w", {nz, ny, nx});
    Field<3> omega1("mpi_column_omega1", {nz, ny, nx});
    Field<3> omega2("mpi_column_omega2", {nz, ny, nx});
    Field<3> wind1("mpi_column_wind1", {nz, ny, nx});
    Field<3> wind2("mpi_column_wind2", {nz, ny, nx});
    Field<1> spacing("mpi_column_spacing", {nz - 1});

    const std::array<Field<2>*, 4> planes = {&rhs_psi, &rhs_chi, &psi, &chi};
    std::array<Field<2>::HostMirrorType, 4> host;

    for (std::size_t n = 0; n < planes.size(); ++n) {
        host[n] = Kokkos::create_mirror(planes[n]->get_device_data());
    }

    // Independent host allocations also serve as uploaded-input snapshots.
    auto w_host = Kokkos::create_mirror(w.get_device_data());
    auto omega1_host = Kokkos::create_mirror(omega1.get_device_data());
    auto omega2_host = Kokkos::create_mirror(omega2.get_device_data());
    auto spacing_host = Kokkos::create_mirror(spacing.get_device_data());

    for (int j = 0; j < ny; ++j) {
        const double global_j = static_cast<double>(start_j + j - h);
        const double angle_y = two_pi * (global_j + 0.5) / grid.get_global_points_y();

        for (int i = 0; i < nx; ++i) {
            const double global_i = static_cast<double>(start_i + i - h);
            const double angle_z = two_pi * (global_i + 1.0) / grid.get_global_points_x();
            const double angle_t = two_pi * (global_i + 0.5) / grid.get_global_points_x();

            host[0](j, i) = static_cast<Real>(active * 1e-10 * std::sin(angle_z));
            host[1](j, i) = static_cast<Real>(active * -6e-11 * std::cos(2.0 * angle_t));

            // Prescribed initial guesses, identical functions of global indices.
            host[2](j, i) = static_cast<Real>(active * 1000.0 * (std::sin(angle_z) * std::cos(angle_y) + 0.3));
            host[3](j, i) = static_cast<Real>(active * 700.0 * (std::cos(angle_t) * std::sin(angle_y) - 0.2));

            for (int k = 0; k < nz; ++k) {
                const double level = static_cast<double>(k);

                // Prescribed vertical-input halos are initialized analytically.
                // Their production MPI exchange is not under test here.
                w_host(k, j, i) = static_cast<Real>(active * 0.1 * ((1.0 + 0.125 * level) * std::sin(angle_t) + 0.3 * std::cos(angle_y)));
                omega1_host(k, j, i) = static_cast<Real>(active * 2e-4 / radius * (1.0 + 0.125 * level + 0.2 * std::cos(angle_t)));
                omega2_host(k, j, i) = static_cast<Real>(active * -3e-4 / radius * (1.0 + 0.0625 * level + 0.1 * std::sin(angle_y)));
            }
        }
    }

    for (int k = 0; k < nz - 1; ++k) {
        spacing_host(k) = mode == 2 ? real(40.0) * static_cast<Real>(k + 1) : real(100.0);
    }

    for (std::size_t n = 0; n < planes.size(); ++n) {
        Kokkos::deep_copy(planes[n]->get_mutable_device_data(), host[n]);
    }

    Kokkos::deep_copy(w.get_mutable_device_data(), w_host);
    Kokkos::deep_copy(omega1.get_mutable_device_data(), omega1_host);
    Kokkos::deep_copy(omega2.get_mutable_device_data(), omega2_host);
    Kokkos::deep_copy(spacing.get_mutable_device_data(), spacing_host);
    Kokkos::deep_copy(wind1.get_mutable_device_data(), sentinel);
    Kokkos::deep_copy(wind2.get_mutable_device_data(), sentinel);

    HorizontalEllipticSolver::Options options;
    options.iterations = iterations;
    options.diagonal_shift = real(0.25);
    options.refresh_initial_halos = true;

    solver.solve_at_z_and_t(rhs_psi, psi, rhs_chi, chi, options);

    const HorizontalWindColumnRecovery recovery(grid.geometry());
    recovery.recover(psi, chi, w, omega1, omega2, spacing, wind1, wind2, bottom, top);
    Kokkos::fence();

    Result result;
    result.nx = physical_nx;
    result.ny = physical_ny;
    result.start_i = start_i;
    result.start_j = start_j;
    result.mode = mode;
    result.iterations = iterations;

    result.valid = unchanged(rhs_psi, host[0]) && unchanged(rhs_chi, host[1]);
    result.valid = result.valid && unchanged(w, w_host) && unchanged(omega1, omega1_host);
    result.valid = result.valid && unchanged(omega2, omega2_host) && unchanged(spacing, spacing_host);

    const auto psi_host = psi.get_host_data();
    const auto chi_host = chi.get_host_data();
    const auto first = wind1.get_host_data();
    const auto second = wind2.get_host_data();

    const std::size_t cells = static_cast<std::size_t>(physical_nx) * physical_ny;
    result.fields[0].resize(cells);
    result.fields[1].resize(cells);
    result.fields[2].resize(cells * wind_levels);
    result.fields[3].resize(cells * wind_levels);

    for (int j = 0; j < physical_ny; ++j) {
        const double global_j = static_cast<double>(start_j + j);
        const double phi_u = south_edge + (global_j + 0.5) * dq2;
        const double h1_u = spherical ? radius * std::cos(phi_u) : 1.0;
        const double h2 = spherical ? radius : 1.0;

        for (int i = 0; i < physical_nx; ++i) {
            const std::size_t index = static_cast<std::size_t>(j) * physical_nx + i;
            result.fields[0][index] = static_cast<double>(psi_host(j + h, i + h));
            result.fields[1][index] = static_cast<double>(chi_host(j + h, i + h));

            for (int k = bottom; k <= top; ++k) {
                const std::size_t volume_index = static_cast<std::size_t>(k - bottom) * cells + index;
                result.fields[2][volume_index] = static_cast<double>(first(k, j + h, i + h)) / h1_u;
                result.fields[3][volume_index] = static_cast<double>(second(k, j + h, i + h)) / h2;
            }
        }
    }

    for (int k = 0; k < nz; ++k) {
        for (int j = 0; j < ny; ++j) {
            for (int i = 0; i < nx; ++i) {
                const bool written = k >= bottom && k <= top && j >= h && j < ny - h && i >= h && i < nx - h;

                if (!written) {
                    result.valid = result.valid && first(k, j, i) == sentinel && second(k, j, i) == sentinel;
                }
            }
        }
    }

    for (const auto& values : result.fields) {
        for (double value : values) {
            result.valid = result.valid && std::isfinite(value);
            if (mode == 0) result.valid = result.valid && value == 0.0;
        }
    }

    return result;
}

std::vector<Result> run_cases(const Grid& grid, HaloExchanger& halo_exchanger) {
    HorizontalEllipticSolver solver(grid, halo_exchanger);
    std::vector<Result> results;

    // Reuse solver scratch storage, ending with zero inputs.
    for (int mode : {1, 2, 0}) {
        for (int iterations : {1, 4}) {
            results.push_back(run_case(grid, solver, mode, iterations));
        }
    }

    Kokkos::fence();
    return results;
}

#if defined(ENABLE_NCCL)
void require_nccl(ncclResult_t status, const char* operation) {
    if (status != ncclSuccess) {
        throw std::runtime_error(std::string(operation) + ": " + ncclGetErrorString(status));
    }
}
#endif

std::vector<Result> evaluate(const ConfigurationManager& config, MPI_Comm communicator) {
    Grid grid(config, communicator);

#if defined(ENABLE_NCCL)
    int rank = 0;
    int size = 0;
    MPI_Comm_rank(communicator, &rank);
    MPI_Comm_size(communicator, &size);

    ncclUniqueId id;
    if (rank == 0) require_nccl(ncclGetUniqueId(&id), "Create NCCL identifier");
    MPI_Bcast(&id, static_cast<int>(sizeof(id)), MPI_BYTE, 0, communicator);

    ncclComm_t communication = nullptr;
    require_nccl(ncclCommInitRank(&communication, size, id, rank), "Initialize NCCL communicator");

    std::vector<Result> results;

    try {
        const cudaStream_t stream = Kokkos::Cuda().cuda_stream();
        HaloExchanger halo_exchanger(config, grid, communication, stream);
        results = run_cases(grid, halo_exchanger);
    } catch (...) {
        ncclCommAbort(communication);
        throw;
    }

    require_nccl(ncclCommDestroy(communication), "Destroy NCCL communicator");
    return results;
#else
    HaloExchanger halo_exchanger(grid);
    return run_cases(grid, halo_exchanger);
#endif
}

int compare(const Result& reference, const Result& actual, int ranks) {
    std::array<double, 4> local_errors = {};
    std::array<double, 4> reference_scales = {};

    int local_invalid = reference.valid && actual.valid ? 0 : 1;

    if (reference.mode != actual.mode || reference.iterations != actual.iterations
        || reference.start_i != 0 || reference.start_j != 0
        || actual.start_i < 0 || actual.start_j < 0
        || actual.start_i + actual.nx > reference.nx
        || actual.start_j + actual.ny > reference.ny) {
        throw std::runtime_error("Invalid reference/local result mapping.");
    }

    const std::size_t reference_cells = static_cast<std::size_t>(reference.nx) * reference.ny;
    const std::size_t local_cells = static_cast<std::size_t>(actual.nx) * actual.ny;

    for (std::size_t component = 0; component < reference.fields.size(); ++component) {
        for (double value : reference.fields[component]) {
            if (std::isfinite(value)) {
                reference_scales[component] = std::max(reference_scales[component], std::abs(value));
            } else {
                local_invalid = 1;
            }
        }

        const int levels = component < 2 ? 1 : wind_levels;

        for (int k = 0; k < levels; ++k) {
            for (int j = 0; j < actual.ny; ++j) {
                for (int i = 0; i < actual.nx; ++i) {
                    const std::size_t local_index = static_cast<std::size_t>(k) * local_cells + static_cast<std::size_t>(j) * actual.nx + i;
                    const std::size_t global_index = static_cast<std::size_t>(k) * reference_cells
                        + static_cast<std::size_t>(actual.start_j + j) * reference.nx + actual.start_i + i;

                    const double expected = reference.fields[component][global_index];
                    const double value = actual.fields[component][local_index];

                    if (!std::isfinite(expected) || !std::isfinite(value)) {
                        local_invalid = 1;
                    } else {
                        local_errors[component] = std::max(local_errors[component], std::abs(value - expected));
                    }
                }
            }
        }
    }

    std::array<double, 4> global_errors = {};
    std::array<double, 4> global_scales = {};
    int global_invalid = 0;

    MPI_Allreduce(local_errors.data(), global_errors.data(), 4, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    MPI_Allreduce(reference_scales.data(), global_scales.data(), 4, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    MPI_Allreduce(&local_invalid, &global_invalid, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);

    const double tolerance = sizeof(Real) == sizeof(float) ? 5e-4 : 5e-10;
    bool passed = global_invalid == 0;

    for (std::size_t component = 0; component < global_errors.size(); ++component) {
        global_errors[component] /= std::max(1e-12, global_scales[component]);
        passed = passed && global_errors[component] <= tolerance;
    }

    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (rank == 0) {
        std::printf("ranks=%d mode=%d iterations=%d psi=%.3e chi=%.3e wind_e=%.3e wind_n=%.3e inputs_regions_finite=%d %s\n",
            ranks, actual.mode, actual.iterations, global_errors[0], global_errors[1],
            global_errors[2], global_errors[3], global_invalid == 0 ? 1 : 0, passed ? "PASS" : "FAIL");
    }

    return passed ? 0 : 1;
}

} // namespace

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);
    Kokkos::initialize(argc, argv);

    int rank = 0;
    int ranks = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);

    int failures = 0;

    try {
        if (argc < 2) {
            throw std::invalid_argument("Usage: test_horizontal_wind_solver_column_mpi <configuration.json>");
        }

        if (ranks != 2 && ranks != 4) {
            throw std::invalid_argument("Run this decomposition test with 2 or 4 MPI ranks.");
        }

        const ConfigurationManager config(argv[1]);

        // Each rank uses its assigned device for a full-domain reference.
        // All reference communicators/solver objects are destroyed before
        // constructing the distributed communicator and solver.
        const auto reference = evaluate(config, MPI_COMM_SELF);
        MPI_Barrier(MPI_COMM_WORLD);

        const auto distributed = evaluate(config, MPI_COMM_WORLD);

        if (reference.size() != distributed.size()) {
            throw std::runtime_error("Reference and distributed case counts differ.");
        }

        for (std::size_t n = 0; n < reference.size(); ++n) {
            failures += compare(reference[n], distributed[n], ranks);
        }
    } catch (const std::exception& error) {
        std::fprintf(stderr, "Rank %d test_horizontal_wind_solver_column_mpi: %s\n", rank, error.what());

        // An unexpected rank-local exception must not leave peers waiting
        // indefinitely in another communication operation.
        MPI_Abort(MPI_COMM_WORLD, 1);
        return 1;
    }

    Kokkos::finalize();
    MPI_Finalize();

    return failures == 0 ? 0 : 1;
}
