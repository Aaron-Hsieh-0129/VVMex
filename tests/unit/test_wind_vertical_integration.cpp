#include "core/Grid.hpp"
#include "core/Parameters.hpp"
#include "core/State.hpp"
#include "core/haloexchange/HaloExchanger.hpp"
#include "dynamics/solvers/WindSolver.hpp"
#include "utils/ConfigurationManager.hpp"

#include <Kokkos_Core.hpp>
#include <mpi.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using VVM::Real;
using VVM::real;
using VVM::Core::Grid;
using VVM::Core::HaloExchanger;
using VVM::Core::Parameters;
using VVM::Core::State;
using VVM::Dynamics::WindSolver;
using VVM::Utils::ConfigurationManager;

int failures = 0;

struct Communication {
#if defined(ENABLE_NCCL)
    ncclComm_t comm = nullptr;
    cudaStream_t stream = nullptr;

    Communication() {
        ncclUniqueId id;
        require(ncclGetUniqueId(&id), "ncclGetUniqueId");
        require(ncclCommInitRank(&comm, 1, id, 0), "ncclCommInitRank");
        stream = Kokkos::Cuda().cuda_stream();
    }

    ~Communication() {
        if (comm) ncclCommDestroy(comm);
    }

    static void require(ncclResult_t status, const char* operation) {
        if (status != ncclSuccess) {
            throw std::runtime_error(std::string(operation) + ": " + ncclGetErrorString(status));
        }
    }
#endif
};

Real gradient_x(int mode, int k) {
    return (mode & 2) ? real(1e-4) * (real(1.0) + real(0.125) * k) : real(0.0);
}

Real gradient_y(int mode, int k) {
    return (mode & 2) ? real(-2e-4) * (real(1.0) + real(0.0625) * k) : real(0.0);
}

std::vector<Real> snapshot_field(const VVM::Core::Field<3>& field) {
    const auto host = field.get_host_data();
    std::vector<Real> values;
    values.reserve(host.size());

    for (std::size_t k = 0; k < host.extent(0); ++k) {
        for (std::size_t j = 0; j < host.extent(1); ++j) {
            for (std::size_t i = 0; i < host.extent(2); ++i) {
                values.push_back(host(k, j, i));
            }
        }
    }

    return values;
}

void run_tests(const ConfigurationManager& config, const Communication& communication) {
    const Grid grid(config);
    Parameters parameters(config, grid);

#if defined(ENABLE_NCCL)
    State state(config, parameters, grid, communication.comm, communication.stream);
    HaloExchanger halo(config, grid, communication.comm, communication.stream);
#else
    (void)communication;
    State state(config, parameters, grid);
    HaloExchanger halo(grid);
#endif

    WindSolver solver(grid, config, parameters, halo, state);

    const int nz = grid.get_local_total_points_z();
    const int ny = grid.get_local_total_points_y();
    const int nx = grid.get_local_total_points_x();
    const int h = grid.get_halo_cells();
    const int top = nz - h - 1;

    const Real dx_value = parameters.get_value_host(parameters.dx);
    const Real dy_value = parameters.get_value_host(parameters.dy);
    const Real dz_value = parameters.get_value_host(parameters.dz);
    const Real sentinel = real(-999.0);
    const Real tolerance = sizeof(Real) == sizeof(double) ? real(1e-11) : real(5e-5);

    auto u = state.get_field<3>("u").get_mutable_device_data();
    auto v = state.get_field<3>("v").get_mutable_device_data();
    auto w = state.get_field<3>("w").get_mutable_device_data();
    auto xi = state.get_field<3>("xi_topo").get_mutable_device_data();
    auto eta = state.get_field<3>("eta_topo").get_mutable_device_data();
    auto flex = parameters.flex_height_coef_up.get_mutable_device_data();

    auto u_host = Kokkos::create_mirror_view(u);
    auto v_host = Kokkos::create_mirror_view(v);
    auto w_host = Kokkos::create_mirror_view(w);
    auto xi_host = Kokkos::create_mirror_view(xi);
    auto eta_host = Kokkos::create_mirror_view(eta);
    auto flex_host = Kokkos::create_mirror_view(flex);

    Kokkos::View<Real***> reference_u("reference_u", nz, ny, nx);
    Kokkos::View<Real***> reference_v("reference_v", nz, ny, nx);

    const auto rdx = parameters.rdx;
    const auto rdy = parameters.rdy;
    const auto dz = parameters.dz;

    for (bool stretched : {false, true}) {
        std::vector<Real> z(nz, real(0.0));

        for (int k = 0; k < nz; ++k) {
            flex_host(k) = stretched
                ? (k % 3 == 0 ? real(0.8) : (k % 3 == 1 ? real(1.1) : real(1.4)))
                : real(1.0);
        }

        for (int k = 0; k < nz - 1; ++k) {
            z[k + 1] = z[k] + dz_value / flex_host(k);
        }

        Kokkos::deep_copy(flex, flex_host);

        for (int mode = 0; mode < 4; ++mode) {
            const Real shear_u = (mode & 1) ? real(1e-4) : real(0.0);
            const Real shear_v = (mode & 1) ? real(-2e-4) : real(0.0);

            // Manufacture:
            //   u = 12 + shear_u*z
            //   v = -8 + shear_v*z
            //   w = gradient_x(k)*x + gradient_y(k)*y
            //
            // Legacy Cartesian VVMex definitions:
            //   xi  = dw/dy - dv/dz
            //   eta = dw/dx - du/dz
            for (int k = 0; k < nz; ++k) {
                const Real a = gradient_x(mode, k);
                const Real b = gradient_y(mode, k);

                for (int j = 0; j < ny; ++j) {
                    const Real y = static_cast<Real>(j - h) * dy_value;

                    for (int i = 0; i < nx; ++i) {
                        const Real x = static_cast<Real>(i - h) * dx_value;
                        w_host(k, j, i) = a * x + b * y;
                        xi_host(k, j, i) = b - shear_v;
                        eta_host(k, j, i) = a - shear_u;

                        const bool top_physical = k == top && j >= h && j < ny - h && i >= h && i < nx - h;
                        u_host(k, j, i) = top_physical ? real(12.0) + shear_u * z[k] : sentinel;
                        v_host(k, j, i) = top_physical ? real(-8.0) + shear_v * z[k] : sentinel;
                    }
                }
            }

            Kokkos::deep_copy(u, u_host);
            Kokkos::deep_copy(v, v_host);
            Kokkos::deep_copy(w, w_host);
            Kokkos::deep_copy(xi, xi_host);
            Kokkos::deep_copy(eta, eta_host);
            Kokkos::deep_copy(reference_u, u_host);
            Kokkos::deep_copy(reference_v, v_host);
            const auto original_w = snapshot_field(state.get_field<3>("w"));
            const auto original_xi = snapshot_field(state.get_field<3>("xi_topo"));
            const auto original_eta = snapshot_field(state.get_field<3>("eta_topo"));

            // Literal reference arithmetic from solve_uv before extraction.
            Kokkos::parallel_for("OriginalWindVerticalIntegrationReference",
                Kokkos::MDRangePolicy<Kokkos::Rank<2>>({h, h}, {ny-h, nx-h}),
                KOKKOS_LAMBDA(const int j, const int i) {
                    for (int k = nz-h-2; k >= h-1; --k) {
                        reference_u(k,j,i) = reference_u(k+1,j,i)
                            - ((w(k,j,i+1) - w(k,j,i))*rdx() - eta(k,j,i)) * dz() / flex(k);
                        reference_v(k,j,i) = reference_v(k+1,j,i)
                            - ((w(k,j+1,i) - w(k,j,i))*rdy() - xi(k,j,i)) * dz() / flex(k);
                    }

                    reference_u(nz-h,j,i) = reference_u(nz-h-1,j,i)
                        + ((w(nz-h-1,j,i+1) - w(nz-h-1,j,i))*rdx() - eta(nz-h-1,j,i)) * dz() / flex(nz-h-1);
                    reference_v(nz-h,j,i) = reference_v(nz-h-1,j,i)
                        + ((w(nz-h-1,j+1,i) - w(nz-h-1,j,i))*rdy() - xi(nz-h-1,j,i)) * dz() / flex(nz-h-1);
                }
            );

            solver.integrate_uv_from_top();
            Kokkos::fence();

            const auto actual_u = state.get_field<3>("u").get_host_data();
            const auto actual_v = state.get_field<3>("v").get_host_data();
            const auto actual_w = state.get_field<3>("w").get_host_data();
            const auto actual_xi = state.get_field<3>("xi_topo").get_host_data();
            const auto actual_eta = state.get_field<3>("eta_topo").get_host_data();
            const auto expected_u = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), reference_u);
            const auto expected_v = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), reference_v);

            bool exact_reference = true;
            bool untouched_correct = true;
            bool inputs_unchanged = true;
            bool finite = true;
            Real analytic_error = real(0.0);

            for (int k = 0; k < nz; ++k) {
                const Real a = gradient_x(mode, k);
                const Real b = gradient_y(mode, k);

                for (int j = 0; j < ny; ++j) {
                    const Real y = static_cast<Real>(j - h) * dy_value;

                    for (int i = 0; i < nx; ++i) {
                        const Real x = static_cast<Real>(i - h) * dx_value;

                        exact_reference = exact_reference
                            && actual_u(k, j, i) == expected_u(k, j, i)
                            && actual_v(k, j, i) == expected_v(k, j, i);

                        const std::size_t index = (static_cast<std::size_t>(k) * ny + j) * nx + i;

                        inputs_unchanged = inputs_unchanged
                            && actual_w(k, j, i) == original_w[index]
                            && actual_xi(k, j, i) == original_xi[index]
                            && actual_eta(k, j, i) == original_eta[index];

                        const bool integrated_region = j >= h && j < ny - h
                            && i >= h && i < nx - h && k >= h - 1 && k <= nz - h;

                        if (integrated_region) {
                            const Real analytic_u = real(12.0) + shear_u * z[k];
                            const Real analytic_v = real(-8.0) + shear_v * z[k];

                            if (!std::isfinite(actual_u(k, j, i)) || !std::isfinite(actual_v(k, j, i))) {
                                finite = false;
                            } else {
                                analytic_error = std::max(analytic_error,
                                    std::abs(actual_u(k, j, i) - analytic_u) / std::max(real(1.0), std::abs(analytic_u)));
                                analytic_error = std::max(analytic_error,
                                    std::abs(actual_v(k, j, i) - analytic_v) / std::max(real(1.0), std::abs(analytic_v)));
                            }
                        } else {
                            untouched_correct = untouched_correct
                                && actual_u(k, j, i) == sentinel
                                && actual_v(k, j, i) == sentinel;
                        }
                    }
                }
            }

            const bool passed = exact_reference && untouched_correct && inputs_unchanged
                && finite && analytic_error <= tolerance;

            std::printf("stretched=%d mode=%d exact_reference=%d untouched=%d inputs=%d analytic_error=%.3e %s\n",
                static_cast<int>(stretched), mode, static_cast<int>(exact_reference),
                static_cast<int>(untouched_correct), static_cast<int>(inputs_unchanged),
                static_cast<double>(analytic_error), passed ? "PASS" : "FAIL");

            if (!passed) ++failures;
        }
    }
}

} // namespace

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);
    Kokkos::initialize(argc, argv);

    try {
        int size = 0;
        MPI_Comm_size(MPI_COMM_WORLD, &size);

        if (size != 1) throw std::runtime_error("This test requires one MPI rank.");
        if (argc != 2) throw std::runtime_error("Usage: test_wind_vertical_integration <configuration.json>");

        Communication communication;
        const ConfigurationManager config(argv[1]);
        run_tests(config, communication);
    } catch (const std::exception& error) {
        ++failures;
        std::fprintf(stderr, "test_wind_vertical_integration: %s\n", error.what());
    }

    int global_failures = 0;
    MPI_Allreduce(&failures, &global_failures, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

    if (global_failures == 0) std::puts("test_wind_vertical_integration: PASS");

    Kokkos::finalize();
    MPI_Finalize();
    return global_failures == 0 ? 0 : 1;
}
