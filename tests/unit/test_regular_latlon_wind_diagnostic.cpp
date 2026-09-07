#include "core/Field.hpp"
#include "core/Grid.hpp"
#include "core/haloexchange/HaloExchanger.hpp"
#include "core/boundary/HorizontalBoundaryStencils.hpp"
#include "core/geometry/HorizontalLocation.hpp"
#include "dynamics/solvers/HorizontalEllipticSolver.hpp"
#include "dynamics/solvers/HorizontalWindStateAdapter.hpp"
#include "dynamics/solvers/VerticalEllipticSolver.hpp"
#include "dynamics/solvers/WindSolver.hpp"
#include "utils/ConfigurationManager.hpp"

#include <Kokkos_Core.hpp>
#include <mpi.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <limits>

namespace {

using VVM::Real;
using VVM::real;
using VVM::Core::Field;
using VVM::Core::Grid;
using VVM::Core::HaloExchanger;
using VVM::Dynamics::HorizontalEllipticSolver;
using VVM::Dynamics::HorizontalWindStateAdapter;
using VVM::Dynamics::VerticalEllipticSolver;
using VVM::Dynamics::WindSolver;
using VVM::Utils::ConfigurationManager;

constexpr Real sentinel = real(-12345.0);

[[noreturn]] void fatal(const char* message) {
    std::fprintf(stderr, "%s\n", message);
    MPI_Abort(MPI_COMM_WORLD, 2);
    std::abort();
}

#if defined(ENABLE_NCCL)
void cuda_check(cudaError_t error) {
    if (error != cudaSuccess) fatal(cudaGetErrorString(error));
}

void nccl_check(ncclResult_t error) {
    if (error != ncclSuccess) fatal(ncclGetErrorString(error));
}

struct Graph {
    cudaGraph_t graph = nullptr;
    cudaGraphExec_t executable = nullptr;

    ~Graph() {
        if (executable) cudaGraphExecDestroy(executable);
        if (graph) cudaGraphDestroy(graph);
    }
};
#endif

template<std::size_t Dim>
std::vector<Real> snapshot(const Field<Dim>& field) {
    const auto data = field.get_host_data();
    std::vector<Real> values;
    values.reserve(data.size());

    if constexpr (Dim == 0) {
        values.push_back(data());
    } else if constexpr (Dim == 1) {
        for (int k = 0; k < static_cast<int>(data.extent(0)); ++k) {
            values.push_back(data(k));
        }
    } else if constexpr (Dim == 2) {
        for (int j = 0; j < static_cast<int>(data.extent(0)); ++j) {
            for (int i = 0; i < static_cast<int>(data.extent(1)); ++i) {
                values.push_back(data(j, i));
            }
        }
    } else if constexpr (Dim == 3) {
        for (int k = 0; k < static_cast<int>(data.extent(0)); ++k) {
            for (int j = 0; j < static_cast<int>(data.extent(1)); ++j) {
                for (int i = 0; i < static_cast<int>(data.extent(2)); ++i) {
                    values.push_back(data(k, j, i));
                }
            }
        }
    }

    return values;
}

void append(std::vector<Real>& destination, const std::vector<Real>& source) {
    destination.insert(destination.end(), source.begin(), source.end());
}

int wrap(int value, int size) {
    value %= size;
    if (value < 0) value += size;
    return value;
}

int clamp(int value, int size) {
    return std::max(0, std::min(size - 1, value));
}

template<std::size_t Dim>
bool centered_q2_neumann_halos(const Grid& grid, const Field<Dim>& field) {
    static_assert(Dim == 2 || Dim == 3);

    const int h = grid.get_halo_cells();
    const bool south = grid.get_local_physical_start_y() == 0;
    const bool north =
        grid.get_local_physical_end_y() == grid.get_global_points_y() - 1;
    const auto data = field.get_host_data();
    const int ny = static_cast<int>(data.extent(Dim - 2));
    const int nx = static_cast<int>(data.extent(Dim - 1));
    int nz = 1;
    if constexpr (Dim == 3) nz = static_cast<int>(data.extent(0));

    const auto rows_equal = [&](int first_j, int second_j) {
        for (int k = 0; k < nz; ++k) {
            for (int i = h; i < nx - h; ++i) {
                if constexpr (Dim == 2) {
                    if (data(first_j, i) != data(second_j, i)) return false;
                } else {
                    if (data(k, first_j, i) != data(k, second_j, i)) return false;
                }
            }
        }
        return true;
    };

    if (south) {
        for (int distance = 0; distance < h; ++distance) {
            if (!rows_equal(h - 1 - distance, h + distance)) return false;
        }
    }

    if (north) {
        const int wall_j = ny - h - 1;
        for (int distance = 0; distance < h; ++distance) {
            if (!rows_equal(wall_j + 1 + distance, wall_j - distance)) return false;
        }
    }

    return true;
}

template<std::size_t Dim>
bool positive_face_q2_dirichlet_halos(const Grid& grid, const Field<Dim>& field) {
    static_assert(Dim == 2 || Dim == 3);

    const int h = grid.get_halo_cells();
    const bool south = grid.get_local_physical_start_y() == 0;
    const bool north =
        grid.get_local_physical_end_y() == grid.get_global_points_y() - 1;
    const auto data = field.get_host_data();
    const int ny = static_cast<int>(data.extent(Dim - 2));
    const int nx = static_cast<int>(data.extent(Dim - 1));
    int nz = 1;
    if constexpr (Dim == 3) nz = static_cast<int>(data.extent(0));

    const auto value = [&](int k, int j, int i) {
        if constexpr (Dim == 2) return data(j, i);
        else return data(k, j, i);
    };

    const auto wall_is_zero = [&](int wall_j) {
        for (int k = 0; k < nz; ++k) {
            for (int i = h; i < nx - h; ++i) {
                if (value(k, wall_j, i) != real(0.0)) return false;
            }
        }
        return true;
    };

    const auto rows_are_odd = [&](int exterior_j, int interior_j) {
        for (int k = 0; k < nz; ++k) {
            for (int i = h; i < nx - h; ++i) {
                if (value(k, exterior_j, i) != -value(k, interior_j, i)) return false;
            }
        }
        return true;
    };

    if (south) {
        const int wall_j = h - 1;
        if (!wall_is_zero(wall_j)) return false;
        for (int distance = 1; distance < h; ++distance) {
            if (!rows_are_odd(wall_j - distance, wall_j + distance)) return false;
        }
    }

    if (north) {
        const int wall_j = ny - h - 1;
        if (!wall_is_zero(wall_j)) return false;
        for (int distance = 1; distance <= h; ++distance) {
            if (!rows_are_odd(wall_j + distance, wall_j - distance)) return false;
        }
    }

    return true;
}

bool free_slip_physical_wind_halos(
    const Grid& grid, const Field<3>& u, const Field<3>& v) {
    const int h = grid.get_halo_cells();
    const bool south = grid.get_local_physical_start_y() == 0;
    const bool north =
        grid.get_local_physical_end_y() == grid.get_global_points_y() - 1;
    const auto u_data = u.get_host_data();
    const auto v_data = v.get_host_data();
    const int nz = static_cast<int>(u_data.extent(0));
    const int ny = static_cast<int>(u_data.extent(1));
    const int nx = static_cast<int>(u_data.extent(2));
    const auto h1 = Kokkos::create_mirror_view_and_copy(
        Kokkos::HostSpace(),
        grid.geometry().device_view(VVM::Core::Geometry::HorizontalLocation::U)
            .contravariant_to_physical.a11.one_dimensional);

    const auto physical_u_equal = [&](int k, int exterior_j, int interior_j, int i) {
        const Real actual = h1(exterior_j) * u_data(k, exterior_j, i);
        const Real expected = h1(interior_j) * u_data(k, interior_j, i);
        const Real scale = std::max(real(1.0), std::max(Kokkos::abs(actual), Kokkos::abs(expected)));
        return Kokkos::abs(actual - expected) <=
            real(512.0) * std::numeric_limits<Real>::epsilon() * scale;
    };

    for (int k = 0; k < nz; ++k) {
        for (int i = h; i < nx - h; ++i) {
            if (south) {
                const int wall_j = h - 1;
                if (v_data(k, wall_j, i) != real(0.0)) return false;
                for (int distance = 0; distance < h; ++distance) {
                    const int exterior_j = wall_j - distance;
                    if (!physical_u_equal(k, exterior_j, h + distance, i)) return false;
                    if (distance > 0 &&
                        v_data(k, exterior_j, i) != -v_data(k, wall_j + distance, i)) return false;
                }
            }
            if (north) {
                const int wall_j = ny - h - 1;
                if (v_data(k, wall_j, i) != real(0.0)) return false;
                for (int distance = 0; distance < h; ++distance) {
                    if (!physical_u_equal(k, wall_j + 1 + distance, wall_j - distance, i)) return false;
                }
                for (int distance = 1; distance <= h; ++distance) {
                    if (v_data(k, wall_j + distance, i) != -v_data(k, wall_j - distance, i)) return false;
                }
            }
        }
    }

    return true;
}

struct Sources {
    Field<3> xi;
    Field<3> eta;
    Field<1> rhobar;
    Field<1> rhobar_up;
    Field<1> flex_mid;
    Field<1> flex_up;
    Field<1> spacing;
    Field<0> zonal_covariant_increment;

    explicit Sources(const Grid& grid)
        : xi("combined_xi", {grid.get_local_total_points_z(), grid.get_local_total_points_y(), grid.get_local_total_points_x()}),
          eta("combined_eta", {grid.get_local_total_points_z(), grid.get_local_total_points_y(), grid.get_local_total_points_x()}),
          rhobar("combined_rhobar", {grid.get_local_total_points_z()}),
          rhobar_up("combined_rhobar_up", {grid.get_local_total_points_z()}),
          flex_mid("combined_flex_mid", {grid.get_local_total_points_z()}),
          flex_up("combined_flex_up", {grid.get_local_total_points_z()}),
          spacing("combined_spacing", {grid.get_local_total_points_z()}),
          zonal_covariant_increment("combined_zonal_covariant_increment", {}) {}

    void initialize_profiles(bool stretched) {
        const int nz = static_cast<int>(rhobar.get_device_data().extent(0));

        auto rho = Kokkos::create_mirror(rhobar.get_device_data());
        auto rho_up = Kokkos::create_mirror(rhobar_up.get_device_data());
        auto mid = Kokkos::create_mirror(flex_mid.get_device_data());
        auto up = Kokkos::create_mirror(flex_up.get_device_data());

        for (int k = 0; k < nz; ++k) {
            rho(k) = stretched ? real(1.40) - real(0.02) * k : real(1.0);
            rho_up(k) = stretched ? real(1.39) - real(0.02) * k : real(1.0);
            mid(k) = stretched ? real(1.10) + real(0.02) * k : real(1.0);
            up(k) = stretched ? real(0.90) + real(0.03) * k : real(1.0);
        }

        Kokkos::deep_copy(rhobar.get_mutable_device_data(), rho);
        Kokkos::deep_copy(rhobar_up.get_mutable_device_data(), rho_up);
        Kokkos::deep_copy(flex_mid.get_mutable_device_data(), mid);
        Kokkos::deep_copy(flex_up.get_mutable_device_data(), up);

        HorizontalWindStateAdapter::initialize_spacing(
            real(100.0),
            flex_up,
            spacing);
    }

    Real factor(int step) const {
        return step == 2 ? real(0.0) : Real(step + 1);
    }

    Real top_zeta_value(const Grid& grid, int step, int global_j, int global_i) const {
        const int nx = grid.get_global_points_x();
        const int ny = grid.get_global_points_y();

        global_i = wrap(global_i, nx);
        global_j = clamp(global_j, ny);

        if (global_j == ny - 1) {
            return real(0.0);
        }

        const Real angle =
            real(2.0) * real(std::acos(-1.0)) * Real(global_i) / Real(nx);

        return factor(step) *
            (real(1.0e-5) * Kokkos::sin(angle) +
             real(2.0e-6) * Real(global_j - ny / 2));
    }

    void initialize_step(const Grid& grid, HaloExchanger& halo, int step) {
        initialize_step(grid, step);

        halo.exchange_multiple_halos(
            std::vector<Field<3>*>{
                &xi,
                &eta
            });

        VVM::Core::Boundary::HorizontalBoundaryStencils boundary(grid);
        boundary.fill_positive_face_q2_homogeneous_dirichlet_halos(xi);
        boundary.fill_centered_q2_neumann_halos(eta);
    }

    bool free_slip_boundaries(const Grid& grid) const {
        return
            positive_face_q2_dirichlet_halos(grid, xi) &&
            centered_q2_neumann_halos(grid, eta);
    }

    void initialize_step(const Grid& grid, int step) {
        const int nz = grid.get_local_total_points_z();
        const int ny = grid.get_local_total_points_y();
        const int nx = grid.get_local_total_points_x();
        const int h = grid.get_halo_cells();
        const int global_nx = grid.get_global_points_x();
        const int global_ny = grid.get_global_points_y();
        const Real scale = factor(step);
        const Real two_pi = real(2.0) * real(std::acos(-1.0));

        auto x = Kokkos::create_mirror(xi.get_device_data());
        auto e = Kokkos::create_mirror(eta.get_device_data());

        for (int k = 0; k < nz; ++k) {
            for (int j = 0; j < ny; ++j) {
                const int global_j = clamp(
                    grid.get_local_physical_start_y() + j - h,
                    global_ny);

                for (int i = 0; i < nx; ++i) {
                    const int global_i = wrap(
                        grid.get_local_physical_start_x() + i - h,
                        global_nx);

                    const Real longitude =
                        two_pi * Real(global_i) / Real(global_nx);

                    x(k, j, i) = scale *
                        (real(2.0e-4) * Kokkos::sin(longitude) +
                         real(4.0e-6) * Real(global_j) +
                         real(1.0e-6) * Real(k));

                    e(k, j, i) = scale *
                        (real(1.5e-4) * Kokkos::cos(longitude) -
                         real(3.0e-6) * Real(global_j) +
                         real(2.0e-6) * Real(k));
                }
            }
        }

        Kokkos::deep_copy(xi.get_mutable_device_data(), x);
        Kokkos::deep_copy(eta.get_mutable_device_data(), e);
        Kokkos::deep_copy(
            zonal_covariant_increment.get_mutable_device_data(),
            scale * real(1.0e6));
    }

    std::vector<Real> values() const {
        std::vector<Real> result;

        append(result, snapshot(xi));
        append(result, snapshot(eta));
        append(result, snapshot(rhobar));
        append(result, snapshot(rhobar_up));
        append(result, snapshot(flex_mid));
        append(result, snapshot(flex_up));
        append(result, snapshot(spacing));
        append(result, snapshot(zonal_covariant_increment));

        return result;
    }
};

template<std::size_t Dim>
bool constant_q2_halos(const Grid& grid, const Field<Dim>& field) {
    static_assert(Dim == 2 || Dim == 3);

    const int h = grid.get_halo_cells();
    const bool south =
        grid.get_local_physical_start_y() == 0;
    const bool north =
        grid.get_local_physical_end_y() ==
        grid.get_global_points_y() - 1;

    const auto data = field.get_host_data();

    if constexpr (Dim == 2) {
        const int ny = static_cast<int>(data.extent(0));
        const int nx = static_cast<int>(data.extent(1));

        if (south) {
            for (int j = 0; j < h; ++j) {
                for (int i = 0; i < nx; ++i) {
                    if (data(j, i) != data(h, i)) return false;
                }
            }
        }

        if (north) {
            const int last = ny - h - 1;

            for (int j = ny - h; j < ny; ++j) {
                for (int i = 0; i < nx; ++i) {
                    if (data(j, i) != data(last, i)) return false;
                }
            }
        }
    }

    if constexpr (Dim == 3) {
        const int nz = static_cast<int>(data.extent(0));
        const int ny = static_cast<int>(data.extent(1));
        const int nx = static_cast<int>(data.extent(2));

        if (south) {
            for (int k = 0; k < nz; ++k) {
                for (int j = 0; j < h; ++j) {
                    for (int i = 0; i < nx; ++i) {
                        if (data(k, j, i) != data(k, h, i)) return false;
                    }
                }
            }
        }

        if (north) {
            const int last = ny - h - 1;

            for (int k = 0; k < nz; ++k) {
                for (int j = ny - h; j < ny; ++j) {
                    for (int i = 0; i < nx; ++i) {
                        if (data(k, j, i) != data(k, last, i)) return false;
                    }
                }
            }
        }
    }

    return true;
}

struct DiagnosticState {
    Field<2> psi;
    Field<2> psi_previous;
    Field<2> chi;
    Field<2> chi_previous;
    Field<3> zeta;
    Field<3> w;
    Field<3> w_previous;
    Field<3> u;
    Field<3> v;
    Field<2> rhs_psi;
    Field<2> rhs_chi;
    Field<2> solution_psi;
    Field<2> solution_chi;

    explicit DiagnosticState(const Grid& grid)
        : psi("combined_psi", {grid.get_local_total_points_y(), grid.get_local_total_points_x()}),
          psi_previous("combined_psi_previous", {grid.get_local_total_points_y(), grid.get_local_total_points_x()}),
          chi("combined_chi", {grid.get_local_total_points_y(), grid.get_local_total_points_x()}),
          chi_previous("combined_chi_previous", {grid.get_local_total_points_y(), grid.get_local_total_points_x()}),
          zeta("combined_zeta", {grid.get_local_total_points_z(), grid.get_local_total_points_y(), grid.get_local_total_points_x()}),
          w("combined_w", {grid.get_local_total_points_z(), grid.get_local_total_points_y(), grid.get_local_total_points_x()}),
          w_previous("combined_w_previous", {grid.get_local_total_points_z(), grid.get_local_total_points_y(), grid.get_local_total_points_x()}),
          u("combined_u", {grid.get_local_total_points_z(), grid.get_local_total_points_y(), grid.get_local_total_points_x()}),
          v("combined_v", {grid.get_local_total_points_z(), grid.get_local_total_points_y(), grid.get_local_total_points_x()}),
          rhs_psi("combined_rhs_psi", {grid.get_local_total_points_y(), grid.get_local_total_points_x()}),
          rhs_chi("combined_rhs_chi", {grid.get_local_total_points_y(), grid.get_local_total_points_x()}),
          solution_psi("combined_solution_psi", {grid.get_local_total_points_y(), grid.get_local_total_points_x()}),
          solution_chi("combined_solution_chi", {grid.get_local_total_points_y(), grid.get_local_total_points_x()}) {}

    void reset_all() {
        psi.set_to_zero();
        psi_previous.set_to_zero();
        chi.set_to_zero();
        chi_previous.set_to_zero();
        zeta.set_to_zero();
        w.set_to_zero();
        w_previous.set_to_zero();
        u.set_to_zero();
        v.set_to_zero();
        rhs_psi.set_to_zero();
        rhs_chi.set_to_zero();
        solution_psi.set_to_zero();
        solution_chi.set_to_zero();
    }

    void prepare_step(const Grid& grid, const Sources& sources, int step) {
        const int nz = grid.get_local_total_points_z();
        const int ny = grid.get_local_total_points_y();
        const int nx = grid.get_local_total_points_x();
        const int h = grid.get_halo_cells();
        const int top = nz - h - 1;

        auto zeta_host = Kokkos::create_mirror(zeta.get_device_data());

        for (int k = 0; k < nz; ++k) {
            for (int j = 0; j < ny; ++j) {
                const int global_j =
                    grid.get_local_physical_start_y() + j - h;

                for (int i = 0; i < nx; ++i) {
                    const int global_i =
                        grid.get_local_physical_start_x() + i - h;

                    zeta_host(k, j, i) =
                        k == top
                        ? sources.top_zeta_value(
                            grid,
                            step,
                            global_j,
                            global_i)
                        : sentinel;
                }
            }
        }

        Kokkos::deep_copy(
            zeta.get_mutable_device_data(),
            zeta_host);

        Kokkos::deep_copy(
            u.get_mutable_device_data(),
            sentinel);

        Kokkos::deep_copy(
            v.get_mutable_device_data(),
            sentinel);
    }

    WindSolver::RegularLatLonDiagnosticFields bind(const Sources& sources) {
        return {
            psi,
            psi_previous,
            chi,
            chi_previous,
            zeta,
            w,
            w_previous,
            sources.xi,
            sources.eta,
            u,
            v,
            sources.rhobar,
            sources.rhobar_up,
            sources.flex_mid,
            sources.spacing,
            sources.zonal_covariant_increment
        };
    }

    WindSolver::HorizontalDiagnosticWorkspace workspace() {
        return {
            rhs_psi,
            rhs_chi,
            solution_psi,
            solution_chi
        };
    }

    std::vector<Real> values() const {
        std::vector<Real> result;

        append(result, snapshot(psi));
        append(result, snapshot(psi_previous));
        append(result, snapshot(chi));
        append(result, snapshot(chi_previous));
        append(result, snapshot(zeta));
        append(result, snapshot(w));
        append(result, snapshot(w_previous));
        append(result, snapshot(u));
        append(result, snapshot(v));
        append(result, snapshot(rhs_psi));
        append(result, snapshot(rhs_chi));
        append(result, snapshot(solution_psi));
        append(result, snapshot(solution_chi));

        return result;
    }

    bool free_slip_boundaries(const Grid& grid) const {
        return
            positive_face_q2_dirichlet_halos(grid, psi) &&
            positive_face_q2_dirichlet_halos(grid, psi_previous) &&
            centered_q2_neumann_halos(grid, chi) &&
            centered_q2_neumann_halos(grid, chi_previous) &&
            positive_face_q2_dirichlet_halos(grid, zeta) &&
            centered_q2_neumann_halos(grid, w) &&
            centered_q2_neumann_halos(grid, w_previous) &&
            free_slip_physical_wind_halos(grid, u, v);
    }

    bool physical_checks(const Grid& grid, const Sources& sources, int step) const {
        const int nz = grid.get_local_total_points_z();
        const int ny = grid.get_local_total_points_y();
        const int nx = grid.get_local_total_points_x();
        const int h = grid.get_halo_cells();
        const int bottom = h - 1;
        const int top = nz - h - 1;

        const auto zeta_data = zeta.get_host_data();
        const auto w_data = w.get_host_data();
        const auto u_data = u.get_host_data();
        const auto v_data = v.get_host_data();

        bool valid = true;

        for (int j = h; j < ny - h; ++j) {
            const int global_j =
                grid.get_local_physical_start_y() + j - h;

            for (int i = h; i < nx - h; ++i) {
                const int global_i =
                    grid.get_local_physical_start_x() + i - h;

                valid = valid &&
                    zeta_data(top, j, i) ==
                    sources.top_zeta_value(
                        grid,
                        step,
                        global_j,
                        global_i);

                valid = valid &&
                    w_data(bottom, j, i) == real(0.0) &&
                    w_data(top, j, i) == real(0.0);

                for (int k = h; k < top; ++k) {
                    valid = valid &&
                        std::isfinite(w_data(k, j, i));
                }

                for (int k = bottom; k <= top; ++k) {
                    valid = valid &&
                        std::isfinite(zeta_data(k, j, i)) &&
                        std::isfinite(u_data(k, j, i)) &&
                        std::isfinite(v_data(k, j, i)) &&
                        u_data(k, j, i) != sentinel &&
                        v_data(k, j, i) != sentinel;
                }

                valid = valid &&
                    std::isfinite(zeta_data(top + 1, j, i)) &&
                    zeta_data(bottom, j, i) != sentinel;
            }
        }

        return valid;
    }
};

int run_case(const Grid& grid, HaloExchanger& halo, bool stretched) {
    Sources sources(grid);
    sources.initialize_profiles(stretched);

    DiagnosticState direct(grid);
    DiagnosticState replayed(grid);

    const Real shift =
        stretched ? real(0.25) : real(0.0);

    VerticalEllipticSolver direct_vertical(
        grid,
        halo,
        sources.rhobar,
        sources.rhobar_up,
        sources.flex_mid,
        sources.flex_up,
        real(0.01),
        shift);

    VerticalEllipticSolver replayed_vertical(
        grid,
        halo,
        sources.rhobar,
        sources.rhobar_up,
        sources.flex_mid,
        sources.flex_up,
        real(0.01),
        shift);

    HorizontalEllipticSolver direct_horizontal(
        grid,
        halo);

    HorizontalEllipticSolver replayed_horizontal(
        grid,
        halo);

    WindSolver::RegularLatLonDiagnosticOptions options;
    options.vertical_iterations = 4;
    options.horizontal.iterations = 4;
    options.horizontal.diagonal_shift = shift;
    options.horizontal.refresh_initial_halos = true;
    options.inverse_dz = real(0.01);
    options.boundary_policy =
        WindSolver::HorizontalDiagnosticBoundaryPolicy::
            RegularLatLonFreeSlipChannel;

    const auto execute = [&](VerticalEllipticSolver& vertical,
                             HorizontalEllipticSolver& horizontal,
                             DiagnosticState& state) {
        WindSolver::diagnose_regular_latlon_wind(
            grid,
            halo,
            vertical,
            horizontal,
            state.bind(sources),
            state.workspace(),
            options);
    };

    sources.initialize_step(grid, halo, 0);
    direct.reset_all();
    replayed.reset_all();
    direct.prepare_step(grid, sources, 0);
    replayed.prepare_step(grid, sources, 0);

    WindSolver::prepare_regular_latlon_diagnostic_execution();

    // Explicitly prepare the same solver and communication operations outside
    // capture, consistent with the existing VVMex composed replay contract.
    execute(
        direct_vertical,
        direct_horizontal,
        direct);

    Kokkos::fence();

    execute(
        replayed_vertical,
        replayed_horizontal,
        replayed);

    Kokkos::fence();

    direct.reset_all();
    replayed.reset_all();
    Kokkos::fence();

#if defined(ENABLE_NCCL)
    Graph graph;
    const auto stream =
        Kokkos::Cuda().cuda_stream();

    MPI_Barrier(grid.get_comm());

    cuda_check(
        cudaStreamBeginCapture(
            stream,
            cudaStreamCaptureModeGlobal));

    execute(
        replayed_vertical,
        replayed_horizontal,
        replayed);

    cuda_check(
        cudaStreamEndCapture(
            stream,
            &graph.graph));

    if (!graph.graph) {
        fatal("Regular latitude-longitude diagnostic capture returned no graph.");
    }

    cuda_check(
        cudaGraphInstantiate(
            &graph.executable,
            graph.graph,
            nullptr,
            nullptr,
            0));

    const char* execution = "cuda_graph";
#else
    const char* execution = "direct_repeat";
#endif

    int failures = 0;

    for (int step = 0; step < 3; ++step) {
        sources.initialize_step(grid, halo, step);
        direct.prepare_step(grid, sources, step);
        replayed.prepare_step(grid, sources, step);
        Kokkos::fence();

        const auto before_sources =
            sources.values();

        MPI_Barrier(grid.get_comm());

        execute(
            direct_vertical,
            direct_horizontal,
            direct);

        Kokkos::fence();
        MPI_Barrier(grid.get_comm());

#if defined(ENABLE_NCCL)
        cuda_check(
            cudaGraphLaunch(
                graph.executable,
                stream));

        cuda_check(
            cudaStreamSynchronize(
                stream));
#else
        execute(
            replayed_vertical,
            replayed_horizontal,
            replayed);

        Kokkos::fence();
#endif

        const bool exact =
            direct.values() == replayed.values();

        const bool sources_preserved =
            sources.values() == before_sources;

        const bool physical =
            direct.physical_checks(
                grid,
                sources,
                step) &&
            replayed.physical_checks(
                grid,
                sources,
                step);

        const bool free_slip_walls =
            sources.free_slip_boundaries(grid) &&
            direct.free_slip_boundaries(grid) &&
            replayed.free_slip_boundaries(grid);

        int local_flags[4] = {
            int(exact),
            int(sources_preserved),
            int(physical),
            int(free_slip_walls)
        };

        int global_flags[4] = {};

        MPI_Allreduce(
            local_flags,
            global_flags,
            4,
            MPI_INT,
            MPI_MIN,
            grid.get_comm());

        const bool pass =
            global_flags[0] &&
            global_flags[1] &&
            global_flags[2] &&
            global_flags[3];

        if (grid.get_mpi_rank() == 0) {
            std::printf(
                "%s ranks=%d stretched=%d step=%d exact=%d sources=%d physical=%d free_slip_walls=%d %s\n",
                execution,
                grid.get_mpi_size(),
                int(stretched),
                step,
                global_flags[0],
                global_flags[1],
                global_flags[2],
                global_flags[3],
                pass ? "PASS" : "FAIL");
        }

        failures += !pass;
    }

    Kokkos::fence();
    MPI_Barrier(grid.get_comm());

    return failures;
}

int run(const Grid& grid, HaloExchanger& halo) {
    int failures = 0;

    for (bool stretched : {false, true}) {
        failures +=
            run_case(
                grid,
                halo,
                stretched);
    }

    return failures;
}

} // namespace

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank = 0;
    int ranks = 0;
    int failures = 0;

    MPI_Comm_rank(
        MPI_COMM_WORLD,
        &rank);

    MPI_Comm_size(
        MPI_COMM_WORLD,
        &ranks);

    try {
        Kokkos::initialize(argc, argv);

        if (argc != 2 ||
            (ranks != 1 &&
             ranks != 2 &&
             ranks != 4)) {

            fatal(
                "Provide one RLL configuration and use 1, 2, or 4 ranks.");
        }

        {
            ConfigurationManager config(argv[1]);
            Grid grid(config);

#if defined(ENABLE_NCCL)
            ncclUniqueId id;

            if (rank == 0) {
                nccl_check(
                    ncclGetUniqueId(
                        &id));
            }

            MPI_Bcast(
                &id,
                int(sizeof(id)),
                MPI_BYTE,
                0,
                grid.get_comm());

            ncclComm_t communicator;

            nccl_check(
                ncclCommInitRank(
                    &communicator,
                    ranks,
                    id,
                    rank));

            {
                HaloExchanger halo(
                    config,
                    grid,
                    communicator,
                    Kokkos::Cuda().cuda_stream());

                failures =
                    run(
                        grid,
                        halo);

                Kokkos::fence();
            }

            nccl_check(
                ncclCommDestroy(
                    communicator));
#else
            HaloExchanger halo(grid);

            failures =
                run(
                    grid,
                    halo);
#endif
        }

        Kokkos::finalize();
    } catch (const std::exception& error) {
        fatal(error.what());
    }

    MPI_Finalize();

    return failures == 0 ? 0 : 1;
}
