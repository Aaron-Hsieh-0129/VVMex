
#include "dynamics/solvers/VerticalEllipticSolver.hpp"

#include <cmath>
#include <stdexcept>
#include <utility>
#include <vector>

#if defined(KOKKOS_ENABLE_CUDA)
#include <cuda_runtime.h>
#endif

namespace VVM {
namespace Dynamics {
namespace {

void
outside_capture() {
#if defined(KOKKOS_ENABLE_CUDA)
    cudaStreamCaptureStatus status;

    const auto error = cudaStreamIsCapturing(Kokkos::Cuda().cuda_stream(), &status);

    if (error != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(error));
    }

    if (status != cudaStreamCaptureStatusNone) {
        throw std::logic_error("Vertical solver initialization "
                               "must be outside capture.");
    }
#endif
}

void
positive(Real value) {
    if (!std::isfinite(value) || value <= real(0.0)) {
        throw std::invalid_argument("Vertical solver requires finite "
                                    "positive density, stretching and "
                                    "pivots.");
    }
}

const Core::Grid&
checked_grid(const Core::Grid& grid) {
    outside_capture();
    return grid;
}

struct BuildGeneralizedVerticalLineFactorsFunctor {
    VerticalWindDiagnosticDeviceView operation;

    Core::Field<1>::ViewType rho;
    Core::Field<1>::ViewType rho_up;
    Core::Field<1>::ViewType flex_mid;
    Core::Field<1>::ViewType flex_up;

    Kokkos::View<Real**> lower;
    Kokkos::View<Real**> pivot;
    Kokkos::View<Real**> upper;

    Real inverse_dz;
    Real shift;

    int halo;
    int last;
    int physical_nx;

    KOKKOS_INLINE_FUNCTION
    void
    operator()(const int column) const {
        const int j = halo + column / physical_nx;

        const int i = halo + column % physical_nx;

        for (int k = halo; k <= last; ++k) {

            const auto row =
                operation
                    .calculate_row_at_t(rho, rho_up, flex_mid, flex_up, inverse_dz, shift, k, j, i);

            lower(column, k) = row.lower;

            const Real diagonal =
                row.diagonal - (k == halo ? real(0.0) : row.lower * upper(column, k - 1));

            pivot(column, k) = diagonal;

            upper(column, k) = row.upper / diagonal;
        }
    }
};

} // namespace

VerticalEllipticSolver::VerticalEllipticSolver(const Core::Grid& grid,
    Core::HaloExchanger& halo,
    const Core::Field<1>& rhobar,
    const Core::Field<1>& rhobar_up,
    const Core::Field<1>& flex_mid,
    const Core::Field<1>& flex_up,
    Real inverse_dz,
    Real diagonal_shift)
    : grid_(checked_grid(grid)), halo_(halo),
      operation_(make_vertical_wind_diagnostic_device_view(grid.geometry())),
      shift_(diagonal_shift), rhs_("vertical_rhs",
                                  {grid.get_local_total_points_z(),
                                      grid.get_local_total_points_y(),
                                      grid.get_local_total_points_x()}),
      first_("vertical_first",
          {grid.get_local_total_points_z(),
              grid.get_local_total_points_y(),
              grid.get_local_total_points_x()}),
      second_("vertical_second",
          {grid.get_local_total_points_z(),
              grid.get_local_total_points_y(),
              grid.get_local_total_points_x()}) {

    outside_capture();

    positive(inverse_dz);

    if (!std::isfinite(shift_) || shift_ < real(0.0)) {
        throw std::invalid_argument("Invalid vertical relaxation shift.");
    }

    const auto& horizontal = grid.horizontal_specification();

    if (horizontal.topology.q1 != Core::HorizontalEdgeTopology::Periodic) {
        throw std::invalid_argument("Vertical solver requires periodic q1.");
    }

    if (horizontal.topology.q2 == Core::HorizontalEdgeTopology::Bounded) {

        boundary_ = std::make_unique<Core::Boundary::HorizontalBoundaryStencils>(grid);
    }

    const int nz = grid.get_local_total_points_z();
    const int ny = grid.get_local_total_points_y();
    const int nx = grid.get_local_total_points_x();
    const int h = grid.get_halo_cells();

    const int last = nz - h - 2;
    if (last < h) {
        throw std::invalid_argument("Vertical solver requires an "
                                    "interior w level.");
    }

    for (const auto* field : {&rhobar, &rhobar_up, &flex_mid, &flex_up}) {
        if (static_cast<int>(field->get_device_data().extent(0)) != nz) {
            throw std::invalid_argument("Vertical solver profile "
                                        "extent mismatch.");
        }
    }

    const auto rho = rhobar.get_host_data();
    const auto rho_up = rhobar_up.get_host_data();
    const auto mid = flex_mid.get_host_data();
    const auto up = flex_up.get_host_data();

    for (int k = h; k <= last + 1; ++k) {
        positive(rho(k));
        positive(mid(k));

        if (k <= last) {
            positive(rho_up(k));
            positive(up(k));
        }
    }

    // The horizontal metric may vary in both q1 and q2.
    //
    // Therefore each horizontal column needs its own
    // vertical tridiagonal factorization.
    const int ni = nx - 2 * h;
    const int columns = ni * (ny - 2 * h);

    density_up_ = Kokkos::View<Real*>("vertical_density_up", nz);
    lower_ = Kokkos::View<Real**>("vertical_lower", columns, nz);
    pivot_ = Kokkos::View<Real**>("vertical_pivot", columns, nz);
    upper_normalized_ = Kokkos::View<Real**>("vertical_upper_normalized", columns, nz);
    temporary_ = Kokkos::View<Real**>("vertical_forward", columns, nz);

    Kokkos::deep_copy(density_up_, rhobar_up.get_device_data());
    Kokkos::deep_copy(lower_, real(0.0));
    Kokkos::deep_copy(pivot_, real(0.0));
    Kokkos::deep_copy(upper_normalized_, real(0.0));

    const BuildGeneralizedVerticalLineFactorsFunctor build_factors{operation_,
        rhobar.get_device_data(),
        rhobar_up.get_device_data(),
        flex_mid.get_device_data(),
        flex_up.get_device_data(),
        lower_,
        pivot_,
        upper_normalized_,
        inverse_dz,
        shift_,
        h,
        last,
        ni};

    Kokkos::parallel_for("BuildGeneralizedVerticalLineFactors",
        Kokkos::Experimental::require(
            Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(0, columns),
            Kokkos::Experimental::WorkItemProperty::HintLightWeight),
        build_factors);

    Kokkos::fence("Build generalized vertical line factors");

    // Construction-time validation only.
    // No host synchronization is added to the captured
    // execution path.
    const auto lower_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), lower_);
    const auto pivot_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), pivot_);
    const auto upper_host =
        Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), upper_normalized_);

    for (int column = 0; column < columns; ++column) {
        for (int k = h; k <= last; ++k) {
            positive(pivot_host(column, k));
            if (!std::isfinite(lower_host(column, k)) || !std::isfinite(upper_host(column, k))) {
                throw std::invalid_argument("Nonfinite generalized "
                                            "vertical solver coefficient.");
            }
        }
    }
}

void
VerticalEllipticSolver::prepare_execution() {

    outside_capture();

#if defined(KOKKOS_ENABLE_CUDA)

    Kokkos::parallel_for("PrepareVerticalEllipticSolver",
        Kokkos::RangePolicy<Kokkos::Cuda>(0, 1),
        KOKKOS_LAMBDA(int){});

    Kokkos::Cuda().fence("Prepare vertical solver "
                         "compilation unit");

#endif
}

void
VerticalEllipticSolver::validate(const Core::Field<3>& field) const {

    const auto data = field.get_device_data();

    if (static_cast<int>(data.extent(0)) != grid_.get_local_total_points_z() ||
        static_cast<int>(data.extent(1)) != grid_.get_local_total_points_y() ||
        static_cast<int>(data.extent(2)) != grid_.get_local_total_points_x()) {

        throw std::invalid_argument("Vertical solver volume "
                                    "extent mismatch.");
    }
}

void
VerticalEllipticSolver::refresh(WorkField& field, int depth) {

    halo_.exchange_halos(field, depth);

    if (boundary_) {
        boundary_->fill_constant_q2_halos(field);
    }
}

void
VerticalEllipticSolver::solve(const Core::Field<3>& xi,
    const Core::Field<3>& eta,
    Core::Field<3>& w,
    Core::Field<3>& previous_w,
    const int iterations) {

    solve_impl(xi, eta, w, previous_w, iterations, VorticityInputRepresentation::PhysicalLegacy);
}

void
VerticalEllipticSolver::solve_from_vvm_contravariant_state(const Core::Field<3>& xi_con,
    const Core::Field<3>& eta_con,
    Core::Field<3>& w,
    Core::Field<3>& previous_w,
    const int iterations) {

    solve_impl(xi_con,
        eta_con,
        w,
        previous_w,
        iterations,
        VorticityInputRepresentation::VvmContravariant);
}

void
VerticalEllipticSolver::solve_impl(const Core::Field<3>& q1_vorticity,
    const Core::Field<3>& q2_vorticity,
    Core::Field<3>& w,
    Core::Field<3>& previous_w,
    const int iterations,
    VorticityInputRepresentation representation) {

    if (iterations <= 0) {
        throw std::invalid_argument("Vertical solver iteration "
                                    "count must be positive.");
    }

    for (const auto* field : {&q1_vorticity,
             &q2_vorticity,
             static_cast<const Core::Field<3>*>(&w),
             static_cast<const Core::Field<3>*>(&previous_w)}) {
        validate(*field);
    }

    const Real* pointers[] = {q1_vorticity.get_device_data().data(),
        q2_vorticity.get_device_data().data(),
        w.get_device_data().data(),
        previous_w.get_device_data().data()};

    for (int a = 0; a < 4; ++a) {
        for (int b = a + 1; b < 4; ++b) {
            if (pointers[a] == pointers[b]) {
                throw std::invalid_argument("Vertical solver fields "
                                            "must not alias.");
            }
        }
    }

    const int nz = grid_.get_local_total_points_z();
    const int ny = grid_.get_local_total_points_y();
    const int nx = grid_.get_local_total_points_x();
    const int h = grid_.get_halo_cells();

    const int last = nz - h - 2;
    const int ni = nx - 2 * h;

    const auto operation = operation_;
    const auto wd = w.get_mutable_device_data();
    const auto history = previous_w.get_mutable_device_data();
    const auto q1 = q1_vorticity.get_device_data();
    const auto q2 = q2_vorticity.get_device_data();
    const auto rhs = rhs_.get_mutable_device_data();
    const auto first = first_.get_mutable_device_data();

    const auto light3 = Kokkos::Experimental::require(
        Kokkos::MDRangePolicy<Kokkos::Rank<3>>({0, 0, 0}, {nz, ny, nx}),
        Kokkos::Experimental::WorkItemProperty::HintLightWeight);

    Kokkos::parallel_for("VerticalGuessAndHistory", light3, KOKKOS_LAMBDA(int k, int j, int i) {
        const Real old = wd(k, j, i);
        Real guess = old;

        if (k >= h && k <= last) {
            guess = real(2.0) * old - history(k, j, i);
        }

        if (k == h - 1 || k == last + 1) {
            guess = real(0.0);
        }
        first(k, j, i) = guess;
        history(k, j, i) = old;
    });

    const auto source_policy = Kokkos::Experimental::require(
        Kokkos::MDRangePolicy<Kokkos::Rank<3>>({h, h, h}, {last + 1, ny - h, nx - h}),
        Kokkos::Experimental::WorkItemProperty::HintLightWeight);

    if (representation == VorticityInputRepresentation::PhysicalLegacy) {
        Kokkos::parallel_for("VerticalWeightedRHS",
            source_policy,
            KOKKOS_LAMBDA(const int k, const int j, const int i) {
                rhs(k, j, i) = operation.calculate_weighted_rhs_at_t(q1, q2, k, j, i);
            });
    }
    else {
        // Canonical:
        //     q1 = xi_con  =  omega^1
        //     q2 = eta_con = -omega^2
        Kokkos::parallel_for("VerticalWeightedRHSContravariant",
            source_policy,
            KOKKOS_LAMBDA(const int k, const int j, const int i) {
                rhs(k, j, i) =
                    operation.calculate_weighted_rhs_from_vvm_contravariant_at_t(q1, q2, k, j, i);
            });
    }

    refresh(first_, 1);

    Kokkos::deep_copy(Kokkos::DefaultExecutionSpace(), second_.get_mutable_device_data(), first);

    WorkField* current = &first_;
    WorkField* previous = &second_;

    const auto lower = lower_;
    const auto pivot = pivot_;
    const auto upper = upper_normalized_;
    const auto temporary = temporary_;
    const auto density = density_up_;
    const Real shift = shift_;

    for (int iteration = 0; iteration < iterations; ++iteration) {
        std::swap(current, previous);
        const auto p = previous->get_device_data();
        const auto c = current->get_mutable_device_data();
        const auto policy =
            Kokkos::Experimental::require(Kokkos::RangePolicy<>(0, (ny - 2 * h) * ni),
                Kokkos::Experimental::WorkItemProperty::HintLightWeight);

        Kokkos::parallel_for("VerticalFixedLineSweep", policy, KOKKOS_LAMBDA(int column) {
            const int j = h + column / ni;
            const int i = h + column % ni;
            for (int k = h; k <= last; ++k) {
                const Real r = shift * p(k, j, i) + rhs(k, j, i) +
                               operation.calculate_horizontal_neighbors_at_t(p, k, j, i);

                temporary(column, k) =
                    (r - (k == h ? real(0.0) : lower(column, k) * temporary(column, k - 1))) /
                    pivot(column, k);
            }

            Real next = temporary(column, last);

            c(last, j, i) = next / density(last);

            for (int k = last - 1; k >= h; --k) {
                const Real value = temporary(column, k) - upper(column, k) * next;
                c(k, j, i) = value / density(k);
                next = value;
            }
            c(h - 1, j, i) = real(0.0);
            c(last + 1, j, i) = real(0.0);
        });

        refresh(*current, iteration + 1 == iterations ? -1 : 1);
    }
    const auto result = current->get_device_data();

    const auto scatter_policy = Kokkos::Experimental::require(
        Kokkos::MDRangePolicy<Kokkos::Rank<3>>({h - 1, 0, 0}, {last + 2, ny, nx}),
        Kokkos::Experimental::WorkItemProperty::HintLightWeight);

    Kokkos::parallel_for("VerticalScatter", scatter_policy, KOKKOS_LAMBDA(int k, int j, int i) {
        wd(k, j, i) = result(k, j, i);
    });
}

} // namespace Dynamics
} // namespace VVM
