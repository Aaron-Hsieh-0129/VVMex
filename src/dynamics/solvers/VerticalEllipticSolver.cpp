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

void outside_capture() {
#if defined(KOKKOS_ENABLE_CUDA)
    cudaStreamCaptureStatus status;
    const auto error = cudaStreamIsCapturing(Kokkos::Cuda().cuda_stream(), &status);
    if (error != cudaSuccess) throw std::runtime_error(cudaGetErrorString(error));
    if (status != cudaStreamCaptureStatusNone) throw std::logic_error("Vertical solver initialization must be outside capture.");
#endif
}

void positive(Real value) {
    if (!std::isfinite(value) || value <= real(0.0)) throw std::invalid_argument("Vertical solver requires finite positive density, stretching and pivots.");
}

const Core::Grid& checked_grid(const Core::Grid& grid) {
    outside_capture();
    return grid;
}

} // namespace

VerticalEllipticSolver::VerticalEllipticSolver(const Core::Grid& grid, Core::HaloExchanger& halo, const Core::Field<1>& rhobar, const Core::Field<1>& rhobar_up, const Core::Field<1>& flex_mid, const Core::Field<1>& flex_up, Real inverse_dz, Real diagonal_shift)
    : grid_(checked_grid(grid)), halo_(halo), operation_(make_vertical_wind_diagnostic_device_view(grid.geometry())), shift_(diagonal_shift),
      rhs_("vertical_rhs", {grid.get_local_total_points_z(), grid.get_local_total_points_y(), grid.get_local_total_points_x()}),
      first_("vertical_first", {grid.get_local_total_points_z(), grid.get_local_total_points_y(), grid.get_local_total_points_x()}),
      second_("vertical_second", {grid.get_local_total_points_z(), grid.get_local_total_points_y(), grid.get_local_total_points_x()}) {
    outside_capture();
    positive(inverse_dz);
    if (!std::isfinite(shift_) || shift_ < real(0.0)) throw std::invalid_argument("Invalid vertical relaxation shift.");
    const auto& horizontal = grid.horizontal_specification();
    if (horizontal.topology.q1 != Core::HorizontalEdgeTopology::Periodic) throw std::invalid_argument("Vertical solver requires periodic q1.");
    if (horizontal.topology.q2 == Core::HorizontalEdgeTopology::Bounded) boundary_ = std::make_unique<Core::Boundary::HorizontalBoundaryStencils>(grid);

    const int nz = grid.get_local_total_points_z(), ny = grid.get_local_total_points_y(), nx = grid.get_local_total_points_x(), h = grid.get_halo_cells();
    const int last = nz - h - 2;
    if (last < h) throw std::invalid_argument("Vertical solver requires an interior w level.");
    for (const auto* field : {&rhobar, &rhobar_up, &flex_mid, &flex_up}) {
        if (int(field->get_device_data().extent(0)) != nz) throw std::invalid_argument("Vertical solver profile extent mismatch.");
    }
    const auto rho = rhobar.get_host_data(), rho_up = rhobar_up.get_host_data();
    const auto mid = flex_mid.get_host_data(), up = flex_up.get_host_data();
    for (int k = h; k <= last + 1; ++k) {
        positive(rho(k));
        positive(mid(k));
        if (k <= last) { positive(rho_up(k)); positive(up(k)); }
    }

    // Coefficients depend on (j,k), not i. Rebind a host copy of the borrowed
    // metric arguments; never dereference device pointers on the host.
    std::vector<Real> jt(ny, real(1.0)), b1(ny, real(1.0)), b2(ny, real(1.0));
    if (grid.geometry().kind() == Core::Geometry::GeometryKind::RegularLatLon) {
        using Core::Geometry::HorizontalLocation;
        const auto t = grid.geometry().device_view(HorizontalLocation::T);
        const auto u = grid.geometry().device_view(HorizontalLocation::U);
        const auto v = grid.geometry().device_view(HorizontalLocation::V);
        const auto a = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), t.sqrt_g.one_dimensional);
        const auto b = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), u.sqrt_g_g_contra.a11.one_dimensional);
        const auto c = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), v.sqrt_g_g_contra.a22.one_dimensional);
        for (int j = 0; j < ny; ++j) { jt[j] = a(j); b1[j] = b(j); b2[j] = c(j); }
    }
    auto host_operation = operation_;
    host_operation.jacobian_t = jt.data();
    host_operation.weighted_inverse_metric_u = b1.data();
    host_operation.weighted_inverse_metric_v = b2.data();

    density_up_ = Kokkos::View<Real*>("vertical_density_up", nz);
    lower_ = Kokkos::View<Real**>("vertical_lower", ny, nz);
    pivot_ = Kokkos::View<Real**>("vertical_pivot", ny, nz);
    upper_normalized_ = Kokkos::View<Real**>("vertical_upper_normalized", ny, nz);
    temporary_ = Kokkos::View<Real**>("vertical_forward", (ny - 2 * h) * (nx - 2 * h), nz);
    auto a = Kokkos::create_mirror(lower_), b = Kokkos::create_mirror(pivot_), c = Kokkos::create_mirror(upper_normalized_);
    Kokkos::deep_copy(a, real(0.0));
    Kokkos::deep_copy(b, real(0.0));
    Kokkos::deep_copy(c, real(0.0));
    for (int j = h; j < ny - h; ++j) {
        for (int k = h; k <= last; ++k) {
            const auto row = host_operation.calculate_row_at_t(rho, rho_up, mid, up, inverse_dz, shift_, k, j, h);
            a(j, k) = row.lower;
            b(j, k) = row.diagonal - (k == h ? real(0.0) : row.lower * c(j, k - 1));
            positive(b(j, k));
            c(j, k) = row.upper / b(j, k);
            if (!std::isfinite(a(j, k)) || !std::isfinite(c(j, k))) throw std::invalid_argument("Nonfinite vertical solver coefficient.");
        }
    }
    Kokkos::deep_copy(density_up_, rhobar_up.get_device_data());
    Kokkos::deep_copy(lower_, a);
    Kokkos::deep_copy(pivot_, b);
    Kokkos::deep_copy(upper_normalized_, c);
}

void VerticalEllipticSolver::prepare_execution() {
    outside_capture();
#if defined(KOKKOS_ENABLE_CUDA)
    Kokkos::parallel_for("PrepareVerticalEllipticSolver", Kokkos::RangePolicy<Kokkos::Cuda>(0, 1), KOKKOS_LAMBDA(int) {});
    Kokkos::Cuda().fence("Prepare vertical solver compilation unit");
#endif
}

void VerticalEllipticSolver::validate(const Core::Field<3>& field) const {
    const auto data = field.get_device_data();
    if (int(data.extent(0)) != grid_.get_local_total_points_z() || int(data.extent(1)) != grid_.get_local_total_points_y() || int(data.extent(2)) != grid_.get_local_total_points_x()) {
        throw std::invalid_argument("Vertical solver volume extent mismatch.");
    }
}

void VerticalEllipticSolver::refresh(WorkField& field, int depth) {
    halo_.exchange_halos(field, depth);
    if (boundary_) boundary_->fill_constant_q2_halos(field);
}

void VerticalEllipticSolver::solve(const Core::Field<3>& xi, const Core::Field<3>& eta, Core::Field<3>& w, Core::Field<3>& previous_w, int iterations) {
    if (iterations <= 0) throw std::invalid_argument("Vertical solver iteration count must be positive.");
    for (const auto* field : {&xi, &eta, static_cast<const Core::Field<3>*>(&w), static_cast<const Core::Field<3>*>(&previous_w)}) validate(*field);
    const Real* pointers[] = {xi.get_device_data().data(), eta.get_device_data().data(), w.get_device_data().data(), previous_w.get_device_data().data()};
    for (int a = 0; a < 4; ++a) for (int b = a + 1; b < 4; ++b) {
        if (pointers[a] == pointers[b]) throw std::invalid_argument("Vertical solver fields must not alias.");
    }
    const int nz = grid_.get_local_total_points_z(), ny = grid_.get_local_total_points_y(), nx = grid_.get_local_total_points_x(), h = grid_.get_halo_cells();
    const int last = nz - h - 2, ni = nx - 2 * h;
    const auto operation = operation_;
    const auto xd = xi.get_device_data(), ed = eta.get_device_data();
    const auto wd = w.get_mutable_device_data(), history = previous_w.get_mutable_device_data();
    const auto rhs = rhs_.get_mutable_device_data(), first = first_.get_mutable_device_data();
    const auto light3 = Kokkos::Experimental::require(Kokkos::MDRangePolicy<Kokkos::Rank<3>>({0, 0, 0}, {nz, ny, nx}), Kokkos::Experimental::WorkItemProperty::HintLightWeight);
    Kokkos::parallel_for("VerticalGuessAndHistory", light3, KOKKOS_LAMBDA(int k, int j, int i) {
        const Real old = wd(k, j, i);
        Real guess = old;
        if (k >= h && k <= last) guess = real(2.0) * old - history(k, j, i);
        if (k == h - 1 || k == last + 1) guess = real(0.0);
        first(k, j, i) = guess;
        history(k, j, i) = old;
    });
    const auto source_policy = Kokkos::Experimental::require(Kokkos::MDRangePolicy<Kokkos::Rank<3>>({h, h, h}, {last + 1, ny - h, nx - h}), Kokkos::Experimental::WorkItemProperty::HintLightWeight);
    Kokkos::parallel_for("VerticalWeightedRHS", source_policy, KOKKOS_LAMBDA(int k, int j, int i) {
        rhs(k, j, i) = operation.calculate_weighted_rhs_at_t(xd, ed, k, j, i);
    });
    refresh(first_, 1);
    Kokkos::deep_copy(Kokkos::DefaultExecutionSpace(), second_.get_mutable_device_data(), first);

    WorkField* current = &first_;
    WorkField* previous = &second_;
    const auto lower = lower_, pivot = pivot_, upper = upper_normalized_, temporary = temporary_;
    const auto density = density_up_;
    const Real shift = shift_;
    for (int iteration = 0; iteration < iterations; ++iteration) {
        std::swap(current, previous);
        const auto p = previous->get_device_data(), c = current->get_mutable_device_data();
        const auto policy = Kokkos::Experimental::require(Kokkos::RangePolicy<>(0, (ny - 2 * h) * ni), Kokkos::Experimental::WorkItemProperty::HintLightWeight);
        Kokkos::parallel_for("VerticalFixedLineSweep", policy, KOKKOS_LAMBDA(int column) {
            const int j = h + column / ni, i = h + column % ni;
            const Real east = operation.coefficient(operation.weighted_inverse_metric_u, j) / (operation.dq1 * operation.dq1);
            const Real north = operation.coefficient(operation.weighted_inverse_metric_v, j) / (operation.dq2 * operation.dq2);
            const Real south = operation.coefficient(operation.weighted_inverse_metric_v, j - 1) / (operation.dq2 * operation.dq2);
            for (int k = h; k <= last; ++k) {
                const Real r = shift * p(k, j, i) + rhs(k, j, i) + east * (p(k, j, i + 1) + p(k, j, i - 1)) + north * p(k, j + 1, i) + south * p(k, j - 1, i);
                temporary(column, k) = (r - (k == h ? real(0.0) : lower(j, k) * temporary(column, k - 1))) / pivot(j, k);
            }
            Real next = temporary(column, last);
            c(last, j, i) = next / density(last);
            for (int k = last - 1; k >= h; --k) {
                const Real value = temporary(column, k) - upper(j, k) * next;
                c(k, j, i) = value / density(k);
                next = value;
            }
            c(h - 1, j, i) = real(0.0);
            c(last + 1, j, i) = real(0.0);
        });
        refresh(*current, iteration + 1 == iterations ? -1 : 1);
    }
    const auto result = current->get_device_data();
    const auto scatter_policy = Kokkos::Experimental::require(Kokkos::MDRangePolicy<Kokkos::Rank<3>>({h - 1, 0, 0}, {last + 2, ny, nx}), Kokkos::Experimental::WorkItemProperty::HintLightWeight);
    Kokkos::parallel_for("VerticalScatter", scatter_policy, KOKKOS_LAMBDA(int k, int j, int i) { wd(k, j, i) = result(k, j, i); });
}

} // namespace Dynamics
} // namespace VVM

