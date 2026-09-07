#ifndef VVM_DYNAMICS_SOLVERS_VERTICAL_WIND_DIAGNOSTIC_HPP
#define VVM_DYNAMICS_SOLVERS_VERTICAL_WIND_DIAGNOSTIC_HPP

#include <cmath>
#include <cstddef>
#include <stdexcept>

#include "core/geometry/HorizontalGeometry.hpp"

namespace VVM {
namespace Dynamics {

// Coefficients for the existing local vertical solve. The line unknown is
// mass flux m(k) = rhobar_up(k) * w(k), not physical w itself.
struct VerticalWindEllipticRow {
    Real lower = real(0.0);
    Real diagonal = real(0.0);
    Real upper = real(0.0);
    Real east = real(0.0);
    Real west = real(0.0);
    Real north = real(0.0);
    Real south = real(0.0);
    Real shift = real(0.0);

    template<typename View>
    KOKKOS_INLINE_FUNCTION
    Real horizontal_neighbors(const View& previous, int k, int j, int i) const noexcept {
        return east * previous(k, j, i + 1) + west * previous(k, j, i - 1)
             + north * previous(k, j + 1, i) + south * previous(k, j - 1, i);
    }

    template<typename View>
    KOKKOS_INLINE_FUNCTION
    Real line_rhs(const View& previous, Real weighted_rhs, int k, int j, int i) const noexcept {
        return shift * previous(k, j, i) + weighted_rhs + horizontal_neighbors(previous, k, j, i);
    }

    template<typename View, typename Profile>
    KOKKOS_INLINE_FUNCTION
    Real jacobi_value(const View& previous, const Profile& rhobar_up, Real weighted_rhs, int k, int j, int i) const noexcept {
        const Real denominator = diagonal * rhobar_up(k) - shift;
        if (denominator == real(0.0)) return previous(k, j, i);

        const Real vertical = -lower * rhobar_up(k - 1) * previous(k - 1, j, i) - upper * rhobar_up(k + 1) * previous(k + 1, j, i);
        return (weighted_rhs + horizontal_neighbors(previous, k, j, i) + vertical) / denominator;
    }
};

// Borrowed compact kernel arguments, not another geometry owner. Construct on
// the host before capture. Geometry storage must outlive kernels and graphs.
// Cartesian/RLL only: the RLL pointers address existing latitude arrays.
// No allocation, launch, halo exchange, synchronization or iteration control.
struct VerticalWindDiagnosticDeviceView {
    const Real* jacobian_t = nullptr;
    const Real* jacobian_z = nullptr;
    const Real* h1_u = nullptr;
    const Real* h1_v = nullptr;
    const Real* weighted_inverse_metric_u = nullptr;
    const Real* weighted_inverse_metric_v = nullptr;
    Real h2 = real(1.0);
    Real dq1 = real(1.0);
    Real dq2 = real(1.0);

    KOKKOS_INLINE_FUNCTION
    Real coefficient(const Real* values, int j) const noexcept {
        return values ? values[j] : real(1.0);
    }

    // CVVM RELAX_3D YTEM after conversion from canonical contravariant
    // vorticity to physical State xi and legacy eta. This RHS ALREADY contains
    // J. Do not multiply by J again in the vertical solver.
    template<typename XiView, typename EtaView>
    KOKKOS_INLINE_FUNCTION
    Real calculate_weighted_rhs_at_t(const XiView& xi, const EtaView& eta, int k, int j, int i) const noexcept {
        return -h2 * (eta(k, j, i) - eta(k, j, i - 1)) / dq1
               -(coefficient(h1_v, j) * xi(k, j, i) - coefficient(h1_v, j - 1) * xi(k, j - 1, i)) / dq2;
    }

    // D = div_h(omega_h); d_z(zeta) = -D. Physical State eta is minus the
    // canonical northward vorticity. Native source staggering is retained.
    template<typename XiView, typename EtaView>
    KOKKOS_INLINE_FUNCTION
    Real calculate_vorticity_divergence_at_z(const XiView& xi, const EtaView& eta, int k, int j, int i) const noexcept {
        return (h2 * (xi(k, j, i + 1) - xi(k, j, i)) / dq1
               -(coefficient(h1_u, j + 1) * eta(k, j + 1, i) - coefficient(h1_u, j) * eta(k, j, i)) / dq2)
               / coefficient(jacobian_z, j);
    }

    // Keep zeta(top) unchanged. spacing(k) = dz/flex_up(k). The optional
    // first upper ghost follows CVVM ZETA_DIAG and VVMex's upward formula.
    // Caller guarantees valid halos, nonaliasing inputs/output, bottom>=0,
    // bottom<=top, and storage through top+1 when extend_upper_ghost is true.
    template<typename XiView, typename EtaView, typename Profile, typename ZetaView>
    KOKKOS_INLINE_FUNCTION
    void integrate_zeta_column(const XiView& xi, const EtaView& eta, const Profile& spacing, const ZetaView& zeta, int bottom, int top, int j, int i, bool extend_upper_ghost) const noexcept {
        for (int k = top - 1; k >= bottom; --k) {
            zeta(k, j, i) = zeta(k + 1, j, i) + spacing(k) * calculate_vorticity_divergence_at_z(xi, eta, k, j, i);
        }

        if (extend_upper_ghost) {
            zeta(top + 1, j, i) = zeta(top, j, i) - spacing(top) * calculate_vorticity_divergence_at_z(xi, eta, k_top(top), j, i);
        }
    }

    // Interior row; physical walls and vertical end rows belong to the caller.
    // Profiles are rhobar(k), rhobar_up(k), flex_mid(k), flex_up(k).
    // All density/stretching entries and inverse_dz must be finite and positive.
    // shift is the CVVM WRXMU coefficient in the J-weighted algebraic equation.
    template<typename Rho, typename RhoUp, typename FlexMid, typename FlexUp>
    KOKKOS_INLINE_FUNCTION
    VerticalWindEllipticRow calculate_row_at_t(const Rho& rho, const RhoUp& rho_up, const FlexMid& flex_mid, const FlexUp& flex_up, Real inverse_dz, Real shift, int k, int j, int i) const noexcept {
        (void)i;

        VerticalWindEllipticRow row;
        row.east = coefficient(weighted_inverse_metric_u, j) / (dq1 * dq1);
        row.west = row.east;
        row.north = coefficient(weighted_inverse_metric_v, j) / (dq2 * dq2);
        row.south = coefficient(weighted_inverse_metric_v, j - 1) / (dq2 * dq2);
        row.shift = shift;

        const Real vertical = coefficient(jacobian_t, j) * flex_up(k) * inverse_dz * inverse_dz;
        row.lower = -vertical * flex_mid(k) / rho(k);
        row.upper = -vertical * flex_mid(k + 1) / rho(k + 1);
        row.diagonal = (shift + row.east + row.west + row.north + row.south) / rho_up(k) - row.lower - row.upper;

        return row;
    }

private:
    KOKKOS_INLINE_FUNCTION
    static int k_top(int top) noexcept {
        return top;
    }
};

inline VerticalWindDiagnosticDeviceView make_vertical_wind_diagnostic_device_view(const Core::Geometry::HorizontalGeometry& geometry) {
    using namespace Core::Geometry;

    if (geometry.kind() != GeometryKind::Cartesian && geometry.kind() != GeometryKind::RegularLatLon) {
        throw std::invalid_argument("Vertical wind diagnostic supports only Cartesian and regular latitude-longitude geometry.");
    }

    if (geometry.layout().halo < 1 || geometry.layout().local_physical_nx < 2 || geometry.layout().local_physical_ny < 2) {
        throw std::invalid_argument("Vertical wind diagnostic requires two active horizontal axes and at least one halo cell.");
    }

    VerticalWindDiagnosticDeviceView result;
    result.dq1 = geometry.dq1();
    result.dq2 = geometry.dq2();

    if (!std::isfinite(result.dq1) || !std::isfinite(result.dq2) || result.dq1 <= real(0.0) || result.dq2 <= real(0.0)) {
        throw std::invalid_argument("Invalid horizontal coordinate increments.");
    }

    if (geometry.kind() == GeometryKind::Cartesian) return result;

    const auto t = geometry.device_view(HorizontalLocation::T);
    const auto u = geometry.device_view(HorizontalLocation::U);
    const auto v = geometry.device_view(HorizontalLocation::V);
    const auto z = geometry.device_view(HorizontalLocation::Z);
    const auto ny = static_cast<std::size_t>(geometry.layout().local_total_ny());

    const auto latitude_data = [ny](const GeometryField2D& field) {
        if (field.layout != GeometryFieldLayout::VaryingJ || field.one_dimensional.extent(0) != ny || !field.one_dimensional.data()) {
            throw std::invalid_argument("RLL vertical diagnostic requires the existing latitude-only metric storage.");
        }
        return field.one_dimensional.data();
    };

    if (v.contravariant_to_physical.a22.layout != GeometryFieldLayout::Constant) {
        throw std::invalid_argument("RLL vertical diagnostic requires constant meridional scale factor.");
    }

    result.h2 = v.contravariant_to_physical.a22.constant;
    if (!std::isfinite(result.h2) || result.h2 <= real(0.0)) throw std::invalid_argument("Invalid meridional scale factor.");

    result.jacobian_t = latitude_data(t.sqrt_g);
    result.jacobian_z = latitude_data(z.sqrt_g);
    result.h1_u = latitude_data(u.contravariant_to_physical.a11);
    result.h1_v = latitude_data(v.contravariant_to_physical.a11);
    result.weighted_inverse_metric_u = latitude_data(u.sqrt_g_g_contra.a11);
    result.weighted_inverse_metric_v = latitude_data(v.sqrt_g_g_contra.a22);

    return result;
}

} // namespace Dynamics
} // namespace VVM

#endif
