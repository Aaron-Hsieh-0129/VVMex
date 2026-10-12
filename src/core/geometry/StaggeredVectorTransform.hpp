#ifndef VVM_CORE_GEOMETRY_STAGGERED_VECTOR_TRANSFORM_HPP
#define VVM_CORE_GEOMETRY_STAGGERED_VECTOR_TRANSFORM_HPP

#include "core/geometry/HorizontalGeometry.hpp"

namespace VVM::Core::Geometry {

// Cross components are weighted at their native face before interpolation.
// Only the two coefficients needed for one output are captured per launch.
// Light-weight policies pass them as kernel arguments during CUDA capture.
// The outermost allocation edge is clamped; physical cells and stencil halos
// use the ordinary four-face average, followed by the caller's halo exchange.
template <bool TargetU, class Volume, class Coefficient>
KOKKOS_INLINE_FUNCTION Real
weighted_cross(const Volume& source, const Coefficient& coefficient, int k, int j, int i) {
    const int nj = static_cast<int>(source.extent(1));
    const int ni = static_cast<int>(source.extent(2));
    const int j2 = TargetU ? (j > 0 ? j - 1 : j) : (j + 1 < nj ? j + 1 : j);
    const int i2 = TargetU ? (i + 1 < ni ? i + 1 : i) : (i > 0 ? i - 1 : i);
    return real(0.25) *
           (coefficient(j, i) * source(k, j, i) + coefficient(j2, i) * source(k, j2, i) +
               coefficient(j, i2) * source(k, j, i2) + coefficient(j2, i2) * source(k, j2, i2));
}

template <bool FirstAtU, class Volume1, class Volume2, class Output1, class Output2>
void
transform_staggered_vector(const Volume1& first,
    const Volume2& second,
    const Output1& output_first,
    const Output2& output_second,
    const Core::Geometry::Matrix2DeviceView& matrix_at_first,
    const Core::Geometry::Matrix2DeviceView& matrix_at_second,
    Real cross_sign = real(1.0),
    bool add = false) {
    const int nz = static_cast<int>(first.extent(0));
    const int ny = static_cast<int>(first.extent(1));
    const int nx = static_cast<int>(first.extent(2));
    const auto a11 = matrix_at_first.a11, a12 = matrix_at_second.a12;
    const auto a21 = matrix_at_first.a21, a22 = matrix_at_second.a22;
    const auto zero = [](const auto& field) {
        return field.layout == GeometryFieldLayout::Constant && field.constant == real(0.0);
    };
    if (zero(a12) && zero(a21)) {
        Kokkos::parallel_for("TransformDiagonalStaggeredVector",
            Kokkos::Experimental::require(
                Kokkos::MDRangePolicy<Kokkos::Rank<3>>({0, 0, 0}, {nz, ny, nx}),
                Kokkos::Experimental::WorkItemProperty::HintLightWeight),
            KOKKOS_LAMBDA(int k, int j, int i) {
                if (add) {
                    output_first(k, j, i) += a11(j, i) * first(k, j, i);
                    output_second(k, j, i) += a22(j, i) * second(k, j, i);
                }
                else {
                    output_first(k, j, i) = a11(j, i) * first(k, j, i);
                    output_second(k, j, i) = a22(j, i) * second(k, j, i);
                }
            });
        return;
    }
    // In/out must not alias: each output uses the other input's neighbors.
    Kokkos::parallel_for("TransformStaggeredVectorFirst",
        Kokkos::Experimental::require(
            Kokkos::MDRangePolicy<Kokkos::Rank<3>>({0, 0, 0}, {nz, ny, nx}),
            Kokkos::Experimental::WorkItemProperty::HintLightWeight),
        KOKKOS_LAMBDA(int k, int j, int i) {
            const Real value = a11(j, i) * first(k, j, i) +
                               cross_sign * weighted_cross<FirstAtU>(second, a12, k, j, i);
            if (add) {
                output_first(k, j, i) += value;
            }
            else {
                output_first(k, j, i) = value;
            }
        });
    Kokkos::parallel_for("TransformStaggeredVectorSecond",
        Kokkos::Experimental::require(
            Kokkos::MDRangePolicy<Kokkos::Rank<3>>({0, 0, 0}, {nz, ny, nx}),
            Kokkos::Experimental::WorkItemProperty::HintLightWeight),
        KOKKOS_LAMBDA(int k, int j, int i) {
            const Real value = a22(j, i) * second(k, j, i) +
                               cross_sign * weighted_cross<!FirstAtU>(first, a21, k, j, i);
            if (add) {
                output_second(k, j, i) += value;
            }
            else {
                output_second(k, j, i) = value;
            }
        });
}

struct InverseMetricCoefficient {
    Core::Geometry::GeometryField2D weighted, inverse_jacobian;
    KOKKOS_INLINE_FUNCTION Real
    operator()(int j, int i) const {
        return weighted(j, i) * inverse_jacobian(j, i);
    }
};

template <class Volume1, class Volume2, class Output1, class Output2>
void
raise_staggered_wind(const Volume1& first,
    const Volume2& second,
    const Output1& output_first,
    const Output2& output_second,
    const Core::Geometry::HorizontalGeometry& geometry) {
    using Core::Geometry::HorizontalLocation;
    const auto gu = geometry.device_view(HorizontalLocation::U);
    const auto gv = geometry.device_view(HorizontalLocation::V);
    const int nz = static_cast<int>(first.extent(0));
    const int ny = static_cast<int>(first.extent(1));
    const int nx = static_cast<int>(first.extent(2));
    // Keep the compact coefficients on device to avoid large graph captures.
    // g^12 is evaluated at the source face before interpolation, matching
    // the generalized Helmholtz reconstruction.
    const auto inv_ju = gu.inv_sqrt_g, inv_jv = gv.inv_sqrt_g;
    const auto c11 = gu.sqrt_g_g_contra.a11, c12v = gv.sqrt_g_g_contra.a12;
    const auto c22 = gv.sqrt_g_g_contra.a22, c12u = gu.sqrt_g_g_contra.a12;
    const auto zero = [](const auto& field) {
        return field.layout == GeometryFieldLayout::Constant && field.constant == real(0.0);
    };
    if (zero(c12v) && zero(c12u)) {
        Kokkos::parallel_for("RaiseDiagonalStaggeredWind",
            Kokkos::Experimental::require(
                Kokkos::MDRangePolicy<Kokkos::Rank<3>>({0, 0, 0}, {nz, ny, nx}),
                Kokkos::Experimental::WorkItemProperty::HintLightWeight),
            KOKKOS_LAMBDA(int k, int j, int i) {
                output_first(k, j, i) = inv_ju(j, i) * c11(j, i) * first(k, j, i);
                output_second(k, j, i) = inv_jv(j, i) * c22(j, i) * second(k, j, i);
            });
        return;
    }
    Kokkos::parallel_for("RaiseStaggeredWindQ1",
        Kokkos::Experimental::require(
            Kokkos::MDRangePolicy<Kokkos::Rank<3>>({0, 0, 0}, {nz, ny, nx}),
            Kokkos::Experimental::WorkItemProperty::HintLightWeight),
        KOKKOS_LAMBDA(int k, int j, int i) {
            output_first(k, j, i) =
                inv_ju(j, i) * c11(j, i) * first(k, j, i) +
                weighted_cross<true>(second, InverseMetricCoefficient{c12v, inv_jv}, k, j, i);
        });
    Kokkos::parallel_for("RaiseStaggeredWindQ2",
        Kokkos::Experimental::require(
            Kokkos::MDRangePolicy<Kokkos::Rank<3>>({0, 0, 0}, {nz, ny, nx}),
            Kokkos::Experimental::WorkItemProperty::HintLightWeight),
        KOKKOS_LAMBDA(int k, int j, int i) {
            output_second(k, j, i) =
                inv_jv(j, i) * c22(j, i) * second(k, j, i) +
                weighted_cross<false>(first, InverseMetricCoefficient{c12u, inv_ju}, k, j, i);
        });
}

} // namespace VVM::Core::Geometry
#endif
