#ifndef VVM_CORE_GEOMETRY_HORIZONTAL_GEOMETRY_HPP
#define VVM_CORE_GEOMETRY_HORIZONTAL_GEOMETRY_HPP

#include <cstdint>
#include <optional>
#include <stdexcept>
#include <string>
#include <Kokkos_Core.hpp>

#include "core/GridStaggering.hpp"
#include "core/vvm_types.hpp"
#include "core/geometry/GeometryKind.hpp"
#include "core/geometry/HorizontalLocation.hpp"
#include "core/geometry/HorizontalStaggering.hpp"


namespace VVM {
namespace Core {
namespace Geometry {

// Geometry only needs layout information. It does not need Grid or MPI.
struct HorizontalDomainLayout {
    int global_nx = 0;
    int global_ny = 0;

    int local_physical_nx = 0;
    int local_physical_ny = 0;

    int global_start_i = 0;
    int global_start_j = 0;

    int halo = 0;

    int panel_id = -1; // For cubed sphere

    int local_total_nx() const noexcept {
        return local_physical_nx + 2 * halo;
    }

    int local_total_ny() const noexcept {
        return local_physical_ny + 2 * halo;
    }
};

// Concrete geometry classes own mutable Kokkos::View<VVM::Real**> arrays.
using GeometryConstView2D = Kokkos::View<const VVM::Real**>;

// A symmetric horizontal tensor:
//
//     [ a11  a12 ]
//     [ a12  a22 ]
//
// Used for:
//   - the covariant metric g_ij
//   - the Jacobian-weighted contravariant metric J*g^ij
struct SymmetricTensorDeviceView {
    GeometryConstView2D a11;
    GeometryConstView2D a12;
    GeometryConstView2D a22;

    KOKKOS_INLINE_FUNCTION
    void apply(
        const int j,
        const int i,
        const VVM::Real x1,
        const VVM::Real x2,
        VVM::Real& y1,
        VVM::Real& y2) const noexcept {

        y1 = a11(j, i) * x1 + a12(j, i) * x2;
        y2 = a12(j, i) * x1 + a22(j, i) * x2;
    }

    KOKKOS_INLINE_FUNCTION
    VVM::Real component_1(
        const int j,
        const int i,
        const VVM::Real x1,
        const VVM::Real x2) const noexcept {

        return a11(j, i) * x1 + a12(j, i) * x2;
    }

    KOKKOS_INLINE_FUNCTION
    VVM::Real component_2(
        const int j,
        const int i,
        const VVM::Real x1,
        const VVM::Real x2) const noexcept {

        return a12(j, i) * x1 + a22(j, i) * x2;
    }
};

// A general 2x2 transformation matrix:
//
//     [ a11  a12 ]
//     [ a21  a22 ]
//
// This is used for transformations between computational contravariant
// components and physical tangent-plane components.
struct Matrix2DeviceView {
    GeometryConstView2D a11;
    GeometryConstView2D a12;
    GeometryConstView2D a21;
    GeometryConstView2D a22;

    KOKKOS_INLINE_FUNCTION
    void apply(
        const int j,
        const int i,
        const VVM::Real x1,
        const VVM::Real x2,
        VVM::Real& y1,
        VVM::Real& y2) const noexcept {

        y1 = a11(j, i) * x1 + a12(j, i) * x2;
        y2 = a21(j, i) * x1 + a22(j, i) * x2;
    }

    KOKKOS_INLINE_FUNCTION
    VVM::Real component_1(
        const int j,
        const int i,
        const VVM::Real x1,
        const VVM::Real x2) const noexcept {

        return a11(j, i) * x1 + a12(j, i) * x2;
    }

    KOKKOS_INLINE_FUNCTION
    VVM::Real component_2(
        const int j,
        const int i,
        const VVM::Real x1,
        const VVM::Real x2) const noexcept {

        return a21(j, i) * x1 + a22(j, i) * x2;
    }
};


// ============================================================================
// Geometry data for one horizontal location
// ============================================================================

// Copying this object copies Kokkos view handles, not the underlying arrays.
// Coordinate conventions:
//   q1, q2
//       Computational coordinates.
//       Cartesian:
//           q1 and q2 are physical distances in metres.
//       Regular latitude-longitude:
//           q1 = longitude and q2 = latitude, in radians.
//       Cubed sphere:
//           q1 and q2 are panel coordinates, usually in radians.
//
//   longitude, latitude
//       Geographic coordinates in radians.
//       Cartesian geometry may fill these with zero or with a configured geographic reference mapping.
//
// Metric conventions:
//   sqrt_g
//       J = sqrt(det(g_ij))
//   inv_sqrt_g
//       1/J
//   g_cov
//       Covariant metric g_ij
//   sqrt_g_g_contra
//       J*g^ij
//
//       These are the coefficients used directly by conservative divergence and Laplace-Beltrami operators.
//
// Component conventions:
//   physical_to_contravariant
//       Maps physical tangent-plane components to computational contravariant components:
//
//           (U,V) -> (u^1,u^2)
//
//   contravariant_to_physical
//       Maps computational contravariant components back to physical
//       tangent-plane components:
//
//           (u^1,u^2) -> (U,V)
//
// For Cartesian geometry, physical components are x/y velocity.
// For spherical geometries, physical components are normally eastward and  northward velocity in metres per second.
struct HorizontalGeometryDeviceView {
    GeometryConstView2D q1;
    GeometryConstView2D q2;

    GeometryConstView2D longitude;
    GeometryConstView2D latitude;

    GeometryConstView2D sqrt_g;
    GeometryConstView2D inv_sqrt_g;

    SymmetricTensorDeviceView g_cov;
    SymmetricTensorDeviceView sqrt_g_g_contra;

    Matrix2DeviceView physical_to_contravariant;
    Matrix2DeviceView contravariant_to_physical;

    VVM::Real dq1 = VVM::real(0.0);
    VVM::Real dq2 = VVM::real(0.0);

    // Recover the unweighted inverse metric when an operator needs g^ij.
    // Most conservative operators should use sqrt_g_g_contra directly.
    KOKKOS_INLINE_FUNCTION
    VVM::Real g_contra_11(const int j, const int i) const noexcept {
        return inv_sqrt_g(j, i) * sqrt_g_g_contra.a11(j, i);
    }

    KOKKOS_INLINE_FUNCTION
    VVM::Real g_contra_12(const int j, const int i) const noexcept {
        return inv_sqrt_g(j, i) * sqrt_g_g_contra.a12(j, i);
    }

    KOKKOS_INLINE_FUNCTION
    VVM::Real g_contra_22(const int j, const int i) const noexcept {
        return inv_sqrt_g(j, i) * sqrt_g_g_contra.a22(j, i);
    }
};

// Abstract host-side geometry interface
// HorizontalGeometry is immutable after construction.
// Runtime geometry selection happens on the host. Device kernels receive a HorizontalGeometryDeviceView and never call a virtual method.
class HorizontalGeometry {
public:
    virtual ~HorizontalGeometry() = default;

    HorizontalGeometry(const HorizontalGeometry&) = delete;
    HorizontalGeometry& operator=(const HorizontalGeometry&) = delete;

    HorizontalGeometry(HorizontalGeometry&&) = delete;
    HorizontalGeometry& operator=(HorizontalGeometry&&) = delete;

    virtual GeometryKind kind() const noexcept = 0;
    virtual const char* name() const noexcept = 0;

    virtual const HorizontalDomainLayout& layout() const noexcept = 0;

    // Uniform computational-coordinate spacing.
    // Cartesian:
    //     dq1 and dq2 are physical distances in metres.
    //
    // Regular latitude-longitude and cubed sphere:
    //     dq1 and dq2 are angular computational-coordinate increments in radians.
    virtual VVM::Real dq1() const noexcept = 0;
    virtual VVM::Real dq2() const noexcept = 0;

    // Select geometry explicitly using T/U/V/Z.
    HorizontalGeometryDeviceView device_view(const HorizontalLocation location) const {
        return device_view_impl(location);
    }

    // Select geometry from existing VVMex FieldMetadata.
    // The field name is optional but produces a more useful exception when
    // the field has Unspecified or NotApplicable staggering.
    HorizontalGeometryDeviceView device_view(const GridStaggering staggering, const std::string& field_name = {}) const {
        return device_view_impl(horizontal_location_or_throw(staggering, field_name));
    }


protected:
    HorizontalGeometry() = default;

    virtual HorizontalGeometryDeviceView device_view_impl(HorizontalLocation location) const = 0;
};

} // namespace Geometry
} // namespace Core
} // namespace VVM

#endif // VVM_CORE_GEOMETRY_HORIZONTAL_GEOMETRY_HPP
