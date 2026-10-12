#ifndef VVM_CORE_GEOMETRY_AFFINE_GEOMETRY_HPP
#define VVM_CORE_GEOMETRY_AFFINE_GEOMETRY_HPP

#include <array>
#include <cmath>
#include <stdexcept>
#include "core/geometry/CartesianGeometry.hpp"

namespace VVM::Core::Geometry {

// A planar mapped chart x=B*q. Computational coordinates have metre units;
// B may rotate, scale and skew the horizontal basis. This provider uses the
// same mapped model backend as geographic charts, including mixed metrics.
class AffineGeometry final : public HorizontalGeometry {
public:
    AffineGeometry(HorizontalDomainLayout layout,
        Real dq1,
        Real dq2,
        const std::array<Real, 4>& b,
        Real longitude = real(0.0),
        Real latitude = real(0.0))
        : coordinates_(layout, dq1, dq2), b_(b), longitude_(longitude), latitude_(latitude) {
        for (const Real value : b_) {
            if (!std::isfinite(value)) {
                throw std::invalid_argument("Affine basis entries must be finite.");
            }
        }
        determinant_ = b_[0] * b_[3] - b_[1] * b_[2];
        if (!std::isfinite(determinant_) || determinant_ <= real(0.0)) {
            throw std::invalid_argument(
                "Affine basis must be nonsingular and positively oriented.");
        }
        const Real g11 = b_[0] * b_[0] + b_[2] * b_[2], g22 = b_[1] * b_[1] + b_[3] * b_[3];
        if (!std::isfinite(g11) || !std::isfinite(g22) ||
            !std::isfinite((g11 / determinant_) / determinant_) ||
            !std::isfinite((g22 / determinant_) / determinant_)) {
            throw std::invalid_argument("Affine metric and inverse metric must be finite.");
        }
        if (!std::isfinite(real(1.0) / determinant_)) {
            throw std::invalid_argument("Affine inverse Jacobian must be finite.");
        }
    }
    GeometryKind
    kind() const noexcept override {
        return GeometryKind::Affine;
    }
    const char*
    name() const noexcept override {
        return "affine";
    }
    const HorizontalDomainLayout&
    layout() const noexcept override {
        return coordinates_.layout();
    }
    Real
    dq1() const noexcept override {
        return coordinates_.dq1();
    }
    Real
    dq2() const noexcept override {
        return coordinates_.dq2();
    }

protected:
    HorizontalGeometryDeviceView
    device_view_impl(HorizontalLocation location) const override {
        const auto c = [](Real value) {
            return GeometryField2D::constant_value(value);
        };
        auto result = coordinates_.device_view(location);
        const Real g11 = b_[0] * b_[0] + b_[2] * b_[2];
        const Real g12 = b_[0] * b_[1] + b_[2] * b_[3];
        const Real g22 = b_[1] * b_[1] + b_[3] * b_[3];
        result.longitude = c(longitude_);
        result.latitude = c(latitude_);
        result.sqrt_g = c(determinant_);
        result.inv_sqrt_g = c(real(1.0) / determinant_);
        result.g_cov = {c(g11), c(g12), c(g22)};
        result.sqrt_g_g_contra = {c(g22 / determinant_),
            c(-g12 / determinant_),
            c(g11 / determinant_)};
        result.contravariant_to_physical = {c(b_[0]), c(b_[1]), c(b_[2]), c(b_[3])};
        result.physical_to_contravariant = {c(b_[3] / determinant_),
            c(-b_[1] / determinant_),
            c(-b_[2] / determinant_),
            c(b_[0] / determinant_)};
        return result;
    }

private:
    CartesianGeometry coordinates_;
    std::array<Real, 4> b_;
    Real determinant_, longitude_, latitude_;
};
}
#endif
