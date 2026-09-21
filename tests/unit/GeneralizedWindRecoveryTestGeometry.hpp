#ifndef VVM_TEST_GENERALIZED_WIND_RECOVERY_GEOMETRY_HPP
#define VVM_TEST_GENERALIZED_WIND_RECOVERY_GEOMETRY_HPP

#include <array>
#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>
#include "core/geometry/HorizontalGeometry.hpp"

namespace WindRecoveryTest {
using VVM::Real;
using VVM::real;
using namespace VVM::Core::Geometry;

inline void
require(bool ok, const std::string& message) {
    if (!ok) {
        throw std::runtime_error(message);
    }
}

struct Tensor {
    Real J, g11, g12, g22;
    Real c11, c12, c22;
};

// Test-only planar chart, NOT a cubed sphere:
// x=2*p+s*q+b*p*q, y=3*q+d*p*q. The varying case has a genuinely
// two-dimensional Jacobian and cross metric; the affine case has b=d=0.
class Geometry final : public HorizontalGeometry {
public:
    Geometry(Real shear, bool varying, int halo = 2)
        : s_(shear), b_(varying ? real(.17) : real(0)), d_(varying ? real(.11) : real(0)) {
        layout_.global_nx = layout_.local_physical_nx = 9;
        layout_.global_ny = layout_.local_physical_ny = 7;
        layout_.halo = halo;
        for (int l = 0; l < 4; ++l) {
            auto& storage = metrics_[l];
            for (auto& field : storage) {
                field = Kokkos::View<Real**>("test_chart_metric", ny(), nx());
            }
            std::array<Kokkos::View<Real**>::HostMirror, 8> mirrors;
            for (int m = 0; m < 8; ++m) {
                mirrors[m] = Kokkos::create_mirror_view(storage[m]);
            }
            for (int j = 0; j < ny(); ++j) {
                for (int i = 0; i < nx(); ++i) {
                    const auto t = metric(l, j, i);
                    const std::array<Real, 8> values = {t.J,
                        real(1) / t.J,
                        t.g11,
                        t.g12,
                        t.g22,
                        t.J * t.c11,
                        t.J * t.c12,
                        t.J * t.c22};
                    for (int m = 0; m < 8; ++m) {
                        mirrors[m](j, i) = values[m];
                    }
                }
            }
            for (int m = 0; m < 8; ++m) {
                Kokkos::deep_copy(storage[m], mirrors[m]);
            }
        }
    }
    // Use the reserved non-Cartesian enum to expose accidental kind guards.
    // This fixture is never installed in Grid or the production factory.
    GeometryKind
    kind() const noexcept override {
        return GeometryKind::CubedSphere;
    }
    const char*
    name() const noexcept override {
        return "test_skew_planar_chart";
    }
    const HorizontalDomainLayout&
    layout() const noexcept override {
        return layout_;
    }
    Real
    dq1() const noexcept override {
        return real(.25);
    }
    Real
    dq2() const noexcept override {
        return real(.375);
    }
    int
    nx() const {
        return layout_.local_total_nx();
    }
    int
    ny() const {
        return layout_.local_total_ny();
    }
    int
    h() const {
        return layout_.halo;
    }
    // Index: T=0, U=1, V=2, Z=3.
    Real
    p(int l, int i) const {
        return (real(i - h()) + (l == 1 || l == 3 ? real(.5) : real(0))) * dq1();
    }
    Real
    q(int l, int j) const {
        return (real(j - h()) + (l == 2 || l == 3 ? real(.5) : real(0))) * dq2();
    }
    Tensor
    metric(int l, int j, int i) const {
        const Real pp = p(l, i), qq = q(l, j);
        const Real x1 = real(2) + b_ * qq, y1 = d_ * qq;
        const Real x2 = s_ + b_ * pp, y2 = real(3) + d_ * pp;
        const Real J = x1 * y2 - x2 * y1;
        const Real g11 = x1 * x1 + y1 * y1, g12 = x1 * x2 + y1 * y2, g22 = x2 * x2 + y2 * y2;
        require(J > real(0), "invalid manufactured chart");
        return {J, g11, g12, g22, g22 / (J * J), -g12 / (J * J), g11 / (J * J)};
    }

protected:
    HorizontalGeometryDeviceView
    device_view_impl(HorizontalLocation loc) const override {
        const int l = loc == HorizontalLocation::T   ? 0
                      : loc == HorizontalLocation::U ? 1
                      : loc == HorizontalLocation::V ? 2
                                                     : 3;
        const auto f = [&](int m) {
            return GeometryField2D::full_2d(metrics_[l][m]);
        };
        HorizontalGeometryDeviceView v;
        v.sqrt_g = f(0);
        v.inv_sqrt_g = f(1);
        v.g_cov = {f(2), f(3), f(4)};
        v.sqrt_g_g_contra = {f(5), f(6), f(7)};
        v.dq1 = dq1();
        v.dq2 = dq2();
        v.rdq1 = real(1) / dq1();
        v.rdq2 = real(1) / dq2();
        // Physical conversion fields deliberately remain unset: recovery
        // must need only metric/J data, not a physical-component adapter.
        return v;
    }

private:
    HorizontalDomainLayout layout_;
    Real s_, b_, d_;
    std::array<std::array<Kokkos::View<Real**>, 8>, 4> metrics_;
};

inline Real
psi(Real p, Real q, int version, int mode) {
    if (mode == 0 || mode == 2) {
        return real(.625);
    }
    return real(1) + real(.125) * version +
           (real(1) + real(.0625) * version) *
               (real(.3) * p * p - real(.2) * q * q + real(.07) * p * q + real(.05) * p * p * q);
}
inline Real
chi(Real p, Real q, int version, int mode) {
    if (mode == 0 || mode == 1) {
        return real(-.875);
    }
    return -real(1) + real(.25) * version +
           (real(1) - real(.03125) * version) *
               (real(.1) * p * p + real(.25) * q * q - real(.13) * p * q + real(.02) * p * q * q);
}

// Independent host oracle: explicitly form true rotational components,
// raise chi using the inverse metric, and lower ONLY the rotational wind
// using the covariant metric. It does not call any production operator or
// use production J*g^ij identities for the covariant reconstruction.
inline std::array<Real, 4>
reference(const Geometry& g, int j, int i, int version, int mode) {
    const auto ps = [&](int y, int x) {
        return psi(g.p(3, x), g.q(3, y), version, mode);
    };
    const auto ch = [&](int y, int x) {
        return chi(g.p(0, x), g.q(0, y), version, mode);
    };
    const auto r1 = [&](int y, int x) {
        return -(ps(y, x) - ps(y - 1, x)) / (g.metric(1, y, x).J * g.dq2());
    };
    const auto r2 = [&](int y, int x) {
        return (ps(y, x) - ps(y, x - 1)) / (g.metric(2, y, x).J * g.dq1());
    };
    const auto d1 = [&](int y, int x) {
        return (ch(y, x + 1) - ch(y, x)) / g.dq1();
    };
    const auto d2 = [&](int y, int x) {
        return (ch(y + 1, x) - ch(y, x)) / g.dq2();
    };
    const auto u = g.metric(1, j, i), v = g.metric(2, j, i);
    Real x1 = real(0), x2 = real(0), c1 = real(0), c2 = real(0);
    for (int dj : {-1, 0}) {
        for (int di : {0, 1}) {
            const auto m = g.metric(2, j + dj, i + di);
            x1 += m.c12 * d2(j + dj, i + di);
            c1 += m.g12 * r2(j + dj, i + di);
        }
    }
    for (int dj : {0, 1}) {
        for (int di : {-1, 0}) {
            const auto m = g.metric(1, j + dj, i + di);
            x2 += m.c12 * d1(j + dj, i + di);
            c2 += m.g12 * r1(j + dj, i + di);
        }
    }
    return {r1(j, i) + u.c11 * d1(j, i) + real(.25) * x1,
        r2(j, i) + v.c22 * d2(j, i) + real(.25) * x2,
        u.g11 * r1(j, i) + real(.25) * c1 + d1(j, i),
        v.g22 * r2(j, i) + real(.25) * c2 + d2(j, i)};
}

} // namespace WindRecoveryTest
#endif
