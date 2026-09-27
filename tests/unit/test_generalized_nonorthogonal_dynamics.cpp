#include "core/geometry/CartesianGeometry.hpp"
#include "dynamics/operators/HorizontalScalarGradient.hpp"
#include "dynamics/operators/HorizontalFluxDivergence.hpp"
#include "dynamics/operators/HorizontalVectorLowering.hpp"
#include "dynamics/operators/HorizontalCurl.hpp"
#include "dynamics/operators/GeneralizedHorizontalVorticityTransport.hpp"
#include "dynamics/operators/GeneralizedHorizontalDeformation.hpp"
#include "dynamics/operators/GeneralizedTopTransport.hpp"
#include "dynamics/operators/GeneralizedTopDeformation.hpp"
#include "dynamics/operators/TakacsScalarTransport.hpp"

#include <Kokkos_Core.hpp>
#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using namespace VVM;
using namespace VVM::Core::Geometry;
namespace O = VVM::Dynamics::Operators;

void
require(bool condition, const std::string& message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

// Test-only stationary chart: x=2*q1+s*q2, y=3*q2; J=6 and g12=2*s.
// The Cartesian enum is just an existing fixture label. No production
// geometry selection or wind-recovery capability is added by this test.
class AffineGeometry final : public HorizontalGeometry {
public:
    AffineGeometry(const HorizontalDomainLayout& layout, Real shear)
        : coordinates_(layout, real(.25), real(.5)), shear_(shear) {}
    GeometryKind
    kind() const noexcept override {
        return GeometryKind::Cartesian;
    }
    const char*
    name() const noexcept override {
        return "test_nonorthogonal_affine";
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
        auto v = coordinates_.device_view(location);
        const auto c = [](Real x) {
            return GeometryField2D::constant_value(x);
        };
        const Real s = shear_;
        v.sqrt_g = c(real(6));
        v.inv_sqrt_g = c(real(1) / real(6));
        v.g_cov = {c(real(4)), c(real(2) * s), c(real(9) + s * s)};
        v.sqrt_g_g_contra = {c((real(9) + s * s) / real(6)), c(-s / real(3)), c(real(2) / real(3))};
        v.contravariant_to_physical = {c(real(2)), c(s), c(real(0)), c(real(3))};
        v.physical_to_contravariant = {c(real(.5)),
            c(-s / real(6)),
            c(real(0)),
            c(real(1) / real(3))};
        return v;
    }

private:
    CartesianGeometry coordinates_;
    Real shear_;
};

// Stored in device memory, not captured by value: several primitive operators
// still carry large geometry views. Only the bundle handle enters the kernel.
struct Operators {
    O::HorizontalScalarGradientDeviceView gradient;
    O::HorizontalFluxDivergenceDeviceView divergence;
    O::HorizontalVectorLoweringDeviceView lowering;
    O::HorizontalCurlDeviceView curl;
    O::GeneralizedHorizontalVorticityTransportDeviceView transport;
    O::GeneralizedHorizontalDeformationDeviceView deformation;
    O::GeneralizedTopTransportDeviceView top_transport;
    O::GeneralizedTopDeformationDeviceView top_deformation;
    O::TakacsScalarTransportDeviceView scalar_transport;
};

struct ConstantScalar {
    Real value;
    KOKKOS_INLINE_FUNCTION Real
    operator()(int, int, int) const {
        return value;
    }
};
template <class Volume>
struct MassFlux {
    Volume wind;
    KOKKOS_INLINE_FUNCTION Real
    operator()(int k, int j, int i) const {
        return real(2) * wind(k, j, i);
    }
};
template <class Volume>
struct Fields {
    Volume u1, u2, w;
    Real omega1, omega2, omega3;
    KOKKOS_INLINE_FUNCTION Real
    omega1_over_rho(int, int, int) const {
        return omega1 / real(2);
    }
    KOKKOS_INLINE_FUNCTION Real
    omega2_over_rho(int, int, int) const {
        return omega2 / real(2);
    }
    KOKKOS_INLINE_FUNCTION Real
    omega3_over_rho(int, int, int) const {
        return omega3 / real(2);
    }
    KOKKOS_INLINE_FUNCTION Real
    f3_at_z(int, int) const {
        return real(.25);
    }
    KOKKOS_INLINE_FUNCTION Real
    rho(int) const {
        return real(2);
    }
    KOKKOS_INLINE_FUNCTION Real
    rho_up(int) const {
        return real(2);
    }
    KOKKOS_INLINE_FUNCTION Real
    fn1(int) const {
        return real(2);
    }
    KOKKOS_INLINE_FUNCTION Real
    fn2(int) const {
        return real(2);
    }
    KOKKOS_INLINE_FUNCTION Real
    inverse_spacing(int) const {
        return real(1);
    }
    KOKKOS_INLINE_FUNCTION Real
    inverse_spacing_mid(int) const {
        return real(1);
    }
    KOKKOS_INLINE_FUNCTION Real
    inverse_spacing_up(int) const {
        return real(1);
    }
};

template <class View>
std::vector<Real>
snapshot(const View& view) {
    const auto host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), view);
    std::vector<Real> values;
    for (std::size_t k = 0; k < host.extent(0); ++k) {
        for (std::size_t j = 0; j < host.extent(1); ++j) {
            for (std::size_t i = 0; i < host.extent(2); ++i) {
                values.push_back(host(k, j, i));
            }
        }
    }
    return values;
}

template <class Layout>
void
run_case(Real shear, Real offset, const char* layout_name) {
    HorizontalDomainLayout domain;
    domain.global_nx = domain.local_physical_nx = 9;
    domain.global_ny = domain.local_physical_ny = 7;
    domain.halo = 2;
    AffineGeometry geometry(domain, shear);
    const int h = domain.halo, nx = domain.local_total_nx(), ny = domain.local_total_ny();
    const int nz = 8, top = nz - h - 1;
    const Real dq1 = geometry.dq1(), dq2 = geometry.dq2();
    using Volume = Kokkos::View<Real***, Layout>;
    using Policy = Kokkos::MDRangePolicy<Kokkos::Rank<3>>;
    Volume u1("affine_u1", nz, ny, nx), u2("affine_u2", nz, ny, nx);
    Volume w("affine_w", nz, ny, nx), scalar("affine_scalar", nz, ny, nx);
    Volume cov1("affine_cov1", nz, ny, nx), cov2("affine_cov2", nz, ny, nx);
    Kokkos::View<Real****, Layout> result("affine_results", 26, nz, ny, nx);
    const Real sentinel = real(-731);

    // Physical affine wind U=a*x+b*y+c*z+offset, V=d*x+e*y+f*z-offset,
    // w=0. This is an operator identity fixture, not an anelastic model
    // solution: divergence is deliberately nonzero to exercise transport.
    const Real a = real(.125), b = real(-.25), c = real(.0625);
    const Real d = real(.375), e = real(-.5), f = real(-.1875);
    const Real sx = real(.75), sy = real(-.625);
    auto uh = Kokkos::create_mirror_view(u1), vh = Kokkos::create_mirror_view(u2);
    auto sh = Kokkos::create_mirror_view(scalar);
    const auto physical_u = [&](Real q1, Real q2, int k) {
        return a * (real(2) * q1 + shear * q2) + b * real(3) * q2 + c * k + offset;
    };
    const auto physical_v = [&](Real q1, Real q2, int k) {
        return d * (real(2) * q1 + shear * q2) + e * real(3) * q2 + f * k - offset;
    };
    for (int k = 0; k < nz; ++k) {
        for (int j = 0; j < ny; ++j) {
            for (int i = 0; i < nx; ++i) {
                const Real qt1 = real(i - h) * dq1, qt2 = real(j - h) * dq2;
                const Real qu1 = (real(i - h) + real(.5)) * dq1,
                           qv2 = (real(j - h) + real(.5)) * dq2;
                uh(k, j, i) =
                    real(.5) * physical_u(qu1, qt2, k) - shear / real(6) * physical_v(qu1, qt2, k);
                vh(k, j, i) = physical_v(qt1, qv2, k) / real(3);
                sh(k, j, i) = sx * (real(2) * qt1 + shear * qt2) + sy * real(3) * qt2;
            }
        }
    }
    Kokkos::deep_copy(u1, uh);
    Kokkos::deep_copy(u2, vh);
    Kokkos::deep_copy(scalar, sh);
    Kokkos::deep_copy(w, real(0));
    Kokkos::deep_copy(result, sentinel);
    const auto saved_u1 = snapshot(u1), saved_u2 = snapshot(u2), saved_s = snapshot(scalar);

    Kokkos::View<Operators> op("affine_operators");
    auto host = Kokkos::create_mirror_view(op);
    host().gradient = O::make_horizontal_scalar_gradient_device_view(geometry);
    host().divergence = O::make_horizontal_flux_divergence_device_view(geometry);
    host().lowering = O::make_horizontal_vector_lowering_device_view(geometry);
    host().curl = O::make_horizontal_curl_device_view(geometry);
    host().transport = O::make_generalized_horizontal_vorticity_transport_device_view(geometry);
    host().deformation = O::make_generalized_horizontal_deformation_device_view(geometry);
    host().top_transport = O::make_generalized_top_transport_device_view(geometry);
    host().top_deformation = O::make_generalized_top_deformation_device_view(geometry);
    host().scalar_transport = O::make_takacs_scalar_transport_device_view(geometry);
    Kokkos::deep_copy(op, host);

    // True vorticity of the physical affine wind; map components explicitly
    // from the analytic chart rather than through the geometry conversion API.
    const Real omega1 = -real(.5) * f - shear * c / real(6), omega2 = c / real(3), omega3 = d - b;
    const Fields<Volume> fields{u1, u2, w, omega1, omega2, omega3};
    Kokkos::parallel_for("AffineLowerBeforeCurl",
        Kokkos::Experimental::require(Policy({h, h - 1, h - 1}, {top + 1, ny - h + 1, nx - h + 1}),
            Kokkos::Experimental::WorkItemProperty::HintLightWeight),
        KOKKOS_LAMBDA(int k, int j, int i) {
            cov1(k, j, i) = op().lowering.calculate_covariant_q1_at_u(u1, u2, k, j, i);
            cov2(k, j, i) = op().lowering.calculate_covariant_q2_at_v(u1, u2, k, j, i);
        });
    Kokkos::parallel_for("AffineGeneralizedDynamics",
        Kokkos::Experimental::require(Policy({h, h, h}, {top + 1, ny - h, nx - h}),
            Kokkos::Experimental::WorkItemProperty::HintLightWeight),
        KOKKOS_LAMBDA(int k, int j, int i) {
            const auto stencil = O::load_scalar_stencil_at_t(scalar, k, j, i);
            result(0, k, j, i) = op().gradient.calculate_q1_at_u(j, i, stencil);
            result(1, k, j, i) = op().gradient.calculate_q2_at_v(j, i, stencil);
            result(2, k, j, i) = op().divergence.at_t(j,
                i,
                u1(k, j, i),
                u1(k, j, i - 1),
                u2(k, j, i),
                u2(k, j - 1, i));
            result(3, k, j, i) = op().curl.calculate_at_z(cov1, cov2, k, j, i);
            result(25, k, j, i) = op().scalar_transport.calculate_horizontal_flux_convergence_at_t(
                ConstantScalar{real(2.75)},
                MassFlux<Volume>{u1},
                MassFlux<Volume>{u2},
                k,
                j,
                i);
            if (k < top) {
                const auto d1 = op().deformation.calculate_omega1_at_v(fields, k, j, i);
                const auto d2 = op().deformation.calculate_omega2_at_u(fields, k, j, i);
                result(4, k, j, i) = d1.stretching;
                result(5, k, j, i) = d1.twisting;
                result(6, k, j, i) = d1.planetary;
                result(7, k, j, i) = d2.stretching;
                result(8, k, j, i) = d2.twisting;
                result(9, k, j, i) = d2.planetary;
                const auto t1 = op().transport.calculate_omega1_at_v(fields, w, k, j, i, h, top);
                const auto t2 = op().transport.calculate_omega2_at_u(fields, w, k, j, i, h, top);
                result(10, k, j, i) = t1.q1;
                result(11, k, j, i) = t1.q2;
                result(12, k, j, i) = t1.vertical;
                result(13, k, j, i) = t2.q1;
                result(14, k, j, i) = t2.q2;
                result(15, k, j, i) = t2.vertical;
            }
            else {
                const auto r = op().top_transport.calculate_at_z(fields, top, j, i, false);
                const auto p = op().top_transport.calculate_at_z(fields, top, j, i, true);
                const auto dtop = op().top_deformation.calculate_at_z(fields, top, j, i);
                result(16, k, j, i) = r.q1;
                result(17, k, j, i) = r.q2;
                result(18, k, j, i) = r.vertical;
                result(19, k, j, i) = p.q1;
                result(20, k, j, i) = p.q2;
                result(21, k, j, i) = p.vertical;
                result(22, k, j, i) = dtop.stretching;
                result(23, k, j, i) = dtop.twisting;
                result(24, k, j, i) = dtop.planetary;
            }
        });
    const auto actual = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), result);
    const auto cov1h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), cov1);
    const auto cov2h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), cov2);
    const Real b11 = a - shear * d / real(3), b12 = (a * shear + real(3) * b) / real(2) -
                                                    shear * (d * shear + real(3) * e) / real(6);
    const Real b21 = real(2) * d / real(3), b22 = d * shear / real(3) + e;
    const Real c1 = c / real(2) - shear * f / real(6), c2 = f / real(3), planet = real(.25);
    const std::array<Real, 26> expected = {sx / real(2) - shear * sy / real(6),
        sy / real(3),
        a + e,
        d - b,
        omega1 * b11,
        omega2 * b12 + omega3 * c1,
        planet * c1,
        omega2 * b22,
        omega1 * b21 + omega3 * c2,
        planet * c2,
        -omega1 * b11,
        -omega1 * b22,
        real(0),
        -omega2 * b11,
        -omega2 * b22,
        real(0),
        -omega3 * b11,
        -omega3 * b22,
        real(0),
        -planet * b11,
        -planet * b22,
        real(0),
        real(0),
        real(0),
        real(0),
        -real(2) * real(2.75) * (a + e)};
    const Real tolerance = real(1024) * std::numeric_limits<Real>::epsilon();
    const auto close = [&](Real x, Real y) {
        return std::isfinite(x) && std::abs(x - y) <= tolerance * std::max(real(1), std::abs(y));
    };
    for (int k = 0; k < nz; ++k) {
        for (int j = 0; j < ny; ++j) {
            for (int i = 0; i < nx; ++i) {
                const bool inside =
                    k >= h && k <= top && j >= h && j < ny - h && i >= h && i < nx - h;
                for (int n = 0; n < 26; ++n) {
                    const bool used =
                        inside && ((n < 4 || n == 25) || (n < 16 ? k < top : k == top));
                    require(used ? close(actual(n, k, j, i), expected[n])
                                 : actual(n, k, j, i) == sentinel,
                        std::string(layout_name) + ": incorrect nonorthogonal component " +
                            std::to_string(n));
                }
                if (inside) {
                    const Real qt1 = real(i - h) * dq1, qt2 = real(j - h) * dq2;
                    const Real qu1 = (real(i - h) + real(.5)) * dq1,
                               qv2 = (real(j - h) + real(.5)) * dq2;
                    require(close(cov1h(k, j, i), real(2) * physical_u(qu1, qt2, k)),
                        "incorrect staggered u_1");
                    require(
                        close(cov2h(k, j, i),
                            shear * physical_u(qt1, qv2, k) + real(3) * physical_v(qt1, qv2, k)),
                        "incorrect staggered u_2");
                }
            }
        }
    }
    require(saved_u1 == snapshot(u1) && saved_u2 == snapshot(u2) && saved_s == snapshot(scalar),
        "nonorthogonal operators modified their inputs");
    std::cout << "PASS: affine generalized dynamics " << layout_name << ", shear=" << shear
              << ", offset=" << offset << '\n';
}
} // namespace

int
main(int argc, char** argv) {
    Kokkos::initialize(argc, argv);
    int result = 0;
    try {
        for (const Real shear : {VVM::real(.75), VVM::real(-1.125)}) {
            run_case<Kokkos::LayoutLeft>(shear, VVM::real(3), "LayoutLeft");
            run_case<Kokkos::LayoutRight>(shear, VVM::real(-3), "LayoutRight");
        }
    }
    catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        result = 1;
    }
    Kokkos::finalize();
    return result;
}
