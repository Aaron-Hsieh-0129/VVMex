#include "core/geometry/HorizontalGeometryBuilders.hpp"
#include "GeneralizedWindRecoveryTestGeometry.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <limits>
#include <stdexcept>

namespace {
using namespace VVM;
using namespace VVM::Core::Geometry;
int failures = 0;
const Real tolerance = real(512) * std::numeric_limits<Real>::epsilon();

void
check(bool condition, const char* message) {
    if (!condition) {
        ++failures;
        std::fprintf(stderr, "FAIL: %s\n", message);
    }
}

bool
close(Real actual, Real expected) {
    return std::abs(actual - expected) <= tolerance * std::max(real(1), std::abs(expected));
}

HorizontalDomainLayout
layout() {
    HorizontalDomainLayout result;
    result.global_nx = result.local_physical_nx = 5;
    result.global_ny = result.local_physical_ny = 4;
    result.halo = 2;
    return result;
}

std::array<HorizontalGeometryInput, 4>
constant_inputs(const std::array<Real, 4>& g,
    const std::array<Real, 4>& b = {real(1), real(0), real(0), real(1)}) {
    const auto c = GeometryField2D::constant_value;
    std::array<HorizontalGeometryInput, 4> inputs;
    for (auto& input : inputs) {
        input.g_cov = {c(g[0]), c(g[1]), c(g[2]), c(g[3])};
        input.contravariant_to_physical = {c(b[0]), c(b[1]), c(b[2]), c(b[3])};
    }
    return inputs;
}

void
test_constant_metric() {
    HorizontalGeometry geometry(layout(),
        real(.25),
        real(.5),
        constant_inputs({real(4), real(1.5), real(1.5), real(9.5625)},
            {real(2), real(.75), real(0), real(3)}),
        GeometryKind::Affine,
        "test_metric");
    for (const auto loc : {HorizontalLocation::T,
             HorizontalLocation::U,
             HorizontalLocation::V,
             HorizontalLocation::Z}) {
        const auto v = geometry.device_view(loc);
        check(close(v.sqrt_g.constant, real(6)), "J must be derived from det(g)");
        check(close(v.sqrt_g_g_contra.a12.constant, real(-.25)),
            "J*g^12 must include the cross metric");
        check(close(v.sqrt_g_g_contra.a21.constant, real(-.25)),
            "J*g^21 must include the cross metric");
        const auto b = v.contravariant_to_physical;
        const auto inverse = v.physical_to_contravariant;
        check(close(b.a11.constant, real(2)) && close(b.a12.constant, real(.75)) &&
                  close(b.a21.constant, real(0)) && close(b.a22.constant, real(3)),
            "Shared geometry must preserve the coordinate basis");
        check(close(inverse.a11.constant, real(.5)) && close(inverse.a12.constant, real(-.125)) &&
                  close(inverse.a21.constant, real(0)) &&
                  close(inverse.a22.constant, real(1) / real(3)),
            "Physical transform must invert the supplied basis");
    }
}

void
test_rotated_basis() {
    // A nonsymmetric coordinate transformation still produces a symmetric metric.
    const std::array<Real, 4> expected{real(2), real(.5), real(-.25), real(1.5)};
    auto geometry = make_affine_geometry(layout(), real(1), real(1), expected);
    const auto b = geometry.device_view(HorizontalLocation::T).contravariant_to_physical;
    check(close(b.a11.constant, expected[0]) && close(b.a12.constant, expected[1]) &&
              close(b.a21.constant, expected[2]) && close(b.a22.constant, expected[3]),
        "Shared geometry must preserve an arbitrary positive affine basis");
}

void
test_varying_metric() {
    // This fixture now supplies only g and the physical basis. The independent map
    // supplies expected J, inverse g, and physical basis at all four staggers.
    WindRecoveryTest::Geometry chart(real(.75), true);
    const int nx = chart.nx(), ny = chart.ny();
    for (int l = 0; l < 4; ++l) {
        const auto v = chart.geometry().device_view(static_cast<HorizontalLocation>(l));
        Kokkos::View<Real***> values("derived_metric_check", ny, nx, 11);
        Kokkos::parallel_for("InspectDerivedMetric",
            Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {ny, nx}),
            KOKKOS_LAMBDA(const int j, const int i) {
                values(j, i, 0) = v.sqrt_g(j, i);
                values(j, i, 1) = v.g_contra_11(j, i);
                values(j, i, 2) = v.g_contra_12(j, i);
                values(j, i, 3) = v.g_contra_21(j, i);
                values(j, i, 4) = v.g_contra_22(j, i);
                const auto b = v.contravariant_to_physical;
                const auto a = v.physical_to_contravariant;
                values(j, i, 5) = b.a11(j, i);
                values(j, i, 6) = b.a12(j, i);
                values(j, i, 7) = b.a21(j, i);
                values(j, i, 8) = b.a22(j, i);
                Real u, w, restored_u, restored_w;
                b.apply(j, i, real(3), real(-2), u, w);
                a.apply(j, i, u, w, restored_u, restored_w);
                values(j, i, 9) = restored_u;
                values(j, i, 10) = restored_w;
            });
        const auto host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), values);
        for (int j = 0; j < ny; ++j) {
            for (int i = 0; i < nx; ++i) {
                const auto m = chart.metric(l, j, i);
                const Real p = chart.p(l, i), q = chart.q(l, j);
                check(close(host(j, i, 0), m.J),
                    "Derived varying J must match the analytic map, including halos");
                check(close(host(j, i, 1), m.c11) && close(host(j, i, 2), m.c12) &&
                          close(host(j, i, 3), m.c12) && close(host(j, i, 4), m.c22),
                    "Derived inverse metric must match the independent map");
                check(close(host(j, i, 5), real(2) + real(.17) * q) &&
                          close(host(j, i, 6), real(.75) + real(.17) * p) &&
                          close(host(j, i, 7), real(.11) * q) &&
                          close(host(j, i, 8), real(3) + real(.11) * p),
                    "Varying physical basis must retain native point placement");
                check(close(host(j, i, 9), real(3)) && close(host(j, i, 10), real(-2)),
                    "Derived physical transforms must round trip");
            }
        }
    }
}

void
test_independent_off_diagonal_entries() {
    // Algebraic sentinel, not the metric of a physical coordinate mapping.
    // Future matrix providers must not have a21 overwritten by a12.
    HorizontalGeometry geometry(layout(),
        real(1),
        real(1),
        constant_inputs({real(4), real(1), real(2), real(9)}),
        GeometryKind::Affine,
        "test_independent_entries");
    const auto v = geometry.device_view(HorizontalLocation::T);
    const Real J = std::sqrt(real(34));
    check(v.g_cov.a12.constant == real(1) && v.g_cov.a21.constant == real(2),
        "Input g12 and g21 must remain independent");
    check(close(v.sqrt_g.constant, J) && close(v.sqrt_g_g_contra.a12.constant, -real(1) / J) &&
              close(v.sqrt_g_g_contra.a21.constant, -real(2) / J),
        "The determinant and inverse must use both off-diagonal entries");
    check(close(real(4) * v.g_contra_11(0, 0) + v.g_contra_21(0, 0), real(1)) &&
              close(real(4) * v.g_contra_12(0, 0) + v.g_contra_22(0, 0), real(0)) &&
              close(real(2) * v.g_contra_11(0, 0) + real(9) * v.g_contra_21(0, 0), real(0)) &&
              close(real(2) * v.g_contra_12(0, 0) + real(9) * v.g_contra_22(0, 0), real(1)),
        "A nonsymmetric matrix times its derived inverse must equal identity");
}

void
test_independent_varying_entries() {
    // Matrix-algebra sentinel: g12 varies with i, g21 with j, and each
    // native location has distinct values. No symmetry or row-only shortcut.
    const auto domain = layout();
    const int nx = domain.local_total_nx(), ny = domain.local_total_ny();
    auto inputs = constant_inputs({real(4), real(0), real(0), real(9)});
    for (int l = 0; l < 4; ++l) {
        Kokkos::View<Real*> g12("independent_g12", nx), g21("independent_g21", ny);
        Kokkos::parallel_for("InitializeIndependentG12", nx, KOKKOS_LAMBDA(const int i) {
            g12(i) = real(.1) + real(.01) * i + real(.02) * l;
        });
        Kokkos::parallel_for("InitializeIndependentG21", ny, KOKKOS_LAMBDA(const int j) {
            g21(j) = real(.2) + real(.02) * j + real(.04) * l;
        });
        inputs[l].g_cov.a12 = GeometryField2D::varying_i(g12);
        inputs[l].g_cov.a21 = GeometryField2D::varying_j(g21);
    }
    HorizontalGeometry geometry(domain,
        real(1),
        real(1),
        inputs,
        GeometryKind::Affine,
        "test_independent_varying_entries");
    for (int l = 0; l < 4; ++l) {
        const auto v = geometry.device_view(static_cast<HorizontalLocation>(l));
        check(v.sqrt_g.layout == GeometryFieldLayout::Full2D,
            "Mixed i/j metric dependence must produce full two-dimensional coefficients");
        Kokkos::View<Real***> values("independent_varying_check", ny, nx, 3);
        Kokkos::parallel_for("InspectIndependentVaryingMetric",
            Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {ny, nx}),
            KOKKOS_LAMBDA(const int j, const int i) {
                values(j, i, 0) = v.sqrt_g(j, i);
                values(j, i, 1) = v.sqrt_g_g_contra.a12(j, i);
                values(j, i, 2) = v.sqrt_g_g_contra.a21(j, i);
            });
        const auto host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), values);
        for (int j = 0; j < ny; ++j) {
            for (int i = 0; i < nx; ++i) {
                const Real g12 = real(.1) + real(.01) * i + real(.02) * l;
                const Real g21 = real(.2) + real(.02) * j + real(.04) * l;
                const Real J = std::sqrt(real(36) - g12 * g21);
                check(close(host(j, i, 0), J) && close(host(j, i, 1), -g12 / J) &&
                          close(host(j, i, 2), -g21 / J),
                    "Independent g12/g21 must be used at every point and native location, "
                    "including halos");
            }
        }
    }
}

void
test_invalid_metrics() {
    const auto c = GeometryField2D::constant_value;
    for (const auto& g : {std::array<Real, 4>{real(1), real(2), real(2), real(1)},
             std::array<Real, 4>{real(1), real(1), real(1), real(1)},
             std::array<Real, 4>{real(-1), real(0), real(0), real(-1)},
             std::array<Real, 4>{std::numeric_limits<Real>::infinity(),
                 real(0),
                 real(0),
                 real(1)}}) {
        const auto inputs = constant_inputs(g);
        bool rejected = false;
        try {
            HorizontalGeometry geometry(layout(),
                real(1),
                real(1),
                inputs,
                GeometryKind::Affine,
                "invalid_metric");
        }
        catch (const std::invalid_argument&) {
            rejected = true;
        }
        check(rejected, "Nonphysical or singular metrics must be rejected");
    }
    std::array<HorizontalGeometryInput, 4> inputs;
    for (auto& input : inputs) {
        input.g_cov = {c(real(4)), c(real(0)), c(real(0)), c(real(1))};
        input.contravariant_to_physical = {c(real(1)), c(real(0)), c(real(0)), c(real(1))};
    }
    bool rejected = false;
    try {
        HorizontalGeometry geometry(layout(),
            real(1),
            real(1),
            inputs,
            GeometryKind::Cartesian,
            "invalid_cartesian");
    }
    catch (const std::invalid_argument&) {
        rejected = true;
    }
    check(rejected, "A changed g must not silently select Cartesian shortcuts");
}
} // namespace

int
main(int argc, char** argv) {
    Kokkos::initialize(argc, argv);
    try {
        test_constant_metric();
        test_rotated_basis();
        test_varying_metric();
        test_independent_off_diagonal_entries();
        test_independent_varying_entries();
        test_invalid_metrics();
    }
    catch (const std::exception& error) {
        ++failures;
        std::fprintf(stderr, "Unexpected exception: %s\n", error.what());
    }
    Kokkos::finalize();
    if (failures == 0) {
        std::puts("test_metric_geometry: PASS");
    }
    return failures == 0 ? 0 : 1;
}
