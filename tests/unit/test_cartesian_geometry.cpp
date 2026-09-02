#include "core/geometry/CartesianGeometry.hpp"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <stdexcept>

#include <Kokkos_Core.hpp>

namespace {

using VVM::Core::Geometry::CartesianGeometry;
using VVM::Core::Geometry::GeometryKind;
using VVM::Core::Geometry::HorizontalDomainLayout;
using VVM::Core::Geometry::HorizontalLocation;

int failures = 0;

void check(
    const bool condition,
    const char* message) {

    if (!condition) {
        std::fprintf(
            stderr,
            "FAIL: %s\n",
            message);

        ++failures;
    }
}

bool close(
    const VVM::Real actual,
    const VVM::Real expected) {

    return std::abs(actual - expected) <
           VVM::real(1.0e-5);
}

HorizontalDomainLayout make_layout() {
    HorizontalDomainLayout layout;

    layout.global_nx = 16;
    layout.global_ny = 8;

    layout.local_physical_nx = 8;
    layout.local_physical_ny = 4;

    layout.global_start_i = 8;
    layout.global_start_j = 4;

    layout.halo = 2;
    layout.panel_id = -1;

    return layout;
}

void test_host_metadata() {
    const CartesianGeometry geometry(
        make_layout(),
        VVM::real(100.0),
        VVM::real(200.0));

    check(
        geometry.kind() ==
            GeometryKind::Cartesian,
        "Cartesian geometry must report Cartesian kind");

    check(
        std::strcmp(
            geometry.name(),
            "cartesian") == 0,
        "Cartesian geometry must report its name");

    check(
        close(
            geometry.dq1(),
            VVM::real(100.0)),
        "dq1 must equal dx");

    check(
        close(
            geometry.dq2(),
            VVM::real(200.0)),
        "dq2 must equal dy");

    check(
        geometry.layout().local_total_nx() == 12,
        "local total nx must include halos");

    check(
        geometry.layout().local_total_ny() == 8,
        "local total ny must include halos");
}

void test_device_values() {
    const HorizontalDomainLayout layout =
        make_layout();

    const CartesianGeometry geometry(
        layout,
        VVM::real(100.0),
        VVM::real(200.0));

    const auto t =
        geometry.device_view(
            HorizontalLocation::T);

    const auto u =
        geometry.device_view(
            HorizontalLocation::U);

    const auto v =
        geometry.device_view(
            HorizontalLocation::V);

    const auto z =
        geometry.device_view(
            HorizontalLocation::Z);

    constexpr int result_count = 32;

    Kokkos::View<VVM::Real*> results(
        "cartesian_geometry_results",
        result_count);

    const int h =
        layout.halo;

    Kokkos::parallel_for(
        "EvaluateCartesianGeometry",
        Kokkos::RangePolicy<>(0, 1),
        KOKKOS_LAMBDA(const int) {

            results(0) =
                t.q1(h, h);

            results(1) =
                t.q2(h, h);

            results(2) =
                u.q1(h, h);

            results(3) =
                u.q2(h, h);

            results(4) =
                v.q1(h, h);

            results(5) =
                v.q2(h, h);

            results(6) =
                z.q1(h, h);

            results(7) =
                z.q2(h, h);

            results(8) =
                t.q1(0, 0);

            results(9) =
                t.q2(0, 0);

            results(10) =
                t.longitude(h, h);

            results(11) =
                t.latitude(h, h);

            results(12) =
                t.sqrt_g(h, h);

            results(13) =
                t.inv_sqrt_g(h, h);

            results(14) =
                t.g_cov.a11(h, h);

            results(15) =
                t.g_cov.a12(h, h);

            results(16) =
                t.g_cov.a22(h, h);

            results(17) =
                t.sqrt_g_g_contra.a11(h, h);

            results(18) =
                t.sqrt_g_g_contra.a12(h, h);

            results(19) =
                t.sqrt_g_g_contra.a22(h, h);

            results(20) =
                t.g_contra_11(h, h);

            results(21) =
                t.g_contra_12(h, h);

            results(22) =
                t.g_contra_22(h, h);

            VVM::Real contra_1 =
                VVM::real(0.0);

            VVM::Real contra_2 =
                VVM::real(0.0);

            t.physical_to_contravariant.apply(
                h,
                h,
                VVM::real(3.0),
                VVM::real(4.0),
                contra_1,
                contra_2);

            results(23) =
                contra_1;

            results(24) =
                contra_2;

            VVM::Real physical_1 =
                VVM::real(0.0);

            VVM::Real physical_2 =
                VVM::real(0.0);

            t.contravariant_to_physical.apply(
                h,
                h,
                VVM::real(5.0),
                VVM::real(6.0),
                physical_1,
                physical_2);

            results(25) =
                physical_1;

            results(26) =
                physical_2;

            results(27) =
                t.dq1;

            results(28) =
                t.dq2;

            results(29) =
                u.q1(h, h) -
                t.q1(h, h);

            results(30) =
                v.q2(h, h) -
                t.q2(h, h);

            results(31) =
                z.q1(h, h) -
                t.q1(h, h);
        });

    const auto host =
        Kokkos::create_mirror_view_and_copy(
            Kokkos::HostSpace(),
            results);

    check(
        close(
            host(0),
            VVM::real(800.0)),
        "T q1 must use the global x index");

    check(
        close(
            host(1),
            VVM::real(800.0)),
        "T q2 must use the global y index");

    check(
        close(
            host(2),
            VVM::real(850.0)),
        "U q1 must have a half-cell q1 offset");

    check(
        close(
            host(3),
            VVM::real(800.0)),
        "U q2 must remain at the T q2 location");

    check(
        close(
            host(4),
            VVM::real(800.0)),
        "V q1 must remain at the T q1 location");

    check(
        close(
            host(5),
            VVM::real(900.0)),
        "V q2 must have a half-cell q2 offset");

    check(
        close(
            host(6),
            VVM::real(850.0)),
        "Z q1 must have a half-cell q1 offset");

    check(
        close(
            host(7),
            VVM::real(900.0)),
        "Z q2 must have a half-cell q2 offset");

    check(
        close(
            host(8),
            VVM::real(600.0)),
        "q1 must be valid in the western halo");

    check(
        close(
            host(9),
            VVM::real(400.0)),
        "q2 must be valid in the southern halo");

    check(
        close(
            host(10),
            VVM::real(0.0)),
        "Cartesian longitude is zero in this milestone");

    check(
        close(
            host(11),
            VVM::real(0.0)),
        "Cartesian latitude is zero in this milestone");

    check(
        close(
            host(12),
            VVM::real(1.0)),
        "Cartesian sqrt_g must be one");

    check(
        close(
            host(13),
            VVM::real(1.0)),
        "Cartesian inv_sqrt_g must be one");

    check(
        close(
            host(14),
            VVM::real(1.0)),
        "Cartesian g_cov a11 must be one");

    check(
        close(
            host(15),
            VVM::real(0.0)),
        "Cartesian g_cov a12 must be zero");

    check(
        close(
            host(16),
            VVM::real(1.0)),
        "Cartesian g_cov a22 must be one");

    check(
        close(
            host(17),
            VVM::real(1.0)),
        "Cartesian Jg^11 must be one");

    check(
        close(
            host(18),
            VVM::real(0.0)),
        "Cartesian Jg^12 must be zero");

    check(
        close(
            host(19),
            VVM::real(1.0)),
        "Cartesian Jg^22 must be one");

    check(
        close(
            host(20),
            VVM::real(1.0)),
        "Cartesian g^11 must be one");

    check(
        close(
            host(21),
            VVM::real(0.0)),
        "Cartesian g^12 must be zero");

    check(
        close(
            host(22),
            VVM::real(1.0)),
        "Cartesian g^22 must be one");

    check(
        close(
            host(23),
            VVM::real(3.0)),
        "physical-to-contravariant component 1 must be identity");

    check(
        close(
            host(24),
            VVM::real(4.0)),
        "physical-to-contravariant component 2 must be identity");

    check(
        close(
            host(25),
            VVM::real(5.0)),
        "contravariant-to-physical component 1 must be identity");

    check(
        close(
            host(26),
            VVM::real(6.0)),
        "contravariant-to-physical component 2 must be identity");

    check(
        close(
            host(27),
            VVM::real(100.0)),
        "device dq1 must equal dx");

    check(
        close(
            host(28),
            VVM::real(200.0)),
        "device dq2 must equal dy");

    check(
        close(
            host(29),
            VVM::real(50.0)),
        "U-to-T q1 offset must equal dx/2");

    check(
        close(
            host(30),
            VVM::real(100.0)),
        "V-to-T q2 offset must equal dy/2");

    check(
        close(
            host(31),
            VVM::real(50.0)),
        "Z-to-T q1 offset must equal dx/2");
}

template<typename Function>
void expect_invalid_argument(
    Function&& function,
    const char* message) {

    bool threw = false;

    try {
        function();
    }
    catch (const std::invalid_argument&) {
        threw = true;
    }

    check(
        threw,
        message);
}

void test_validation() {
    expect_invalid_argument(
        [] {
            auto layout =
                make_layout();

            layout.global_nx = 0;

            CartesianGeometry geometry(
                layout,
                VVM::real(100.0),
                VVM::real(200.0));
        },
        "global_nx == 0 must be rejected");

    expect_invalid_argument(
        [] {
            auto layout =
                make_layout();

            layout.global_start_i = 12;

            CartesianGeometry geometry(
                layout,
                VVM::real(100.0),
                VVM::real(200.0));
        },
        "local x range outside the global range must be rejected");

    expect_invalid_argument(
        [] {
            auto layout =
                make_layout();

            layout.panel_id = 0;

            CartesianGeometry geometry(
                layout,
                VVM::real(100.0),
                VVM::real(200.0));
        },
        "Cartesian panel_id other than -1 must be rejected");

    expect_invalid_argument(
        [] {
            CartesianGeometry geometry(
                make_layout(),
                VVM::real(0.0),
                VVM::real(200.0));
        },
        "dx == 0 must be rejected");

    expect_invalid_argument(
        [] {
            CartesianGeometry geometry(
                make_layout(),
                VVM::real(100.0),
                VVM::real(-1.0));
        },
        "negative dy must be rejected");
}

} // namespace

// main must be in the global namespace.
int main(
    int argc,
    char** argv) {

    Kokkos::initialize(
        argc,
        argv);

    {
        test_host_metadata();
        test_device_values();
        test_validation();
    }

    Kokkos::finalize();

    if (failures == 0) {
        std::puts(
            "test_cartesian_geometry: PASS");

        return 0;
    }

    std::fprintf(
        stderr,
        "test_cartesian_geometry: %d failure(s)\n",
        failures);

    return 1;
}
