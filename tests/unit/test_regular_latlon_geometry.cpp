#include "core/geometry/RegularLatLonGeometry.hpp"

#include <Kokkos_Core.hpp>

#include <cmath>
#include <cstdio>
#include <cstring>
#include <exception>
#include <stdexcept>

namespace {

using VVM::Core::Geometry::GeometryFieldLayout;
using VVM::Core::Geometry::GeometryKind;
using VVM::Core::Geometry::HorizontalDomainLayout;
using VVM::Core::Geometry::HorizontalLocation;
using VVM::Core::Geometry::RegularLatLonGeometry;

int failures = 0;

constexpr VVM::Real relative_tolerance =
    sizeof(VVM::Real) == sizeof(double)
        ? VVM::real(1.0e-12)
        : VVM::real(5.0e-5);

void check(
    const bool condition,
    const char* message) {

    if (condition) {
        return;
    }

    ++failures;
    std::fprintf(
        stderr,
        "FAIL: %s\n",
        message);
}

bool close(
    const VVM::Real actual,
    const VVM::Real expected) {

    if (expected == VVM::real(0.0)) {
        return std::abs(actual) <=
               relative_tolerance;
    }

    return std::abs(actual - expected) <=
           relative_tolerance *
           std::abs(expected);
}

HorizontalDomainLayout make_layout() {
    HorizontalDomainLayout layout;

    layout.global_nx = 400;
    layout.global_ny = 100;

    layout.local_physical_nx = 400;
    layout.local_physical_ny = 100;

    layout.global_start_i = 0;
    layout.global_start_j = 0;

    layout.halo = 2;
    layout.panel_id = -1;

    return layout;
}

RegularLatLonGeometry make_geometry() {
    const VVM::Real pi =
        std::acos(VVM::real(-1.0));

    const VVM::Real spacing =
        VVM::real(0.9) *
        pi /
        VVM::real(180.0);

    return RegularLatLonGeometry(
        make_layout(),
        spacing,
        spacing,
        VVM::real(0.0),
        -pi / VVM::real(4.0),
        VVM::real(6371220.0));
}

void test_host_metadata() {
    const auto geometry =
        make_geometry();

    const VVM::Real pi =
        std::acos(VVM::real(-1.0));

    const VVM::Real spacing =
        VVM::real(0.9) *
        pi /
        VVM::real(180.0);

    check(
        geometry.kind() ==
            GeometryKind::RegularLatLon,
        "RLL geometry must report RegularLatLon kind");

    check(
        std::strcmp(
            geometry.name(),
            "regular_latlon") == 0,
        "RLL geometry must report its name");

    check(
        close(
            geometry.dq1(),
            spacing),
        "RLL dq1 must be longitude spacing in radians");

    check(
        close(
            geometry.dq2(),
            spacing),
        "RLL dq2 must be latitude spacing in radians");

    check(
        close(
            geometry.longitude_west_edge(),
            VVM::real(0.0)),
        "RLL west edge must be retained");

    check(
        close(
            geometry.latitude_south_edge(),
            -pi / VVM::real(4.0)),
        "RLL south edge must be retained");

    check(
        close(
            geometry.radius(),
            VVM::real(6371220.0)),
        "RLL radius must be retained");
}

void test_compact_storage_and_sharing() {
    const auto geometry =
        make_geometry();

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

    check(
        t.q1.layout ==
            GeometryFieldLayout::VaryingI,
        "RLL q1 must use one-dimensional i storage");

    check(
        t.q2.layout ==
            GeometryFieldLayout::VaryingJ,
        "RLL q2 must use one-dimensional j storage");

    check(
        t.sqrt_g.layout ==
            GeometryFieldLayout::VaryingJ,
        "RLL Jacobian must use one-dimensional j storage");

    check(
        t.g_cov.a12.layout ==
            GeometryFieldLayout::Constant,
        "RLL cross metric must use constant storage");

    check(
        t.g_cov.a22.layout ==
            GeometryFieldLayout::Constant,
        "RLL g22 must use constant storage");

    check(
        t.q1.one_dimensional.data() ==
            v.q1.one_dimensional.data(),
        "T and V must share centered q1 storage");

    check(
        u.q1.one_dimensional.data() ==
            z.q1.one_dimensional.data(),
        "U and Z must share staggered q1 storage");

    check(
        t.q2.one_dimensional.data() ==
            u.q2.one_dimensional.data(),
        "T and U must share centered q2 storage");

    check(
        v.q2.one_dimensional.data() ==
            z.q2.one_dimensional.data(),
        "V and Z must share staggered q2 storage");

    check(
        t.sqrt_g.one_dimensional.data() ==
            u.sqrt_g.one_dimensional.data(),
        "T and U must share centered-latitude metrics");

    check(
        v.sqrt_g.one_dimensional.data() ==
            z.sqrt_g.one_dimensional.data(),
        "V and Z must share staggered-latitude metrics");
}

void test_coordinates_and_halos() {
    const HorizontalDomainLayout layout =
        make_layout();

    const auto geometry =
        make_geometry();

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

    Kokkos::View<VVM::Real*> results(
        "regular_latlon_coordinate_results",
        18);

    const int h =
        layout.halo;

    const int last_i =
        h +
        layout.local_physical_nx -
        1;

    const int last_j =
        h +
        layout.local_physical_ny -
        1;

    Kokkos::parallel_for(
        "EvaluateRegularLatLonCoordinates",
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
                t.q1(last_j, last_i);

            results(9) =
                t.q2(last_j, last_i);

            results(10) =
                v.q2(last_j, last_i);

            results(11) =
                t.q1(h, 0);

            results(12) =
                t.q2(0, h);

            results(13) =
                t.longitude(h, h);

            results(14) =
                t.latitude(h, h);

            results(15) =
                u.longitude(h, h);

            results(16) =
                v.latitude(h, h);

            results(17) =
                z.longitude(h, h);
        });

    const auto host =
        Kokkos::create_mirror_view_and_copy(
            Kokkos::HostSpace(),
            results);

    const VVM::Real pi =
        std::acos(VVM::real(-1.0));

    const VVM::Real degrees =
        pi /
        VVM::real(180.0);

    check(
        close(
            host(0),
            VVM::real(0.45) * degrees),
        "first T longitude must be 0.45 degrees");

    check(
        close(
            host(1),
            VVM::real(-44.55) * degrees),
        "first T latitude must be -44.55 degrees");

    check(
        close(
            host(2),
            VVM::real(0.9) * degrees),
        "first U longitude must be 0.9 degrees");

    check(
        close(
            host(3),
            VVM::real(-44.55) * degrees),
        "first U latitude must remain at T latitude");

    check(
        close(
            host(4),
            VVM::real(0.45) * degrees),
        "first V longitude must remain at T longitude");

    check(
        close(
            host(5),
            VVM::real(-44.1) * degrees),
        "first V latitude must be -44.1 degrees");

    check(
        close(
            host(6),
            VVM::real(0.9) * degrees),
        "first Z longitude must be 0.9 degrees");

    check(
        close(
            host(7),
            VVM::real(-44.1) * degrees),
        "first Z latitude must be -44.1 degrees");

    check(
        close(
            host(8),
            VVM::real(359.55) * degrees),
        "last T longitude must be 359.55 degrees");

    check(
        close(
            host(9),
            VVM::real(44.55) * degrees),
        "last T latitude must be 44.55 degrees");

    check(
        close(
            host(10),
            VVM::real(45.0) * degrees),
        "last physical V latitude must lie on the north edge");

    check(
        close(
            host(11),
            VVM::real(-1.35) * degrees),
        "western T halo longitude must remain analytically unwrapped");

    check(
        close(
            host(12),
            VVM::real(-46.35) * degrees),
        "southern T halo latitude must be analytic");

    check(
        close(
            host(13),
            host(0)),
        "RLL longitude must equal q1");

    check(
        close(
            host(14),
            host(1)),
        "RLL latitude must equal q2");

    check(
        close(
            host(15),
            host(2)),
        "U longitude must equal U q1");

    check(
        close(
            host(16),
            host(5)),
        "V latitude must equal V q2");

    check(
        close(
            host(17),
            host(6)),
        "Z longitude must equal Z q1");
}

void test_metric_and_vector_transforms() {
    const HorizontalDomainLayout layout =
        make_layout();

    const auto geometry =
        make_geometry();

    const auto t =
        geometry.device_view(
            HorizontalLocation::T);

    const auto v =
        geometry.device_view(
            HorizontalLocation::V);

    const int j =
        layout.halo + 25;

    const int i =
        layout.halo + 17;

    const int equator_j_at_v =
        layout.halo + 49;

    Kokkos::View<VVM::Real*> results(
        "regular_latlon_metric_results",
        23);

    Kokkos::parallel_for(
        "EvaluateRegularLatLonMetrics",
        Kokkos::RangePolicy<>(0, 1),
        KOKKOS_LAMBDA(const int) {
            results(0) =
                t.latitude(j, i);

            results(1) =
                t.sqrt_g(j, i);

            results(2) =
                t.inv_sqrt_g(j, i);

            results(3) =
                t.g_cov.a11(j, i);

            results(4) =
                t.g_cov.a12(j, i);

            results(5) =
                t.g_cov.a22(j, i);

            results(6) =
                t.sqrt_g_g_contra.a11(j, i);

            results(7) =
                t.sqrt_g_g_contra.a12(j, i);

            results(8) =
                t.sqrt_g_g_contra.a22(j, i);

            results(9) =
                t.g_contra_11(j, i);

            results(10) =
                t.g_contra_12(j, i);

            results(11) =
                t.g_contra_22(j, i);

            results(12) =
                t.sqrt_g_g_contra.a11(j, i) *
                t.sqrt_g_g_contra.a22(j, i) -
                t.sqrt_g_g_contra.a12(j, i) *
                t.sqrt_g_g_contra.a12(j, i);

            VVM::Real contravariant_1;
            VVM::Real contravariant_2;

            t.physical_to_contravariant.apply(
                j,
                i,
                VVM::real(30.0),
                VVM::real(-12.0),
                contravariant_1,
                contravariant_2);

            results(13) =
                contravariant_1;

            results(14) =
                contravariant_2;

            VVM::Real physical_1;
            VVM::Real physical_2;

            t.contravariant_to_physical.apply(
                j,
                i,
                contravariant_1,
                contravariant_2,
                physical_1,
                physical_2);

            results(15) =
                physical_1;

            results(16) =
                physical_2;

            results(17) =
                t.sqrt_g(j, i) *
                t.inv_sqrt_g(j, i);

            results(18) =
                v.latitude(
                    equator_j_at_v,
                    i);

            results(19) =
                v.sqrt_g(
                    equator_j_at_v,
                    i);

            results(20) =
                v.g_cov.a11(
                    equator_j_at_v,
                    i);

            results(21) =
                v.sqrt_g_g_contra.a11(
                    equator_j_at_v,
                    i);

            results(22) =
                v.sqrt_g_g_contra.a22(
                    equator_j_at_v,
                    i);
        });

    const auto host =
        Kokkos::create_mirror_view_and_copy(
            Kokkos::HostSpace(),
            results);

    const VVM::Real radius =
        geometry.radius();

    const VVM::Real cosine =
        std::cos(host(0));

    const VVM::Real radius_squared =
        radius * radius;

    check(
        close(
            host(1),
            radius_squared * cosine),
        "RLL J must equal R^2 cos(latitude)");

    check(
        close(
            host(2),
            VVM::real(1.0) /
                (radius_squared * cosine)),
        "RLL inverse J must be correct");

    check(
        close(
            host(3),
            radius_squared *
                cosine *
                cosine),
        "RLL covariant g11 must equal R^2 cos^2(latitude)");

    check(
        close(
            host(4),
            VVM::real(0.0)),
        "RLL covariant g12 must be zero");

    check(
        close(
            host(5),
            radius_squared),
        "RLL covariant g22 must equal R^2");

    check(
        close(
            host(6),
            VVM::real(1.0) /
                cosine),
        "RLL J g^11 must equal 1/cos(latitude)");

    check(
        close(
            host(7),
            VVM::real(0.0)),
        "RLL J g^12 must be zero");

    check(
        close(
            host(8),
            cosine),
        "RLL J g^22 must equal cos(latitude)");

    check(
        close(
            host(9),
            VVM::real(1.0) /
                (radius_squared *
                 cosine *
                 cosine)),
        "RLL g^11 must be correct");

    check(
        close(
            host(10),
            VVM::real(0.0)),
        "RLL g^12 must be zero");

    check(
        close(
            host(11),
            VVM::real(1.0) /
                radius_squared),
        "RLL g^22 must equal 1/R^2");

    check(
        close(
            host(12),
            VVM::real(1.0)),
        "det(J g^ij) must equal one");

    check(
        close(
            host(13),
            VVM::real(30.0) /
                (radius * cosine)),
        "eastward wind must convert to u^1");

    check(
        close(
            host(14),
            VVM::real(-12.0) /
                radius),
        "northward wind must convert to u^2");

    check(
        close(
            host(15),
            VVM::real(30.0)),
        "u^1 must convert back to eastward wind");

    check(
        close(
            host(16),
            VVM::real(-12.0)),
        "u^2 must convert back to northward wind");

    check(
        close(
            host(17),
            VVM::real(1.0)),
        "J times inverse J must equal one");

    check(
        close(
            host(18),
            VVM::real(0.0)),
        "V latitude index 49 must lie on the equator");

    check(
        close(
            host(19),
            radius_squared),
        "V Jacobian at the equator must equal R^2");

    check(
        close(
            host(20),
            radius_squared),
        "V g11 at the equator must equal R^2");

    check(
        close(
            host(21),
            VVM::real(1.0)),
        "V J g^11 at the equator must equal one");

    check(
        close(
            host(22),
            VVM::real(1.0)),
        "V J g^22 at the equator must equal one");
}

void test_channel_area() {
    const HorizontalDomainLayout layout =
        make_layout();

    const auto geometry =
        make_geometry();

    const auto t =
        geometry.device_view(
            HorizontalLocation::T);

    const int h =
        layout.halo;

    Kokkos::View<VVM::Real> numerical_area(
        "regular_latlon_channel_area");

    Kokkos::parallel_reduce(
        "SumRegularLatLonChannelArea",
        Kokkos::MDRangePolicy<Kokkos::Rank<2>>(
            {h, h},
            {
                h + layout.local_physical_ny,
                h + layout.local_physical_nx
            }),
        KOKKOS_LAMBDA(
            const int j,
            const int i,
            VVM::Real& sum) {

            sum +=
                t.sqrt_g(j, i) *
                t.dq1 *
                t.dq2;
        },
        numerical_area);

    VVM::Real numerical_area_host =
        VVM::real(0.0);

    Kokkos::deep_copy(
        numerical_area_host,
        numerical_area);

    const VVM::Real pi =
        std::acos(VVM::real(-1.0));

    const VVM::Real south =
        -pi / VVM::real(4.0);

    const VVM::Real north =
        pi / VVM::real(4.0);

    const VVM::Real exact_area =
        VVM::real(2.0) *
        pi *
        geometry.radius() *
        geometry.radius() *
        (
            std::sin(north) -
            std::sin(south)
        );

    const VVM::Real relative_error =
        std::abs(
            numerical_area_host -
            exact_area) /
        exact_area;

    const VVM::Real area_tolerance =
        sizeof(VVM::Real) == sizeof(double)
            ? VVM::real(2.0e-5)
            : VVM::real(1.0e-4);

    check(
        relative_error <
            area_tolerance,
        "0.9-degree midpoint cell areas must reproduce "
        "the analytic channel area");
}

void test_invalid_inputs() {
    const VVM::Real pi =
        std::acos(VVM::real(-1.0));

    const VVM::Real spacing =
        VVM::real(0.9) *
        pi /
        VVM::real(180.0);

    HorizontalDomainLayout layout =
        make_layout();

    bool threw = false;

    try {
        RegularLatLonGeometry unused(
            layout,
            spacing,
            spacing,
            VVM::real(0.0),
            -pi / VVM::real(4.0),
            VVM::real(-1.0));
    }
    catch (const std::invalid_argument&) {
        threw = true;
    }

    check(
        threw,
        "RLL geometry must reject a negative radius");

    threw = false;

    try {
        RegularLatLonGeometry unused(
            layout,
            VVM::real(0.0),
            spacing,
            VVM::real(0.0),
            -pi / VVM::real(4.0),
            VVM::real(6371220.0));
    }
    catch (const std::invalid_argument&) {
        threw = true;
    }

    check(
        threw,
        "RLL geometry must reject zero longitude spacing");

    threw = false;

    try {
        RegularLatLonGeometry unused(
            layout,
            spacing,
            spacing,
            VVM::real(0.0),
            VVM::real(-89.0) *
                pi /
                VVM::real(180.0),
            VVM::real(6371220.0));
    }
    catch (const std::invalid_argument&) {
        threw = true;
    }

    check(
        threw,
        "RLL geometry must reject latitude halos that reach a pole");

    threw = false;
    layout.panel_id = 0;

    try {
        RegularLatLonGeometry unused(
            layout,
            spacing,
            spacing,
            VVM::real(0.0),
            -pi / VVM::real(4.0),
            VVM::real(6371220.0));
    }
    catch (const std::invalid_argument&) {
        threw = true;
    }

    check(
        threw,
        "RLL geometry must reject a cubed-sphere panel id");
}

} // namespace

int main(
    int argc,
    char** argv) {

    Kokkos::initialize(
        argc,
        argv);

    {
        try {
            test_host_metadata();
            test_compact_storage_and_sharing();
            test_coordinates_and_halos();
            test_metric_and_vector_transforms();
            test_channel_area();
            test_invalid_inputs();
        }
        catch (const std::exception& error) {
            ++failures;

            std::fprintf(
                stderr,
                "Unexpected exception: %s\n",
                error.what());
        }
    }

    Kokkos::finalize();

    if (failures == 0) {
        std::fprintf(
            stdout,
            "test_regular_latlon_geometry: PASS\n");
    }
    else {
        std::fprintf(
            stderr,
            "test_regular_latlon_geometry: %d failure(s)\n",
            failures);
    }

    return failures == 0
        ? 0
        : 1;
}
