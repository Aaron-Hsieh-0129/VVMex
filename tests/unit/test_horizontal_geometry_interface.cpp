#include "core/geometry/HorizontalGeometry.hpp"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>

namespace {

using VVM::Core::GridStaggering;
using VVM::Core::Geometry::GeometryKind;
using VVM::Core::Geometry::horizontal_location_or_throw;
using VVM::Core::Geometry::HorizontalDomainLayout;
using VVM::Core::Geometry::HorizontalGeometry;
using VVM::Core::Geometry::HorizontalLocation;
using VVM::Core::Geometry::try_horizontal_location;

int failures = 0;

void
check(const bool condition, const char* message) {

    if (!condition) {
        std::fprintf(stderr, "FAIL: %s\n", message);
        ++failures;
    }
}

void
check_mapping(
    const GridStaggering staggering, const HorizontalLocation expected, const char* description) {

    const auto actual = try_horizontal_location(staggering);

    if (!actual.has_value()) {
        std::fprintf(stderr, "FAIL: %s produced no horizontal location\n", description);

        ++failures;
        return;
    }

    if (*actual != expected) {
        std::fprintf(stderr, "FAIL: %s produced the wrong horizontal location\n", description);

        ++failures;
    }
}

// Constant inputs exercise the shared host interface without device allocation.
HorizontalGeometry
make_geometry() {
    HorizontalDomainLayout layout;
    layout.global_nx = 400;
    layout.global_ny = 100;
    layout.local_physical_nx = 100;
    layout.local_physical_ny = 50;
    layout.global_start_i = 200;
    layout.global_start_j = 50;
    layout.halo = 2;
    std::array<VVM::Core::Geometry::HorizontalGeometryInput, 4> inputs;
    const auto c = VVM::Core::Geometry::GeometryField2D::constant_value;
    for (int l = 0; l < 4; ++l) {
        inputs[l].g_cov = {c(VVM::real(1)), c(VVM::real(0)), c(VVM::real(0)), c(VVM::real(1))};
        inputs[l].contravariant_to_physical = inputs[l].g_cov;
        inputs[l].q1 = c(static_cast<VVM::Real>(l));
    }
    return HorizontalGeometry(layout,
        VVM::real(.01),
        VVM::real(.02),
        inputs,
        GeometryKind::RegularLatLon,
        "mock_regular_latlon");
}

void
test_staggering_mapping() {
    check_mapping(GridStaggering::Centered, HorizontalLocation::T, "Centered");

    check_mapping(GridStaggering::StaggeredZ, HorizontalLocation::T, "StaggeredZ");

    check_mapping(GridStaggering::Surface, HorizontalLocation::T, "Surface");

    check_mapping(GridStaggering::StaggeredX, HorizontalLocation::U, "StaggeredX");

    check_mapping(GridStaggering::StaggeredXZ, HorizontalLocation::U, "StaggeredXZ");

    check_mapping(GridStaggering::StaggeredY, HorizontalLocation::V, "StaggeredY");

    check_mapping(GridStaggering::StaggeredYZ, HorizontalLocation::V, "StaggeredYZ");

    check_mapping(GridStaggering::StaggeredXY, HorizontalLocation::Z, "StaggeredXY");

    check_mapping(GridStaggering::StaggeredXYZ, HorizontalLocation::Z, "StaggeredXYZ");

    check(!try_horizontal_location(GridStaggering::Unspecified).has_value(),
        "Unspecified must not have a horizontal location");

    check(!try_horizontal_location(GridStaggering::NotApplicable).has_value(),
        "NotApplicable must not have a horizontal location");
}

void
test_strict_mapping_errors() {
    bool threw = false;

    try {
        (void)horizontal_location_or_throw(GridStaggering::Unspecified, "bad_field");
    }
    catch (const std::invalid_argument& error) {
        threw = true;

        const std::string message = error.what();

        check(message.find("bad_field") != std::string::npos,
            "mapping error must contain the field name");

        check(message.find("unspecified") != std::string::npos,
            "mapping error must contain the staggering name");
    }

    check(threw, "strict conversion must reject Unspecified");

    threw = false;

    try {
        (void)horizontal_location_or_throw(GridStaggering::NotApplicable);
    }
    catch (const std::invalid_argument&) {
        threw = true;
    }

    check(threw, "strict conversion must reject NotApplicable");
}

void
test_layout() {
    const auto geometry = make_geometry();
    const auto& layout = geometry.layout();

    check(layout.global_nx == 400, "global_nx must be preserved");

    check(layout.global_ny == 100, "global_ny must be preserved");

    check(layout.local_total_nx() == 104, "local_total_nx must include both halos");

    check(layout.local_total_ny() == 54, "local_total_ny must include both halos");

    check(layout.global_start_i == 200, "global_start_i must be preserved");

    check(layout.global_start_j == 50, "global_start_j must be preserved");

    check(layout.panel_id == -1, "non-panel geometry must use panel_id -1");
}

void
test_geometry_identity() {
    const auto geometry = make_geometry();

    check(geometry.kind() == GeometryKind::RegularLatLon, "geometry kind must be returned");

    check(std::strcmp(geometry.name(), "mock_regular_latlon") == 0,
        "geometry name must be returned");

    check(std::abs(geometry.dq1() - VVM::real(0.01)) < VVM::real(1.0e-12), "dq1 must be returned");

    check(std::abs(geometry.dq2() - VVM::real(0.02)) < VVM::real(1.0e-12), "dq2 must be returned");
}

void
test_explicit_location_dispatch() {
    auto geometry = make_geometry();

    const auto result = geometry.device_view(HorizontalLocation::U);

    check(result.q1.constant == VVM::real(1), "explicit U lookup must dispatch to U storage");

    check(std::abs(result.dq1 - geometry.dq1()) < VVM::real(1.0e-12),
        "device view must contain dq1");

    check(std::abs(result.dq2 - geometry.dq2()) < VVM::real(1.0e-12),
        "device view must contain dq2");
}

void
test_grid_staggering_dispatch() {
    auto geometry = make_geometry();

    const auto t = geometry.device_view(GridStaggering::Centered, "th");

    check(t.q1.constant == VVM::real(0), "Centered field must dispatch to T geometry");

    const auto u = geometry.device_view(GridStaggering::StaggeredX, "u");

    check(u.q1.constant == VVM::real(1), "StaggeredX field must dispatch to U geometry");

    const auto v = geometry.device_view(GridStaggering::StaggeredYZ, "xi");

    check(v.q1.constant == VVM::real(2), "StaggeredYZ field must dispatch to V geometry");

    const auto z = geometry.device_view(GridStaggering::StaggeredXY, "zeta");

    check(z.q1.constant == VVM::real(3), "StaggeredXY field must dispatch to Z geometry");
}

void
test_string_conversions() {
    check(std::strcmp(VVM::Core::grid_staggering_to_string(GridStaggering::Surface), "surface") ==
              0,
        "Surface must have a string representation");

    check(std::strcmp(VVM::Core::Geometry::horizontal_location_to_string(HorizontalLocation::Z),
              "Z") == 0,
        "Z must have a string representation");
}

} // namespace

int
main() {
    test_staggering_mapping();
    test_strict_mapping_errors();
    test_layout();
    test_geometry_identity();
    test_explicit_location_dispatch();
    test_grid_staggering_dispatch();
    test_string_conversions();

    if (failures == 0) {
        std::puts("test_horizontal_geometry_interface: PASS");
    }

    return failures == 0 ? 0 : 1;
}
