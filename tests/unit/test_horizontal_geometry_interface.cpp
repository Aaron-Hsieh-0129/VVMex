#include "core/geometry/HorizontalGeometry.hpp"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>

namespace {

using VVM::Core::GridStaggering;
using VVM::Core::Geometry::GeometryKind;
using VVM::Core::Geometry::HorizontalDomainLayout;
using VVM::Core::Geometry::HorizontalGeometry;
using VVM::Core::Geometry::HorizontalGeometryDeviceView;
using VVM::Core::Geometry::HorizontalLocation;
using VVM::Core::Geometry::horizontal_location_or_throw;
using VVM::Core::Geometry::try_horizontal_location;

int failures = 0;

void check(
    const bool condition,
    const char* message) {

    if (!condition) {
        std::fprintf(stderr, "FAIL: %s\n", message);
        ++failures;
    }
}

void check_mapping(
    const GridStaggering staggering,
    const HorizontalLocation expected,
    const char* description) {

    const auto actual = try_horizontal_location(staggering);

    if (!actual.has_value()) {
        std::fprintf(
            stderr,
            "FAIL: %s produced no horizontal location\n",
            description);

        ++failures;
        return;
    }

    if (*actual != expected) {
        std::fprintf(
            stderr,
            "FAIL: %s produced the wrong horizontal location\n",
            description);

        ++failures;
    }
}

// Minimal concrete geometry used only to test the host-side interface.
//
// It does not allocate metric fields or launch Kokkos kernels.
class MockGeometry final : public HorizontalGeometry {
public:
    MockGeometry() {
        layout_.global_nx = 400;
        layout_.global_ny = 100;

        layout_.local_physical_nx = 100;
        layout_.local_physical_ny = 50;

        layout_.global_start_i = 200;
        layout_.global_start_j = 50;

        layout_.halo = 2;
        layout_.panel_id = -1;
    }

    GeometryKind kind() const noexcept override {
        return GeometryKind::RegularLatLon;
    }

    const char* name() const noexcept override {
        return "mock_regular_latlon";
    }

    const HorizontalDomainLayout& layout() const noexcept override {
        return layout_;
    }

    VVM::Real dq1() const noexcept override {
        return VVM::real(0.01);
    }

    VVM::Real dq2() const noexcept override {
        return VVM::real(0.02);
    }

    HorizontalLocation last_location() const noexcept {
        return last_location_;
    }

private:
    HorizontalGeometryDeviceView device_view_impl(
        const HorizontalLocation location) const override {

        last_location_ = location;

        HorizontalGeometryDeviceView result;
        result.dq1 = dq1();
        result.dq2 = dq2();

        return result;
    }

    HorizontalDomainLayout layout_;

    mutable HorizontalLocation last_location_ =
        HorizontalLocation::T;
};

void test_staggering_mapping() {
    check_mapping(
        GridStaggering::Centered,
        HorizontalLocation::T,
        "Centered");

    check_mapping(
        GridStaggering::StaggeredZ,
        HorizontalLocation::T,
        "StaggeredZ");

    check_mapping(
        GridStaggering::Surface,
        HorizontalLocation::T,
        "Surface");

    check_mapping(
        GridStaggering::StaggeredX,
        HorizontalLocation::U,
        "StaggeredX");

    check_mapping(
        GridStaggering::StaggeredXZ,
        HorizontalLocation::U,
        "StaggeredXZ");

    check_mapping(
        GridStaggering::StaggeredY,
        HorizontalLocation::V,
        "StaggeredY");

    check_mapping(
        GridStaggering::StaggeredYZ,
        HorizontalLocation::V,
        "StaggeredYZ");

    check_mapping(
        GridStaggering::StaggeredXY,
        HorizontalLocation::Z,
        "StaggeredXY");

    check_mapping(
        GridStaggering::StaggeredXYZ,
        HorizontalLocation::Z,
        "StaggeredXYZ");

    check(
        !try_horizontal_location(
            GridStaggering::Unspecified).has_value(),
        "Unspecified must not have a horizontal location");

    check(
        !try_horizontal_location(
            GridStaggering::NotApplicable).has_value(),
        "NotApplicable must not have a horizontal location");
}

void test_strict_mapping_errors() {
    bool threw = false;

    try {
        (void)horizontal_location_or_throw(
            GridStaggering::Unspecified,
            "bad_field");
    } catch (const std::invalid_argument& error) {
        threw = true;

        const std::string message = error.what();

        check(
            message.find("bad_field") != std::string::npos,
            "mapping error must contain the field name");

        check(
            message.find("unspecified") != std::string::npos,
            "mapping error must contain the staggering name");
    }

    check(
        threw,
        "strict conversion must reject Unspecified");

    threw = false;

    try {
        (void)horizontal_location_or_throw(
            GridStaggering::NotApplicable);
    } catch (const std::invalid_argument&) {
        threw = true;
    }

    check(
        threw,
        "strict conversion must reject NotApplicable");
}

void test_layout() {
    const MockGeometry geometry;
    const auto& layout = geometry.layout();

    check(
        layout.global_nx == 400,
        "global_nx must be preserved");

    check(
        layout.global_ny == 100,
        "global_ny must be preserved");

    check(
        layout.local_total_nx() == 104,
        "local_total_nx must include both halos");

    check(
        layout.local_total_ny() == 54,
        "local_total_ny must include both halos");

    check(
        layout.global_start_i == 200,
        "global_start_i must be preserved");

    check(
        layout.global_start_j == 50,
        "global_start_j must be preserved");

    check(
        layout.panel_id == -1,
        "non-panel geometry must use panel_id -1");
}

void test_geometry_identity() {
    const MockGeometry geometry;

    check(
        geometry.kind() == GeometryKind::RegularLatLon,
        "geometry kind must be returned");

    check(
        std::strcmp(
            geometry.name(),
            "mock_regular_latlon") == 0,
        "geometry name must be returned");

    check(
        std::abs(
            geometry.dq1() -
            VVM::real(0.01)) < VVM::real(1.0e-12),
        "dq1 must be returned");

    check(
        std::abs(
            geometry.dq2() -
            VVM::real(0.02)) < VVM::real(1.0e-12),
        "dq2 must be returned");
}

void test_explicit_location_dispatch() {
    MockGeometry geometry;

    const auto result =
        geometry.device_view(HorizontalLocation::U);

    check(
        geometry.last_location() == HorizontalLocation::U,
        "explicit U lookup must dispatch to U storage");

    check(
        std::abs(
            result.dq1 -
            geometry.dq1()) < VVM::real(1.0e-12),
        "device view must contain dq1");

    check(
        std::abs(
            result.dq2 -
            geometry.dq2()) < VVM::real(1.0e-12),
        "device view must contain dq2");
}

void test_grid_staggering_dispatch() {
    MockGeometry geometry;

    (void)geometry.device_view(
        GridStaggering::Centered,
        "th");

    check(
        geometry.last_location() == HorizontalLocation::T,
        "Centered field must dispatch to T geometry");

    (void)geometry.device_view(
        GridStaggering::StaggeredX,
        "u");

    check(
        geometry.last_location() == HorizontalLocation::U,
        "StaggeredX field must dispatch to U geometry");

    (void)geometry.device_view(
        GridStaggering::StaggeredYZ,
        "xi");

    check(
        geometry.last_location() == HorizontalLocation::V,
        "StaggeredYZ field must dispatch to V geometry");

    (void)geometry.device_view(
        GridStaggering::StaggeredXY,
        "zeta");

    check(
        geometry.last_location() == HorizontalLocation::Z,
        "StaggeredXY field must dispatch to Z geometry");
}

void test_string_conversions() {
    check(
        std::strcmp(
            VVM::Core::grid_staggering_to_string(
                GridStaggering::Surface),
            "surface") == 0,
        "Surface must have a string representation");

    check(
        std::strcmp(
            VVM::Core::Geometry::geometry_kind_to_string(
                GeometryKind::RegularLatLon),
            "regular_latlon") == 0,
        "RegularLatLon must have a string representation");

    check(
        std::strcmp(
            VVM::Core::Geometry::horizontal_location_to_string(
                HorizontalLocation::Z),
            "Z") == 0,
        "Z must have a string representation");
}

} // namespace

int main() {
    test_staggering_mapping();
    test_strict_mapping_errors();
    test_layout();
    test_geometry_identity();
    test_explicit_location_dispatch();
    test_grid_staggering_dispatch();
    test_string_conversions();

    if (failures == 0) {
        std::puts(
            "test_horizontal_geometry_interface: PASS");
    }

    return failures == 0 ? 0 : 1;
}
