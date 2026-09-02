#include "core/geometry/GeometryField2D.hpp"

#include <cmath>
#include <cstdio>

#include <Kokkos_Core.hpp>

namespace {

using VVM::Core::Geometry::GeometryField2D;
using VVM::Core::Geometry::GeometryFieldLayout;

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
           VVM::real(1.0e-6);
}

void run_test() {
    constexpr int nx = 4;
    constexpr int ny = 3;

    Kokkos::View<VVM::Real*> values_i(
        "geometry_values_i",
        nx);

    Kokkos::View<VVM::Real*> values_j(
        "geometry_values_j",
        ny);

    Kokkos::View<VVM::Real**> values_2d(
        "geometry_values_2d",
        ny,
        nx);

    Kokkos::parallel_for(
        "InitializeGeometryValuesI",
        Kokkos::RangePolicy<>(0, nx),
        KOKKOS_LAMBDA(const int i) {
            values_i(i) =
                VVM::real(10.0) +
                static_cast<VVM::Real>(i);
        });

    Kokkos::parallel_for(
        "InitializeGeometryValuesJ",
        Kokkos::RangePolicy<>(0, ny),
        KOKKOS_LAMBDA(const int j) {
            values_j(j) =
                VVM::real(20.0) +
                VVM::real(2.0) *
                    static_cast<VVM::Real>(j);
        });

    Kokkos::parallel_for(
        "InitializeGeometryValues2D",
        Kokkos::MDRangePolicy<Kokkos::Rank<2>>(
            {0, 0},
            {ny, nx}),
        KOKKOS_LAMBDA(
            const int j,
            const int i) {

            values_2d(j, i) =
                VVM::real(100.0) +
                VVM::real(10.0) *
                    static_cast<VVM::Real>(j) +
                static_cast<VVM::Real>(i);
        });

    const GeometryField2D constant_field =
        GeometryField2D::constant_value(
            VVM::real(7.0));

    const GeometryField2D i_field =
        GeometryField2D::varying_i(
            values_i);

    const GeometryField2D j_field =
        GeometryField2D::varying_j(
            values_j);

    const GeometryField2D full_field =
        GeometryField2D::full_2d(
            values_2d);

    Kokkos::View<VVM::Real*> results(
        "geometry_field_results",
        4);

    Kokkos::parallel_for(
        "EvaluateGeometryFieldLayouts",
        Kokkos::RangePolicy<>(0, 1),
        KOKKOS_LAMBDA(const int) {
            constexpr int j = 2;
            constexpr int i = 3;

            results(0) =
                constant_field(j, i);

            results(1) =
                i_field(j, i);

            results(2) =
                j_field(j, i);

            results(3) =
                full_field(j, i);
        });

    const auto host_results =
        Kokkos::create_mirror_view_and_copy(
            Kokkos::HostSpace(),
            results);

    check(
        constant_field.layout ==
            GeometryFieldLayout::Constant,
        "constant factory must select Constant layout");

    check(
        i_field.layout ==
            GeometryFieldLayout::VaryingI,
        "i factory must select VaryingI layout");

    check(
        j_field.layout ==
            GeometryFieldLayout::VaryingJ,
        "j factory must select VaryingJ layout");

    check(
        full_field.layout ==
            GeometryFieldLayout::Full2D,
        "2-D factory must select Full2D layout");

    check(
        close(
            host_results(0),
            VVM::real(7.0)),
        "constant field must return its scalar value");

    check(
        close(
            host_results(1),
            VVM::real(13.0)),
        "VaryingI field must use the i index");

    check(
        close(
            host_results(2),
            VVM::real(24.0)),
        "VaryingJ field must use the j index");

    check(
        close(
            host_results(3),
            VVM::real(123.0)),
        "Full2D field must use both j and i");
}

} // namespace

int main(
    int argc,
    char** argv) {

    Kokkos::initialize(
        argc,
        argv);

    {
        run_test();
    }

    Kokkos::finalize();

    if (failures == 0) {
        std::puts(
            "test_geometry_field_2d: PASS");

        return 0;
    }

    std::fprintf(
        stderr,
        "test_geometry_field_2d: %d failure(s)\n",
        failures);

    return 1;
}
