#include <cmath>
#include <stdexcept>
#include <iostream>

#include "Parameters.hpp"
#include "core/geometry/GeometryKind.hpp"

namespace VVM {
namespace Core {

Parameters::Parameters(const Utils::ConfigurationManager& config, const Grid& grid) :
    gravity("gravity"),
    Rd("Rd"),
    PSFC("PSFC"),
    P0("P0"),
    Cp("Cp"),
    Lv("Lv"),
    dx("dx"),
    dy("dy"),
    dz("dz"),
    dt("dt"),
    rdx("rdx"),
    rdy("rdy"),
    rdz("rdz"),
    rdx2("rdx2"),
    rdy2("rdy2"),
    rdz2("rdz2"),
    WRXMU("WRXMU"),
    fact1_zeta("fact1_zeta", {}),
    fact2_zeta("fact2_zeta", {}),
    z_mid("z_mid", {grid.get_local_total_points_z()}),
    z_up("z_up", {grid.get_local_total_points_z()}),
    flex_height_coef_mid("flex_height_coef_mid", {grid.get_local_total_points_z()}),
    flex_height_coef_up("flex_height_coef_up", {grid.get_local_total_points_z()}),
    dz_mid("dz_mid", {grid.get_local_total_points_z()}),
    dz_up("dz_up", {grid.get_local_total_points_z()}),
    fact1_xi_eta("fact1_xi_eta", {grid.get_local_total_points_z()}),
    fact2_xi_eta("fact2_xi_eta", {grid.get_local_total_points_z()}),
    AGAU("AGAU", {grid.get_local_total_points_z()}),
    BGAU("BGAU", {grid.get_local_total_points_z()}),
    CGAU("CGAU", {grid.get_local_total_points_z()}),
    bn_new("bn_new", {grid.get_local_total_points_z()}),
    cn_new("cn_new", {grid.get_local_total_points_z()})
{
    const auto& horizontal = grid.horizontal_specification();
    const auto& vertical = grid.vertical_specification();

    // Existing consumers interpret Parameters::dx/dy as Cartesian distances.
    // RLL angular increments must not silently enter those legacy kernels.
    // Geometry and standalone elliptic/operator tests do not need Parameters.
    if (horizontal.geometry.kind != Geometry::GeometryKind::Cartesian) {
        throw std::runtime_error(
            "Parameters still exposes Cartesian dx/dy in metres. Non-Cartesian Parameters construction "
            "is not enabled until its spacing-dependent consumers have been migrated. "
            "Use Grid geometry directly in the existing RLL component tests.");
    }


    Kokkos::deep_copy(gravity, config.get_value<VVM::Real>("constants.gravity"));
    Kokkos::deep_copy(Rd, config.get_value<VVM::Real>("constants.Rd"));
    Kokkos::deep_copy(P0, config.get_value<VVM::Real>("constants.P0"));
    Kokkos::deep_copy(Cp, config.get_value<VVM::Real>("constants.Cp"));

    const VVM::Real dx_val = horizontal.geometry.dq1;
    const VVM::Real dy_val = horizontal.geometry.dq2;
    const VVM::Real dz_val = vertical.dz;

    VVM::Real dt_val = config.get_value<VVM::Real>("simulation.dt_s");

    const auto require_finite_positive = [](VVM::Real value, const char* key) {
        if (!std::isfinite(value) || value <= VVM::real(0.0)) {
            throw std::runtime_error(
                std::string("Configuration error: '") + key +
                "' must be finite and greater than zero.");
        }
    };
    require_finite_positive(dx_val, "grid.dx");
    require_finite_positive(dy_val, "grid.dy");
    require_finite_positive(dz_val, "grid.dz");
    require_finite_positive(dt_val, "simulation.dt_s");

    Kokkos::deep_copy(dx, dx_val);
    Kokkos::deep_copy(dy, dy_val);
    Kokkos::deep_copy(dz, dz_val);
    Kokkos::deep_copy(dt, dt_val);

    Kokkos::deep_copy(rdx, real(1.0) / dx_val);
    Kokkos::deep_copy(rdy, real(1.0) / dy_val);
    Kokkos::deep_copy(rdz, real(1.0) / dz_val);
    
    Kokkos::deep_copy(rdx2, real(1.0) / (dx_val * dx_val));
    Kokkos::deep_copy(rdy2, real(1.0) / (dy_val * dy_val));
    Kokkos::deep_copy(rdz2, real(1.0) / (dz_val * dz_val));

    VVM::Real WRXMU_val = config.get_value<VVM::Real>("dynamics.solver.WRXMU");
    Kokkos::deep_copy(WRXMU, WRXMU_val);

    solver_iteration = config.get_value<int>("dynamics.solver.iteration");

    Kokkos::fence();
}

VVM::Real Parameters::get_value_host(const Kokkos::View<VVM::Real>& device_view) const {
    VVM::Real host_value;
    Kokkos::deep_copy(host_value, device_view);
    return host_value;
}

} // namespace Core
} // namespace VVM
