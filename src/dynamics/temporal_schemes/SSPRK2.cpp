#include "SSPRK2.hpp"

#include <stdexcept>
#include <utility>

namespace VVM {
namespace Dynamics {

SSPRK2::SSPRK2(
    std::string var_name,
    const std::array<int, 3>& dimensions)
    : variable_name_(std::move(var_name)),
      original_("ssprk2_original_" + variable_name_, dimensions),
      stage_one_("ssprk2_stage_one_" + variable_name_, dimensions) {}

void SSPRK2::step(
    Core::State&,
    const Core::Grid&,
    const Core::Parameters&,
    VVM::Real) const {
    throw std::runtime_error(
        "SSPRK2 for '" + variable_name_ +
        "' must be orchestrated by TimeIntegrator with tendency recomputation.");
}

void SSPRK2::begin_multistage_step(
    Core::State& state,
    const Core::Grid&,
    const Core::Parameters&) const {
    Kokkos::deep_copy(
        Kokkos::DefaultExecutionSpace(),
        original_.get_mutable_device_data(),
        state.get_field<3>(variable_name_).get_device_data());
}

void SSPRK2::advance_multistage(
    Core::State& state,
    const Core::Grid& grid,
    const Core::Parameters&,
    const Core::Field<3>& tendency,
    VVM::Real dt,
    int stage) const {
    if (stage != 0 && stage != 1) {
        throw std::runtime_error(
            "SSPRK2 for '" + variable_name_ +
            "' received an invalid stage index.");
    }

    const int nz = grid.get_local_total_points_z();
    const int ny = grid.get_local_total_points_y();
    const int nx = grid.get_local_total_points_x();
    const int h = grid.get_halo_cells();

    auto q = state.get_field<3>(variable_name_).get_mutable_device_data();
    const auto qn = original_.get_device_data();
    auto q1 = stage_one_.get_mutable_device_data();
    const auto rhs = tendency.get_device_data();

    if (stage == 0) {
        Kokkos::parallel_for("SSPRK2_stage_one_" + variable_name_,
            Kokkos::MDRangePolicy<Kokkos::Rank<3>>({h, h, h}, {nz - h, ny - h, nx - h}),
            KOKKOS_LAMBDA(const int k, const int j, const int i) {
                q1(k, j, i) = qn(k, j, i) + dt * rhs(k, j, i);
                q(k, j, i) = q1(k, j, i);
            });
        return;
    }

    Kokkos::parallel_for("SSPRK2_stage_two_" + variable_name_,
        Kokkos::MDRangePolicy<Kokkos::Rank<3>>({h, h, h}, {nz - h, ny - h, nx - h}),
        KOKKOS_LAMBDA(const int k, const int j, const int i) {
            q(k, j, i) = VVM::real(0.5) * qn(k, j, i) +
                         VVM::real(0.5) * (q1(k, j, i) + dt * rhs(k, j, i));
        });
}

} // namespace Dynamics
} // namespace VVM
