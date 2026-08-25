#include "TracerSource.hpp"

#include <Kokkos_Core.hpp>

namespace VVM {
namespace Dynamics {

TracerSource::TracerSource(
    const Core::Grid& grid,
    const Core::State& state)
    : grid_(grid),
      target_vars_(state.get_tracer_source_targets()) {}

void TracerSource::apply(Core::State& state, VVM::Real dt) const {
    const int nz = grid_.get_local_total_points_z();
    const int ny = grid_.get_local_total_points_y();
    const int nx = grid_.get_local_total_points_x();
    const int h = grid_.get_halo_cells();
    const auto ITYPEW = ITYPEW_ref_.get(state, "ITYPEW").get_device_data();

    if (source_pairs_.empty()) {
        source_pairs_.reserve(target_vars_.size());
        for (const auto& tracer_name : target_vars_) {
            source_pairs_.push_back({&state.get_field<3>(tracer_name),
                                     &state.get_field<3>(tracer_name + "_source"),
                                     "apply_tracer_source_" + tracer_name});
        }
    }

    for (const auto& pair : source_pairs_) {
        auto& tracer = pair.tracer->get_mutable_device_data();
        const auto source = pair.source->get_device_data();

        // Source values are concentration per second, so each forcing call
        // advances the tracer by dt times the prescribed source field.
        Kokkos::parallel_for(pair.kernel_label,
            Kokkos::MDRangePolicy<Kokkos::Rank<3>>({h, h, h}, {nz - h, ny - h, nx - h}),
            KOKKOS_LAMBDA(const int k, const int j, const int i) {
                if (ITYPEW(k, j, i) == VVM::real(1.0)) {
                    tracer(k, j, i) += dt * source(k, j, i);
                }
            });
    }
}

} // namespace Dynamics
} // namespace VVM
