#include "VVM_cleo_process_interface.hpp"
#include "core/State.hpp"
#include "core/vvm_types.hpp"

namespace VVM::Physics {

// PIMPL
struct VVM_CLEO_Interface::Impl {
    Impl() {}

    void
    initialize(Core::State& state) {}
    void
    run(Core::State& state) {}
    void
    finalize() {}
};

VVM_CLEO_Interface::VVM_CLEO_Interface(const VVM::Utils::ConfigurationManager& config,
    const VVM::Core::Grid& grid,
    const VVM::Core::Parameters& params,
    Core::HaloExchanger& halo_exchanger,
    Core::State& state) {

    const std::array<int, 3> shape{grid.get_local_total_points_z(),
        grid.get_local_total_points_y(),
        grid.get_local_total_points_x()};

    const auto add_3d_field = [&](const char* name, const char* units, const char* long_name) {
        if (!state.has_field(name)) {
            state.add_field<3>(name,
                shape,
                Core::FieldMetadata{Core::GridStaggering::Centered, units, long_name});
        }
    };
    add_3d_field("qc", "kg kg-1", "cloud liquid water mass mixing ratio (radius <= 40um)");
    add_3d_field("qr", "kg kg-1", "rain water mass mixing ratio (radius > 40um");
    add_3d_field("qp", "kg kg-1", "total hydrometeor mass mixing ratio");

    add_3d_field("nc", "kg-1", "cloud droplet number mixing ratio");
    add_3d_field("nr", "kg-1", "rain droplet number mixing ratio");
    // No ice in CLEO, always zero
    add_3d_field("qi", "kg kg-1", "total ice mass mixing ratio");
    add_3d_field("diag_eff_radius_qc", "um", "cloud droplet effective radius");
    add_3d_field("diag_eff_radius_qr", "um", "rain drop effective radius");
    add_3d_field("diag_eff_radius_qi", "um", "ice particle effective radius");

    // no ice grid set to 25um, consistent with P3
    Kokkos::deep_copy(state.get_field<3>("diag_eff_radius_qi").get_mutable_device_data(),
        real(25.0));

    const std::array<int, 2> sfc_shape{grid.get_local_total_points_y(),
        grid.get_local_total_points_x()};
    const auto add_2d_field = [&](const char* name, const char* long_name) {
        if (!state.has_field(name)) {
            state.add_field<2>(name,
                sfc_shape,
                Core::FieldMetadata{Core::GridStaggering::Surface, "kg m-2 s-1", long_name});
        }
    };
    add_2d_field("precip_liq_surf_mass", "time-averaged surface liquid precipitation flux");
    add_2d_field("precip_liq_surf_flux", "surface liquid precipitation flux");
    // always zero
    add_2d_field("precip_ice_surf_mass", "time-averaged surface ice precipitation flux");
    add_2d_field("precip_ice_surf_flux", "surface ice precipitation flux");
}

VVM_CLEO_Interface::~VVM_CLEO_Interface() = default;

void
VVM_CLEO_Interface::initialize(Core::State& state) {
    impl_->initialize(state);
}

void
VVM_CLEO_Interface::run(Core::State& state, Real dt) {
    impl_->run(state);
}

void
VVM_CLEO_Interface::finalize() {
    impl_->finalize();
}

}
