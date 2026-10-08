#include "Model.hpp"
#include "utils/ProcessScheduling.hpp"
#include "utils/Timer.hpp"
#include "core/RegularLatLonModelConfiguration.hpp"
#include <cmath>
#include <sstream>

namespace VVM {
namespace Driver {

Model::Model(const Utils::ConfigurationManager& config,
    Core::Parameters& params,
    const Core::Grid& grid,
    Core::State& state,
    Core::HaloExchanger& halo_exchanger)
    : config_(config), params_(params), grid_(grid), state_(state), halo_exchanger_(halo_exchanger),
      bc_manager_(grid, Core::is_rll_idealized(config)) {
    VVM::Real dt_s = params_.get_value_host(params_.dt);

    std::string mode = config_.get_value<std::string>("simulation.idealized_test", "none");
    std::vector<std::string> no_solver_mode = {"advection_u",
        "advection_v",
        "advection_w",
        "stretching",
        "twisting"};
    auto it = std::find(no_solver_mode.begin(), no_solver_mode.end(), mode);
    if (it != no_solver_mode.end()) {
        wind_solver_ = false;
    }

    dycore_ = std::make_unique<Dynamics::DynamicalCore>(config_,
        grid_,
        params_,
        state_,
        halo_exchanger_,
        bc_manager_);
    if (config_.get_value<bool>("physics.p3.enable_p3", false)) {
        microphysics_ = std::make_unique<Physics::VVM_P3_Interface>(config_,
            grid_,
            params_,
            halo_exchanger_,
            state_);
    }

    if (config_.get_value<bool>("physics.turbulence.enable_turbulence", false)) {
        turbulence_ = std::make_unique<Physics::TurbulenceProcess>(config_,
            grid_,
            params_,
            halo_exchanger_,
            state_);
    }

    if (config_.get_value<bool>("physics.rrtmgp.enable_rrtmgp", false)) {
        radiation_ =
            std::make_unique<Physics::RRTMGP::RRTMGPRadiation>(config_, grid_, params_, state_);

        VVM::Real rad_freq_s = config_.get_value<VVM::Real>("physics.rrtmgp.rad_frequency_s", 1.0);

        rad_freq_in_steps_ =
            Utils::interval_steps_from_frequency(rad_freq_s, dt_s, "RRTMGP radiation");
    }

    if (config_.get_value<bool>("dynamics.forcings.sponge_layer.enable", false)) {
        sponge_layer_ = std::make_unique<Dynamics::SpongeLayer>(config_,
            grid_,
            params_,
            halo_exchanger_,
            state_);
    }

    if (config_.get_value<bool>("dynamics.forcings.lateral_boundary_nudging.enable", false)) {
        lateral_boundary_nudging_ =
            std::make_unique<Dynamics::LateralBoundaryNudging>(config_, grid_, params_, state_);
    }

    uvtau_ = config.get_value<VVM::Real>("dynamics.forcings.areamn.uvtau", 0.0);
    if (config_.get_value<bool>("dynamics.forcings.areamn.enable", false)) {
        area_mean_nudging_ = std::make_unique<Dynamics::AreaMeanNudging>(config_, grid_, params_);
    }

    if (config_.get_value<bool>("dynamics.forcings.random_perturbation.enable", false)) {
        random_forcing_ = std::make_unique<Dynamics::RandomForcing>(config_, grid_, params_);
    }
    if (!state_.get_tracer_source_targets().empty()) {
        tracer_source_ = std::make_unique<Dynamics::TracerSource>(grid_, state_);
    }

    dynamics_vars_ = {"xi", "eta", "zeta"};
    thermodynamics_vars_ = {"th", "qv"};
    if (config.get_value<bool>("physics.p3.enable_p3", false)) {
        thermodynamics_vars_.insert(thermodynamics_vars_.end(),
            {"qc", "qr", "qi", "nc", "nr", "ni", "bm", "qm"});
    }
    if (turbulence_) {
        thermodynamics_vars_ = turbulence_->get_thermodynamics_vars();
    }

    sfc_thermodynamics_vars_ = {"th", "qv"};
    sfc_dynamics_vars_ = {"xi", "eta"};
    enable_surface_process_ = config.get_value<bool>("physics.surface_process.enable", false);
    std::string land_scheme =
        config.get_value<std::string>("physics.surface_process.land_scheme", "none");
    std::string ocean_scheme =
        config.get_value<std::string>("physics.surface_process.ocean_scheme", "none");
    if (enable_surface_process_) {
        surface_ = std::make_unique<Physics::SurfaceProcess>(config_,
            grid_,
            params_,
            halo_exchanger_,
            state_);

        if (land_scheme == "noahlsm") {
            land_ = std::make_unique<Physics::LandProcess>(config_,
                grid_,
                params_,
                halo_exchanger_,
                state_,
                ocean_scheme);
        }

        surface_process_s_ = config_.get_value<VVM::Real>("physics.surface_process.frequency_s", 1);

        surface_process_steps_ =
            Utils::interval_steps_from_frequency(surface_process_s_, dt_s, "surface process");
    }
}

void
Model::init() {
    int rank = grid_.get_mpi_rank();
    if (rank == 0) {
        std::cout << "\n=== Initializing VVM Model ===" << std::endl;
    }

    if (rank == 0) {
        std::cout << "Loading Initial Conditions..." << std::endl;
    }
    Core::Initializer initializer(config_, grid_, params_, state_, halo_exchanger_);
    initializer.initialize_state();

    if (microphysics_) {
        microphysics_->initialize(state_);
    }
    if (turbulence_) {
        turbulence_->initialize(state_);
    }
    if (radiation_) {
        radiation_->initialize(state_);
    }
    if (sponge_layer_) {
        sponge_layer_->initialize(state_);
    }
    if (lateral_boundary_nudging_) {
        lateral_boundary_nudging_->initialize(state_);
    }
    if (area_mean_nudging_) {
        area_mean_nudging_->initialize(state_);
    }
    if (surface_) {
        surface_->initialize(state_);
    }
    if (land_) {
        land_->init();
    }
    if (random_forcing_) {
        random_forcing_->initialize(state_);
    }

    const bool restart_enabled = config_.get_value<bool>("restart.enable", false);
    if (restart_enabled) {
        ensure_field_cache();
        // Thermodynamic physical state
        halo_exchanger_.exchange_multiple_halos(thermo_boundary_fields_);

        for (const auto& target : thermo_boundary_targets_) {
            if (grid_.geometry().kind() == Core::Geometry::GeometryKind::RegularLatLon) {
                bc_manager_.apply_horizontal_bcs(*target.field);
            }

            if (target.zero_gradient_top) {
                bc_manager_.apply_zero_gradient(*target.field);
            }
            else {
                bc_manager_.apply_zero_gradient_bottom_zero_top(*target.field);
            }
        }

        // ---------------------------------------------------------
        // Physical vorticity
        // ---------------------------------------------------------
        halo_exchanger_.exchange_multiple_halos(dynamics_boundary_fields_);
        for (const auto& target : dynamics_boundary_targets_) {
            if (grid_.geometry().kind() == Core::Geometry::GeometryKind::RegularLatLon) {
                bc_manager_.apply_horizontal_bcs(*target.field);
            }
            if (target.name == "zeta") {
                // Both wind solvers use physical top zeta as the horizontal
                // elliptic right-hand side. Preserve it across a restart.
                bc_manager_.apply_zero_gradient(*target.field);
            }
            else {
                bc_manager_.apply_vorticity_bc(*target.field);
            }
        }

        // qp is diagnostic and is not part of the checkpoint.
        if (microphysics_) {
            microphysics_->refresh_total_condensate(state_);
        }
    }

    if (rank == 0) {
        std::cout << "=== Model Initialization Complete ===\n" << std::endl;
    }

    int nz = grid_.get_local_total_points_z();
    int ny = grid_.get_local_total_points_y();
    int nx = grid_.get_local_total_points_x();
    int h = grid_.get_halo_cells();
    if (!state_.has_field("th_perturb")) {
        state_.add_field<3>("th_perturb", {nz, ny, nx});
    }
    if (Core::is_rll_bubble(config_)) {
        state_.add_field<3>("qv_perturb", {nz,ny,nx},
            {Core::GridStaggering::Centered,"kg kg-1","water vapor anomaly from reference profile"});
    }

    if (area_mean_nudging_ && uvtau_ == 0.0) {
        predict_uvtopmn_ = false;
    }

    dycore_->sync_contravariant_vorticity_from_physical();
    if (restart_enabled) {
        dycore_->update_contravariant_wind_shadow_state();
    }
    else if (config_.get_value<bool>("initial_conditions.diagnose_wind_from_vorticity", false) ||
             Core::is_rll_idealized(config_)) {
        dycore_->compute_wind_fields();
    }
    else {
        dycore_->update_contravariant_wind_shadow_state();
    }
    dycore_->compute_diagnostic_fields();
    update_thermodynamic_perturbations();

    if (restart_enabled) {
        dycore_->initialize_restart_history();
    }
}

void
Model::ensure_field_cache() {
    if (field_cache_ready_) {
        return;
    }

    auto build = [&](const std::vector<std::string>& var_names, bool with_fe_tendency) {
        std::vector<FeTarget> targets;
        targets.reserve(var_names.size());
        for (const auto& var_name : var_names) {
            FeTarget target;
            target.name = var_name;
            target.field = &state_.get_field<3>(var_name);
            target.zero_gradient_top = (var_name == "th" || var_name == "qv");
            if (with_fe_tendency) {
                const std::string fe_name = "fe_tendency_" + var_name;
                if (var_name == "zeta") {
                    target.fe_2d = &state_.get_field<2>(fe_name);
                }
                else {
                    target.fe_3d = &state_.get_field<3>(fe_name);
                }
            }
            targets.push_back(std::move(target));
        }
        return targets;
    };
    auto fields_of = [](const std::vector<FeTarget>& targets) {
        std::vector<Core::Field<3>*> fields;
        fields.reserve(targets.size());
        for (const auto& target : targets) {
            fields.push_back(target.field);
        }
        return fields;
    };

    if (tracer_source_) {
        tracer_source_targets_ = build(tracer_source_->get_target_vars(), false);
        tracer_source_fields_ = fields_of(tracer_source_targets_);
    }
    if (turbulence_) {
        turbulence_thermo_targets_ = build(turbulence_->get_thermodynamics_vars(), true);
        turbulence_dynamics_targets_ = build(turbulence_->get_dynamics_vars(), true);
    }
    if (enable_surface_process_) {
        surface_thermo_targets_ = build(sfc_thermodynamics_vars_, true);
        surface_dynamics_targets_ = build(sfc_dynamics_vars_, true);
    }
    if (turbulence_ || enable_surface_process_) {
        integrate_thermo_targets_ =
            turbulence_ ? turbulence_thermo_targets_ : surface_thermo_targets_;
        integrate_dynamics_targets_ =
            turbulence_ ? turbulence_dynamics_targets_ : surface_dynamics_targets_;
    }
    if (sponge_layer_) {
        sponge_thermo_targets_ = build(sponge_layer_->get_thermodynamics_vars(), true);
        sponge_dynamics_targets_ = build(sponge_layer_->get_dynamics_vars(), true);
    }
    if (lateral_boundary_nudging_) {
        lateral_nudging_targets_ = build(lateral_boundary_nudging_->get_target_vars(), true);
    }

    thermo_boundary_targets_ = build(thermodynamics_vars_, false);
    thermo_boundary_fields_ = fields_of(thermo_boundary_targets_);
    dynamics_boundary_targets_ = build(dynamics_vars_, false);
    dynamics_boundary_fields_ = fields_of(dynamics_boundary_targets_);

    field_cache_ready_ = true;
}

void Model::enable_transport_flux_capture() {
    if (!captured_fluxes_.empty()) return;
    for (const std::string name : {"th", "qv"}) {
        const auto key="dynamics.prognostic_variables."+name+".tendency_terms.advection.";
        if (config_.get_value<std::string>(key+"spatial_scheme","")!="MUSCL" ||
            config_.get_value<std::string>(key+"temporal_scheme","")!="SSPRK2")
            throw std::runtime_error("Coupling flux diagnostics require MUSCL/SSPRK2.");
        for (const std::string axis : {"x","y","z"}) {
            const auto label="coupling_flux_"+axis+"_"+name;
            state_.add_field<3>(label,{grid_.get_local_total_points_z(),grid_.get_local_total_points_y(),grid_.get_local_total_points_x()});
            captured_fluxes_.push_back(&state_.get_field<3>(label));
        }
    }
}

void Model::prepare_projected_state() {
    ensure_field_cache();
    halo_exchanger_.exchange_multiple_halos(thermo_boundary_fields_);
    for (const auto& target : thermo_boundary_targets_) {
        bc_manager_.apply_horizontal_bcs(*target.field);
        if(target.zero_gradient_top) bc_manager_.apply_zero_gradient(*target.field);
        else bc_manager_.apply_zero_gradient_bottom_zero_top(*target.field);
    }
    for (const char* name : {"u","v","w"}) {
        auto& field=state_.get_field<3>(name);
        halo_exchanger_.exchange_halos(field);
        bc_manager_.apply_horizontal_bcs(field);
    }
    dycore_->update_contravariant_wind_shadow_state();
    if (microphysics_) microphysics_->refresh_total_condensate(state_);
}

void
Model::run_step(VVM::Real dt, const ThermodynamicFeedback* feedback, bool thermodynamics_only) {
    for (auto* flux : captured_fluxes_) flux->set_to_zero();
    ensure_field_cache();

    size_t current_step = state_.get_step();
    VVM::Real current_time = state_.get_time();

    if (lateral_boundary_nudging_) {
        VVM::Utils::Timer timer("lateral_boundary_nudging");
        lateral_boundary_nudging_->update_large_scale_forcing(state_, current_time);
    }

    if (area_mean_nudging_) {
        VVM::Utils::Timer timer("area_mean_nudging");
        area_mean_nudging_->update_forcing_target(state_, current_time);
    }

    // Caculate tendencies of thermodynamics variables
    {
        VVM::Utils::Timer timer("dynamics_thermo");
        dycore_->calculate_thermo_tendencies();
    }

    // Calculate radiation based on t
    if (radiation_) {
        VVM::Utils::Timer timer("radiation");

        // Update net heating used for calculating th tendency
        if (Utils::is_process_step(current_step, rad_freq_in_steps_)) {
            radiation_->run(state_, dt);
        }

        // Update forward th tendency
        // The effects of radiation is updated in update_thermodynamics
        radiation_->calculate_tendencies(state_);
    }

    // Update thermodynamics variables using tendencies above
    {
        VVM::Utils::Timer timer("dynamics_thermo");
        dycore_->update_thermodynamics(dt);
    }

    if (tracer_source_) {
        VVM::Utils::Timer timer("tracer_source");
        tracer_source_->apply(state_, dt);
        halo_exchanger_.exchange_multiple_halos(tracer_source_fields_);
        for (const auto& target : tracer_source_targets_) {
            bc_manager_.apply_horizontal_bcs(*target.field);
            bc_manager_.apply_zero_gradient_bottom_zero_top(*target.field);
        }
    }

    if (random_forcing_) {
        VVM::Utils::Timer timer("random_perturbation");
        random_forcing_->apply(state_);
    }

    if (feedback && feedback->transport && feedback->strength != real(0.)) {
        feedback->transport->apply(state_, dt, feedback->strength);
        halo_exchanger_.exchange_multiple_halos(thermo_boundary_fields_);
        for (const auto& target : thermo_boundary_targets_) {
            bc_manager_.apply_horizontal_bcs(*target.field);
            if(target.zero_gradient_top) bc_manager_.apply_zero_gradient(*target.field);
            else bc_manager_.apply_zero_gradient_bottom_zero_top(*target.field);
        }
    }

    begin_physics_capture(false);

    // P3 Microphysics based on (t+1) thermodynamics variables
    if (microphysics_) {
        VVM::Utils::Timer timer("microphysics");
        if (capture_physics_feedback_) {
            Kokkos::deep_copy(Kokkos::DefaultExecutionSpace(),
                pre_p3_theta_->get_mutable_device_data(), state_.get_field<3>("th").get_device_data());
            Kokkos::deep_copy(Kokkos::DefaultExecutionSpace(),
                pre_p3_qv_->get_mutable_device_data(), state_.get_field<3>("qv").get_device_data());
        }
        if(pre_p3_water_) {
            const auto q=state_.get_field<3>("qv").get_device_data(),c=state_.get_field<3>("qc").get_device_data(),
                r=state_.get_field<3>("qr").get_device_data(),ice=state_.get_field<3>("qi").get_device_data();
            const auto out=pre_p3_water_->get_mutable_device_data();
            Kokkos::parallel_for("SnapshotP3TotalWater",Kokkos::RangePolicy<>(0,out.size()),
                KOKKOS_LAMBDA(size_t n){out.data()[n]=q.data()[n]+c.data()[n]+r.data()[n]+ice.data()[n];});
        }
        microphysics_->run(state_, dt);
        if(pre_p3_water_) {
            const auto q=state_.get_field<3>("qv").get_device_data(),c=state_.get_field<3>("qc").get_device_data(),
                r=state_.get_field<3>("qr").get_device_data(),ice=state_.get_field<3>("qi").get_device_data(),
                before=pre_p3_water_->get_device_data();
            const auto out=state_.get_field<3>("physics_tendency_qt").get_mutable_device_data();
            Kokkos::parallel_for("CaptureP3TotalWater",Kokkos::RangePolicy<>(0,out.size()),
                KOKKOS_LAMBDA(size_t n){out.data()[n]=(q.data()[n]+c.data()[n]+r.data()[n]+ice.data()[n]-before.data()[n])/dt;});
        }
        if (capture_physics_feedback_) {
            const int h = grid_.get_halo_cells();
            const int nz = grid_.get_local_total_points_z(), ny = grid_.get_local_total_points_y(),
                      nx = grid_.get_local_total_points_x();
            const auto before_th = pre_p3_theta_->get_device_data();
            const auto before_q = pre_p3_qv_->get_device_data();
            const auto th = state_.get_field<3>("th").get_device_data();
            const auto q = state_.get_field<3>("qv").get_device_data();
            const auto active = state_.get_field<3>("ITYPEW").get_device_data();
            const auto ft = state_.get_field<3>("physics_tendency_th").get_mutable_device_data();
            const auto fq = state_.get_field<3>("physics_tendency_qv").get_mutable_device_data();
            Kokkos::parallel_for("CaptureP3Feedback",
                Kokkos::MDRangePolicy<Kokkos::Rank<3>>({h, h, h}, {nz-h, ny-h, nx-h}),
                KOKKOS_LAMBDA(int k, int j, int i) {
                    ft(k,j,i) = active(k,j,i) == 1 ? (th(k,j,i)-before_th(k,j,i))/dt : real(0.);
                    fq(k,j,i) = active(k,j,i) == 1 ? (q(k,j,i)-before_q(k,j,i))/dt : real(0.);
                });
        }
    }

    // Match the fine model's P3 split stage. Buoyancy and wind recovery below
    // see this update during the same step; AB2 transport histories do not.
    if (feedback && !feedback->transport) {
        apply_thermodynamic_feedback(dt, *feedback);
    }

    // Turbulence diffusion on thermodynamics variables
    if (turbulence_) {
        VVM::Utils::Timer timer("turbulence");
        turbulence_->compute_coefficients(state_, dt);
        for (const auto& target : turbulence_thermo_targets_) {
            target.fe_3d->set_to_zero();
            turbulence_->calculate_tendencies(state_, target.name, *target.fe_3d);
        }
    }

    // Surface process (sea/land/ice)
    if (enable_surface_process_) {
        // FIXME: (surface-scheduling): this is the v1.0.0 schedule, kept on purpose
        // so results stay reproducible for the paper under review. It is wrong:
        // it fires at 1, N+1, 2N+1, ... and skips step 0 for N of 24 and 120, so
        // the first step runs with zero surface fluxes -- calculate_tendencies()
        // below reads sfc_flux_*, and compute_coefficients() is their only writer.
        //
        // The fix is to swap this one line for
        //     Utils::is_process_step(current_step, surface_process_steps_)
        // which gives 0, N, 2N, ... and matches radiation above. It changes every
        // surface case's answers, so regenerate tests/references/ in the same
        // commit (check_output.py --update). See utils/ProcessScheduling.hpp.
        // So do this in the furture:
        // const bool is_compute_step = Utils::is_process_step(current_step, surface_process_steps_);
        const bool is_compute_step =
            Utils::is_legacy_surface_compute_step(current_step, surface_process_steps_);
        if (is_compute_step) {
            // NOTE: Even the configuration specified tco_ocean model which is not from surface_, surface_ stil calculates surface friction for xi and eta.
            // note that the dt for land module should be calling time step because the soil T needs to be updated
            if (land_) {
                VVM::Utils::Timer timer("land");
                land_->run(surface_process_s_);
            }

            {
                VVM::Utils::Timer timer("surface");
                surface_->compute_coefficients(state_);
            }
        }

        for (const auto& target : surface_thermo_targets_) {
            if (!turbulence_) {
                target.fe_3d->set_to_zero();
            }
            if (surface_) {
                VVM::Utils::Timer timer("surface");
                surface_->calculate_tendencies(state_, target.name, *target.fe_3d);
            }
            if (land_) {
                VVM::Utils::Timer timer("land");
                land_->calculate_tendencies(target.name, *target.fe_3d);
            }
        }
    }

    if (turbulence_ || enable_surface_process_) {
        VVM::Utils::Timer timer("time_integrator_thermo");
        for (const auto& target : integrate_thermo_targets_) {
            VVM::Dynamics::TimeIntegrator::apply_forward_update(*target.field,
                target.name,
                grid_,
                dt,
                *target.fe_3d);
        }
    }

    finish_physics_capture(false, dt);

    // Apply sponge layer
    if (sponge_layer_) {
        VVM::Utils::Timer timer("sponge_layer");
        for (const auto& target : sponge_thermo_targets_) {
            target.fe_3d->set_to_zero();
            sponge_layer_->calculate_tendencies(state_, target.name, *target.fe_3d);

            VVM::Dynamics::TimeIntegrator::apply_forward_update(*target.field,
                target.name,
                grid_,
                dt,
                *target.fe_3d);
        }
    }

    // Apply lateral boundary nudge
    if (lateral_boundary_nudging_) {
        VVM::Utils::Timer timer("lateral_boundary_nudging");
        for (const auto& target : lateral_nudging_targets_) {
            target.fe_3d->set_to_zero();
            lateral_boundary_nudging_->calculate_tendencies(state_, target.name, *target.fe_3d);
            VVM::Dynamics::TimeIntegrator::apply_forward_update(*target.field,
                target.name,
                grid_,
                dt,
                *target.fe_3d);
        }
    }

    if (turbulence_ || sponge_layer_ || enable_surface_process_ || lateral_boundary_nudging_) {
        VVM::Utils::Timer timer("halo_exchange");
        halo_exchanger_.exchange_multiple_halos(thermo_boundary_fields_);
        for (const auto& target : thermo_boundary_targets_) {
            if (grid_.geometry().kind() == Core::Geometry::GeometryKind::RegularLatLon) {
                bc_manager_.apply_horizontal_bcs(*target.field);
            }
            if (target.zero_gradient_top) {
                bc_manager_.apply_zero_gradient(*target.field);
            }
            else {
                bc_manager_.apply_zero_gradient_bottom_zero_top(*target.field);
            }
        }
    }

    if (microphysics_ && grid_.geometry().kind() != Core::Geometry::GeometryKind::Cartesian) {
        // Buoyancy must see condensate after diffusion/sponge, not P3's earlier snapshot.
        microphysics_->refresh_total_condensate(state_);
    }

    if (thermodynamics_only) { update_thermodynamic_perturbations(); return; }

    // Calculate buoyancy based on thermodynamics variables at t+1
    // dycore_->update_buoyancy_term(state_);
    // This is included in calculate vorticity tendencies

    {
        VVM::Utils::Timer timer("dynamics_vorticity");
        // Caulcate vorticity tendencies using variables at t
        dycore_->calculate_vorticity_tendencies();
        // Update vorticity to t+1
        dycore_->update_vorticity(dt);
    }

    begin_physics_capture(true);

    // Vorticity diffusion
    if (turbulence_) {
        VVM::Utils::Timer timer("turbulence");
        for (const auto& target : turbulence_dynamics_targets_) {
            if (target.fe_2d != nullptr) {
                target.fe_2d->set_to_zero();
                turbulence_->calculate_tendencies(state_, target.name, *target.fe_2d);
            }
            else {
                target.fe_3d->set_to_zero();
                turbulence_->calculate_tendencies(state_, target.name, *target.fe_3d);
            }
        }
    }

    if (enable_surface_process_) {
        VVM::Utils::Timer timer("surface");
        for (const auto& target : surface_dynamics_targets_) {
            if (!turbulence_) {
                target.fe_3d->set_to_zero();
            }
            surface_->calculate_tendencies(state_, target.name, *target.fe_3d);
        }
    }

    if (turbulence_ || enable_surface_process_) {
        VVM::Utils::Timer timer("time_integrator_vorticity");
        for (const auto& target : integrate_dynamics_targets_) {
            if (target.fe_2d != nullptr) {
                VVM::Dynamics::TimeIntegrator::apply_forward_update(*target.field,
                    target.name,
                    grid_,
                    dt,
                    *target.fe_2d);
            }
            else {
                VVM::Dynamics::TimeIntegrator::apply_forward_update(*target.field,
                    target.name,
                    grid_,
                    dt,
                    *target.fe_3d);
            }
        }
    }

    finish_physics_capture(true, dt);
    if (feedback && !feedback->dynamics.empty()) apply_dynamic_feedback(dt, *feedback);

    if (sponge_layer_) {
        VVM::Utils::Timer timer("sponge_layer");
        for (const auto& target : sponge_dynamics_targets_) {
            if (target.fe_2d != nullptr) {
                target.fe_2d->set_to_zero();
                sponge_layer_->calculate_tendencies(state_, target.name, *target.fe_2d);
                VVM::Dynamics::TimeIntegrator::apply_forward_update(*target.field,
                    target.name,
                    grid_,
                    dt,
                    *target.fe_2d);
            }
            else {
                target.fe_3d->set_to_zero();
                sponge_layer_->calculate_tendencies(state_, target.name, *target.fe_3d);
                VVM::Dynamics::TimeIntegrator::apply_forward_update(*target.field,
                    target.name,
                    grid_,
                    dt,
                    *target.fe_3d);
            }
        }
    }

    if (area_mean_nudging_) {
        VVM::Utils::Timer timer("area_mean_nudging");
        area_mean_nudging_->apply_vorticity(state_, dt);
    }

    if (turbulence_ || sponge_layer_ || enable_surface_process_ || area_mean_nudging_ ||
        (feedback && feedback->strength != real(0.) && !feedback->dynamics.empty())) {
        VVM::Utils::Timer timer("halo_exchange");
        halo_exchanger_.exchange_multiple_halos(dynamics_boundary_fields_);
        for (const auto& target : dynamics_boundary_targets_) {
            if (grid_.geometry().kind() == Core::Geometry::GeometryKind::RegularLatLon) {
                bc_manager_.apply_horizontal_bcs(*target.field);
            }
            if (target.name == "zeta" &&
                grid_.geometry().kind() == Core::Geometry::GeometryKind::RegularLatLon) {
                // RLL advances physical top zeta as the horizontal elliptic
                // right-hand side. Keep that value for the wind recovery.
                bc_manager_.apply_zero_gradient(*target.field);
            }
            else {
                bc_manager_.apply_vorticity_bc(*target.field);
            }
        }
        dycore_->compute_zeta_vertical_structure(state_);

        dycore_->sync_contravariant_vorticity_from_physical();
    }

    if (wind_solver_) {
        VVM::Utils::Timer timer("dynamics_wind_total");
        if (predict_uvtopmn_) {
            dycore_->compute_uvtopmn();
        }
        if (area_mean_nudging_) {
            area_mean_nudging_->apply_uvtopmn(state_, dt);
        }
        dycore_->compute_wind_fields();
    }
    {
        VVM::Utils::Timer timer("dynamics_diagnostics");
        dycore_->compute_diagnostic_fields();
    }
    update_thermodynamic_perturbations();
}

void
Model::update_thermodynamic_perturbations() {
    if (!Core::is_rll_bubble(config_)) return;
    const auto th = state_.get_field<3>("th").get_device_data();
    const auto qv = state_.get_field<3>("qv").get_device_data();
    const auto thbar = state_.get_field<1>("thbar").get_device_data();
    const auto qvbar = state_.get_field<1>("qvbar").get_device_data();
    const auto tp = state_.get_field<3>("th_perturb").get_mutable_device_data();
    const auto qp = state_.get_field<3>("qv_perturb").get_mutable_device_data();
    Kokkos::parallel_for("RLLBubbleThermodynamicPerturbations",
        Kokkos::MDRangePolicy<Kokkos::Rank<3>>({0,0,0},
            {grid_.get_local_total_points_z(),grid_.get_local_total_points_y(),grid_.get_local_total_points_x()}),
        KOKKOS_LAMBDA(int k,int j,int i) {
            tp(k,j,i) = th(k,j,i)-thbar(k);
            qp(k,j,i) = qv(k,j,i)-qvbar(k);
        });
}

void
Model::enable_physics_tendency_capture(const std::vector<std::string>& variables) {
    if (!physics_captures_.empty()) throw std::runtime_error("Physics tendency capture already enabled.");
    const std::array<int,3> dims{grid_.get_local_total_points_z(),
        grid_.get_local_total_points_y(),grid_.get_local_total_points_x()};
    std::set<std::string> seen;
    for (const auto& name : variables) {
        if (!seen.insert(name).second) throw std::runtime_error("Duplicate physics feedback variable: "+name);
        const bool dynamics = name == "xi" || name == "eta" || name == "zeta";
        if ((name != "th" && name != "qv" && !dynamics) || !state_.has_field(name))
            throw std::runtime_error("Unsupported physics feedback prognostic variable: "+name);
        auto metadata = state_.get_field<3>(name).get_metadata();
        metadata.units = dynamics ? "s-2" : name == "th" ? "K s-1" : "kg kg-1 s-1";
        metadata.long_name = "net physical-process tendency of "+name;
        state_.add_field<3>("coupling_physics_"+name,dims,metadata);
        physics_captures_.push_back({name,dynamics,
            std::make_unique<Core::Field<3>>("before_physics_"+name,dims)});
    }
}

void
Model::begin_physics_capture(bool dynamics) {
    for (auto& entry : physics_captures_) if (entry.dynamics == dynamics)
        Kokkos::deep_copy(Kokkos::DefaultExecutionSpace(), entry.before->get_mutable_device_data(),
            state_.get_field<3>(entry.variable).get_device_data());
}

void
Model::finish_physics_capture(bool dynamics, VVM::Real dt) {
    for (auto& entry : physics_captures_) if (entry.dynamics == dynamics) {
        const auto before = entry.before->get_device_data();
        const auto after = state_.get_field<3>(entry.variable).get_device_data();
        const auto tendency = state_.get_field<3>("coupling_physics_"+entry.variable).get_mutable_device_data();
        Kokkos::parallel_for("CaptureNetPhysicsIncrement",Kokkos::RangePolicy<>(0,after.size()),
            KOKKOS_LAMBDA(size_t n) { tendency.data()[n]=(after.data()[n]-before.data()[n])/dt; });
    }
}

void
Model::apply_dynamic_feedback(VVM::Real dt, const ThermodynamicFeedback& feedback) {
    if (feedback.strength == real(0.)) return;
    const int h=grid_.get_halo_cells(),nz=grid_.get_local_total_points_z(),
        ny=grid_.get_local_total_points_y(),nx=grid_.get_local_total_points_x();
    for (const auto& entry : feedback.dynamics) {
        const bool horizontal = entry.variable == "xi" || entry.variable == "eta";
        if ((!horizontal && entry.variable != "zeta") || !entry.field)
            throw std::runtime_error("Unsupported dynamic feedback variable: "+entry.variable);
        const auto source=entry.field->get_device_data();
        const auto target=state_.get_field<3>(entry.variable).get_mutable_device_data();
        if (source.extent_int(0)!=nz || source.extent_int(1)!=ny || source.extent_int(2)!=nx)
            throw std::runtime_error("Dynamic feedback dimensions do not match receiver.");
        // xi/eta are prognostic on vertical interfaces. Only top zeta is an
        // independent prognostic; its column structure is reconstructed below.
        const int first=horizontal?h:nz-h-1,last=horizontal?nz-h-1:nz-h;
        const Real strength=feedback.strength,increment=dt*strength;
        int invalid=0;
        Kokkos::parallel_reduce("CheckDynamicFeedback",Kokkos::MDRangePolicy<Kokkos::Rank<3>>(
            {first,h,h},{last,ny-h,nx-h}),KOKKOS_LAMBDA(int k,int j,int i,int& count) {
                if(!Kokkos::isfinite(target(k,j,i)+increment*source(k,j,i))) ++count;
            },invalid);
        if(invalid) throw std::runtime_error("Nonfinite dynamic coupling tendency: "+entry.variable);
        const auto applied=state_.get_field<3>("feedback_applied_"+entry.variable).get_mutable_device_data();
        Kokkos::parallel_for("ApplyDynamicFeedback",Kokkos::MDRangePolicy<Kokkos::Rank<3>>(
            {first,h,h},{last,ny-h,nx-h}),KOKKOS_LAMBDA(int k,int j,int i) {
                target(k,j,i)+=increment*source(k,j,i);
                applied(k,j,i)=strength*source(k,j,i);
            });
    }
}

void
Model::enable_physics_feedback_capture(bool water_budget) {
    if (!microphysics_) {
        throw std::runtime_error("Physics feedback capture requires P3.");
    }
    const std::array<int, 3> dims{grid_.get_local_total_points_z(),
        grid_.get_local_total_points_y(), grid_.get_local_total_points_x()};
    pre_p3_theta_ = std::make_unique<Core::Field<3>>("pre_p3_theta", dims);
    pre_p3_qv_ = std::make_unique<Core::Field<3>>("pre_p3_qv", dims);
    state_.add_field<3>("physics_tendency_th", dims,
        {Core::GridStaggering::Centered, "K s-1", "step-averaged net P3 potential temperature tendency"});
    state_.add_field<3>("physics_tendency_qv", dims,
        {Core::GridStaggering::Centered, "kg kg-1 s-1", "step-averaged net P3 water vapor tendency"});
    if(water_budget) {
        pre_p3_water_=std::make_unique<Core::Field<3>>("pre_p3_water",dims);
        state_.add_field<3>("physics_tendency_qt",dims,
            {Core::GridStaggering::Centered,"kg kg-1 s-1","net P3 total water tendency including fallout"});
    }
    capture_physics_feedback_ = true;
}

void
Model::apply_thermodynamic_feedback(VVM::Real dt, const ThermodynamicFeedback& feedback) {
    if (!std::isfinite(feedback.strength) || feedback.strength < real(0.) ||
        !std::isfinite(dt) || dt <= real(0.)) {
        throw std::runtime_error("Thermodynamic feedback requires finite nonnegative strength and positive dt.");
    }
    using View = Core::Field<3>::ViewType;
    View applied_q, unapplied_q, applied_th, unapplied_th;
    if (feedback.limit_vapor || state_.has_field("feedback_applied_qv")) {
        const std::array<int,3> dims{grid_.get_local_total_points_z(),
            grid_.get_local_total_points_y(),grid_.get_local_total_points_x()};
        for(const char* name : {"feedback_applied_qv","feedback_unapplied_qv",
                               "feedback_applied_theta","feedback_unapplied_theta"}) {
            if(!state_.has_field(name)) state_.add_field<3>(name,dims,
                {Core::GridStaggering::Centered,std::string(name).find("qv")!=std::string::npos ? "kg kg-1 s-1" : "K s-1",
                 "scaled applied or unapplied combined feedback tendency"});
            state_.get_field<3>(name).set_to_zero();
        }
        applied_q=state_.get_field<3>("feedback_applied_qv").get_mutable_device_data();
        unapplied_q=state_.get_field<3>("feedback_unapplied_qv").get_mutable_device_data();
        applied_th=state_.get_field<3>("feedback_applied_theta").get_mutable_device_data();
        unapplied_th=state_.get_field<3>("feedback_unapplied_theta").get_mutable_device_data();
    }
    // An exact bypass keeps the alpha=0 coarse solution identical to standalone.
    if (feedback.strength == real(0.) || (!feedback.theta && !feedback.qv)) return;
    if (microphysics_) throw std::runtime_error("The feedback receiver must have P3 disabled.");
    const int h = grid_.get_halo_cells();
    const int nz = grid_.get_local_total_points_z(), ny = grid_.get_local_total_points_y(),
              nx = grid_.get_local_total_points_x();
    const auto th = state_.get_field<3>("th").get_mutable_device_data();
    const auto q = state_.get_field<3>("qv").get_mutable_device_data();
    const auto active = state_.get_field<3>("ITYPEW").get_device_data();
    const auto ft = feedback.theta ? feedback.theta->get_device_data() : View{};
    const auto fq = feedback.qv ? feedback.qv->get_device_data() : View{};
    for (const auto& v : {ft, fq}) {
        if (v.data() && (v.extent_int(0) != nz || v.extent_int(1) != ny || v.extent_int(2) != nx))
            throw std::runtime_error("Thermodynamic feedback field dimensions do not match receiver.");
    }
    const auto increment = dt * feedback.strength;
    const bool limit_vapor=feedback.limit_vapor;
    int invalid = 0;
    Kokkos::parallel_reduce("CheckThermodynamicFeedback",
        Kokkos::MDRangePolicy<Kokkos::Rank<3>>({h,h,h}, {nz-h,ny-h,nx-h}),
        KOKKOS_LAMBDA(int k, int j, int i, int& errors) {
            if (active(k,j,i) != 1) return;
            const auto t = th(k,j,i) + (ft.data() ? increment*ft(k,j,i) : real(0.));
            const auto vapor = q(k,j,i) + (fq.data() ? increment*fq(k,j,i) : real(0.));
            if (!Kokkos::isfinite(t) || !Kokkos::isfinite(vapor)) ++errors;
            if(limit_vapor && q(k,j,i)<real(0.)) ++errors;
            // Negative vapor is intentionally allowed in this one-way experiment:
            // the receiver applies the prescribed tendency without clipping.
            // if (vapor < real(0.)) ++errors;
        }, invalid);
    if (invalid) {
        VVM::Real min_before, min_after;
        Kokkos::parallel_reduce("FeedbackMinimumVaporBefore",
            Kokkos::MDRangePolicy<Kokkos::Rank<3>>({h,h,h}, {nz-h,ny-h,nx-h}),
            KOKKOS_LAMBDA(int k, int j, int i, VVM::Real& minimum) {
                if (active(k,j,i) == 1 && q(k,j,i) < minimum) minimum = q(k,j,i);
            }, Kokkos::Min<VVM::Real>(min_before));
        Kokkos::parallel_reduce("FeedbackMinimumVaporAfter",
            Kokkos::MDRangePolicy<Kokkos::Rank<3>>({h,h,h}, {nz-h,ny-h,nx-h}),
            KOKKOS_LAMBDA(int k, int j, int i, VVM::Real& minimum) {
                if (active(k,j,i) != 1) return;
                const auto vapor = q(k,j,i) + (fq.data() ? increment*fq(k,j,i) : real(0.));
                if (vapor < minimum) minimum = vapor;
            }, Kokkos::Min<VVM::Real>(min_after));
        std::ostringstream message;
        message << "Coupling produced nonfinite thermodynamics; no clipping is applied. "
                << "Invalid cells=" << invalid << ", minimum qv before feedback=" << min_before
                << ", minimum qv after feedback=" << min_after << ", dt=" << dt;
        throw std::runtime_error(message.str());
    }
    Kokkos::parallel_for("ApplyThermodynamicFeedback",
        Kokkos::MDRangePolicy<Kokkos::Rank<3>>({h,h,h}, {nz-h,ny-h,nx-h}),
        KOKKOS_LAMBDA(int k, int j, int i) {
            if (active(k,j,i) != 1) return;
            const auto requested_t=ft.data() ? increment*ft(k,j,i) : real(0.);
            const auto requested_q=fq.data() ? increment*fq(k,j,i) : real(0.);
            Real factor=real(1.);
            if(limit_vapor && requested_q<real(0.) && q(k,j,i)+requested_q<real(0.))
                factor=q(k,j,i)/(-requested_q);
            const auto applied_t=factor*requested_t;
            const auto actual_q=factor<real(1.) ? -q(k,j,i) : requested_q;
            th(k,j,i)+=applied_t;
            if(factor<real(1.)) q(k,j,i)=real(0.);
            else q(k,j,i)+=actual_q;
            if(applied_q.data()) {
                applied_th(k,j,i)=applied_t/dt;unapplied_th(k,j,i)=(requested_t-applied_t)/dt;
                applied_q(k,j,i)=actual_q/dt;unapplied_q(k,j,i)=(requested_q-actual_q)/dt;
            }
        });
    std::vector<Core::Field<3>*> fields{&state_.get_field<3>("th"), &state_.get_field<3>("qv")};
    halo_exchanger_.exchange_multiple_halos(fields);
    for (auto* field : fields) {
        bc_manager_.apply_horizontal_bcs(*field);
        bc_manager_.apply_zero_gradient(*field);
    }
}

void
Model::finalize() {
    if (microphysics_) {
        microphysics_->finalize();
    }
    if (radiation_) {
        radiation_->finalize();
    }
    if (land_) {
        land_->finalize();
    }
}

}
}
