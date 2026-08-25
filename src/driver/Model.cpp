#include "Model.hpp"
#include "utils/ProcessScheduling.hpp"
#include "utils/Timer.hpp"

namespace VVM {
namespace Driver {

Model::Model(const Utils::ConfigurationManager& config,
             Core::Parameters& params,
             const Core::Grid& grid,
             Core::State& state,
             Core::HaloExchanger& halo_exchanger)
    : config_(config),
      params_(params),
      grid_(grid),
      state_(state),
      halo_exchanger_(halo_exchanger), bc_manager_(grid)
{
    std::string x_bc = config.get_value<std::string>("grid.boundary_condition.x", "periodic");
    std::string y_bc = config.get_value<std::string>("grid.boundary_condition.y", "periodic");
    bc_manager_.initialize_bc_types(x_bc, y_bc);
    VVM::Real dt_s = params_.get_value_host(params_.dt);

    std::string mode = config_.get_value<std::string>("simulation.idealized_test", "none");
    std::vector<std::string> no_solver_mode = {"advection_u", "advection_v", "advection_w", "stretching", "twisting"};
    auto it = std::find(no_solver_mode.begin(), no_solver_mode.end(), mode);
    if (it != no_solver_mode.end()) {
        wind_solver_ = false;
    }

    dycore_ = std::make_unique<Dynamics::DynamicalCore>(config_, grid_, params_, state_, halo_exchanger_, bc_manager_);
    if (config_.get_value<bool>("physics.p3.enable_p3", false)) {
        microphysics_ = std::make_unique<Physics::VVM_P3_Interface>(config_, grid_, params_, halo_exchanger_, state_);
    }

    if (config_.get_value<bool>("physics.turbulence.enable_turbulence", false)) {
        turbulence_ = std::make_unique<Physics::TurbulenceProcess>(config_, grid_, params_, halo_exchanger_, state_);
    }

    if (config_.get_value<bool>("physics.rrtmgp.enable_rrtmgp", false)) {
        radiation_ = std::make_unique<Physics::RRTMGP::RRTMGPRadiation>(config_, grid_, params_, state_);

        VVM::Real rad_freq_s = config_.get_value<VVM::Real>("physics.rrtmgp.rad_frequency_s", 1.0);

        rad_freq_in_steps_ = Utils::interval_steps_from_frequency(rad_freq_s, dt_s, "RRTMGP radiation");
    }

    if (config_.get_value<bool>("dynamics.forcings.sponge_layer.enable", false)) {
        sponge_layer_ = std::make_unique<Dynamics::SpongeLayer>(config_, grid_, params_, halo_exchanger_, state_);
    }

    if (config_.get_value<bool>("dynamics.forcings.lateral_boundary_nudging.enable", false)) {
        lateral_boundary_nudging_ = std::make_unique<Dynamics::LateralBoundaryNudging>(config_, grid_, params_, state_);
    }

    uvtau_ = config.get_value<VVM::Real>("dynamics.forcings.areamn.uvtau", 0.0);
    if (config_.get_value<bool>("dynamics.forcings.areamn.enable", false)) {
        area_mean_nudging_ = std::make_unique<Dynamics::AreaMeanNudging>(config_, grid_, params_);
    }

    if (config_.get_value<bool>("dynamics.forcings.random_perturbation.enable", false)) {
        random_forcing_ = std::make_unique<Dynamics::RandomForcing>(config_, grid_, params_);
    }
    if (!state_.get_tracer_source_targets().empty()) {
        tracer_source_ = std::make_unique<Dynamics::TracerSource>(
            grid_, state_);
    }

    dynamics_vars_ = {"xi", "eta", "zeta"};
    thermodynamics_vars_ = {"th", "qv"};
    if (config.get_value<bool>("physics.p3.enable_p3", false)) {
        thermodynamics_vars_.insert(thermodynamics_vars_.end(), {"qc", "qr", "qi", "nc", "nr", "ni", "bm", "qm"});
    }
    if (turbulence_) {
        thermodynamics_vars_ = turbulence_->get_thermodynamics_vars();
    }

    sfc_thermodynamics_vars_ = {"th", "qv"};
    sfc_dynamics_vars_ = {"xi", "eta"};
    enable_surface_process_ = config.get_value<bool>("physics.surface_process.enable", false);
    std::string land_scheme  = config.get_value<std::string>("physics.surface_process.land_scheme", "none");
    std::string ocean_scheme = config.get_value<std::string>("physics.surface_process.ocean_scheme", "none");
    if (enable_surface_process_) {
        surface_ = std::make_unique<Physics::SurfaceProcess>(config_, grid_, params_, halo_exchanger_, state_);

        if (land_scheme == "noahlsm") {
            land_ = std::make_unique<Physics::LandProcess>(config_, grid_, params_, halo_exchanger_, state_, ocean_scheme);
        }

        surface_process_s_ = config_.get_value<VVM::Real>("physics.surface_process.frequency_s", 1);

        surface_process_steps_ = Utils::interval_steps_from_frequency(surface_process_s_, dt_s, "surface process");
    }
}

void Model::init() {
    int rank = grid_.get_mpi_rank();
    if (rank == 0) std::cout << "\n=== Initializing VVM Model ===" << std::endl;

    if (rank == 0) std::cout << "Loading Initial Conditions..." << std::endl;
    Core::Initializer initializer(config_, grid_, params_, state_, halo_exchanger_);
    initializer.initialize_state();

    if (microphysics_) microphysics_->initialize(state_);
    if (turbulence_) turbulence_->initialize(state_);
    if (radiation_) radiation_->initialize(state_);
    if (sponge_layer_) sponge_layer_->initialize(state_);
    if (lateral_boundary_nudging_) lateral_boundary_nudging_->initialize(state_);
    if (area_mean_nudging_) area_mean_nudging_->initialize(state_);
    if (surface_) surface_->initialize(state_);
    if (land_) land_->init();
    if (random_forcing_) random_forcing_->initialize(state_);
    
    if (rank == 0) std::cout << "=== Model Initialization Complete ===\n" << std::endl;

    int nz = grid_.get_local_total_points_z();
    int ny = grid_.get_local_total_points_y();
    int nx = grid_.get_local_total_points_x();
    int h = grid_.get_halo_cells();
    if (!state_.has_field("th_perturb")) state_.add_field<3>("th_perturb", {nz, ny, nx});

    if (area_mean_nudging_ && uvtau_ == 0.0) {
        predict_uvtopmn_ = false;
    }

    if (config_.get_value<bool>(
            "initial_conditions.diagnose_wind_from_vorticity", false)) {
        dycore_->compute_wind_fields();
    }
    dycore_->compute_diagnostic_fields();
    if (config_.get_value<bool>("restart.enable", false)) {
        dycore_->initialize_restart_history();
    }
}

void Model::ensure_field_cache() {
    if (field_cache_ready_) return;

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
                if (var_name == "zeta") target.fe_2d = &state_.get_field<2>(fe_name);
                else target.fe_3d = &state_.get_field<3>(fe_name);
            }
            targets.push_back(std::move(target));
        }
        return targets;
    };
    auto fields_of = [](const std::vector<FeTarget>& targets) {
        std::vector<Core::Field<3>*> fields;
        fields.reserve(targets.size());
        for (const auto& target : targets) fields.push_back(target.field);
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

void Model::run_step(VVM::Real dt) {
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

    // P3 Microphysics based on (t+1) thermodynamics variables
    if (microphysics_) {
        VVM::Utils::Timer timer("microphysics");
        microphysics_->run(state_, dt);
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
        const bool is_compute_step = Utils::is_legacy_surface_compute_step(current_step, surface_process_steps_);
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
            if (!turbulence_) target.fe_3d->set_to_zero();
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
            VVM::Dynamics::TimeIntegrator::apply_forward_update(*target.field, target.name, grid_, dt, *target.fe_3d);
        }
    }

    // Apply sponge layer
    if (sponge_layer_) {
        VVM::Utils::Timer timer("sponge_layer");
        for (const auto& target : sponge_thermo_targets_) {
            target.fe_3d->set_to_zero();
            sponge_layer_->calculate_tendencies(state_, target.name, *target.fe_3d);

            VVM::Dynamics::TimeIntegrator::apply_forward_update(*target.field, target.name, grid_, dt, *target.fe_3d);
        }
    }
    
    // Apply lateral boundary nudge
    if (lateral_boundary_nudging_) {
        VVM::Utils::Timer timer("lateral_boundary_nudging");
        for (const auto& target : lateral_nudging_targets_) {
            target.fe_3d->set_to_zero();
            lateral_boundary_nudging_->calculate_tendencies(state_, target.name, *target.fe_3d);
            VVM::Dynamics::TimeIntegrator::apply_forward_update(*target.field, target.name, grid_, dt, *target.fe_3d);
        }
    }

    if (turbulence_ || sponge_layer_ || enable_surface_process_ || lateral_boundary_nudging_) {
        VVM::Utils::Timer timer("halo_exchange");
        halo_exchanger_.exchange_multiple_halos(thermo_boundary_fields_);
        for (const auto& target : thermo_boundary_targets_) {
            if (target.zero_gradient_top) {
                 bc_manager_.apply_zero_gradient(*target.field);
            }
            else {
                bc_manager_.apply_zero_gradient_bottom_zero_top(*target.field);
            }
        }
    }

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
            if (!turbulence_) target.fe_3d->set_to_zero();
            surface_->calculate_tendencies(state_, target.name, *target.fe_3d);
        }
    }

    if (turbulence_ || enable_surface_process_) {
        VVM::Utils::Timer timer("time_integrator_vorticity");
        for (const auto& target : integrate_dynamics_targets_) {
            if (target.fe_2d != nullptr) {
                VVM::Dynamics::TimeIntegrator::apply_forward_update(*target.field, target.name, grid_, dt, *target.fe_2d);
            } 
            else {
                VVM::Dynamics::TimeIntegrator::apply_forward_update(*target.field, target.name, grid_, dt, *target.fe_3d);
            }
        }
    }

    if (sponge_layer_) {
        VVM::Utils::Timer timer("sponge_layer");
        for (const auto& target : sponge_dynamics_targets_) {
            if (target.fe_2d != nullptr) {
                target.fe_2d->set_to_zero();
                sponge_layer_->calculate_tendencies(state_, target.name, *target.fe_2d);
                VVM::Dynamics::TimeIntegrator::apply_forward_update(*target.field, target.name, grid_, dt, *target.fe_2d);
            }
            else {
                target.fe_3d->set_to_zero();
                sponge_layer_->calculate_tendencies(state_, target.name, *target.fe_3d);
                VVM::Dynamics::TimeIntegrator::apply_forward_update(*target.field, target.name, grid_, dt, *target.fe_3d);
            }
        }
    }

    if (area_mean_nudging_) {
        VVM::Utils::Timer timer("area_mean_nudging");
        area_mean_nudging_->apply_vorticity(state_, dt);
    }

    if (turbulence_ || sponge_layer_ || enable_surface_process_ || area_mean_nudging_) {
        VVM::Utils::Timer timer("halo_exchange");
        halo_exchanger_.exchange_multiple_halos(dynamics_boundary_fields_);
        for (const auto& target : dynamics_boundary_targets_) {
            bc_manager_.apply_vorticity_bc(*target.field);
        }
        dycore_->compute_zeta_vertical_structure(state_);
    }

    if (wind_solver_) {
        VVM::Utils::Timer timer("dynamics_wind_total");
        if (predict_uvtopmn_) dycore_->compute_uvtopmn();
        if (area_mean_nudging_) area_mean_nudging_->apply_uvtopmn(state_, dt);
        dycore_->compute_wind_fields();
    }
    {
        VVM::Utils::Timer timer("dynamics_diagnostics");
        dycore_->compute_diagnostic_fields();
    }
}

void Model::finalize() {
    if (microphysics_) microphysics_->finalize();
    if (radiation_) radiation_->finalize();
    if (land_) land_->finalize();
}

}
}
