#include "DynamicalCore.hpp"
#include "numerical_methods/NumericalMethodFactory.hpp"
#include "spatial_schemes/Takacs.hpp"
#include "core/HaloExchanger.hpp"
#include <stdexcept>
#include <iostream>
#include <unordered_set>

namespace VVM {
namespace Dynamics {

bool startsWith(const std::string& fullString, const std::string& prefix) {
    if (fullString.length() < prefix.length()) return false;
    return fullString.compare(0, prefix.length(), prefix) == 0;
}

DynamicalCore::DynamicalCore(const Utils::ConfigurationManager& config, 
                             const Core::Grid& grid, 
                             const Core::Parameters& params,
                             Core::State& state, 
                             Core::HaloExchanger& halo_exchanger, 
                             const Core::BoundaryConditionManager& bc_manager)
    : config_(config), grid_(grid), params_(params), state_(state), 
      wind_solver_(std::make_unique<WindSolver>(grid, config, params, halo_exchanger, state)), 
      halo_exchanger_(halo_exchanger), bc_manager_(bc_manager) {

    int rank = grid_.get_mpi_rank();
    if (rank == 0) std::cout << "\n--- Initializing Dynamical Core ---" << std::endl;

    int nz = grid_.get_local_total_points_z();
    int ny = grid_.get_local_total_points_y();
    int nx = grid_.get_local_total_points_x();
    auto dims = std::array<int, 3>{
          grid.get_local_total_points_z(),
          grid.get_local_total_points_y(),
          grid.get_local_total_points_x()
    };

    std::vector<std::string> common_thermo = {"th", "qv"};
    
    auto prognostic_config = config_.get_value<nlohmann::json>("dynamics.prognostic_variables");
    if (!config.get_value<bool>("physics.p3.enable_p3", false)) {
        if (rank == 0) std::cout << "[WARNING] P3 is not turned on but the P3 variables are listed in prognostic variables so they are deleted!!" << std::endl;
        std::unordered_set<std::string> P3toRemove = {"qc", "qi", "qr", "qm", "nc", "ni", "nr", "bm"};
        std::vector<std::string> keysToDelete;
        for (auto& [key, value] : prognostic_config.items()) {
            for (const auto& prefix : P3toRemove) {
                if (startsWith(key, prefix)) {
                    keysToDelete.push_back(key);
                    break;
                }
            }
        }
        for (const auto& key : keysToDelete) {
            prognostic_config.erase(key);
        }
    }
    else common_thermo.insert(common_thermo.end(), {"qc", "qr", "qi", "nc", "nr", "ni", "qm", "bm"});

    if (config_.has_key("dynamics.tracers")) {
        const auto tracer_config = config_.get_value<nlohmann::json>("dynamics.tracers");
        for (const auto& tracer_name : state_.get_tracer_names()) {
            prognostic_config[tracer_name] = tracer_config.at(tracer_name);
        }
    }

    bool coriolis_xi = config.get_value<bool>("dynamics.prognostic_variables.xi.tendency_terms.coriolis.enable", false);
    bool coriolis_eta = config.get_value<bool>("dynamics.prognostic_variables.eta.tendency_terms.coriolis.enable", false);
    bool coriolis_zeta = config.get_value<bool>("dynamics.prognostic_variables.zeta.tendency_terms.coriolis.enable", false);
    enable_coriolis_ = coriolis_xi && coriolis_eta && coriolis_zeta;
    enable_turbulence_ = config.get_value<bool>("physics.turbulence.enable_turbulence", false);

    diagnostic_scheme_ = std::make_unique<Takacs>(config_, grid_, halo_exchanger_, bc_manager_);
    
    mean_wind_state_ = std::make_shared<MeanWindState>();
    NumericalMethodFactory method_factory(config_, grid_, halo_exchanger_, bc_manager_,
                                          mean_wind_state_);

    for (auto& [var_name, var_conf] : prognostic_config.items()) {
        if (rank == 0) {
            std::cout << "  * Loading prognostic variable: "
                      << var_name << std::endl;
        }

        const bool has_external_forward_euler = var_name == "th" && config.get_value<bool>("physics.rrtmgp.enable_rrtmgp", false);
        if (has_external_forward_euler && rank == 0) {
            std::cout << "    - Enabled radiation forcing integration. " << std::endl;
        }

        const bool is_tracer = state_.is_tracer(var_name);
        const bool is_thermo = is_tracer || std::find(common_thermo.begin(), common_thermo.end(), var_name) != common_thermo.end();

        if (is_thermo) thermo_vars_.push_back(var_name);
        else vorticity_vars_.push_back(var_name);

        if (!state_.has_field(var_name)) state_.add_field<3>(var_name, dims);

        auto numerical_method = method_factory.create(var_name, var_conf, is_tracer, is_thermo, dims, has_external_forward_euler);
        const auto requirements = numerical_method->state_requirements();

        if (requirements.previous_state) {
            state_.add_field<3>(var_name + "_m", dims);
        }
        if (requirements.ab2_tendency_history) {
            state_.add_field<3>("d_" + var_name + "_0", dims);
            state_.add_field<3>("d_" + var_name + "_1", dims);
        }
        if (requirements.forward_euler_tendency) {
            state_.add_field<3>("fe_tendency_" + var_name, dims);
        }

        numerical_methods_[var_name] = std::move(numerical_method);
    }
    // This is for predict utopmn and vtopmn
    state_.add_field<1>("d_utopmn", {2});
    state_.add_field<1>("d_vtopmn", {2});
    state_.add_field<0>("utopmn_m", {});
    state_.add_field<0>("vtopmn_m", {});


    if (!state.has_field("RKM")) state.add_field<3>("RKM", dims);
    if (!state.has_field("RKH")) state.add_field<3>("RKH", dims);
    if (!state.has_field("tempu")) state.add_field<2>("tempu", {ny, nx}, Core::FieldMetadata{Core::GridStaggering::StaggeredX, "m s-1", "temporary top-boundary x wind work field"});
    if (!state.has_field("tempv")) state.add_field<2>("tempv", {ny, nx}, Core::FieldMetadata{Core::GridStaggering::StaggeredY, "m s-1", "temporary top-boundary y wind work field"});

    Kokkos::deep_copy(Kokkos::DefaultExecutionSpace(), utopmn_m_ref_.get(state_, "utopmn_m").get_mutable_device_data(), utopmn_ref_.get(state_, "utopmn").get_mutable_device_data());
    Kokkos::deep_copy(Kokkos::DefaultExecutionSpace(), vtopmn_m_ref_.get(state_, "vtopmn_m").get_mutable_device_data(), vtopmn_ref_.get(state_, "vtopmn").get_mutable_device_data());
}

DynamicalCore::~DynamicalCore() = default;

void DynamicalCore::compute_diagnostic_fields() const {
    auto& R_xi_field = R_xi_ref_.get(state_, "R_xi");
    auto& R_eta_field = R_eta_ref_.get(state_, "R_eta");
    auto& R_zeta_field = R_zeta_ref_.get(state_, "R_zeta");

    diagnostic_scheme_->calculate_R_xi(state_, grid_, params_, R_xi_field);
    diagnostic_scheme_->calculate_R_eta(state_, grid_, params_, R_eta_field);
    diagnostic_scheme_->calculate_R_zeta(state_, grid_, params_, R_zeta_field);
}

void DynamicalCore::initialize_restart_history() {
    int rank = grid_.get_mpi_rank();
    if (rank == 0) {
        std::cout << "  [WARNING] Restart files do not preserve the previous AB2 "
                     "tendency. The first step after restart uses first-order history "
                     "initialization and may differ from an uninterrupted run." << std::endl;
    }

    for (const auto& item : numerical_methods_) {
        const std::string& var_name = item.first;

        if (state_.has_field(var_name + "_m")) {
            Kokkos::deep_copy(Kokkos::DefaultExecutionSpace(),
                              state_.get_field<3>(var_name + "_m").get_mutable_device_data(),
                              state_.get_field<3>(var_name).get_device_data());
        }

        item.second->calculate_tendencies(state_, grid_, params_);

        // NOTE: Restart files currently store the prognostic state but not the
        // previous AB2 tendency. The tendency evaluated from the restart state is
        // therefore copied into both history slots. This makes the first resumed
        // step equivalent to a first-order startup step; normal AB2 integration
        // resumes afterward. Restart is intended for recovery, not bitwise-exact
        // continuation.
        if (state_.has_field("d_" + var_name + "_0")) {
            const size_t now_idx = state_.get_step() % 2;
            const std::string now_name  = "d_" + var_name + (now_idx == 0 ? "_0" : "_1");
            const std::string prev_name = "d_" + var_name + (now_idx == 0 ? "_1" : "_0");
            Kokkos::deep_copy(Kokkos::DefaultExecutionSpace(),
                              state_.get_field<3>(prev_name).get_mutable_device_data(),
                              state_.get_field<3>(now_name).get_device_data());
        }
    }

    if (state_.has_field("utopmn_m")) {
        Kokkos::deep_copy(Kokkos::DefaultExecutionSpace(),
                          utopmn_m_ref_.get(state_, "utopmn_m").get_mutable_device_data(),
                          utopmn_ref_.get(state_, "utopmn").get_device_data());
    }
    if (state_.has_field("vtopmn_m")) {
        Kokkos::deep_copy(Kokkos::DefaultExecutionSpace(),
                          vtopmn_m_ref_.get(state_, "vtopmn_m").get_mutable_device_data(),
                          vtopmn_ref_.get(state_, "vtopmn").get_device_data());
    }
}

void DynamicalCore::compute_zeta_vertical_structure(Core::State& state) const {
    auto& zeta_field = zeta_ref_.get(state, "zeta");
    auto zeta_data = zeta_field.get_mutable_device_data();
    const auto& xi = xi_ref_.get(state, "xi").get_device_data();
    const auto& eta = eta_ref_.get(state, "eta").get_device_data();

    const int nz = grid_.get_local_total_points_z();
    const int ny = grid_.get_local_total_points_y();
    const int nx = grid_.get_local_total_points_x();
    const int h = grid_.get_halo_cells();
    
    const VVM::Real dz = grid_.get_dz();
    const VVM::Real dy = grid_.get_dy();
    const VVM::Real dx = grid_.get_dx();
    const auto& rdx = params_.rdx;
    const auto& rdy = params_.rdy;

    const auto& flex_height_coef_up = params_.flex_height_coef_up.get_device_data();

    Kokkos::parallel_for("zeta_downward_integration",
        Kokkos::MDRangePolicy<Kokkos::Rank<2>>({h, h}, {ny-h, nx-h}),
        KOKKOS_LAMBDA(const int j, const int i) {
            // The for-loop inside is to prevent racing condition because lower layers depend on upper layers.
            for (int k = nz-h-2; k >= h-1; --k) {
                // zeta_data(k,j,i) = zeta_data(k+1,j,i) + rhs_data(k,j,i) * -dz / flex_height_coef_up(k);
                zeta_data(k,j,i) = zeta_data(k+1,j,i) 
                                 + ( xi(k,j,i+1) -  xi(k,j,i)) * rdx() * dz / (flex_height_coef_up(k))
                                 - (eta(k,j+1,i) - eta(k,j,i)) * rdy() * dz / (flex_height_coef_up(k));
            }
            // WARNING: NK3 has a upward integration in original VVM code.
            // zeta_data(nz-h,j,i) = zeta_data(nz-h-1,j,i) + rhs_data(nz-h-1,j,i) * dz / flex_height_coef_up(nz-h-1);
            zeta_data(nz-h,j,i) = zeta_data(nz-h-1,j,i) 
                             - ( xi(nz-h-1,j,i+1) -  xi(nz-h-1,j,i)) * dz * rdx() / (flex_height_coef_up(nz-h-1))
                             + (eta(nz-h-1,j+1,i) - eta(nz-h-1,j,i)) * dz * rdy() / (flex_height_coef_up(nz-h-1));

            // for (int k = 0; k < nz; k++) {
            //     if (k != nz-h-1) zeta_data(k,j,i) = 0;
            // }
        }
    );
    halo_exchanger_.exchange_halos(zeta_field);
    bc_manager_.apply_horizontal_bcs(zeta_field);
}

void DynamicalCore::compute_wind_fields() {
    // Assign wind for topography 
    const auto& ITYPEU = ITYPEU_ref_.get(state_, "ITYPEU").get_device_data();
    const auto& ITYPEV = ITYPEV_ref_.get(state_, "ITYPEV").get_device_data();
    const auto& ITYPEW = ITYPEW_ref_.get(state_, "ITYPEW").get_device_data();
    const auto& max_topo_idx = params_.max_topo_idx;

    auto& u_topo = u_topo_ref_.get(state_, "u_topo").get_mutable_device_data();
    const auto& u = u_ref_.get(state_, "u").get_device_data();
    auto& v_topo = v_topo_ref_.get(state_, "v_topo").get_mutable_device_data();
    const auto& v = v_ref_.get(state_, "v").get_device_data();
    auto& w_topo = w_topo_ref_.get(state_, "w_topo").get_mutable_device_data();
    const auto& w = w_ref_.get(state_, "w").get_device_data();
    Kokkos::deep_copy(Kokkos::DefaultExecutionSpace(), u_topo, u);
    Kokkos::deep_copy(Kokkos::DefaultExecutionSpace(), v_topo, v);
    Kokkos::deep_copy(Kokkos::DefaultExecutionSpace(), w_topo, w);

    const int nz = grid_.get_local_total_points_z();
    const int ny = grid_.get_local_total_points_y();
    const int nx = grid_.get_local_total_points_x();
    const int h = grid_.get_halo_cells();

    Kokkos::parallel_for("wind_topo",
        Kokkos::MDRangePolicy<Kokkos::Rank<3>>({h-1, 0, 0}, {max_topo_idx+2, ny, nx}),
        KOKKOS_LAMBDA(const int k, const int j, const int i) {
            if (ITYPEU(k,j,i) != 1) u_topo(k,j,i) = 0;
            else u_topo(k,j,i) = u(k,j,i);

            if (ITYPEV(k,j,i) != 1) v_topo(k,j,i) = 0;
            else v_topo(k,j,i) = v(k,j,i);

            if (ITYPEW(k,j,i) != 1) w_topo(k,j,i) = 0;
            else w_topo(k,j,i) = w(k,j,i);
        }
    );

    auto& xi_topo = xi_topo_ref_.get(state_, "xi_topo").get_mutable_device_data();
    const auto& xi = xi_ref_.get(state_, "xi").get_device_data();
    auto& eta_topo = eta_topo_ref_.get(state_, "eta_topo").get_mutable_device_data();
    const auto& eta = eta_ref_.get(state_, "eta").get_device_data();
    const auto& rdx = params_.rdx;
    const auto& rdy = params_.rdy;
    const auto& rdz = params_.rdz;
    const auto& flex_height_coef_up = params_.flex_height_coef_up.get_device_data();

    // Assign vorticity for topography
    Kokkos::parallel_for("vorticity_topo",
        Kokkos::MDRangePolicy<Kokkos::Rank<3>>({h-1, h, h}, {nz-h, ny-h, nx-h}),
        KOKKOS_LAMBDA(const int k, const int j, const int i) {
            if (ITYPEV(k,j,i) != 1) {
                xi_topo(k,j,i) = (w_topo(k,j+1,i) - w_topo(k,j,i)) * rdy()
                               - (v_topo(k+1,j,i) - v_topo(k,j,i)) * rdz() * flex_height_coef_up(k);
            }
            else xi_topo(k,j,i) = xi(k,j,i);
            

            if (ITYPEU(k,j,i) != 1) {
                eta_topo(k,j,i) = (w_topo(k,j,i+1) - w_topo(k,j,i)) * rdx()
                                - (u_topo(k+1,j,i) - u_topo(k,j,i)) * rdz() * flex_height_coef_up(k);
            }
            else eta_topo(k,j,i) = eta(k,j,i);
        }
    );
    halo_exchanger_.exchange_halos(xi_topo_ref_.get(state_, "xi_topo"));
    halo_exchanger_.exchange_halos(eta_topo_ref_.get(state_, "eta_topo"));
    bc_manager_.apply_horizontal_bcs(xi_topo_ref_.get(state_, "xi_topo"));
    bc_manager_.apply_horizontal_bcs(eta_topo_ref_.get(state_, "eta_topo"));

    wind_solver_->solve_w();
    wind_solver_->solve_uv();

    mean_wind_state_->invalidate();
}

void DynamicalCore::compute_uvtopmn() {
    const int nz = grid_.get_local_total_points_z();
    const int ny = grid_.get_local_total_points_y();
    const int nx = grid_.get_local_total_points_x();
    const int h = grid_.get_halo_cells();
    const auto& dt = params_.dt;

    const auto& u = u_ref_.get(state_, "u").get_device_data();
    const auto& v = v_ref_.get(state_, "v").get_device_data();
    const auto& w = w_ref_.get(state_, "w").get_device_data();
    const auto& flex_height_coef_mid = params_.flex_height_coef_mid.get_device_data();
    const auto& rhobar = rhobar_ref_.get(state_, "rhobar").get_device_data();
    const auto& rhobar_up = rhobar_up_ref_.get(state_, "rhobar_up").get_device_data();
    const auto& rdz = params_.rdz;

    auto& tempu_field = tempu_ref_.get(state_, "tempu");
    auto& tempv_field = tempv_ref_.get(state_, "tempv");
    auto& tempu = tempu_field.get_mutable_device_data();
    auto& tempv = tempv_field.get_mutable_device_data();

    auto &utopmn = utopmn_ref_.get(state_, "utopmn");
    auto &vtopmn = vtopmn_ref_.get(state_, "vtopmn");

    Kokkos::parallel_for("calculate_utopmn",
        Kokkos::MDRangePolicy<Kokkos::Rank<2>>({h,h}, {ny-h, nx-h}),
        KOKKOS_LAMBDA(const int j, const int i) {
            tempu(j,i) = (rhobar(nz-h-2)*u(nz-h-2,j,i) + rhobar(nz-h-1)*u(nz-h-1,j,i)) 
                       * (w(nz-h-2,j,i)+w(nz-h-2,j,i+1));
            tempv(j,i) = (rhobar(nz-h-2)*v(nz-h-2,j,i) + rhobar(nz-h-1)*v(nz-h-1,j,i)) 
                       * (w(nz-h-2,j,i)+w(nz-h-2,j+1,i));
        }
    );
    state_.calculate_horizontal_mean(tempu_field, tempumn_);
    state_.calculate_horizontal_mean(tempv_field, tempvmn_);

    auto tempumn = tempumn_;
    auto tempvmn = tempvmn_;
    Kokkos::parallel_for("DataClipZero", 1, KOKKOS_LAMBDA(const int i) {
        if (Kokkos::abs(tempumn()) < 1e-15) {
            tempumn() = real(0.0);
        }
        if (Kokkos::abs(tempvmn()) < 1e-15) {
            tempvmn() = real(0.0);
        }
    });

    int NK2 = nz-h-1;
    int NK1 = nz-h-2;

    const auto& RKM = RKM_ref_.get(state_, "RKM").get_device_data();
    const auto& RKH = RKH_ref_.get(state_, "RKH").get_device_data();
    const auto& R_xi = R_xi_ref_.get(state_, "R_xi").get_device_data();
    const auto& R_eta = R_eta_ref_.get(state_, "R_eta").get_device_data();
    const auto& f = f_ref_.get(state_, "f").get_device_data();

    // Diffusion
    auto mean_u_turb = mean_u_turb_;
    auto mean_v_turb = mean_v_turb_;
    if (enable_turbulence_) {
        Kokkos::parallel_for("calculate_utopmn_diffusion",
            Kokkos::MDRangePolicy<Kokkos::Rank<2>>({h,h}, {ny-h, nx-h}),
            KOKKOS_LAMBDA(const int j, const int i) {
                tempu(j,i) = (RKM(NK1,j,i)+RKM(NK1,j,i+1)+RKM(NK2,j,i)+RKM(NK2,j,i+1))
                             *R_eta(NK1,j,i)*rhobar_up(NK1);
                tempv(j,i) = (RKM(NK1,j,i)+RKM(NK1,j+1,i)+RKM(NK2,j,i)+RKM(NK2,j+1,i))
                             *R_xi(NK1,j,i)*rhobar_up(NK1);
            }
        );
        state_.calculate_horizontal_mean(tempu_field, mean_u_turb_);
        state_.calculate_horizontal_mean(tempv_field, mean_v_turb_);
        Kokkos::parallel_for("DataClipZero", 1, KOKKOS_LAMBDA(const int i) {
            if (Kokkos::abs(mean_u_turb()) < real(1e-15)) {
                mean_u_turb() = real(0.0);
            }
            if (Kokkos::abs(mean_v_turb()) < real(1e-15)) {
                mean_v_turb() = real(0.0);
            }
        });
    }

    Kokkos::deep_copy(Kokkos::DefaultExecutionSpace(), mean_u_coriolis_, real(0.0));
    Kokkos::deep_copy(Kokkos::DefaultExecutionSpace(), mean_v_coriolis_, real(0.0));

    auto mean_u_coriolis = mean_u_coriolis_;
    auto mean_v_coriolis = mean_v_coriolis_;
    if (enable_coriolis_) {
        // Coriolis force
        Kokkos::parallel_for("calculate_utopmn_coriolis",
            Kokkos::MDRangePolicy<Kokkos::Rank<2>>({h,h}, {ny-h, nx-h}),
            KOKKOS_LAMBDA(const int j, const int i) {
                tempu(j,i) = f(j) * v(NK2, j, i);
                tempv(j,i) = f(j) * u(NK2, j, i);
            }
        );
        state_.calculate_horizontal_mean(tempu_field, mean_u_coriolis_);
        state_.calculate_horizontal_mean(tempv_field, mean_v_coriolis_);
        Kokkos::parallel_for("DataClipZero", 1, KOKKOS_LAMBDA(const int i) {
            if (Kokkos::abs(mean_u_coriolis()) < real(1e-15)) {
                mean_u_coriolis() = real(0.0);
            }
            if (Kokkos::abs(mean_v_coriolis()) < real(1e-15)) {
                mean_v_coriolis() = real(0.0);
            }
        });
    }

    auto& utopmn_to_update = utopmn_ref_.get(state_, "utopmn");
    auto& utopmn_new_view = utopmn_to_update.get_mutable_device_data();
    auto& utopmn_prev_step = utopmn_m_ref_.get(state_, "utopmn_m");
    auto& vtopmn_to_update = vtopmn_ref_.get(state_, "vtopmn");
    auto& vtopmn_new_view = vtopmn_to_update.get_mutable_device_data();
    auto& vtopmn_prev_step = vtopmn_m_ref_.get(state_, "vtopmn_m");

    // update utopmn, vtopmn
    Kokkos::deep_copy(Kokkos::DefaultExecutionSpace(), utopmn_prev_step.get_mutable_device_data(), utopmn_to_update.get_device_data());
    auto& utopmn_old_view = utopmn_prev_step.get_device_data();
    Kokkos::deep_copy(Kokkos::DefaultExecutionSpace(), vtopmn_prev_step.get_mutable_device_data(), vtopmn_to_update.get_device_data());
    auto& vtopmn_old_view = vtopmn_prev_step.get_device_data();

    auto& d_utopmn = d_utopmn_ref_.get(state_, "d_utopmn").get_mutable_device_data();
    auto& d_vtopmn = d_vtopmn_ref_.get(state_, "d_vtopmn").get_mutable_device_data();

    size_t now_idx = state_.get_step() % 2;
    size_t prev_idx = (state_.get_step() + 1) % 2;

    if (state_.get_step() == 0) {
        Kokkos::parallel_for("Cauculate_uvtopmn", 
            1, 
            KOKKOS_LAMBDA(const int i) {
                d_utopmn(now_idx) = real(0.25) * flex_height_coef_mid(NK2) * tempumn() * rdz() / rhobar(NK2)
                                   -real(0.25) * flex_height_coef_mid(NK2) * mean_u_turb() * rdz() / rhobar(NK2)
                                   +mean_u_coriolis();
                d_vtopmn(now_idx) = real(0.25) * flex_height_coef_mid(NK2) * tempvmn() * rdz() / rhobar(NK2)
                                   -real(0.25) * flex_height_coef_mid(NK2) * mean_v_turb() * rdz() / rhobar(NK2)
                                   -mean_v_coriolis();

                utopmn_new_view() = utopmn_old_view() + dt() * d_utopmn(now_idx);
                vtopmn_new_view() = vtopmn_old_view() + dt() * d_vtopmn(now_idx);
            }
        );
    }
    else {
        Kokkos::parallel_for("Cauculate_uvtopmn", 
            1, 
            KOKKOS_LAMBDA(const int i) {
                d_utopmn(now_idx) = real(0.25) * flex_height_coef_mid(NK2) * tempumn() * rdz() / rhobar(NK2)
                                   -real(0.25) * flex_height_coef_mid(NK2) * mean_u_turb() * rdz() / rhobar(NK2)
                                   +mean_u_coriolis();
                d_vtopmn(now_idx) = real(0.25) * flex_height_coef_mid(NK2) * tempvmn() * rdz() / rhobar(NK2)
                                   -real(0.25) * flex_height_coef_mid(NK2) * mean_v_turb() * rdz() / rhobar(NK2)
                                   -mean_v_coriolis();

                utopmn_new_view() = utopmn_old_view() 
                        + dt() * (real(1.5) * d_utopmn(now_idx) - real(0.5) * d_utopmn(prev_idx));
                vtopmn_new_view() = vtopmn_old_view() 
                        + dt() * (real(1.5) * d_vtopmn(now_idx) - real(0.5) * d_vtopmn(prev_idx));
            }
        );
    }
    return;
}


void DynamicalCore::ensure_field_cache() {
    if (field_cache_ready_) return;

    auto build = [&](const std::vector<std::string>& var_names,
                     std::vector<VariableCache>& out,
                     bool with_fe_tendency) {
        out.reserve(var_names.size());
        for (const auto& var_name : var_names) {
            VariableCache entry;
            entry.name = var_name;
            entry.field = &state_.get_field<3>(var_name);
            const auto method_it = numerical_methods_.find(var_name);
            entry.method = (method_it == numerical_methods_.end())
                           ? nullptr : method_it->second.get();
            if (with_fe_tendency) {
                const std::string fe_name = "fe_tendency_" + var_name;
                entry.fe_tendency = state_.has_field(fe_name)
                                    ? &state_.get_field<3>(fe_name) : nullptr;
            }
            entry.is_th = (var_name == "th");
            entry.is_xi = (var_name == "xi");
            entry.is_eta = (var_name == "eta");
            entry.zero_gradient_top = (var_name == "th" || var_name == "qv");
            out.push_back(std::move(entry));
        }
    };

    build(thermo_vars_, thermo_cache_, true);
    build(vorticity_vars_, vorticity_cache_, false);

    for (const auto& var : thermo_cache_) {
        if (var.method == nullptr || var.method->uses_multistage_scheme()) continue;
        single_stage_thermo_.push_back(&var);
        single_stage_thermo_fields_.push_back(var.field);
    }

    field_cache_ready_ = true;
}

void DynamicalCore::calculate_thermo_tendencies() {
    ensure_field_cache();

    for (const auto& var : thermo_cache_) {
        if (var.fe_tendency != nullptr) {
            auto data = var.fe_tendency->get_mutable_device_data();
            Kokkos::deep_copy(Kokkos::DefaultExecutionSpace(), data, real(0.0));
        }
    }

    for (const auto& var : thermo_cache_) {
        if (var.method != nullptr) {
            var.method->calculate_tendencies(state_, grid_, params_);
        }
    }
}

void DynamicalCore::update_thermodynamics(VVM::Real dt) {
    ensure_field_cache();

    const int h = grid_.get_halo_cells();
    const auto& max_topo_idx = params_.max_topo_idx;
    const int nz = grid_.get_local_total_points_z();
    const int ny = grid_.get_local_total_points_y();
    const int nx = grid_.get_local_total_points_x();

    auto apply_topographic_mask = [&](const VariableCache& var_cache) {
        const auto& ITYPEW = ITYPEW_ref_.get(state_, "ITYPEW").get_device_data();
        auto& var = var_cache.field->get_mutable_device_data();

        if (var_cache.is_th) {
            const auto thbar = thbar_ref_.get(state_, "thbar").get_device_data();
            Kokkos::parallel_for("topo_bc_th",
                Kokkos::MDRangePolicy<Kokkos::Rank<3>>({h, h, h}, {max_topo_idx + 1, ny - h, nx - h}),
                KOKKOS_LAMBDA(const int k, const int j, const int i) {
                    if (ITYPEW(k, j, i) != VVM::real(1.0)) {
                        var(k, j, i) = thbar(k);
                    }
                });
        } 
        else {
            Kokkos::parallel_for("topo_thermodynamic",
                Kokkos::MDRangePolicy<Kokkos::Rank<3>>({h, h, h}, {max_topo_idx + 1, ny - h, nx - h}),
                KOKKOS_LAMBDA(const int k, const int j, const int i) {
                    if (ITYPEW(k, j, i) != VVM::real(1.0)) {
                        var(k, j, i) = VVM::real(0.0);
                    }
                });
        }
    };

    auto process_stage_field = [&](const VariableCache& var_cache) {
        apply_topographic_mask(var_cache);
        // Preserve the established final-update ordering: exchange physical
        // values first, then fill global horizontal and vertical boundaries.
        halo_exchanger_.exchange_halos(*var_cache.field);
        bc_manager_.apply_horizontal_bcs(*var_cache.field);
        if (var_cache.zero_gradient_top) bc_manager_.apply_zero_gradient(*var_cache.field);
        else bc_manager_.apply_zero_gradient_bottom_zero_top(*var_cache.field);
    };

    for (const auto& var : thermo_cache_) {
        if (var.method == nullptr) continue;

        auto& numerical_method = *var.method;
        if (numerical_method.uses_multistage_scheme()) {
            numerical_method.advance(
                state_, grid_, params_, dt,
                [&]() { process_stage_field(var); });
        }
        else {
            numerical_method.advance(state_, grid_, params_, dt);
            apply_topographic_mask(var);
        }
    }

    // Single-stage fields retain the established batched halo path.
    if (!single_stage_thermo_fields_.empty()) {
        halo_exchanger_.exchange_multiple_halos(single_stage_thermo_fields_);
    }
    for (const auto* var : single_stage_thermo_) {
        bc_manager_.apply_horizontal_bcs(*var->field);
        if (var->zero_gradient_top) {
            bc_manager_.apply_zero_gradient(*var->field);
        }
        else {
            bc_manager_.apply_zero_gradient_bottom_zero_top(*var->field);
        }
    }
}

void DynamicalCore::calculate_vorticity_tendencies() {
    ensure_field_cache();

    const int nz = grid_.get_local_total_points_z();
    const int ny = grid_.get_local_total_points_y();
    const int nx = grid_.get_local_total_points_x();
    const int h = grid_.get_halo_cells();
    const auto& rhobar_up = rhobar_up_ref_.get(state_, "rhobar_up").get_device_data();
    const auto& rhobar = rhobar_ref_.get(state_, "rhobar").get_device_data();

    auto& xi = xi_ref_.get(state_, "xi").get_mutable_device_data();
    auto& eta = eta_ref_.get(state_, "eta").get_mutable_device_data();
    auto& zeta = zeta_ref_.get(state_, "zeta").get_mutable_device_data();
    
    // Divide by density
    Kokkos::parallel_for("divide_by_density_xi_eta",
        Kokkos::MDRangePolicy<Kokkos::Rank<3>>({h-1, 0, 0}, {nz-h, ny, nx}),
        KOKKOS_LAMBDA(const int k, const int j, const int i) {
            xi(k, j, i) /= rhobar_up(k);
            eta(k, j, i) /= rhobar_up(k);
        }
    );
    Kokkos::parallel_for("divide_by_density_zeta",
        Kokkos::MDRangePolicy<Kokkos::Rank<3>>({h-1, 0, 0}, {nz-h+1, ny, nx}),
        KOKKOS_LAMBDA(const int k, const int j, const int i) {
            zeta(k, j, i) /= rhobar(k);
        }
    );

    // Calculate vorticity tendency
    for (const auto& var : vorticity_cache_) {
        if (var.method != nullptr) {
            var.method->calculate_tendencies(state_, grid_, params_);
        }
    }
}

void DynamicalCore::update_vorticity(VVM::Real dt) {
    ensure_field_cache();

    const int nz = grid_.get_local_total_points_z();
    const int ny = grid_.get_local_total_points_y();
    const int nx = grid_.get_local_total_points_x();
    const int h = grid_.get_halo_cells();
    const auto& rhobar_up = rhobar_up_ref_.get(state_, "rhobar_up").get_device_data();
    const auto& rhobar = rhobar_ref_.get(state_, "rhobar").get_device_data();

    auto& xi = xi_ref_.get(state_, "xi").get_mutable_device_data();
    auto& eta = eta_ref_.get(state_, "eta").get_mutable_device_data();
    auto& zeta = zeta_ref_.get(state_, "zeta").get_mutable_device_data();

    Kokkos::parallel_for("multiply_density_xi",
        Kokkos::MDRangePolicy<Kokkos::Rank<3>>({h-1, 0, 0}, {nz-h, ny, nx}),
        KOKKOS_LAMBDA(const int k, const int j, const int i) {
            xi(k, j, i) *= rhobar_up(k);
        }
    );
    Kokkos::parallel_for("multiply_density_eta",
        Kokkos::MDRangePolicy<Kokkos::Rank<3>>({h-1, 0, 0}, {nz-h, ny, nx}),
        KOKKOS_LAMBDA(const int k, const int j, const int i) {
            eta(k, j, i) *= rhobar_up(k);
        }
    );
    Kokkos::parallel_for("multiply_density_zeta",
        Kokkos::MDRangePolicy<Kokkos::Rank<3>>({h-1, 0, 0}, {nz-h+1, ny, nx}),
        KOKKOS_LAMBDA(const int k, const int j, const int i) {
            zeta(k, j, i) *= rhobar(k);
        }
    );

    for (const auto& var : vorticity_cache_) {
        if (var.method != nullptr) {
            var.method->advance(state_, grid_, params_, dt);

            auto& var_data = var.field->get_mutable_device_data();
            const auto& max_topo_idx = params_.max_topo_idx;
            const int h = grid_.get_halo_cells();
            if (var.is_xi) {
                const auto& ITYPEV = ITYPEV_ref_.get(state_, "ITYPEV").get_device_data();
                Kokkos::parallel_for("mask_xi_topo",
                    Kokkos::MDRangePolicy<Kokkos::Rank<3>>({h-1, 0, 0}, {max_topo_idx+2, ny, nx}),
                    KOKKOS_LAMBDA(const int k, const int j, const int i) {
                        if (ITYPEV(k, j, i) != 1) var_data(k, j, i) = real(0.0);
                    }
                );
            } 
            else if (var.is_eta) {
                const auto& ITYPEU = ITYPEU_ref_.get(state_, "ITYPEU").get_device_data();
                Kokkos::parallel_for("mask_eta_topo",
                    Kokkos::MDRangePolicy<Kokkos::Rank<3>>({h-1, 0, 0}, {max_topo_idx+2, ny, nx}),
                    KOKKOS_LAMBDA(const int k, const int j, const int i) {
                        if (ITYPEU(k, j, i) != 1) var_data(k, j, i) = real(0.0);
                    }
                );
            }



            halo_exchanger_.exchange_halos(*var.field);
            bc_manager_.apply_horizontal_bcs(*var.field);
        }
    }
    bc_manager_.apply_vorticity_bc(xi_ref_.get(state_, "xi"));
    bc_manager_.apply_vorticity_bc(eta_ref_.get(state_, "eta"));
 
    if (config_.get_value<std::string>("simulation.idealized_test", "none") != "twisting") {
        compute_zeta_vertical_structure(state_);
    }
}

void DynamicalCore::diagnose_wind_fields(Core::State& state) {
    compute_uvtopmn(); 
    compute_wind_fields();
}


} // namespace Dynamics
} // namespace VVM
