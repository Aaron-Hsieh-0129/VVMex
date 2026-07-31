#include "State.hpp"
#include "BoundaryConditionManager.hpp"
#include <unordered_set>

namespace VVM {
namespace Core {

#if defined(ENABLE_NCCL)
State::State(const Utils::ConfigurationManager& config, const Parameters& params, const Grid& grid, ncclComm_t nccl_comm, cudaStream_t nccl_stream)
    : config_ref_(config), parameters_(params), grid_(grid), nccl_comm_(nccl_comm), nccl_stream_(nccl_stream) {
#else
State::State(const Utils::ConfigurationManager& config, const Parameters& params, const Grid& grid)
    : config_ref_(config), parameters_(params), grid_(grid) {
#endif
    int rank;
    MPI_Comm_rank(grid.get_comm(), &rank);

    // Get local points including halo cells
    int nx_total = grid.get_local_total_points_x();
    int ny_total = grid.get_local_total_points_y();
    int nz_total = grid.get_local_total_points_z();
    // 0D field
    add_field<0>("utopmn", {}, FieldMetadata{GridStaggering::StaggeredX, "m s-1", "predicted domain-mean top-boundary x wind", "ubartop", ""});
    add_field<0>("vtopmn", {}, FieldMetadata{GridStaggering::StaggeredY, "m s-1", "predicted domain-mean top-boundary y wind", "vbartop", ""});

    // 1D field
    add_field<1>("Tbar", {nz_total}, FieldMetadata{GridStaggering::Centered, "K", "reference-state air temperature profile", "", ""});
    add_field<1>("Tvbar", {nz_total}, FieldMetadata{GridStaggering::Centered, "K", "reference-state virtual temperature profile", "", ""});
    add_field<1>("thbar", {nz_total}, FieldMetadata{GridStaggering::Centered, "K", "reference-state air potential temperature profile", "", ""});
    add_field<1>("rhobar", {nz_total}, FieldMetadata{GridStaggering::Centered, "kg m-3", "reference-state air density"});
    add_field<1>("rhobar_up", {nz_total}, FieldMetadata{GridStaggering::StaggeredZ, "kg m-3", "reference-state air density at vertical interfaces"});
    add_field<1>("pbar", {nz_total}, FieldMetadata{GridStaggering::Centered, "Pa", "reference-state air pressure"});
    add_field<1>("pbar_up", {nz_total}, FieldMetadata{GridStaggering::StaggeredZ, "Pa", "reference-state air pressure at vertical interfaces"});
    add_field<1>("dpbar_mid", {nz_total}, FieldMetadata{GridStaggering::Centered, "Pa", "reference-state pressure layer thickness"});
    add_field<1>("pibar", {nz_total}, FieldMetadata{GridStaggering::Centered, "1", "reference-state Exner function"});
    add_field<1>("pibar_up", {nz_total}, FieldMetadata{GridStaggering::StaggeredZ, "1", "reference-state Exner function at vertical interfaces"});
    add_field<1>("qvbar", {nz_total}, FieldMetadata{GridStaggering::Centered, "kg kg-1", "reference-state water vapor mixing ratio"});
    add_field<1>("U", {nz_total}, FieldMetadata{GridStaggering::StaggeredX, "m s-1", "Initial x wind profile"});
    add_field<1>("V", {nz_total}, FieldMetadata{GridStaggering::StaggeredY, "m s-1", "Initial y wind profile"});

    add_field<2>("lon", {ny_total, nx_total}, FieldMetadata{GridStaggering::Centered, "degrees_east", "longitude"});
    add_field<2>("lat", {ny_total, nx_total}, FieldMetadata{GridStaggering::Centered, "degrees_north", "latitude"});
    add_field<1>("f", {ny_total}, FieldMetadata{GridStaggering::Centered, "s-1", "Coriolis parameter"});
    add_field<2>("f_2d", {ny_total, nx_total}, FieldMetadata{GridStaggering::Centered, "s-1", "Coriolis parameter on the horizontal grid"});

    // 2D field
    add_field<2>("psi", {ny_total, nx_total}, FieldMetadata{GridStaggering::StaggeredXY, "m2 s-1", "top-boundary horizontal streamfunction"});
    add_field<2>("psinm1", {ny_total, nx_total}, FieldMetadata{GridStaggering::StaggeredXY, "m2 s-1", "previous iteration top-boundary horizontal streamfunction"});
    add_field<2>("chi", {ny_total, nx_total}, FieldMetadata{GridStaggering::Centered, "m2 s-1", "top-boundary horizontal velocity potential"});
    add_field<2>("chinm1", {ny_total, nx_total}, FieldMetadata{GridStaggering::Centered, "m2 s-1", "previous iteration top-boundary horizontal velocity potential"});
    add_field<2>("utop", {ny_total, nx_total}, FieldMetadata{GridStaggering::StaggeredX, "m s-1", "top-boundary x wind perturbation"});
    add_field<2>("vtop", {ny_total, nx_total}, FieldMetadata{GridStaggering::StaggeredY, "m s-1", "top-boundary y wind perturbation"});
    add_field<2>("Tg", {ny_total, nx_total}, FieldMetadata{GridStaggering::Centered, "K", "surface skin temperature"});

    // 3D field
    add_field<3>("th", {nz_total, ny_total, nx_total}, FieldMetadata{GridStaggering::Centered, "K", "air potential temperature", "air_potential_temperature", ""});
    add_field<3>("qv", {nz_total, ny_total, nx_total}, FieldMetadata{GridStaggering::Centered, "kg kg-1", "water vapor mixing ratio"});
    add_field<3>("T", {nz_total, ny_total, nx_total}, FieldMetadata{GridStaggering::Centered, "K", "air temperature"});
    add_field<3>("xi", {nz_total, ny_total, nx_total}, FieldMetadata{GridStaggering::StaggeredYZ, "s-1", "x component of relative vorticity"});
    add_field<3>("eta", {nz_total, ny_total, nx_total}, FieldMetadata{GridStaggering::StaggeredXZ, "s-1", "y component of relative vorticity"});
    add_field<3>("zeta", {nz_total, ny_total, nx_total}, FieldMetadata{GridStaggering::StaggeredXY, "s-1", "z component of relative vorticity"});
    add_field<3>("u", {nz_total, ny_total, nx_total}, FieldMetadata{GridStaggering::StaggeredX, "m s-1", "x wind"});
    add_field<3>("v", {nz_total, ny_total, nx_total}, FieldMetadata{GridStaggering::StaggeredY, "m s-1", "y wind"});
    add_field<3>("w", {nz_total, ny_total, nx_total}, FieldMetadata{GridStaggering::StaggeredZ, "m s-1", "vertical wind"});
    add_field<3>("u_mean", {nz_total, ny_total, nx_total}, FieldMetadata{GridStaggering::StaggeredX, "kg m-2 s-1", "mean x wind for Takacs advection"});
    add_field<3>("v_mean", {nz_total, ny_total, nx_total}, FieldMetadata{GridStaggering::StaggeredY, "kg m-2 s-1", "mean y wind for Takacs advection"});
    add_field<3>("w_mean", {nz_total, ny_total, nx_total}, FieldMetadata{GridStaggering::StaggeredZ, "kg m-2 s-1", "mean z wind for Takacs advection"});

    // Rotation term
    add_field<3>("R_xi", {nz_total, ny_total, nx_total}, FieldMetadata{GridStaggering::StaggeredYZ, "s-1", "Shear strain rate for y-z plane at xi place"});
    add_field<3>("R_eta", {nz_total, ny_total, nx_total}, FieldMetadata{GridStaggering::StaggeredXZ, "s-1", "Shear strain rate for x-z plane at eta place"});
    add_field<3>("R_zeta", {nz_total, ny_total, nx_total}, FieldMetadata{GridStaggering::StaggeredXY, "s-1", "Shear strain rate for x-y plane at zeta place"});

    // Topography
    add_field<2>("topo", {ny_total, nx_total}, FieldMetadata{GridStaggering::Centered, "index", "topographic vertical-level index"});
    add_field<3>("ITYPEU", {nz_total, ny_total, nx_total}, FieldMetadata{GridStaggering::StaggeredX, "index", "x-wind fluid-cell mask"});
    add_field<3>("ITYPEV", {nz_total, ny_total, nx_total}, FieldMetadata{GridStaggering::StaggeredY, "index", "y-wind fluid-cell mask"});
    add_field<3>("ITYPEW", {nz_total, ny_total, nx_total}, FieldMetadata{GridStaggering::StaggeredZ, "index", "vertical-wind fluid-cell mask"});

    if (config.has_key("dynamics.tracers")) {
        const auto tracer_config = config.get_value<nlohmann::json>("dynamics.tracers");
        if (!tracer_config.is_object()) {
            throw std::runtime_error("Configuration error: 'dynamics.tracers' must be an object.");
        }

        // These names are allocated outside State or are reserved by the initial-file
        // and output formats. They remain unavailable even when optional physics is off.
        static const std::unordered_set<std::string> reserved_names = {
            "nx", "ny", "nz", "x", "y", "z_mid", "time",
            "coordinates/x", "coordinates/y", "coordinates/z_mid",
            "mask", "height", "sea_land_ice_mask",
            "vegtype", "soiltype", "slopetype", "albedo", "gvf", "lai",
            "shdmax", "shdmin", "sm1", "sm2", "sm3", "sm4",
            "sl1", "sl2", "sl3", "sl4", "st1", "st2", "st3", "st4",
            "slc1", "slc2", "slc3", "slc4", "sfemis", "zorl", "cmx",
            "chx", "canopy", "snwdph", "sneqv", "hfx", "le", "gfx",
            "qc", "qr", "qi", "qm", "nc", "nr", "ni", "bm", "qp",
            "th_perturb", "th_m_diag", "qv_m_diag", "qv_after_p3",
            "qc_after_p3", "qi_after_p3", "qr_after_p3", "nc_after_p3",
            "nr_after_p3", "ni_after_p3", "qm_after_p3", "bm_after_p3",
            "th_after_p3", "diag_eff_radius_qc", "diag_eff_radius_qi",
            "diag_eff_radius_qr", "P_wet", "precip_liq_surf_mass",
            "precip_ice_surf_mass", "precip_liq_surf_flux",
            "precip_ice_surf_flux", "RKM", "RKH", "topou", "topov",
            "sfc_flux_th", "sfc_flux_qv", "sfc_flux_u", "sfc_flux_v",
            "gwet", "zrough", "VEN2D", "ustar", "molen", "sw_heating",
            "lw_heating", "net_heating", "net_sw_flux", "net_lw_flux",
            "swdn", "lwdn", "lwup", "swup_toa", "swdn_toa", "lwup_toa",
            "lwdn_toa", "swup_sfc", "swdn_sfc", "lwup_sfc", "lwdn_sfc",
            "Q1", "Q2", "CGR_thermo", "CGR_vort", "lbn_weight",
            "areamn_xi0", "areamn_eta0", "areamn_zeta0_top",
            "areamn_local_sum_xi", "areamn_global_sum_xi",
            "areamn_local_sum_eta", "areamn_global_sum_eta",
            "areamn_local_sum_zeta_top", "areamn_global_sum_zeta_top",
            "areamn_utopmn0", "areamn_vtopmn0"
        };

        nlohmann::json prognostic_config = nlohmann::json::object();
        if (config.has_key("dynamics.prognostic_variables")) {
            prognostic_config = config.get_value<nlohmann::json>("dynamics.prognostic_variables");
        }

        for (const auto& item : tracer_config.items()) {
            const std::string& tracer_name = item.key();
            const auto& tracer_options = item.value();
            if (!tracer_options.is_object()) {
                throw std::runtime_error("Configuration error for tracer '" + tracer_name +
                                         "': tracer options must be an object.");
            }

            bool enabled = true;
            if (tracer_options.contains("enable")) {
                try {
                    enabled = tracer_options.at("enable").get<bool>();
                } 
                catch (const nlohmann::json::exception& e) {
                    throw std::runtime_error("Configuration error for tracer '" + tracer_name +
                                             "': option 'enable' must be boolean. " + e.what());
                }
            }

            const bool uses_internal_name = tracer_name.rfind("d_", 0) == 0 || tracer_name.rfind("fe_tendency_", 0) == 0 ||
                                            tracer_name.find("_ls") != std::string::npos || 
                                           (tracer_name.size() >= 2 && tracer_name.compare(tracer_name.size() - 2, 2, "_m") == 0);
            if (tracer_name.empty() || has_field(tracer_name) ||
                reserved_names.count(tracer_name) != 0 ||
                prognostic_config.contains(tracer_name) || uses_internal_name) {
                throw std::runtime_error("Configuration error for tracer '" + tracer_name +
                                         "': the name collides with an existing or reserved VVMex field/NetCDF name.");
            }
            if (!enabled) continue;

            bool source_enabled = false;
            if (tracer_options.contains("source")) {
                const auto& source_options = tracer_options.at("source");
                if (!source_options.is_object()) {
                    throw std::runtime_error(
                        "Configuration error for tracer '" + tracer_name +
                        "': source options must be an object.");
                }
                if (source_options.contains("enable")) {
                    try {
                        source_enabled =
                            source_options.at("enable").get<bool>();
                    } catch (const nlohmann::json::exception& e) {
                        throw std::runtime_error(
                            "Configuration error for tracer '" + tracer_name +
                            "': source option 'enable' must be boolean. " +
                            e.what());
                    }
                } else {
                    source_enabled = true;
                }
            }

            tracer_names_.push_back(tracer_name);
            if (source_enabled) {
                tracer_source_targets_.push_back(tracer_name);
            }
        }

        for (const auto& tracer_name : tracer_names_) {
            add_field<3>(tracer_name, {nz_total, ny_total, nx_total}, FieldMetadata{GridStaggering::Centered, "1", "tracer " + tracer_name});
        }
        for (const auto& tracer_name : tracer_source_targets_) {
            const std::string source_name = tracer_name + "_source";
            if (has_field(source_name) ||
                prognostic_config.contains(source_name)) {
                throw std::runtime_error(
                    "Configuration error: tracer source field '" + source_name +
                    "' collides with an existing VVMex field/NetCDF name.");
            }
            tracer_source_names_.push_back(source_name);
            add_field<3>(source_name, {nz_total, ny_total, nx_total}, FieldMetadata{GridStaggering::Centered, "s-1", "source tendency for tracer " + tracer_name});
        }
    }
}

} // namespace Core
} // namespace VVM
