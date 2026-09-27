// Turbulence scheme for Shuts & Grey (1994)

#include "TurbulenceProcess.hpp"
#include "core/geometry/GeometryKind.hpp"
#include "core/geometry/HorizontalLocation.hpp"
#include "dynamics/operators/HorizontalFluxDivergence.hpp"
#include "dynamics/operators/HorizontalScalarGradient.hpp"

#include <iostream>
#include <cmath>
#include <algorithm>

namespace VVM {
namespace Physics {

namespace {

struct CovariantHorizontalStrain {
    VVM::Real r11 = VVM::real(0.0);
    VVM::Real r12 = VVM::real(0.0);
    VVM::Real r22 = VVM::real(0.0);
};

template <typename U1View, typename U2View>
KOKKOS_INLINE_FUNCTION CovariantHorizontalStrain
horizontal_strain_at_t(const U1View& u1,
    const U2View& u2,
    const int k,
    const int j,
    const int i,
    const Core::Geometry::HorizontalGeometryDeviceView& geom) noexcept {
    const VVM::Real u1_t = VVM::real(0.5) * (u1(k, j, i) + u1(k, j, i - 1));
    const VVM::Real u2_t = VVM::real(0.5) * (u2(k, j, i) + u2(k, j - 1, i));

    const VVM::Real du1_dq1 = (u1(k, j, i) - u1(k, j, i - 1)) * geom.rdq1;
    const VVM::Real du2_dq2 = (u2(k, j, i) - u2(k, j - 1, i)) * geom.rdq2;

    const VVM::Real du1_dq2 =
        VVM::real(0.25) *
        (u1(k, j + 1, i) + u1(k, j + 1, i - 1) - u1(k, j - 1, i) - u1(k, j - 1, i - 1)) * geom.rdq2;
    const VVM::Real du2_dq1 =
        VVM::real(0.25) *
        (u2(k, j, i + 1) + u2(k, j - 1, i + 1) - u2(k, j, i - 1) - u2(k, j - 1, i - 1)) * geom.rdq1;

    const VVM::Real half_rdq1 = VVM::real(0.5) * geom.rdq1;
    const VVM::Real half_rdq2 = VVM::real(0.5) * geom.rdq2;

    const VVM::Real dg11_dq1 = (geom.g_cov.a11(j, i + 1) - geom.g_cov.a11(j, i - 1)) * half_rdq1;
    const VVM::Real dg11_dq2 = (geom.g_cov.a11(j + 1, i) - geom.g_cov.a11(j - 1, i)) * half_rdq2;

    const VVM::Real dg12_dq1 = (geom.g_cov.a12(j, i + 1) - geom.g_cov.a12(j, i - 1)) * half_rdq1;
    const VVM::Real dg12_dq2 = (geom.g_cov.a12(j + 1, i) - geom.g_cov.a12(j - 1, i)) * half_rdq2;

    const VVM::Real dg22_dq1 = (geom.g_cov.a22(j, i + 1) - geom.g_cov.a22(j, i - 1)) * half_rdq1;
    const VVM::Real dg22_dq2 = (geom.g_cov.a22(j + 1, i) - geom.g_cov.a22(j - 1, i)) * half_rdq2;

    const VVM::Real g11 = geom.g_cov.a11(j, i);
    const VVM::Real g12 = geom.g_cov.a12(j, i);
    const VVM::Real g22 = geom.g_cov.a22(j, i);

    CovariantHorizontalStrain result;

    // R_ab = nabla_a u_b + nabla_b u_a
    //      = (L_u g)_ab
    //
    // This is the symmetric deformation tensor expressed
    // in the covariant generalized-coordinate basis.
    result.r11 =
        u1_t * dg11_dq1 + u2_t * dg11_dq2 + VVM::real(2.0) * (g11 * du1_dq1 + g12 * du2_dq1);
    result.r22 =
        u1_t * dg22_dq1 + u2_t * dg22_dq2 + VVM::real(2.0) * (g12 * du1_dq2 + g22 * du2_dq2);
    result.r12 = u1_t * dg12_dq1 + u2_t * dg12_dq2 + g12 * du1_dq1 + g22 * du2_dq1 + g11 * du1_dq2 +
                 g12 * du2_dq2;

    return result;
}

template <typename U1View, typename U2View>
KOKKOS_INLINE_FUNCTION CovariantHorizontalStrain
horizontal_strain_at_z(const U1View& u1,
    const U2View& u2,
    const int k,
    const int j,
    const int i,
    const Core::Geometry::HorizontalGeometryDeviceView& geom) noexcept {
    const VVM::Real u1_z = VVM::real(0.5) * (u1(k, j, i) + u1(k, j + 1, i));
    const VVM::Real u2_z = VVM::real(0.5) * (u2(k, j, i) + u2(k, j, i + 1));

    const VVM::Real du1_dq1 =
        VVM::real(0.25) *
        (u1(k, j, i + 1) + u1(k, j + 1, i + 1) - u1(k, j, i - 1) - u1(k, j + 1, i - 1)) * geom.rdq1;
    const VVM::Real du2_dq2 =
        VVM::real(0.25) *
        (u2(k, j + 1, i) + u2(k, j + 1, i + 1) - u2(k, j - 1, i) - u2(k, j - 1, i + 1)) * geom.rdq2;

    const VVM::Real du1_dq2 = (u1(k, j + 1, i) - u1(k, j, i)) * geom.rdq2;
    const VVM::Real du2_dq1 = (u2(k, j, i + 1) - u2(k, j, i)) * geom.rdq1;

    const VVM::Real half_rdq1 = VVM::real(0.5) * geom.rdq1;
    const VVM::Real half_rdq2 = VVM::real(0.5) * geom.rdq2;

    const VVM::Real dg11_dq1 = (geom.g_cov.a11(j, i + 1) - geom.g_cov.a11(j, i - 1)) * half_rdq1;
    const VVM::Real dg11_dq2 = (geom.g_cov.a11(j + 1, i) - geom.g_cov.a11(j - 1, i)) * half_rdq2;

    const VVM::Real dg12_dq1 = (geom.g_cov.a12(j, i + 1) - geom.g_cov.a12(j, i - 1)) * half_rdq1;
    const VVM::Real dg12_dq2 = (geom.g_cov.a12(j + 1, i) - geom.g_cov.a12(j - 1, i)) * half_rdq2;

    const VVM::Real dg22_dq1 = (geom.g_cov.a22(j, i + 1) - geom.g_cov.a22(j, i - 1)) * half_rdq1;
    const VVM::Real dg22_dq2 = (geom.g_cov.a22(j + 1, i) - geom.g_cov.a22(j - 1, i)) * half_rdq2;

    const VVM::Real g11 = geom.g_cov.a11(j, i);
    const VVM::Real g12 = geom.g_cov.a12(j, i);
    const VVM::Real g22 = geom.g_cov.a22(j, i);

    CovariantHorizontalStrain result;

    result.r11 =
        u1_z * dg11_dq1 + u2_z * dg11_dq2 + VVM::real(2.0) * (g11 * du1_dq1 + g12 * du2_dq1);
    result.r22 =
        u1_z * dg22_dq1 + u2_z * dg22_dq2 + VVM::real(2.0) * (g12 * du1_dq2 + g22 * du2_dq2);
    result.r12 = u1_z * dg12_dq1 + u2_z * dg12_dq2 + g12 * du1_dq1 + g22 * du2_dq1 + g11 * du1_dq2 +
                 g12 * du2_dq2;

    return result;
}

KOKKOS_INLINE_FUNCTION
VVM::Real
physical_horizontal_strain_norm_squared(const CovariantHorizontalStrain& strain,
    const Core::Geometry::HorizontalGeometryDeviceView& geom,
    const int j,
    const int i) noexcept {
    const VVM::Real b11 = geom.physical_to_contravariant.a11(j, i);
    const VVM::Real b12 = geom.physical_to_contravariant.a12(j, i);
    const VVM::Real b21 = geom.physical_to_contravariant.a21(j, i);
    const VVM::Real b22 = geom.physical_to_contravariant.a22(j, i);

    const VVM::Real p11 =
        b11 * b11 * strain.r11 + VVM::real(2.0) * b11 * b21 * strain.r12 + b21 * b21 * strain.r22;
    const VVM::Real p12 =
        b11 * b12 * strain.r11 + (b11 * b22 + b21 * b12) * strain.r12 + b21 * b22 * strain.r22;
    const VVM::Real p22 =
        b12 * b12 * strain.r11 + VVM::real(2.0) * b12 * b22 * strain.r12 + b22 * b22 * strain.r22;

    return p11 * p11 + VVM::real(2.0) * p12 * p12 + p22 * p22;
}

KOKKOS_INLINE_FUNCTION
VVM::Real
physical_horizontal_shear_12(const CovariantHorizontalStrain& strain,
    const Core::Geometry::HorizontalGeometryDeviceView& geom,
    const int j,
    const int i) noexcept {
    const VVM::Real b11 = geom.physical_to_contravariant.a11(j, i);
    const VVM::Real b12 = geom.physical_to_contravariant.a12(j, i);
    const VVM::Real b21 = geom.physical_to_contravariant.a21(j, i);
    const VVM::Real b22 = geom.physical_to_contravariant.a22(j, i);

    return b11 * b12 * strain.r11 + (b11 * b22 + b21 * b12) * strain.r12 + b21 * b22 * strain.r22;
}

} // namespace

TurbulenceProcess::TurbulenceProcess(const Utils::ConfigurationManager& config,
    const Core::Grid& grid,
    const Core::Parameters& params,
    Core::HaloExchanger& halo_exchanger,
    Core::State& state)
    : config_(config), grid_(grid), params_(params), halo_exchanger_(halo_exchanger),
      masks_(grid.get_local_total_points_z(),
          grid.get_local_total_points_y(),
          grid.get_local_total_points_x()) {
    int nz = grid_.get_local_total_points_z();
    int ny = grid_.get_local_total_points_y();
    int nx = grid_.get_local_total_points_x();

    auto dims = std::array<int, 3>{grid.get_local_total_points_z(),
        grid.get_local_total_points_y(),
        grid.get_local_total_points_x()};
    if (!state.has_field("RKM")) {
        state.add_field<3>("RKM",
            dims,
            Core::FieldMetadata{Core::GridStaggering::Centered,
                "m2 s-1",
                "turbulent eddy viscosity"});
    }
    if (!state.has_field("RKH")) {
        state.add_field<3>("RKH",
            dims,
            Core::FieldMetadata{Core::GridStaggering::Centered,
                "m2 s-1",
                "turbulent eddy diffusivity"});
    }

    dynamics_vars_ = {"xi", "eta", "zeta"};
    thermodynamics_vars_ = {"th", "qv"};
    if (config.get_value<bool>("physics.p3.enable_p3", false)) {
        thermodynamics_vars_.insert(thermodynamics_vars_.end(),
            {"qc", "qr", "qi", "nc", "nr", "ni", "bm", "qm"});
    }
    const auto& tracer_names = state.get_tracer_names();
    thermodynamics_vars_.insert(thermodynamics_vars_.end(),
        tracer_names.begin(),
        tracer_names.end());
}

void
TurbulenceProcess::initialize(Core::State& state) {
    int nz = grid_.get_local_total_points_z();
    int ny = grid_.get_local_total_points_y();
    int nx = grid_.get_local_total_points_x();

    auto dims = std::array<int, 3>{nz, ny, nx};
    const auto tendency_metadata = [&state](const std::string& var_name) {
        const auto& metadata = state.get_field<3>(var_name).get_metadata();
        std::string units;
        if (metadata.units == "1") {
            units = "s-1";
        }
        else if (metadata.units == "s-1") {
            units = "s-2";
        }
        else if (!metadata.units.empty()) {
            units = metadata.units + " s-1";
        }
        const std::string long_name =
            metadata.long_name.empty() ? var_name + " tendency" : metadata.long_name + " tendency";
        return Core::FieldMetadata{metadata.grid_staggering, units, long_name};
    };
    for (const auto& var_name : thermodynamics_vars_) {
        std::string fe_tendency_name = "fe_tendency_" + var_name;
        if (!state.has_field(fe_tendency_name)) {
            state.add_field<3>(fe_tendency_name, dims, tendency_metadata(var_name));
        }
    }
    for (const auto& var_name : dynamics_vars_) {
        std::string fe_tendency_name = "fe_tendency_" + var_name;
        if (!state.has_field(fe_tendency_name)) {
            if (var_name == "zeta") {
                state.add_field<2>(fe_tendency_name, {ny, nx}, tendency_metadata(var_name));
            }
            else {
                state.add_field<3>(fe_tendency_name, dims, tendency_metadata(var_name));
            }
        }
    }
    Kokkos::deep_copy(dx_, params_.dx);
    Kokkos::deep_copy(dy_, params_.dy);
    Kokkos::deep_copy(dz_, params_.dz);
    Kokkos::deep_copy(rdx_, params_.rdx);
    Kokkos::deep_copy(rdy_, params_.rdy);
    Kokkos::deep_copy(rdz_, params_.rdz);
    Kokkos::deep_copy(rdx2_, params_.rdx2);
    Kokkos::deep_copy(rdy2_, params_.rdy2);
    Kokkos::deep_copy(rdz2_, params_.rdz2);
    Kokkos::deep_copy(grav_, params_.gravity);

    vk_ = 0.4;
    deld_ = std::pow(dx_ * dy_ * dz_, 1.0 / 3.0);
    ramd0s_ = std::pow(0.23 * deld_, 2.0);
    critmn_ = 1.0;

    init_boundary_masks(state);
    return;
}

void
TurbulenceProcess::init_boundary_masks(Core::State& state) {
    const auto& ITYPEU = ITYPEU_ref_.get(state, "ITYPEU").get_device_data();
    const auto& ITYPEV = ITYPEV_ref_.get(state, "ITYPEV").get_device_data();
    const auto& ITYPEW = ITYPEW_ref_.get(state, "ITYPEW").get_device_data();
    const auto& hx = topo_ref_.get(state, "topo").get_device_data();

    int nz = grid_.get_local_total_points_z();
    int ny = grid_.get_local_total_points_y();
    int nx = grid_.get_local_total_points_x();
    int h = grid_.get_halo_cells();
    int max_topo = params_.max_topo_idx;

    auto masks = masks_;

    masks_.reset_to_ones();

    Kokkos::parallel_for("Init_DH_Topo_Inside",
        Kokkos::MDRangePolicy<Kokkos::Rank<2>>({{h, h}}, {{ny - h, nx - h}}),
        KOKKOS_LAMBDA(const int j, const int i) {
            int NN = static_cast<int>(hx(j, i));
            if (NN != 0) {
                for (int k = h; k < NN; ++k) {
                    masks.turn_off_all(k, j, i);
                }
            }
        });

    Kokkos::parallel_for("Init_DH_ITYPEW",
        Kokkos::MDRangePolicy<Kokkos::Rank<3>>({{h, h, h}}, {{max_topo + 1, ny - h, nx - h}}),
        KOKKOS_LAMBDA(const int k, const int j, const int i) {
            if (ITYPEW(k, j, i) != 1) {
                masks.turn_off(k, j, i - 1, UU1);
                masks.turn_off(k, j, i - 1, UV1);
                masks.turn_off(k, j, i - 1, UW1);
                masks.turn_off(k, j - 1, i, VU1);
                masks.turn_off(k, j - 1, i, VV1);
                masks.turn_off(k, j - 1, i, VW1);

                masks.turn_off(k, j, i - 1, UU2);
                masks.turn_off(k, j, i - 1, UV2);
                masks.turn_off(k, j, i - 1, UW2);
                masks.turn_off(k, j - 1, i, VU2);
                masks.turn_off(k, j - 1, i, VV2);
                masks.turn_off(k, j - 1, i, VW2);

                masks.turn_off(k, j, i + 1, WU2);
                masks.turn_off(k, j, i - 1, WU1);
                masks.turn_off(k, j + 1, i, WV2);
                masks.turn_off(k, j - 1, i, WV1);
                masks.turn_off(k + 1, j, i, WW2);
                masks.turn_off(k - 1, j, i, WW1);
            }
        });

    Kokkos::parallel_for("Init_DH_ITYPEU",
        Kokkos::MDRangePolicy<Kokkos::Rank<3>>({{h, h, h}}, {{nz - h, ny - h, nx - h}}),
        KOKKOS_LAMBDA(const int k, const int j, const int i) {
            if (ITYPEU(k, j, i) != 1) {
                masks.turn_off(k, j, i + 1, UU2);
                masks.turn_off(k, j, i - 1, UU1);
                masks.turn_off(k, j + 1, i, UV2);
                masks.turn_off(k, j - 1, i, UV1);
                masks.turn_off(k + 1, j, i, UW2);
                masks.turn_off(k - 1, j, i, UW1);
            }
        });

    Kokkos::parallel_for("Init_DH_ITYPEV",
        Kokkos::MDRangePolicy<Kokkos::Rank<3>>({{h, h, h}}, {{nz - h, ny - h, nx - h}}),
        KOKKOS_LAMBDA(const int k, const int j, const int i) {
            if (ITYPEV(k, j, i) != 1) {
                masks.turn_off(k, j, i + 1, VU2);
                masks.turn_off(k, j, i - 1, VU1);
                masks.turn_off(k, j + 1, i, VV2);
                masks.turn_off(k, j - 1, i, VV1);
                masks.turn_off(k + 1, j, i, VW2);
                masks.turn_off(k - 1, j, i, VW1);
            }
        });

    Kokkos::parallel_for("Init_DH_ITYPEW_2D",
        Kokkos::MDRangePolicy<Kokkos::Rank<2>>({{0, 0}}, {{nz, nx}}),
        KOKKOS_LAMBDA(const int k, const int i) {
            // North Boundary
            if (ITYPEW(k, ny - h, i) != 1) {
                masks.turn_off(k, ny - h - 1, i, VU1);
                masks.turn_off(k, ny - h - 1, i, VV1);
                masks.turn_off(k, ny - h - 1, i, VW1);
                masks.turn_off(k, ny - h - 1, i, VU2);
                masks.turn_off(k, ny - h - 1, i, VV2);
                masks.turn_off(k, ny - h - 1, i, VW2);
                masks.turn_off(k, ny - h - 1, i, WV1);
            }
            // South Boundary
            if (ITYPEW(k, h - 1, i) != 1) {
                masks.turn_off(k, h, i, WV2);
            }

            if (ITYPEU(k, ny - h, i) != 1) {
                masks.turn_off(k, ny - h - 1, i, UV1);
            }
            if (ITYPEU(k, h - 1, i) != 1) {
                masks.turn_off(k, h, i, UV2);
            }

            if (ITYPEV(k, ny - h, i) != 1) {
                masks.turn_off(k, ny - h - 1, i, VV1);
            }
            if (ITYPEV(k, h - 1, i) != 1) {
                masks.turn_off(k, h, i, VV2);
            }
        });

    Kokkos::parallel_for("Init_DH_Edge_X",
        Kokkos::MDRangePolicy<Kokkos::Rank<2>>({{0, 0}}, {{nz, ny}}),
        KOKKOS_LAMBDA(const int k, const int j) {
            // East Boundary
            if (ITYPEW(k, j, nx - h) != 1) {
                masks.turn_off(k, j, nx - h - 1, UU1);
                masks.turn_off(k, j, nx - h - 1, UV1);
                masks.turn_off(k, j, nx - h - 1, UW1);
                masks.turn_off(k, j, nx - h - 1, UU2);
                masks.turn_off(k, j, nx - h - 1, UV2);
                masks.turn_off(k, j, nx - h - 1, UW2);
                masks.turn_off(k, j, nx - h - 1, WU1);
            }
            // West Boundary
            if (ITYPEW(k, j, h - 1) != 1) {
                masks.turn_off(k, j, h, WU2);
            }

            if (ITYPEU(k, j, nx - h) != 1) {
                masks.turn_off(k, j, nx - h - 1, UU1);
            }
            if (ITYPEU(k, j, h - 1) != 1) {
                masks.turn_off(k, j, h, UU2);
            }

            if (ITYPEV(k, j, nx - h) != 1) {
                masks.turn_off(k, j, nx - h - 1, VU1);
            }
            if (ITYPEV(k, j, h - 1) != 1) {
                masks.turn_off(k, j, h, VU2);
            }
        });

    Kokkos::parallel_for("Init_DH_WW_Bound",
        Kokkos::MDRangePolicy<Kokkos::Rank<2>>({{0, 0}}, {{ny, nx}}),
        KOKKOS_LAMBDA(const int j, const int i) {
            int NN = static_cast<int>(hx(j, i));
            if (NN > 1 && ITYPEW(NN - 1, j, i) == 0) {
                masks.turn_off(NN, j, i, WW1);
                masks.turn_off(NN, j, i, WU1);
                masks.turn_off(NN, j, i, WU2);
                masks.turn_off(NN, j, i, WV1);
                masks.turn_off(NN, j, i, WV2);
            }
            if (NN > 1 && ITYPEV(NN, j, i) == 0) {
                masks.turn_off(NN, j, i, VU1);
                masks.turn_off(NN, j, i, VU2);
                masks.turn_off(NN, j, i, VV1);
                masks.turn_off(NN, j, i, VW1);
            }
            if (NN > 1 && ITYPEU(NN, j, i) == 0) {
                masks.turn_off(NN, j, i, UU1);
                masks.turn_off(NN, j, i, UU2);
                masks.turn_off(NN, j, i, UV1);
                masks.turn_off(NN, j, i, UV2);
                masks.turn_off(NN, j, i, UW1);
            }
        });
}

void
TurbulenceProcess::compute_coefficients(Core::State& state, VVM::Real dt) {
    if (grid_.geometry().kind() != Core::Geometry::GeometryKind::Cartesian) {
        compute_generalized_deformation(state);
        compute_generalized_coefficients(state, dt);
        return;
    }

    const int nz = grid_.get_local_total_points_z();
    const int ny = grid_.get_local_total_points_y();
    const int nx = grid_.get_local_total_points_x();
    const int h = grid_.get_halo_cells();

    const auto& u = u_ref_.get(state, "u").get_device_data();
    const auto& v = v_ref_.get(state, "v").get_device_data();
    const auto& w = w_ref_.get(state, "w").get_device_data();
    const auto& R_xi = R_xi_ref_.get(state, "R_xi").get_device_data();
    const auto& R_eta = R_eta_ref_.get(state, "R_eta").get_device_data();
    const auto& R_zeta = R_zeta_ref_.get(state, "R_zeta").get_device_data();
    const auto& th = th_ref_.get(state, "th").get_device_data();
    const auto& z_mid = params_.z_mid.get_device_data();
    const auto& flex_height_coef_mid = params_.flex_height_coef_mid.get_device_data();
    const auto& flex_height_coef_up = params_.flex_height_coef_up.get_device_data();
    const auto& ITYPEU = ITYPEU_ref_.get(state, "ITYPEU").get_device_data();
    const auto& ITYPEV = ITYPEV_ref_.get(state, "ITYPEV").get_device_data();
    const auto& ITYPEW = ITYPEW_ref_.get(state, "ITYPEW").get_device_data();

    auto& rkm = RKM_ref_.get(state, "RKM").get_mutable_device_data();
    auto& rkh = RKH_ref_.get(state, "RKH").get_mutable_device_data();

    const VVM::Real rdx = rdx_;
    const VVM::Real rdy = rdy_;
    const VVM::Real rdz = rdz_;

    const VVM::Real grav = grav_;
    const VVM::Real vk = vk_;
    const VVM::Real ramd0s = ramd0s_;
    const VVM::Real critmn = critmn_;
    const VVM::Real critmx = real(0.8) * deld_ * deld_ / dt;

    const auto& masks = masks_;
    Kokkos::parallel_for("ShuttsGray_Coeffs",
        Kokkos::MDRangePolicy<Kokkos::Rank<3>>({{h, h, h}}, {{nz - h, ny - h, nx - h}}),
        KOKKOS_LAMBDA(const int k, const int j, const int i) {
            VVM::Real du_dx = (u(k, j, i) - u(k, j, i - 1)) * rdx;
            VVM::Real dv_dy = (v(k, j, i) - v(k, j - 1, i)) * rdy;
            VVM::Real dw_dz = flex_height_coef_mid(k) * (w(k, j, i) - w(k - 1, j, i)) * rdz;

            VVM::Real TERM1 =
                Kokkos::pow(R_zeta(k, j - 1, i - 1), real(2.)) +
                Kokkos::pow(R_zeta(k, j, i - 1), real(2.)) +
                Kokkos::pow(R_zeta(k, j - 1, i), real(2.)) +
                Kokkos::pow(R_zeta(k, j, i), real(2.)) + Kokkos::pow(R_eta(k, j, i - 1), real(2.)) +
                Kokkos::pow(R_eta(k, j, i), real(2.)) +
                Kokkos::pow(R_eta(k - 1, j, i - 1), real(2.)) +
                Kokkos::pow(R_eta(k - 1, j, i), real(2.)) +
                Kokkos::pow(R_xi(k, j - 1, i), real(2.)) + Kokkos::pow(R_xi(k, j, i), real(2.)) +
                Kokkos::pow(R_xi(k - 1, j - 1, i), real(2.)) +
                Kokkos::pow(R_xi(k - 1, j, i), real(2.));

            TERM1 =
                real(0.25) * TERM1 + real(2.0) * (du_dx * du_dx + dv_dy * dv_dy + dw_dz * dw_dz);

            if (ITYPEW(k, j, i) != 1) {
                TERM1 = real(0.);
            }

            // N^2
            VVM::Real DDY_top = grav * flex_height_coef_up(k) * (th(k + 1, j, i) - th(k, j, i)) *
                                rdz / (th(k + 1, j, i) + th(k, j, i)) * masks.val(k, j, i, WW1);

            VVM::Real DDY_bot = grav * flex_height_coef_up(k - 1) *
                                (th(k, j, i) - th(k - 1, j, i)) * rdz /
                                (th(k, j, i) + th(k - 1, j, i)) * masks.val(k, j, i, WW2);

            VVM::Real DDY = DDY_top + DDY_bot;

            // Mixing Length
            VVM::Real z = z_mid(k);
            VVM::Real ZROUGH = real(2e-4);
            VVM::Real DDX = (ramd0s * vk * vk * Kokkos::pow(z + ZROUGH, real(2.))) /
                            (ramd0s + vk * vk * Kokkos::pow(z + ZROUGH, real(2.)));

            // Richardson Number
            VVM::Real Ri = DDY / TERM1;

            // E. Km, Kh
            VVM::Real rkm_val = real(0.0);
            VVM::Real rkh_val = real(0.0);
            VVM::Real sqrt_TERM1 = Kokkos::sqrt(TERM1);

            if (TERM1 == real(0.)) {
                rkm_val = real(0.);
                rkh_val = real(0.);
            }
            else {
                if (Ri < real(0.0)) {
                    rkm_val = sqrt_TERM1 * DDX * Kokkos::sqrt(real(1.0) - real(16.0) * Ri);
                    rkh_val =
                        sqrt_TERM1 * DDX * real(1.4) * Kokkos::sqrt(real(1.0) - real(40.0) * Ri);
                }
                else if (Ri < real(0.25)) {
                    rkm_val = sqrt_TERM1 * DDX * Kokkos::pow(real(1.) - real(4.) * Ri, real(4.));
                    rkh_val = sqrt_TERM1 * DDX * real(1.4) * (real(1.0) - real(1.2) * Ri) *
                              Kokkos::pow(real(1.) - real(4.) * Ri, real(4.));
                }
                else {
                    rkm_val = real(0.0);
                    rkh_val = real(0.0);
                }
            }

            // Limiter
            rkm_val = Kokkos::max(rkm_val, critmn);
            rkh_val = Kokkos::max(rkh_val, critmn);
            rkm_val = Kokkos::min(rkm_val, critmx);
            rkh_val = Kokkos::min(rkh_val, critmx);

            if (ITYPEW(k, j, i) != 1) {
                rkh_val = real(0.);
                rkm_val = real(0.);
            }

            rkm(k, j, i) = rkm_val;
            rkh(k, j, i) = rkh_val;
        });

    halo_exchanger_.exchange_halos(RKM_ref_.get(state, "RKM"));
    halo_exchanger_.exchange_halos(RKH_ref_.get(state, "RKH"));
}

template <size_t Dim>
void
TurbulenceProcess::calculate_tendencies(
    Core::State& state, const std::string& var_name, Core::Field<Dim>& out_tendency) {
    const auto& RKM = RKM_ref_.get(state, "RKM").get_device_data();
    const auto& RKH = RKH_ref_.get(state, "RKH").get_device_data();
    const auto& var = state.get_field<3>(var_name).get_device_data();
    auto& tend = out_tendency.get_mutable_device_data();
    const auto& flex_height_coef_mid = params_.flex_height_coef_mid.get_device_data();
    const auto& flex_height_coef_up = params_.flex_height_coef_up.get_device_data();
    const auto& rhobar_up = rhobar_up_ref_.get(state, "rhobar_up").get_device_data();
    const auto& rhobar = rhobar_ref_.get(state, "rhobar").get_device_data();
    const auto& hx = topo_ref_.get(state, "topo").get_device_data();
    const auto& hxv = topov_ref_.get(state, "topov").get_device_data();

    const int nz = grid_.get_local_total_points_z();
    const int ny = grid_.get_local_total_points_y();
    const int nx = grid_.get_local_total_points_x();
    const int h = grid_.get_halo_cells();

    const VVM::Real rdx2 = rdx2_;
    const VVM::Real rdy2 = rdy2_;
    const VVM::Real rdz2 = rdz2_;

    const auto masks = masks_;

    int NK2 = nz - h - 1;
    int NK1 = nz - h - 2;
    const auto& ITYPEW = ITYPEW_ref_.get(state, "ITYPEW").get_device_data();
    auto step = state.get_step();

    if constexpr (Dim == 3) {
        if (grid_.geometry().kind() != Core::Geometry::GeometryKind::Cartesian) {
            if (var_name == "xi") {
                calculate_generalized_xi_tendency(state, out_tendency);
            }
            else if (var_name == "eta") {
                calculate_generalized_eta_tendency(state, out_tendency);
            }
            else {
                calculate_generalized_scalar_tendency(state, var_name, out_tendency);
            }
            return;
        }

        if (var_name == "xi") {
            Kokkos::parallel_for("Compute_Diff_Tendency_" + var_name,
                Kokkos::MDRangePolicy<Kokkos::Rank<3>>({{h, h, h}}, {{nz - h - 1, ny - h, nx - h}}),
                KOKKOS_LAMBDA(const int k, const int j, const int i) {
                    VVM::Real d2dx2 =
                        ((RKM(k, j, i) + RKM(k, j, i + 1) + RKM(k, j + 1, i) +
                             RKM(k, j + 1, i + 1) + RKM(k + 1, j, i) + RKM(k + 1, j, i + 1) +
                             RKM(k + 1, j + 1, i) + RKM(k + 1, j + 1, i + 1)) *
                                (var(k, j, i + 1) - var(k, j, i)) * masks.val(k, j, i, VU1) -
                            (RKM(k, j, i - 1) + RKM(k, j, i) + RKM(k, j + 1, i - 1) +
                                RKM(k, j + 1, i) + RKM(k + 1, j, i - 1) + RKM(k + 1, j, i) +
                                RKM(k + 1, j + 1, i - 1) + RKM(k + 1, j + 1, i)) *
                                (var(k, j, i) - var(k, j, i - 1)) * masks.val(k, j, i, VU2)) *
                        real(0.125) * rdx2;

                    VVM::Real d2dy2 =
                        ((RKM(k, j + 1, i) + RKM(k + 1, j + 1, i)) *
                                (var(k, j + 1, i) - var(k, j, i)) * masks.val(k, j, i, VV1) -
                            (RKM(k, j, i) + RKM(k + 1, j, i)) * (var(k, j, i) - var(k, j - 1, i)) *
                                masks.val(k, j, i, VV2)) *
                        real(0.5) * rdy2;

                    VVM::Real d2dz2 =
                        (flex_height_coef_mid(k + 1) * rhobar(k + 1) *
                                (RKM(k + 1, j, i) + RKM(k + 1, j + 1, i)) *
                                (var(k + 1, j, i) - var(k, j, i)) * masks.val(k, j, i, VW1) -
                            flex_height_coef_mid(k) * rhobar(k) *
                                (RKM(k, j, i) + RKM(k, j + 1, i)) *
                                (var(k, j, i) - var(k - 1, j, i)) * masks.val(k, j, i, VW2)) *
                        real(0.5) * rdz2 / (rhobar_up(k)) * flex_height_coef_up(k);

                    tend(k, j, i) = d2dx2 + d2dy2 + d2dz2;
                });
        }
        else if (var_name == "eta") {
            Kokkos::parallel_for("Compute_Diff_Tendency_" + var_name,
                Kokkos::MDRangePolicy<Kokkos::Rank<3>>({{h, h, h}}, {{nz - h - 1, ny - h, nx - h}}),
                KOKKOS_LAMBDA(const int k, const int j, const int i) {
                    VVM::Real d2dx2 =
                        ((RKM(k, j, i + 1) + RKM(k + 1, j, i + 1)) *
                                (var(k, j, i + 1) - var(k, j, i)) * masks.val(k, j, i, UU1) -
                            (RKM(k, j, i) + RKM(k + 1, j, i)) * (var(k, j, i) - var(k, j, i - 1)) *
                                masks.val(k, j, i, UU2)) *
                        real(0.5) * rdx2;

                    VVM::Real d2dy2 =
                        ((RKM(k, j, i) + RKM(k, j, i + 1) + RKM(k, j + 1, i) +
                             RKM(k, j + 1, i + 1) + RKM(k + 1, j, i) + RKM(k + 1, j, i + 1) +
                             RKM(k + 1, j + 1, i) + RKM(k + 1, j + 1, i + 1)) *
                                (var(k, j + 1, i) - var(k, j, i)) * masks.val(k, j, i, UV1) -
                            (RKM(k, j - 1, i) + RKM(k, j - 1, i + 1) + RKM(k, j, i) +
                                RKM(k, j, i + 1) + RKM(k + 1, j - 1, i) + RKM(k + 1, j - 1, i + 1) +
                                RKM(k + 1, j, i) + RKM(k + 1, j, i + 1)) *
                                (var(k, j, i) - var(k, j - 1, i)) * masks.val(k, j, i, UV2)) *
                        real(0.125) * rdy2;

                    VVM::Real d2dz2 =
                        (flex_height_coef_mid(k + 1) * rhobar(k + 1) *
                                (RKM(k + 1, j, i) + RKM(k + 1, j, i + 1)) *
                                (var(k + 1, j, i) - var(k, j, i)) * masks.val(k, j, i, UW1) -
                            flex_height_coef_mid(k) * rhobar(k) *
                                (RKM(k, j, i) + RKM(k, j, i + 1)) *
                                (var(k, j, i) - var(k - 1, j, i)) * masks.val(k, j, i, UW2)) *
                        real(0.5) * rdz2 / rhobar_up(k) * flex_height_coef_up(k);

                    tend(k, j, i) = d2dx2 + d2dy2 + d2dz2;
                });
        }
        else {
            Kokkos::parallel_for("Compute_Diff_Tendency_" + var_name,
                Kokkos::MDRangePolicy<Kokkos::Rank<3>>({{h, h, h}}, {{nz - h, ny - h, nx - h}}),
                KOKKOS_LAMBDA(const int k, const int j, const int i) {
                    VVM::Real d2dx2 =
                        real(0.5) *
                        ((RKH(k, j, i + 1) + RKH(k, j, i)) * (var(k, j, i + 1) - var(k, j, i)) *
                                masks.val(k, j, i, WU1) -
                            (RKH(k, j, i) + RKH(k, j, i - 1)) * (var(k, j, i) - var(k, j, i - 1)) *
                                masks.val(k, j, i, WU2)) *
                        rdx2;

                    VVM::Real d2dy2 =
                        real(0.5) *
                        ((RKH(k, j + 1, i) + RKH(k, j, i)) * (var(k, j + 1, i) - var(k, j, i)) *
                                masks.val(k, j, i, WV1) -
                            (RKH(k, j, i) + RKH(k, j - 1, i)) * (var(k, j, i) - var(k, j - 1, i)) *
                                masks.val(k, j, i, WV2)) *
                        rdy2;

                    VVM::Real d2dz2 = real(0.);

                    if (k == nz - h - 1) {
                        d2dz2 = -real(0.5) * flex_height_coef_mid(NK2) *
                                (flex_height_coef_up(NK1) * rhobar_up(NK1) *
                                    (RKH(NK2, j, i) + RKH(NK1, j, i)) *
                                    (var(NK2, j, i) - var(NK1, j, i))) /
                                rhobar(NK2) * rdz2;
                    }
                    else {
                        d2dz2 =
                            real(0.5) * flex_height_coef_mid(k) *
                            (flex_height_coef_up(k) * rhobar_up(k) *
                                    (RKH(k + 1, j, i) + RKH(k, j, i)) *
                                    (var(k + 1, j, i) - var(k, j, i)) * masks.val(k, j, i, WW1) -
                                flex_height_coef_up(k - 1) * rhobar_up(k - 1) *
                                    (RKH(k, j, i) + RKH(k - 1, j, i)) *
                                    (var(k, j, i) - var(k - 1, j, i)) * masks.val(k, j, i, WW2)) /
                            rhobar(k) * rdz2;
                    }
                    tend(k, j, i) = d2dx2 + d2dy2 + d2dz2;
                });
        }
    }
    else if constexpr (Dim == 2) {
        if (var_name == "zeta") {
            if (grid_.geometry().kind() != Core::Geometry::GeometryKind::Cartesian) {
                calculate_generalized_zeta_tendency(state, out_tendency);
                return;
            }

            Kokkos::parallel_for("Compute_Diff_Tendency_" + var_name,
                Kokkos::MDRangePolicy<Kokkos::Rank<2>>({{h, h}}, {{ny - h, nx - h}}),
                KOKKOS_LAMBDA(const int j, const int i) {
                    VVM::Real d2dx2 = ((RKM(NK2, j, i + 1) + RKM(NK2, j + 1, i + 1)) *
                                              (var(NK2, j, i + 1) - var(NK2, j, i)) -
                                          (RKM(NK2, j, i) + RKM(NK2, j + 1, i)) *
                                              (var(NK2, j, i) - var(NK2, j, i - 1))) *
                                      real(0.5) * rdx2;

                    VVM::Real d2dy2 = ((RKM(NK2, j + 1, i) + RKM(NK2, j + 1, i + 1)) *
                                              (var(NK2, j + 1, i) - var(NK2, j, i)) -
                                          (RKM(NK2, j, i) + RKM(NK2, j, i + 1)) *
                                              (var(NK2, j, i) - var(NK2, j - 1, i))) *
                                      real(0.5) * rdy2;

                    VVM::Real d2dz2 =
                        -(flex_height_coef_up(NK1) * rhobar_up(NK1) *
                            (RKM(NK2, j, i) + RKM(NK2, j, i + 1) + RKM(NK2, j + 1, i) +
                                RKM(NK2, j + 1, i + 1) + RKM(NK1, j, i) + RKM(NK1, j, i + 1) +
                                RKM(NK1, j + 1, i) + RKM(NK1, j + 1, i + 1)) *
                            (var(NK2, j, i) - var(NK1, j, i))) *
                        real(0.125) * rdz2 / rhobar(NK2) * flex_height_coef_mid(NK2);

                    tend(j, i) = d2dx2 + d2dy2 + d2dz2;
                });
        }
    }
}

void
TurbulenceProcess::compute_generalized_deformation(Core::State& state) {
    using Core::Geometry::HorizontalLocation;

    const int nz = grid_.get_local_total_points_z();
    const int ny = grid_.get_local_total_points_y();
    const int nx = grid_.get_local_total_points_x();
    const int h = grid_.get_halo_cells();

    const auto& u1 = u_con_ref_.get(state, "u_con").get_device_data();
    const auto& u2 = v_con_ref_.get(state, "v_con").get_device_data();
    const auto& w = w_ref_.get(state, "w").get_device_data();

    auto& R_xi = R_xi_ref_.get(state, "R_xi").get_mutable_device_data();
    auto& R_eta = R_eta_ref_.get(state, "R_eta").get_mutable_device_data();
    auto& R_zeta = R_zeta_ref_.get(state, "R_zeta").get_mutable_device_data();

    const auto& flex_height_coef_up = params_.flex_height_coef_up.get_device_data();

    const auto geom_u = grid_.geometry().device_view(HorizontalLocation::U);
    const auto geom_v = grid_.geometry().device_view(HorizontalLocation::V);
    const auto geom_z = grid_.geometry().device_view(HorizontalLocation::Z);

    const VVM::Real rdz = rdz_;

    auto policy = Kokkos::Experimental::require(
        Kokkos::MDRangePolicy<Kokkos::Rank<3>>({{h - 1, h - 1, h - 1}}, {{nz - h, ny - h, nx - h}}),
        Kokkos::Experimental::WorkItemProperty::HintLightWeight);

    Kokkos::parallel_for("Generalized_Turbulence_Deformation",
        policy,
        KOKKOS_LAMBDA(const int k, const int j, const int i) {
            const VVM::Real vertical_scale = flex_height_coef_up(k) * rdz;

            // ----------------------------------------------------
            // R_eta:
            //
            // R_a3 =
            //     partial_a(w)
            //   + g_ab partial_z(u^b)
            //
            // evaluated at U.
            // ----------------------------------------------------

            const VVM::Real dw_dq1_u = (w(k, j, i + 1) - w(k, j, i)) * geom_u.rdq1;

            const VVM::Real dw_dq2_u =
                real(0.25) *
                (w(k, j + 1, i) + w(k, j + 1, i + 1) - w(k, j - 1, i) - w(k, j - 1, i + 1)) *
                geom_u.rdq2;

            const VVM::Real du1_dz_u = (u1(k + 1, j, i) - u1(k, j, i)) * vertical_scale;

            const VVM::Real du2_dz_u =
                real(0.25) *
                (u2(k + 1, j, i) + u2(k + 1, j, i + 1) + u2(k + 1, j - 1, i) +
                    u2(k + 1, j - 1, i + 1) - u2(k, j, i) - u2(k, j, i + 1) - u2(k, j - 1, i) -
                    u2(k, j - 1, i + 1)) *
                vertical_scale;

            const VVM::Real r13_u =
                dw_dq1_u + geom_u.g_cov.a11(j, i) * du1_dz_u + geom_u.g_cov.a12(j, i) * du2_dz_u;

            const VVM::Real r23_u =
                dw_dq2_u + geom_u.g_cov.a12(j, i) * du1_dz_u + geom_u.g_cov.a22(j, i) * du2_dz_u;

            R_eta(k, j, i) = geom_u.physical_to_contravariant.a11(j, i) * r13_u +
                             geom_u.physical_to_contravariant.a21(j, i) * r23_u;

            // ----------------------------------------------------
            // R_xi at V.
            // ----------------------------------------------------

            const VVM::Real dw_dq1_v =
                real(0.25) *
                (w(k, j, i + 1) + w(k, j + 1, i + 1) - w(k, j, i - 1) - w(k, j + 1, i - 1)) *
                geom_v.rdq1;

            const VVM::Real dw_dq2_v = (w(k, j + 1, i) - w(k, j, i)) * geom_v.rdq2;

            const VVM::Real du1_dz_v =
                real(0.25) *
                (u1(k + 1, j, i) + u1(k + 1, j + 1, i) + u1(k + 1, j, i - 1) +
                    u1(k + 1, j + 1, i - 1) - u1(k, j, i) - u1(k, j + 1, i) - u1(k, j, i - 1) -
                    u1(k, j + 1, i - 1)) *
                vertical_scale;

            const VVM::Real du2_dz_v = (u2(k + 1, j, i) - u2(k, j, i)) * vertical_scale;

            const VVM::Real r13_v =
                dw_dq1_v + geom_v.g_cov.a11(j, i) * du1_dz_v + geom_v.g_cov.a12(j, i) * du2_dz_v;

            const VVM::Real r23_v =
                dw_dq2_v + geom_v.g_cov.a12(j, i) * du1_dz_v + geom_v.g_cov.a22(j, i) * du2_dz_v;

            R_xi(k, j, i) = geom_v.physical_to_contravariant.a12(j, i) * r13_v +
                            geom_v.physical_to_contravariant.a22(j, i) * r23_v;

            // ----------------------------------------------------
            // Horizontal shear.
            //
            // No explicit RLL tan(phi)/R term is needed.
            // Metric derivatives provide the connection contribution.
            // ----------------------------------------------------

            const auto horizontal_strain = horizontal_strain_at_z(u1, u2, k, j, i, geom_z);

            R_zeta(k, j, i) = physical_horizontal_shear_12(horizontal_strain, geom_z, j, i);
        });
}

void
TurbulenceProcess::compute_generalized_coefficients(Core::State& state, VVM::Real dt) {
    using Core::Geometry::HorizontalLocation;

    const int nz = grid_.get_local_total_points_z();
    const int ny = grid_.get_local_total_points_y();
    const int nx = grid_.get_local_total_points_x();
    const int h = grid_.get_halo_cells();

    const auto& u1 = u_con_ref_.get(state, "u_con").get_device_data();
    const auto& u2 = v_con_ref_.get(state, "v_con").get_device_data();
    const auto& w = w_ref_.get(state, "w").get_device_data();

    const auto& R_xi = R_xi_ref_.get(state, "R_xi").get_device_data();
    const auto& R_eta = R_eta_ref_.get(state, "R_eta").get_device_data();

    const auto& th = th_ref_.get(state, "th").get_device_data();

    const auto& z_mid = params_.z_mid.get_device_data();
    const auto& dz_mid = params_.dz_mid.get_device_data();

    const auto& flex_height_coef_mid = params_.flex_height_coef_mid.get_device_data();
    const auto& flex_height_coef_up = params_.flex_height_coef_up.get_device_data();

    const auto& ITYPEW = ITYPEW_ref_.get(state, "ITYPEW").get_device_data();

    auto& rkm = RKM_ref_.get(state, "RKM").get_mutable_device_data();
    auto& rkh = RKH_ref_.get(state, "RKH").get_mutable_device_data();

    const auto geom_t = grid_.geometry().device_view(HorizontalLocation::T);

    const VVM::Real rdz = rdz_;
    const VVM::Real grav = grav_;
    const VVM::Real vk = vk_;
    const VVM::Real critmn = critmn_;

    const auto masks = masks_;

    auto policy = Kokkos::Experimental::require(
        Kokkos::MDRangePolicy<Kokkos::Rank<3>>({{h, h, h}}, {{nz - h, ny - h, nx - h}}),
        Kokkos::Experimental::WorkItemProperty::HintLightWeight);

    Kokkos::parallel_for("Generalized_ShuttsGray_Coeffs",
        policy,
        KOKKOS_LAMBDA(const int k, const int j, const int i) {
            const auto horizontal_strain = horizontal_strain_at_t(u1, u2, k, j, i, geom_t);

            const VVM::Real horizontal_term =
                real(0.5) *
                physical_horizontal_strain_norm_squared(horizontal_strain, geom_t, j, i);

            const VVM::Real vertical_shear_term =
                real(0.25) *
                (R_eta(k, j, i - 1) * R_eta(k, j, i - 1) + R_eta(k, j, i) * R_eta(k, j, i) +
                    R_eta(k - 1, j, i - 1) * R_eta(k - 1, j, i - 1) +
                    R_eta(k - 1, j, i) * R_eta(k - 1, j, i) +
                    R_xi(k, j - 1, i) * R_xi(k, j - 1, i) + R_xi(k, j, i) * R_xi(k, j, i) +
                    R_xi(k - 1, j - 1, i) * R_xi(k - 1, j - 1, i) +
                    R_xi(k - 1, j, i) * R_xi(k - 1, j, i));

            const VVM::Real dw_dz = flex_height_coef_mid(k) * (w(k, j, i) - w(k - 1, j, i)) * rdz;

            VVM::Real TERM1 = horizontal_term + vertical_shear_term + real(2.0) * dw_dz * dw_dz;
            TERM1 = Kokkos::max(TERM1, real(0.0));

            if (ITYPEW(k, j, i) != 1) {
                TERM1 = real(0.0);
            }

            const VVM::Real DDY_top = grav * flex_height_coef_up(k) *
                                      (th(k + 1, j, i) - th(k, j, i)) * rdz /
                                      (th(k + 1, j, i) + th(k, j, i)) * masks.val(k, j, i, WW1);

            const VVM::Real DDY_bot = grav * flex_height_coef_up(k - 1) *
                                      (th(k, j, i) - th(k - 1, j, i)) * rdz /
                                      (th(k, j, i) + th(k - 1, j, i)) * masks.val(k, j, i, WW2);

            const VVM::Real DDY = DDY_top + DDY_bot;

            // Physical horizontal area:
            //
            // dA = sqrt(g) dq1 dq2
            //
            // This remains valid when g12 != 0.
            const VVM::Real horizontal_cell_area = geom_t.sqrt_g(j, i) * geom_t.dq1 * geom_t.dq2;

            const VVM::Real deld =
                Kokkos::pow(horizontal_cell_area * dz_mid(k), real(1.0) / real(3.0));

            const VVM::Real ramd0s = Kokkos::pow(real(0.23) * deld, real(2.0));

            const VVM::Real critmx = real(0.8) * deld * deld / dt;

            const VVM::Real z = z_mid(k);

            const VVM::Real ZROUGH = real(2e-4);

            const VVM::Real vk2z2 = vk * vk * Kokkos::pow(z + ZROUGH, real(2.0));

            const VVM::Real DDX = ramd0s * vk2z2 / (ramd0s + vk2z2);

            VVM::Real rkm_val = real(0.0);

            VVM::Real rkh_val = real(0.0);

            if (TERM1 > real(0.0)) {
                const VVM::Real Ri = DDY / TERM1;

                const VVM::Real sqrt_TERM1 = Kokkos::sqrt(TERM1);

                if (Ri < real(0.0)) {
                    rkm_val = sqrt_TERM1 * DDX * Kokkos::sqrt(real(1.0) - real(16.0) * Ri);
                    rkh_val =
                        sqrt_TERM1 * DDX * real(1.4) * Kokkos::sqrt(real(1.0) - real(40.0) * Ri);
                }
                else if (Ri < real(0.25)) {
                    const VVM::Real f = Kokkos::pow(real(1.0) - real(4.0) * Ri, real(4.0));
                    rkm_val = sqrt_TERM1 * DDX * f;
                    rkh_val = sqrt_TERM1 * DDX * real(1.4) * (real(1.0) - real(1.2) * Ri) * f;
                }
            }

            rkm_val = Kokkos::max(rkm_val, critmn);
            rkh_val = Kokkos::max(rkh_val, critmn);

            rkm_val = Kokkos::min(rkm_val, critmx);
            rkh_val = Kokkos::min(rkh_val, critmx);

            if (ITYPEW(k, j, i) != 1) {
                rkm_val = real(0.0);
                rkh_val = real(0.0);
            }

            rkm(k, j, i) = rkm_val;
            rkh(k, j, i) = rkh_val;
        });

    halo_exchanger_.exchange_halos(RKM_ref_.get(state, "RKM"));
    halo_exchanger_.exchange_halos(RKH_ref_.get(state, "RKH"));
}

void
TurbulenceProcess::calculate_generalized_scalar_tendency(
    Core::State& state, const std::string& var_name, Core::Field<3>& out_tendency) {
    using Dynamics::Operators::load_scalar_stencil_at_t;
    using Dynamics::Operators::make_horizontal_flux_divergence_device_view;
    using Dynamics::Operators::make_horizontal_scalar_gradient_device_view;

    const auto& RKH = RKH_ref_.get(state, "RKH").get_device_data();
    const auto& var = state.get_field<3>(var_name).get_device_data();
    auto& tend = out_tendency.get_mutable_device_data();

    const auto& flex_height_coef_mid = params_.flex_height_coef_mid.get_device_data();
    const auto& flex_height_coef_up = params_.flex_height_coef_up.get_device_data();
    const auto& rhobar_up = rhobar_up_ref_.get(state, "rhobar_up").get_device_data();
    const auto& rhobar = rhobar_ref_.get(state, "rhobar").get_device_data();

    const int nz = grid_.get_local_total_points_z();
    const int ny = grid_.get_local_total_points_y();
    const int nx = grid_.get_local_total_points_x();
    const int h = grid_.get_halo_cells();

    const int NK2 = nz - h - 1;
    const int NK1 = nz - h - 2;
    const VVM::Real rdz2 = rdz2_;

    const auto masks = masks_;

    const auto gradient = make_horizontal_scalar_gradient_device_view(grid_.geometry());

    const auto divergence = make_horizontal_flux_divergence_device_view(grid_.geometry());

    auto policy = Kokkos::Experimental::require(
        Kokkos::MDRangePolicy<Kokkos::Rank<3>>({{h, h, h}}, {{nz - h, ny - h, nx - h}}),
        Kokkos::Experimental::WorkItemProperty::HintLightWeight);

    Kokkos::parallel_for("Generalized_Compute_Diff_Tendency_" + var_name,
        policy,
        KOKKOS_LAMBDA(const int k, const int j, const int i) {
            const auto grad =
                gradient.calculate_uv_around_t(j, i, load_scalar_stencil_at_t(var, k, j, i));

            const VVM::Real kh_e =
                real(0.5) * (RKH(k, j, i + 1) + RKH(k, j, i)) * masks.val(k, j, i, WU1);

            const VVM::Real kh_w =
                real(0.5) * (RKH(k, j, i) + RKH(k, j, i - 1)) * masks.val(k, j, i, WU2);

            const VVM::Real kh_n =
                real(0.5) * (RKH(k, j + 1, i) + RKH(k, j, i)) * masks.val(k, j, i, WV1);

            const VVM::Real kh_s =
                real(0.5) * (RKH(k, j, i) + RKH(k, j - 1, i)) * masks.val(k, j, i, WV2);

            const VVM::Real horizontal = divergence.at_t(j,
                i,
                kh_e * grad.q1_at_u_j_i,
                kh_w * grad.q1_at_u_j_im1,
                kh_n * grad.q2_at_v_j_i,
                kh_s * grad.q2_at_v_jm1_i);

            VVM::Real vertical = real(0.0);

            if (k == NK2) {
                vertical =
                    -real(0.5) * flex_height_coef_mid(NK2) *
                    (flex_height_coef_up(NK1) * rhobar_up(NK1) * (RKH(NK2, j, i) + RKH(NK1, j, i)) *
                        (var(NK2, j, i) - var(NK1, j, i))) /
                    rhobar(NK2) * rdz2;
            }
            else {
                vertical =
                    real(0.5) * flex_height_coef_mid(k) *
                    (flex_height_coef_up(k) * rhobar_up(k) * (RKH(k + 1, j, i) + RKH(k, j, i)) *
                            (var(k + 1, j, i) - var(k, j, i)) * masks.val(k, j, i, WW1) -
                        flex_height_coef_up(k - 1) * rhobar_up(k - 1) *
                            (RKH(k, j, i) + RKH(k - 1, j, i)) * (var(k, j, i) - var(k - 1, j, i)) *
                            masks.val(k, j, i, WW2)) /
                    rhobar(k) * rdz2;
            }

            tend(k, j, i) = horizontal + vertical;
        });
}

void
TurbulenceProcess::calculate_generalized_zeta_tendency(Core::State& state,
    Core::Field<2>& out_tendency) {

    using Dynamics::Operators::load_scalar_stencil_at_z;
    using Dynamics::Operators::make_horizontal_flux_divergence_device_view;
    using Dynamics::Operators::make_horizontal_scalar_gradient_device_view;

    const auto& RKM = RKM_ref_.get(state, "RKM").get_device_data();
    const auto& zeta = state.get_field<3>("zeta").get_device_data();

    auto& tend = out_tendency.get_mutable_device_data();

    const auto& flex_height_coef_mid = params_.flex_height_coef_mid.get_device_data();
    const auto& flex_height_coef_up = params_.flex_height_coef_up.get_device_data();
    const auto& rhobar_up = rhobar_up_ref_.get(state, "rhobar_up").get_device_data();
    const auto& rhobar = rhobar_ref_.get(state, "rhobar").get_device_data();

    const int nz = grid_.get_local_total_points_z();
    const int ny = grid_.get_local_total_points_y();
    const int nx = grid_.get_local_total_points_x();
    const int h = grid_.get_halo_cells();

    const int NK2 = nz - h - 1;
    const int NK1 = nz - h - 2;

    const VVM::Real rdz2 = rdz2_;

    const auto gradient = make_horizontal_scalar_gradient_device_view(grid_.geometry());

    const auto divergence = make_horizontal_flux_divergence_device_view(grid_.geometry());

    auto policy = Kokkos::Experimental::require(
        Kokkos::MDRangePolicy<Kokkos::Rank<2>>({{h, h}}, {{ny - h, nx - h}}),
        Kokkos::Experimental::WorkItemProperty::HintLightWeight);

    Kokkos::parallel_for("Generalized_Turbulence_Zeta",
        policy,
        KOKKOS_LAMBDA(const int j, const int i) {
            const auto grad =
                gradient.calculate_vu_around_z(j, i, load_scalar_stencil_at_z(zeta, NK2, j, i));

            const VVM::Real km_e = real(0.5) * (RKM(NK2, j, i + 1) + RKM(NK2, j + 1, i + 1));

            const VVM::Real km_w = real(0.5) * (RKM(NK2, j, i) + RKM(NK2, j + 1, i));

            const VVM::Real km_n = real(0.5) * (RKM(NK2, j + 1, i) + RKM(NK2, j + 1, i + 1));

            const VVM::Real km_s = real(0.5) * (RKM(NK2, j, i) + RKM(NK2, j, i + 1));

            const VVM::Real horizontal = divergence.at_z(j,
                i,
                km_e * grad.q1_at_v_j_ip1,
                km_w * grad.q1_at_v_j_i,
                km_n * grad.q2_at_u_jp1_i,
                km_s * grad.q2_at_u_j_i);

            const VVM::Real km_vertical =
                (RKM(NK2, j, i) + RKM(NK2, j, i + 1) + RKM(NK2, j + 1, i) + RKM(NK2, j + 1, i + 1) +
                    RKM(NK1, j, i) + RKM(NK1, j, i + 1) + RKM(NK1, j + 1, i) +
                    RKM(NK1, j + 1, i + 1)) *
                real(0.125);

            const VVM::Real vertical = -flex_height_coef_up(NK1) * rhobar_up(NK1) *
                                       flex_height_coef_mid(NK2) * km_vertical *
                                       (zeta(NK2, j, i) - zeta(NK1, j, i)) * rdz2 / rhobar(NK2);

            tend(j, i) = horizontal + vertical;
        });
}

void
TurbulenceProcess::calculate_generalized_xi_tendency(Core::State& state,
    Core::Field<3>& out_tendency) {
    using Dynamics::Operators::load_scalar_stencil_at_v;
    using Dynamics::Operators::make_horizontal_flux_divergence_device_view;
    using Dynamics::Operators::make_horizontal_scalar_gradient_at_uv_device_view;

    const auto& RKM = RKM_ref_.get(state, "RKM").get_device_data();

    const auto& xi = state.get_field<3>("xi").get_device_data();
    auto& tend = out_tendency.get_mutable_device_data();

    const auto& flex_height_coef_mid = params_.flex_height_coef_mid.get_device_data();
    const auto& flex_height_coef_up = params_.flex_height_coef_up.get_device_data();

    const auto& rhobar = rhobar_ref_.get(state, "rhobar").get_device_data();
    const auto& rhobar_up = rhobar_up_ref_.get(state, "rhobar_up").get_device_data();

    const int nz = grid_.get_local_total_points_z();
    const int ny = grid_.get_local_total_points_y();
    const int nx = grid_.get_local_total_points_x();
    const int h = grid_.get_halo_cells();

    const VVM::Real rdz2 = rdz2_;
    const auto masks = masks_;

    const auto gradient = make_horizontal_scalar_gradient_at_uv_device_view(grid_.geometry());

    const auto divergence = make_horizontal_flux_divergence_device_view(grid_.geometry());

    auto policy = Kokkos::Experimental::require(
        Kokkos::MDRangePolicy<Kokkos::Rank<3>>({{h, h, h}}, {{nz - h - 1, ny - h, nx - h}}),
        Kokkos::Experimental::WorkItemProperty::HintLightWeight);

    Kokkos::parallel_for("Generalized_Compute_Diff_Tendency_xi",
        policy,
        KOKKOS_LAMBDA(const int k, const int j, const int i) {
            const auto grad =
                gradient.calculate_zt_around_v(j, i, load_scalar_stencil_at_v(xi, k, j, i));

            const VVM::Real km_e =
                real(0.125) * (RKM(k, j, i) + RKM(k, j, i + 1) + RKM(k, j + 1, i) +
                                  RKM(k, j + 1, i + 1) + RKM(k + 1, j, i) + RKM(k + 1, j, i + 1) +
                                  RKM(k + 1, j + 1, i) + RKM(k + 1, j + 1, i + 1));

            const VVM::Real km_w =
                real(0.125) * (RKM(k, j, i - 1) + RKM(k, j, i) + RKM(k, j + 1, i - 1) +
                                  RKM(k, j + 1, i) + RKM(k + 1, j, i - 1) + RKM(k + 1, j, i) +
                                  RKM(k + 1, j + 1, i - 1) + RKM(k + 1, j + 1, i));

            const VVM::Real km_n = real(0.5) * (RKM(k, j + 1, i) + RKM(k + 1, j + 1, i));

            const VVM::Real km_s = real(0.5) * (RKM(k, j, i) + RKM(k + 1, j, i));

            const VVM::Real horizontal = divergence.at_v(j,
                i,
                km_e * grad.q1_at_z_j_i * masks.val(k, j, i, VU1),
                km_w * grad.q1_at_z_j_im1 * masks.val(k, j, i, VU2),
                km_n * grad.q2_at_t_jp1_i * masks.val(k, j, i, VV1),
                km_s * grad.q2_at_t_j_i * masks.val(k, j, i, VV2));

            const VVM::Real vertical =
                real(0.5) * flex_height_coef_up(k) *
                (flex_height_coef_mid(k + 1) * rhobar(k + 1) *
                        (RKM(k + 1, j, i) + RKM(k + 1, j + 1, i)) *
                        (xi(k + 1, j, i) - xi(k, j, i)) * masks.val(k, j, i, VW1) -
                    flex_height_coef_mid(k) * rhobar(k) * (RKM(k, j, i) + RKM(k, j + 1, i)) *
                        (xi(k, j, i) - xi(k - 1, j, i)) * masks.val(k, j, i, VW2)) *
                rdz2 / rhobar_up(k);

            tend(k, j, i) = horizontal + vertical;
        });
}

void
TurbulenceProcess::calculate_generalized_eta_tendency(Core::State& state,
    Core::Field<3>& out_tendency) {
    using Dynamics::Operators::load_scalar_stencil_at_u;
    using Dynamics::Operators::make_horizontal_flux_divergence_device_view;
    using Dynamics::Operators::make_horizontal_scalar_gradient_at_uv_device_view;

    const auto& RKM = RKM_ref_.get(state, "RKM").get_device_data();

    const auto& eta = state.get_field<3>("eta").get_device_data();
    auto& tend = out_tendency.get_mutable_device_data();

    const auto& flex_height_coef_mid = params_.flex_height_coef_mid.get_device_data();
    const auto& flex_height_coef_up = params_.flex_height_coef_up.get_device_data();
    const auto& rhobar = rhobar_ref_.get(state, "rhobar").get_device_data();
    const auto& rhobar_up = rhobar_up_ref_.get(state, "rhobar_up").get_device_data();

    const int nz = grid_.get_local_total_points_z();
    const int ny = grid_.get_local_total_points_y();
    const int nx = grid_.get_local_total_points_x();
    const int h = grid_.get_halo_cells();

    const VVM::Real rdz2 = rdz2_;

    const auto masks = masks_;

    const auto gradient = make_horizontal_scalar_gradient_at_uv_device_view(grid_.geometry());

    const auto divergence = make_horizontal_flux_divergence_device_view(grid_.geometry());

    auto policy = Kokkos::Experimental::require(
        Kokkos::MDRangePolicy<Kokkos::Rank<3>>({{h, h, h}}, {{nz - h - 1, ny - h, nx - h}}),
        Kokkos::Experimental::WorkItemProperty::HintLightWeight);

    Kokkos::parallel_for("Generalized_Compute_Diff_Tendency_eta",
        policy,
        KOKKOS_LAMBDA(const int k, const int j, const int i) {
            const auto grad =
                gradient.calculate_tz_around_u(j, i, load_scalar_stencil_at_u(eta, k, j, i));

            const VVM::Real km_e = real(0.5) * (RKM(k, j, i + 1) + RKM(k + 1, j, i + 1));

            const VVM::Real km_w = real(0.5) * (RKM(k, j, i) + RKM(k + 1, j, i));

            const VVM::Real km_n =
                real(0.125) * (RKM(k, j, i) + RKM(k, j + 1, i) + RKM(k, j, i + 1) +
                                  RKM(k, j + 1, i + 1) + RKM(k + 1, j, i) + RKM(k + 1, j + 1, i) +
                                  RKM(k + 1, j, i + 1) + RKM(k + 1, j + 1, i + 1));

            const VVM::Real km_s =
                real(0.125) * (RKM(k, j - 1, i) + RKM(k, j, i) + RKM(k, j - 1, i + 1) +
                                  RKM(k, j, i + 1) + RKM(k + 1, j - 1, i) + RKM(k + 1, j, i) +
                                  RKM(k + 1, j - 1, i + 1) + RKM(k + 1, j, i + 1));

            const VVM::Real horizontal = divergence.at_u(j,
                i,
                km_e * grad.q1_at_t_j_ip1 * masks.val(k, j, i, UU1),
                km_w * grad.q1_at_t_j_i * masks.val(k, j, i, UU2),
                km_n * grad.q2_at_z_j_i * masks.val(k, j, i, UV1),
                km_s * grad.q2_at_z_jm1_i * masks.val(k, j, i, UV2));

            const VVM::Real vertical =
                real(0.5) * flex_height_coef_up(k) *
                (flex_height_coef_mid(k + 1) * rhobar(k + 1) *
                        (RKM(k + 1, j, i) + RKM(k + 1, j, i + 1)) *
                        (eta(k + 1, j, i) - eta(k, j, i)) * masks.val(k, j, i, UW1) -
                    flex_height_coef_mid(k) * rhobar(k) * (RKM(k, j, i) + RKM(k, j, i + 1)) *
                        (eta(k, j, i) - eta(k - 1, j, i)) * masks.val(k, j, i, UW2)) *
                rdz2 / rhobar_up(k);

            tend(k, j, i) = horizontal + vertical;
        });
}

template void TurbulenceProcess::calculate_tendencies(
    Core::State& state, const std::string& var_name, Core::Field<2ul>& out_tendency);
template void TurbulenceProcess::calculate_tendencies(
    Core::State& state, const std::string& var_name, Core::Field<3ul>& out_tendency);

} // namespace Physics
} // namespace VVM
