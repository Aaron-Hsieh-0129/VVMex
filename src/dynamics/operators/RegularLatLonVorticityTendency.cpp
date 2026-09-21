#include "dynamics/operators/RegularLatLonVorticityTendency.hpp"

namespace VVM::Dynamics::Operators {
namespace {

using Volume = Core::Field<3>::ViewType;
using Profile = Core::Field<1>::ViewType;
using Plane = Core::Field<2>::ViewType;
using Term = RegularLatLonVorticityTendency::Term;
using Policy = Kokkos::MDRangePolicy<Kokkos::Rank<3>>;

struct ProductProfile {
    Profile first, second;
    int offset = 0;

    KOKKOS_INLINE_FUNCTION Real
    operator()(int k) const {
        return first(k) * second(k + offset);
    }
};

struct InverseSpacing {
    Profile flex;
    Kokkos::View<Real> inverse_dz;

    KOKKOS_INLINE_FUNCTION Real
    operator()(int k) const {
        return flex(k) * inverse_dz();
    }
};

// Preserve the accepted horizontal-transport input/launch contract.
struct CanonicalHorizontalTransportFields {
    Volume u1, u2, w, omega1, negative_omega2;
    Profile rho_up;
    ProductProfile fn1, fn2;
    InverseSpacing inverse_spacing;

    KOKKOS_INLINE_FUNCTION Real
    omega1_over_rho(int k, int j, int i) const noexcept {
        return omega1(k, j, i) / rho_up(k);
    }

    KOKKOS_INLINE_FUNCTION Real
    omega2_over_rho(int k, int j, int i) const noexcept {
        return -negative_omega2(k, j, i) / rho_up(k);
    }
};

CanonicalHorizontalTransportFields
bind_horizontal_transport_fields(const Core::State& state, const Core::Parameters& params) {
    CanonicalHorizontalTransportFields fields{};

    fields.u1 = state.get_field<3>("u_con").get_device_data();
    fields.u2 = state.get_field<3>("v_con").get_device_data();
    fields.w = state.get_field<3>("w").get_device_data();

    fields.omega1 = state.get_field<3>("xi_con").get_device_data();
    fields.negative_omega2 = state.get_field<3>("eta_con").get_device_data();
    fields.rho_up = state.get_field<1>("rhobar_up").get_device_data();

    const auto rho = state.get_field<1>("rhobar").get_device_data();
    fields.fn1 = {params.fact1_xi_eta.get_device_data(), rho, 1};
    fields.fn2 = {params.fact2_xi_eta.get_device_data(), rho, 0};

    fields.inverse_spacing = {params.flex_height_coef_up.get_device_data(), params.rdz};

    return fields;
}

struct HorizontalTransportFunctor {
    Kokkos::View<const GeneralizedHorizontalVorticityTransportDeviceView> transport;
    CanonicalHorizontalTransportFields fields;
    Core::Geometry::GeometryField2D physical_scale;
    Volume output;

    bool first_component = true;
    int begin = 0;
    int end = 0;
    bool evaluate = true;

    KOKKOS_INLINE_FUNCTION void
    operator()(int k, int j, int i) const {
        if (!evaluate) {
            return;
        }

        const auto t =
            first_component
                ? transport().calculate_omega1_at_v(fields, fields.w, k, j, i, begin, end)
                : transport().calculate_omega2_at_u(fields, fields.w, k, j, i, begin, end);

        const Real scale = first_component ? physical_scale(j, i) : -physical_scale(j, i);

        output(k, j, i) += scale * (t.q1 + t.q2 + t.vertical);
    }
};

// State names and the VVM eta sign are resolved only at this boundary.
// All normalization is lazy and read-only; zeta is divided by rho ONCE.
struct CanonicalVorticityFields {
    Volume omega1, negative_omega2, omega3;
    Plane f_at_z;
    Profile rho, rho_up;

    KOKKOS_INLINE_FUNCTION Real
    omega1_over_rho(int k, int j, int i) const noexcept {
        return omega1(k, j, i) / rho_up(k);
    }

    KOKKOS_INLINE_FUNCTION Real
    omega2_over_rho(int k, int j, int i) const noexcept {
        return -negative_omega2(k, j, i) / rho_up(k);
    }

    KOKKOS_INLINE_FUNCTION Real
    omega3_over_rho(int k, int j, int i) const noexcept {
        return omega3(k, j, i) / rho(k);
    }

    KOKKOS_INLINE_FUNCTION Real
    f3_at_z(int j, int i) const noexcept {
        return f_at_z(j, i);
    }
};

CanonicalVorticityFields
bind_vorticity_fields(const Core::State& state) {
    CanonicalVorticityFields fields{};

    fields.omega1 = state.get_field<3>("xi_con").get_device_data();
    fields.negative_omega2 = state.get_field<3>("eta_con").get_device_data();
    fields.omega3 = state.get_field<3>("zeta").get_device_data();

    fields.f_at_z = state.get_field<2>("f_2d").get_device_data();
    fields.rho = state.get_field<1>("rhobar").get_device_data();
    fields.rho_up = state.get_field<1>("rhobar_up").get_device_data();

    return fields;
}

struct CanonicalHorizontalDeformationFields : CanonicalVorticityFields {
    Volume u1, u2;
    ProductProfile fn1, fn2;
    InverseSpacing inverse_spacing;
};

CanonicalHorizontalDeformationFields
bind_horizontal_deformation_fields(const Core::State& state, const Core::Parameters& params) {
    CanonicalHorizontalDeformationFields fields{};
    static_cast<CanonicalVorticityFields&>(fields) = bind_vorticity_fields(state);

    fields.u1 = state.get_field<3>("u_con").get_device_data();
    fields.u2 = state.get_field<3>("v_con").get_device_data();

    fields.fn1 = {params.fact1_xi_eta.get_device_data(), fields.rho, 1};
    fields.fn2 = {params.fact2_xi_eta.get_device_data(), fields.rho, 0};

    fields.inverse_spacing = {params.flex_height_coef_up.get_device_data(), params.rdz};

    return fields;
}

// Generalized top deformation consumes the canonical accessors. The
// remaining legacy top transport reads physical u/v and normalized zeta.
struct TopFields : CanonicalVorticityFields {
    Volume u, v, w;
    InverseSpacing inverse_spacing_mid, inverse_spacing_up;

    KOKKOS_INLINE_FUNCTION Real
    zeta(int k, int j, int i) const noexcept {
        return omega3_over_rho(k, j, i);
    }
};

TopFields
bind_top_fields(const Core::State& state, const Core::Parameters& params) {
    TopFields fields{};
    static_cast<CanonicalVorticityFields&>(fields) = bind_vorticity_fields(state);

    fields.u = state.get_field<3>("u").get_device_data();
    fields.v = state.get_field<3>("v").get_device_data();
    fields.w = state.get_field<3>("w").get_device_data();

    fields.inverse_spacing_mid = {params.flex_height_coef_mid.get_device_data(), params.rdz};
    fields.inverse_spacing_up = {params.flex_height_coef_up.get_device_data(), params.rdz};

    return fields;
}

struct HorizontalDeformationFunctor {
    GeneralizedHorizontalDeformationDeviceView deformation;
    CanonicalHorizontalDeformationFields fields;
    Core::Geometry::GeometryField2D physical_scale;
    Volume output;

    bool first_component = true;
    Term term = Term::Stretching;
    bool evaluate = true;

    KOKKOS_INLINE_FUNCTION void
    operator()(int k, int j, int i) const {
        if (!evaluate) {
            return;
        }

        const auto t = first_component ? deformation.calculate_omega1_at_v(fields, k, j, i)
                                       : deformation.calculate_omega2_at_u(fields, k, j, i);

        const Real value = term == Term::Stretching ? t.stretching
                           : term == Term::Twisting ? t.twisting
                                                    : t.planetary;

        const Real scale = first_component ? physical_scale(j, i) : -physical_scale(j, i);

        output(k, j, i) += scale * value;
    }
};

struct TopTendencyFunctor {
    Kokkos::View<const RegularLatLonTopTransportDeviceView> transport;
    GeneralizedTopDeformationDeviceView deformation;
    TopFields fields;
    Volume output;

    Term term = Term::Transport;
    bool evaluate = true;

    KOKKOS_INLINE_FUNCTION void
    operator()(int k, int j, int i) const {
        if (!evaluate) {
            return;
        }

        Real value = real(0.0);

        if (term == Term::Transport || term == Term::Planetary) {
            const auto t = transport().calculate_at_z(fields, k, j, i, term == Term::Planetary);

            value = t.q1 + t.q2 + t.vertical;
        }

        if (term != Term::Transport) {
            const auto t = deformation.calculate_at_z(fields, k, j, i);

            value += term == Term::Stretching ? t.stretching
                     : term == Term::Twisting ? t.twisting
                                              : t.planetary;
        }

        // omega^3 is physical vertical vorticity for this mapping.
        // Planetary transport and stretching have each been added once.
        output(k, j, i) += value;
    }
};

} // namespace

RegularLatLonVorticityTendency::RegularLatLonVorticityTendency(
    const Core::Geometry::HorizontalGeometry& geometry)
    : horizontal_transport_("Generalized horizontal vorticity transport"),
      transport_("RLL top vorticity transport"),
      horizontal_deformation_(make_generalized_horizontal_deformation_device_view(geometry)),
      top_deformation_(make_generalized_top_deformation_device_view(geometry)) {
    auto ht = Kokkos::create_mirror_view(horizontal_transport_);
    ht() = make_generalized_horizontal_vorticity_transport_device_view(geometry);
    Kokkos::deep_copy(horizontal_transport_, ht);

    auto t = Kokkos::create_mirror_view(transport_);
    t() = make_regular_lat_lon_top_transport_device_view(geometry);
    Kokkos::deep_copy(transport_, t);

    prepare_execution();
}

void
RegularLatLonVorticityTendency::prepare_execution() {
    // Prepare every actual launch type before CUDA graph capture.
    // evaluate=false prevents access to the empty preparation views.
    HorizontalTransportFunctor transport_preparation{};
    transport_preparation.evaluate = false;

    Kokkos::parallel_for("PrepareGeneralizedHorizontalVorticityTransport",
        Kokkos::Experimental::require(Policy({0, 0, 0}, {1, 1, 1}),
            Kokkos::Experimental::WorkItemProperty::HintLightWeight),
        transport_preparation);

    HorizontalDeformationFunctor deformation_preparation{};
    deformation_preparation.evaluate = false;

    Kokkos::parallel_for("PrepareGeneralizedHorizontalDeformation",
        Kokkos::Experimental::require(Policy({0, 0, 0}, {1, 1, 1}),
            Kokkos::Experimental::WorkItemProperty::HintLightWeight),
        deformation_preparation);

    TopTendencyFunctor top_preparation{};
    top_preparation.evaluate = false;

    Kokkos::parallel_for("PrepareRLLTopVorticityTendency",
        Kokkos::Experimental::require(Policy({0, 0, 0}, {1, 1, 1}),
            Kokkos::Experimental::WorkItemProperty::HintLightWeight),
        top_preparation);

    Kokkos::fence();
}

void
RegularLatLonVorticityTendency::add_from_canonical_state(const Core::State& state,
    const Core::Grid& grid,
    const Core::Parameters& params,
    Core::Field<3>& output,
    const std::string& variable,
    Term term) const {
    const int component = variable == "xi"     ? 0
                          : variable == "eta"  ? 1
                          : variable == "zeta" ? 2
                                               : -1;

    if (component < 0 || grid.geometry().kind() != Core::Geometry::GeometryKind::RegularLatLon) {
        throw std::invalid_argument(
            "RLL vorticity tendencies require xi, eta or zeta on RLL geometry.");
    }

    const int h = grid.get_halo_cells();
    const int top = grid.get_local_total_points_z() - h - 1;
    const bool terrain = state.has_field("rll_terrain_height");

    if (h < 2 || top - h < 2 || (params.max_topo_idx != h && !terrain) ||
        (terrain && (params.max_topo_idx < h || params.max_topo_idx >= top))) {
        throw std::invalid_argument(
            "RLL vorticity tendencies require flat or initialized RLL mountain terrain below the "
            "lid, two halos and at least three wind levels.");
    }

    if (component != 2) {
        using Core::Geometry::HorizontalLocation;

        const auto location = component == 0 ? HorizontalLocation::V : HorizontalLocation::U;
        const auto geometry = grid.geometry().device_view(location);

        const auto scale = component == 0 ? geometry.contravariant_to_physical.a11
                                          : geometry.contravariant_to_physical.a22;

        const auto policy = Kokkos::Experimental::require(
            Policy({h, h, h},
                {top, grid.get_local_total_points_y() - h, grid.get_local_total_points_x() - h}),
            Kokkos::Experimental::WorkItemProperty::HintLightWeight);

        if (term == Term::Transport) {
            Kokkos::parallel_for("GeneralizedHorizontalVorticityTransport",
                policy,
                HorizontalTransportFunctor{horizontal_transport_,
                    bind_horizontal_transport_fields(state, params),
                    scale,
                    output.get_mutable_device_data(),
                    component == 0,
                    h,
                    top});
        }
        else {
            Kokkos::parallel_for("GeneralizedHorizontalDeformation",
                policy,
                HorizontalDeformationFunctor{horizontal_deformation_,
                    bind_horizontal_deformation_fields(state, params),
                    scale,
                    output.get_mutable_device_data(),
                    component == 0,
                    term});
        }

        return;
    }

    Kokkos::parallel_for("RLLTopVorticityTendency",
        Kokkos::Experimental::require(Policy({top, h, h},
                                          {top + 1,
                                              grid.get_local_total_points_y() - h,
                                              grid.get_local_total_points_x() - h}),
            Kokkos::Experimental::WorkItemProperty::HintLightWeight),
        TopTendencyFunctor{transport_,
            top_deformation_,
            bind_top_fields(state, params),
            output.get_mutable_device_data(),
            term});
}

} // namespace VVM::Dynamics::Operators
