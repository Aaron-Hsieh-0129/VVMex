#include "dynamics/operators/GeneralizedVorticityTendency.hpp"

#include <cstddef>
#include <stdexcept>

namespace VVM::Dynamics::Operators {
namespace {

using Volume = Core::Field<3>::ViewType;
using Profile = Core::Field<1>::ViewType;
using Plane = Core::Field<2>::ViewType;
using Weight = Core::Geometry::GeometryField2D;
using Term = GeneralizedVorticityTendency::Term;
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

// Bind State names and the VVM eta sign at the orchestration boundary.
// Numerical kernels receive actual omega^1/omega^2 components over density.
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

struct TopFields : CanonicalVorticityFields {
    Volume u1, u2, w;
    InverseSpacing inverse_spacing_mid, inverse_spacing_up;
};

TopFields
bind_top_fields(const Core::State& state, const Core::Parameters& params) {
    TopFields fields{};
    static_cast<CanonicalVorticityFields&>(fields) = bind_vorticity_fields(state);

    fields.u1 = state.get_field<3>("u_con").get_device_data();
    fields.u2 = state.get_field<3>("v_con").get_device_data();
    fields.w = state.get_field<3>("w").get_device_data();

    fields.inverse_spacing_mid = {params.flex_height_coef_mid.get_device_data(), params.rdz};
    fields.inverse_spacing_up = {params.flex_height_coef_up.get_device_data(), params.rdz};

    return fields;
}

struct HorizontalTransportFunctor {
    Kokkos::View<const GeneralizedHorizontalVorticityTransportDeviceView> transport;
    CanonicalHorizontalTransportFields fields;
    Weight weight;
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

        // Convert the true omega^2 tendency to VVM's eta_con sign ONCE.
        const Real scale = first_component ? weight(j, i) : -weight(j, i);

        output(k, j, i) += scale * (t.q1 + t.q2 + t.vertical);
    }
};

struct HorizontalDeformationFunctor {
    GeneralizedHorizontalDeformationDeviceView deformation;
    CanonicalHorizontalDeformationFields fields;
    Weight weight;
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

        const Real scale = first_component ? weight(j, i) : -weight(j, i);

        output(k, j, i) += scale * value;
    }
};

struct TopTendencyFunctor {
    Kokkos::View<const GeneralizedTopTransportDeviceView> transport;
    GeneralizedTopDeformationDeviceView deformation;
    TopFields fields;
    Weight weight;
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

        // Relative transport and planetary transport are separate calls.
        // Planetary transport and stretching are each included exactly once.
        output(k, j, i) += weight(j, i) * value;
    }
};

} // namespace

GeneralizedVorticityTendency::GeneralizedVorticityTendency(
    const Core::Geometry::HorizontalGeometry& geometry)
    : horizontal_transport_("Generalized horizontal vorticity transport"),
      top_transport_("Generalized top vorticity transport"),
      horizontal_deformation_(make_generalized_horizontal_deformation_device_view(geometry)),
      top_deformation_(make_generalized_top_deformation_device_view(geometry)) {
    auto ht = Kokkos::create_mirror_view(horizontal_transport_);
    ht() = make_generalized_horizontal_vorticity_transport_device_view(geometry);
    Kokkos::deep_copy(horizontal_transport_, ht);

    auto tt = Kokkos::create_mirror_view(top_transport_);
    tt() = make_generalized_top_transport_device_view(geometry);
    Kokkos::deep_copy(top_transport_, tt);

    prepare_execution();
}

void
GeneralizedVorticityTendency::prepare_execution() {
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

    Kokkos::parallel_for("PrepareGeneralizedTopVorticityTendency",
        Kokkos::Experimental::require(Policy({0, 0, 0}, {1, 1, 1}),
            Kokkos::Experimental::WorkItemProperty::HintLightWeight),
        top_preparation);

    Kokkos::fence();
}

void
GeneralizedVorticityTendency::add_from_canonical_state(const Core::State& state,
    const Core::Grid& grid,
    const Core::Parameters& params,
    Core::Field<3>& output,
    const std::string& variable,
    Term term) const {
    add_weighted_from_canonical_state(state,
        grid,
        params,
        output,
        variable,
        term,
        Weight::constant_value(real(1.0)));
}

void
GeneralizedVorticityTendency::add_weighted_from_canonical_state(const Core::State& state,
    const Core::Grid& grid,
    const Core::Parameters& params,
    Core::Field<3>& output,
    const std::string& variable,
    Term term,
    const Weight& weight) const {
    const int component = variable == "xi"     ? 0
                          : variable == "eta"  ? 1
                          : variable == "zeta" ? 2
                                               : -1;

    if (component < 0) {
        throw std::invalid_argument("Generalized vorticity tendencies require xi, eta or zeta.");
    }

    if (term != Term::Transport && term != Term::Stretching && term != Term::Twisting &&
        term != Term::Planetary) {
        throw std::invalid_argument("Invalid generalized vorticity tendency term.");
    }

    const int h = grid.get_halo_cells();
    const int nz = grid.get_local_total_points_z();
    const int ny = grid.get_local_total_points_y();
    const int nx = grid.get_local_total_points_x();
    const int top = nz - h - 1;

    if (h < 2 || top - h < 2 || ny <= 2 * h || nx <= 2 * h) {
        throw std::invalid_argument(
            "Generalized vorticity tendencies require two horizontal halos, "
            "a nonempty horizontal domain and at least three wind levels.");
    }

    const auto data = output.get_device_data();

    if (data.extent(0) != static_cast<std::size_t>(nz) ||
        data.extent(1) != static_cast<std::size_t>(ny) ||
        data.extent(2) != static_cast<std::size_t>(nx)) {
        throw std::invalid_argument("Vorticity tendency output does not match the Grid extent.");
    }

    if (component != 2) {
        const auto policy = Kokkos::Experimental::require(Policy({h, h, h}, {top, ny - h, nx - h}),
            Kokkos::Experimental::WorkItemProperty::HintLightWeight);

        if (term == Term::Transport) {
            Kokkos::parallel_for("GeneralizedHorizontalVorticityTransport",
                policy,
                HorizontalTransportFunctor{horizontal_transport_,
                    bind_horizontal_transport_fields(state, params),
                    weight,
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
                    weight,
                    output.get_mutable_device_data(),
                    component == 0,
                    term});
        }

        return;
    }

    Kokkos::parallel_for("GeneralizedTopVorticityTendency",
        Kokkos::Experimental::require(Policy({top, h, h}, {top + 1, ny - h, nx - h}),
            Kokkos::Experimental::WorkItemProperty::HintLightWeight),
        TopTendencyFunctor{top_transport_,
            top_deformation_,
            bind_top_fields(state, params),
            weight,
            output.get_mutable_device_data(),
            term});
}

} // namespace VVM::Dynamics::Operators
