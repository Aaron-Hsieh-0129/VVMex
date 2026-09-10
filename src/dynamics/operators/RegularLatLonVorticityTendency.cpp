#include "dynamics/operators/RegularLatLonVorticityTendency.hpp"

namespace VVM::Dynamics::Operators {
namespace {
using Volume = Core::Field<3>::ViewType;
using Profile = Core::Field<1>::ViewType;
using Plane = Core::Field<2>::ViewType;

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
struct Fields {
    Volume u, v, w, xi, eta, zeta;
    Plane f_at_z;
    Profile rho, rho_up;
    ProductProfile fn1, fn2;
    InverseSpacing inverse_spacing, inverse_spacing_mid, inverse_spacing_up;
};
struct TendencyFunctor {
    Kokkos::View<const RegularLatLonTopTransportDeviceView> transport;
    Kokkos::View<const RegularLatLonTopDeformationDeviceView> deformation;
    Fields fields;
    Volume output;
    int variable, begin, end;
    RegularLatLonVorticityTendency::Term term;
    bool evaluate = true;

    KOKKOS_INLINE_FUNCTION void
    operator()(int k, int j, int i) const {
        if (!evaluate) {
            return;
        }
        using Term = RegularLatLonVorticityTendency::Term;
        Real value = real(0.0);
        if (term == Term::Transport || (term == Term::Planetary && variable == 2)) {
            const auto t =
                variable == 2 ? transport().calculate_at_z(fields, k, j, i, term == Term::Planetary)
                : variable == 0
                    ? transport()
                          .horizontal.calculate_xi_at_v(fields, fields.w, k, j, i, begin, end)
                    : transport()
                          .horizontal.calculate_eta_at_u(fields, fields.w, k, j, i, begin, end);
            value = t.q1 + t.q2 + t.vertical;
        }
        if (term != Term::Transport) {
            if (variable == 2) {
                const auto t = deformation().calculate_at_z(fields, k, j, i);
                value += term == Term::Stretching ? t.stretching
                         : term == Term::Twisting ? t.twisting
                                                  : t.planetary;
            }
            else {
                const auto t =
                    variable == 0
                        ? transport().horizontal.components.calculate_xi_at_v(fields, k, j, i)
                        : transport().horizontal.components.calculate_eta_at_u(fields, k, j, i);
                value += term == Term::Stretching ? t.stretching
                         : term == Term::Twisting ? t.twisting
                                                  : t.planetary;
            }
        }
        output(k, j, i) += value;
    }
};
using Policy = Kokkos::MDRangePolicy<Kokkos::Rank<3>>;
}

RegularLatLonVorticityTendency::RegularLatLonVorticityTendency(
    const Core::Geometry::HorizontalGeometry& geometry)
    : transport_("RLL vorticity transport"), deformation_("RLL top deformation") {
    auto t = Kokkos::create_mirror_view(transport_);
    t() = make_regular_lat_lon_top_transport_device_view(geometry);
    Kokkos::deep_copy(transport_, t);
    auto d = Kokkos::create_mirror_view(deformation_);
    d() = make_regular_lat_lon_top_deformation_device_view(geometry);
    Kokkos::deep_copy(deformation_, d);
    prepare_execution();
}

void
RegularLatLonVorticityTendency::prepare_execution() {
    // Prepare the exact functor without reading model fields or evaluating a tendency.
    TendencyFunctor preparation{};
    preparation.evaluate = false;
    Kokkos::parallel_for("PrepareRLLVorticityTendency",
        Kokkos::Experimental::require(Policy({0, 0, 0}, {1, 1, 1}),
            Kokkos::Experimental::WorkItemProperty::HintLightWeight),
        preparation);
    Kokkos::fence();
}

void
RegularLatLonVorticityTendency::add(const Core::State& state,
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
    Fields f{};
    f.u = state.get_field<3>("u").get_device_data();
    f.v = state.get_field<3>("v").get_device_data();
    f.w = state.get_field<3>("w").get_device_data();
    f.xi = state.get_field<3>("xi").get_device_data();
    f.eta = state.get_field<3>("eta").get_device_data();
    f.zeta = state.get_field<3>("zeta").get_device_data();
    f.f_at_z = state.get_field<2>("f_2d").get_device_data();
    f.rho = state.get_field<1>("rhobar").get_device_data();
    f.rho_up = state.get_field<1>("rhobar_up").get_device_data();
    f.fn1 = {params.fact1_xi_eta.get_device_data(), f.rho, 1};
    f.fn2 = {params.fact2_xi_eta.get_device_data(), f.rho, 0};
    f.inverse_spacing = {params.flex_height_coef_up.get_device_data(), params.rdz};
    f.inverse_spacing_up = f.inverse_spacing;
    f.inverse_spacing_mid = {params.flex_height_coef_mid.get_device_data(), params.rdz};
    const int begin = component == 2 ? top : h;
    const int end = component == 2 ? top + 1 : top;
    Kokkos::parallel_for("RLLVorticityTendency",
        Kokkos::Experimental::require(
            Policy({begin, h, h},
                {end, grid.get_local_total_points_y() - h, grid.get_local_total_points_x() - h}),
            Kokkos::Experimental::WorkItemProperty::HintLightWeight),
        TendencyFunctor{transport_,
            deformation_,
            f,
            output.get_mutable_device_data(),
            component,
            h,
            top,
            term});
}
} // namespace VVM::Dynamics::Operators
