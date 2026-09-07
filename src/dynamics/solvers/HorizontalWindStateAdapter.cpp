#include "dynamics/solvers/HorizontalWindStateAdapter.hpp"

#include <cmath>
#include <initializer_list>
#include <stdexcept>
#include <string>

#if defined(KOKKOS_ENABLE_CUDA)
#include <cuda_runtime.h>
#endif

namespace VVM {
namespace Dynamics {
namespace {

void require_outside_capture() {
#if defined(KOKKOS_ENABLE_CUDA)
    if (!Kokkos::is_initialized()) throw std::logic_error("Initialize Kokkos before wind adapter preparation.");

    cudaStreamCaptureStatus status = cudaStreamCaptureStatusNone;
    const auto result = cudaStreamIsCapturing(Kokkos::Cuda().cuda_stream(), &status);

    if (result != cudaSuccess) throw std::runtime_error(cudaGetErrorString(result));
    if (status != cudaStreamCaptureStatusNone) throw std::logic_error("Wind adapter initialization must occur outside capture.");
#endif
}

void validate_storage(const Real* first, const Real* second, std::initializer_list<const Real*> inputs) {
    if (first == second) throw std::invalid_argument("Wind adapter outputs must have distinct storage.");

    for (const Real* input : inputs) {
        if (first == input || second == input) throw std::invalid_argument("Wind adapter inputs and outputs must not alias.");
    }
}

void validate_spacing(const Core::Field<1>& spacing, int required) {
    if (static_cast<int>(spacing.get_device_data().extent(0)) < required) {
        throw std::invalid_argument("Insufficient wind-level spacing entries.");
    }
}

} // namespace

HorizontalWindStateAdapter::HorizontalWindStateAdapter(const Core::Geometry::HorizontalGeometry& geometry)
    : layout_(geometry.layout()),
      reconstruction_(Operators::make_horizontal_wind_reconstruction_device_view(geometry)),
      dq1_(geometry.dq1()), dq2_(geometry.dq2()) {

    // The reconstruction factory rejects unsupported/nonorthogonal geometry.
    if (layout_.halo < 1 || layout_.local_physical_nx < 1 || layout_.local_physical_ny < 1) {
        throw std::invalid_argument("Wind adapter requires physical cells and at least one halo cell.");
    }

    const auto u = geometry.device_view(Core::Geometry::HorizontalLocation::U);
    const auto v = geometry.device_view(Core::Geometry::HorizontalLocation::V);

    inverse_h1_at_u_ = u.physical_to_contravariant.a11;
    inverse_h2_at_v_ = v.physical_to_contravariant.a22;
}

void HorizontalWindStateAdapter::prepare_execution() {
    require_outside_capture();

#if defined(KOKKOS_ENABLE_CUDA)
    // Same compilation unit as all adapter kernels; prepare lazy backend symbols.
    Kokkos::parallel_for("PrepareHorizontalWindStateAdapter", Kokkos::RangePolicy<Kokkos::Cuda>(0, 1), KOKKOS_LAMBDA(const int) {});

    Kokkos::Cuda().fence("Prepare horizontal wind State adapter");

    const auto result = cudaGetLastError();
    if (result != cudaSuccess) throw std::runtime_error(cudaGetErrorString(result));
#endif
}

void HorizontalWindStateAdapter::initialize_spacing(Real dz, const Core::Field<1>& flex_up, Core::Field<1>& spacing) {
    require_outside_capture();

    if (!std::isfinite(dz) || dz <= real(0.0)) throw std::invalid_argument("Reference dz must be positive and finite.");

    const auto& input = flex_up.get_device_data();
    const auto& output = spacing.get_mutable_device_data();

    if (input.data() == output.data()) throw std::invalid_argument("Spacing must not alias flex_up.");
    if (input.extent(0) < output.extent(0)) throw std::invalid_argument("Insufficient flex_up entries.");

    const auto coefficients = flex_up.get_host_data();
    auto values = Kokkos::create_mirror(output);

    for (std::size_t k = 0; k < output.extent(0); ++k) {
        const Real coefficient = coefficients(k);

        if (!std::isfinite(coefficient) || coefficient <= real(0.0)) throw std::invalid_argument("flex_up must be positive and finite.");

        values(k) = dz / coefficient;

        if (!std::isfinite(values(k)) || values(k) <= real(0.0)) throw std::invalid_argument("Invalid derived wind-level spacing.");
    }

    Kokkos::deep_copy(output, values);
}

void HorizontalWindStateAdapter::validate_volume(const Core::Field<3>& field, int nz) const {
    const auto& data = field.get_device_data();

    if (static_cast<int>(data.extent(0)) != nz || static_cast<int>(data.extent(1)) != layout_.local_total_ny() || static_cast<int>(data.extent(2)) != layout_.local_total_nx()) {
        throw std::invalid_argument("Incorrect wind adapter volume extents.");
    }
}

void HorizontalWindStateAdapter::validate_plane(const Core::Field<2>& field) const {
    const auto& data = field.get_device_data();

    if (static_cast<int>(data.extent(0)) != layout_.local_total_ny() || static_cast<int>(data.extent(1)) != layout_.local_total_nx()) {
        throw std::invalid_argument("Incorrect wind adapter plane extents.");
    }
}

void HorizontalWindStateAdapter::diagnose_vorticity(const Core::Field<3>& u, const Core::Field<3>& v, const Core::Field<3>& w,
    const Core::Field<1>& spacing, Core::Field<3>& xi, Core::Field<3>& eta, int first, int last) const {

    const int nz = static_cast<int>(w.get_device_data().extent(0));

    if (first < 0 || last < first || last >= nz - 1) throw std::invalid_argument("Invalid vorticity interface range.");

    for (const auto* field : {&u, &v, &w}) validate_volume(*field, nz);
    validate_volume(xi, nz);
    validate_volume(eta, nz);
    validate_spacing(spacing, last + 1);

    const auto ud = u.get_device_data();
    const auto vd = v.get_device_data();
    const auto wd = w.get_device_data();
    const auto ds = spacing.get_device_data();
    const auto xd = xi.get_mutable_device_data();
    const auto ed = eta.get_mutable_device_data();

    validate_storage(xd.data(), ed.data(), {ud.data(), vd.data(), wd.data(), ds.data()});

    const auto ih1 = inverse_h1_at_u_;
    const auto ih2 = inverse_h2_at_v_;
    const Real dq1 = dq1_;
    const Real dq2 = dq2_;
    const int h = layout_.halo;

    const auto policy = Kokkos::Experimental::require(
        Kokkos::MDRangePolicy<Kokkos::Rank<3>>({first, h, h}, {last + 1, layout_.local_total_ny() - h, layout_.local_total_nx() - h}),
        Kokkos::Experimental::WorkItemProperty::HintLightWeight);

    Kokkos::parallel_for("DiagnosePhysicalHorizontalVorticity", policy, KOKKOS_LAMBDA(const int k, const int j, const int i) {
        // xi=h1*omega^1 and eta=-h2*omega^2; h1/h2 do not vary with height.
        xd(k, j, i) = (wd(k, j + 1, i) - wd(k, j, i)) / dq2 * ih2(j, i) - (vd(k + 1, j, i) - vd(k, j, i)) / ds(k);
        ed(k, j, i) = (wd(k, j, i + 1) - wd(k, j, i)) / dq1 * ih1(j, i) - (ud(k + 1, j, i) - ud(k, j, i)) / ds(k);
    });
}

void HorizontalWindStateAdapter::reconstruct_top(const Core::Field<2>& psi, const Core::Field<2>& chi,
    Core::Field<3>& u, Core::Field<3>& v, int top) const {

    const int nz = static_cast<int>(u.get_device_data().extent(0));

    if (top < 0 || top >= nz) throw std::invalid_argument("Invalid top wind level.");

    validate_plane(psi);
    validate_plane(chi);
    validate_volume(u, nz);
    validate_volume(v, nz);

    const auto pd = psi.get_device_data();
    const auto cd = chi.get_device_data();
    const auto ud = u.get_mutable_device_data();
    const auto vd = v.get_mutable_device_data();

    validate_storage(ud.data(), vd.data(), {pd.data(), cd.data()});

    const auto reconstruction = reconstruction_;
    const auto ih1 = inverse_h1_at_u_;
    const auto ih2 = inverse_h2_at_v_;
    const int h = layout_.halo;

    const auto policy = Kokkos::Experimental::require(
        Kokkos::MDRangePolicy<Kokkos::Rank<2>>({h, h}, {layout_.local_total_ny() - h, layout_.local_total_nx() - h}),
        Kokkos::Experimental::WorkItemProperty::HintLightWeight);

    Kokkos::parallel_for("ReconstructPhysicalTopWind", policy, KOKKOS_LAMBDA(const int j, const int i) {
        ud(top, j, i) = reconstruction.calculate_covariant_q1_at_u(pd, cd, j, i) * ih1(j, i);
        vd(top, j, i) = reconstruction.calculate_covariant_q2_at_v(pd, cd, j, i) * ih2(j, i);
    });
}

void HorizontalWindStateAdapter::integrate_from_top(const Core::Field<3>& w, const Core::Field<3>& xi, const Core::Field<3>& eta,
    const Core::Field<1>& spacing, Core::Field<3>& u, Core::Field<3>& v, int bottom, int top) const {

    const int nz = static_cast<int>(w.get_device_data().extent(0));

    if (bottom < 0 || top < bottom || top >= nz) throw std::invalid_argument("Invalid wind integration range.");

    for (const auto* field : {&w, &xi, &eta}) validate_volume(*field, nz);
    validate_volume(u, nz);
    validate_volume(v, nz);
    validate_spacing(spacing, top);

    const auto wd = w.get_device_data();
    const auto xd = xi.get_device_data();
    const auto ed = eta.get_device_data();
    const auto ds = spacing.get_device_data();
    const auto ud = u.get_mutable_device_data();
    const auto vd = v.get_mutable_device_data();

    validate_storage(ud.data(), vd.data(), {wd.data(), xd.data(), ed.data(), ds.data()});

    const auto ih1 = inverse_h1_at_u_;
    const auto ih2 = inverse_h2_at_v_;
    const Real dq1 = dq1_;
    const Real dq2 = dq2_;
    const int h = layout_.halo;

    const auto policy = Kokkos::Experimental::require(
        Kokkos::MDRangePolicy<Kokkos::Rank<2>>({h, h}, {layout_.local_total_ny() - h, layout_.local_total_nx() - h}),
        Kokkos::Experimental::WorkItemProperty::HintLightWeight);

    Kokkos::parallel_for("IntegratePhysicalWindFromTop", policy, KOKKOS_LAMBDA(const int j, const int i) {
        // Orthogonal, height-independent specialization of CVVM covariant recovery.
        for (int k = top - 1; k >= bottom; --k) {
            const Real du_dz = (wd(k, j, i + 1) - wd(k, j, i)) / dq1 * ih1(j, i) - ed(k, j, i);
            const Real dv_dz = (wd(k, j + 1, i) - wd(k, j, i)) / dq2 * ih2(j, i) - xd(k, j, i);

            ud(k, j, i) = ud(k + 1, j, i) - du_dz * ds(k);
            vd(k, j, i) = vd(k + 1, j, i) - dv_dz * ds(k);
        }
    });
}

} // namespace Dynamics
} // namespace VVM
