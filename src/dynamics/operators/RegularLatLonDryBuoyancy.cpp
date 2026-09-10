#include "dynamics/operators/RegularLatLonDryBuoyancy.hpp"

#include <array>
#include <stdexcept>
#include <string>
#include <type_traits>

#include "core/geometry/GeometryKind.hpp"

#if defined(KOKKOS_ENABLE_CUDA)
#include <cuda_runtime.h>
#endif

namespace VVM {
namespace Dynamics {
namespace Operators {

namespace {

using VolumeView = Core::Field<3>::ViewType;
using ProfileView = Core::Field<1>::ViewType;
using ScalarView = Kokkos::View<VVM::Real>;

using DryBuoyancyPolicy = Kokkos::MDRangePolicy<Kokkos::DefaultExecutionSpace, Kokkos::Rank<3>>;

struct DryBuoyancyFunctor {
    RegularLatLonDryBuoyancyDeviceView operator_view;

    VolumeView th;
    ProfileView thbar;
    ScalarView gravity;
    VolumeView output;
    VolumeView qv, qp, face_mask;
    bool moist = false;
    int max_topo_idx = -1;

    KOKKOS_INLINE_FUNCTION
    VVM::Real moisture(const int k, const int j, const int i) const noexcept {
        return VVM::real(0.608)*qv(k,j,i) - qp(k,j,i);
    }

    bool xi_component = true;
    bool preparation_only = false;

    KOKKOS_INLINE_FUNCTION
    ScalarStencilAtT
    normalized_stencil(const int k, const int j, const int i) const noexcept {

        const VVM::Real inverse_thbar = VVM::real(1.0) / thbar(k);

        ScalarStencilAtT stencil;

        stencil.center = th(k, j, i) * inverse_thbar;
        stencil.west = th(k, j, i - 1) * inverse_thbar;
        stencil.east = th(k, j, i + 1) * inverse_thbar;
        stencil.south = th(k, j - 1, i) * inverse_thbar;
        stencil.north = th(k, j + 1, i) * inverse_thbar;
        stencil.southwest = th(k, j - 1, i - 1) * inverse_thbar;
        stencil.southeast = th(k, j - 1, i + 1) * inverse_thbar;
        stencil.northwest = th(k, j + 1, i - 1) * inverse_thbar;
        stencil.northeast = th(k, j + 1, i + 1) * inverse_thbar;

        if (moist) {
            stencil.center += moisture(k,j,i);
            stencil.west += moisture(k,j,i-1);
            stencil.east += moisture(k,j,i+1);
            stencil.south += moisture(k,j-1,i);
            stencil.north += moisture(k,j+1,i);
            stencil.southwest += moisture(k,j-1,i-1);
            stencil.southeast += moisture(k,j-1,i+1);
            stencil.northwest += moisture(k,j+1,i-1);
            stencil.northeast += moisture(k,j+1,i+1);
        }

        return stencil;
    }

    KOKKOS_INLINE_FUNCTION
    void
    operator()(const int k, const int j, const int i) const {

        if (preparation_only) {
            return;
        }

        const ScalarStencilAtT lower = normalized_stencil(k, j, i);
        const ScalarStencilAtT upper = normalized_stencil(k + 1, j, i);

        if (xi_component) {
            output(k, j, i) += operator_view.calculate_xi_at_v(j, i, lower, upper, gravity());
        }
        else {
            output(k, j, i) += operator_view.calculate_eta_at_u(j, i, lower, upper, gravity());
        }
        if (moist && k <= max_topo_idx && face_mask(k,j,i) == VVM::real(0.)) {
            output(k,j,i) = VVM::real(0.);
        }
    }
};

#if defined(KOKKOS_ENABLE_CUDA)

void
require_cuda_success(const cudaError_t status, const char* operation) {

    if (status != cudaSuccess) {
        throw std::runtime_error(std::string("RegularLatLonDryBuoyancy: ") + operation + ": " +
                                 cudaGetErrorString(status));
    }
}

#endif

} // namespace

RegularLatLonDryBuoyancyDeviceView
make_regular_lat_lon_dry_buoyancy_device_view(const Core::Geometry::HorizontalGeometry& geometry) {

    if (geometry.kind() != Core::Geometry::GeometryKind::RegularLatLon) {

        throw std::invalid_argument("RegularLatLonDryBuoyancy requires "
                                    "regular latitude-longitude geometry.");
    }

    RegularLatLonDryBuoyancyDeviceView result;

    result.gradient = make_horizontal_scalar_gradient_device_view(geometry);

    return result;
}

RegularLatLonDryBuoyancy::RegularLatLonDryBuoyancy(
    const Core::Geometry::HorizontalGeometry& geometry)
    : layout_(geometry.layout()),
      operator_(make_regular_lat_lon_dry_buoyancy_device_view(geometry)) {

    if (layout_.halo < 1) {
        throw std::invalid_argument("RegularLatLonDryBuoyancy requires "
                                    "at least one horizontal halo cell.");
    }

    if (layout_.local_physical_nx < 1 || layout_.local_physical_ny < 1) {

        throw std::invalid_argument("RegularLatLonDryBuoyancy requires "
                                    "a nonempty physical horizontal domain.");
    }
}

void
RegularLatLonDryBuoyancy::prepare_execution() {
#if defined(KOKKOS_ENABLE_CUDA)
    if (!Kokkos::is_initialized()) {
        throw std::logic_error("RegularLatLonDryBuoyancy::prepare_execution "
                               "requires initialized Kokkos.");
    }

    static_assert(std::is_same<Kokkos::DefaultExecutionSpace, Kokkos::Cuda>::value,
        "CUDA preparation requires CUDA as the "
        "default execution space.");

    const Kokkos::Cuda execution;

    cudaStreamCaptureStatus capture_status = cudaStreamCaptureStatusNone;

    require_cuda_success(cudaStreamIsCapturing(execution.cuda_stream(), &capture_status),
        "query capture status");

    if (capture_status != cudaStreamCaptureStatusNone) {

        throw std::logic_error("RegularLatLonDryBuoyancy::prepare_execution "
                               "must run before CUDA graph capture.");
    }

    require_cuda_success(cudaGetLastError(), "CUDA error before backend preparation");

    DryBuoyancyFunctor preparation_functor;
    preparation_functor.preparation_only = true;

    const auto preparation_policy =
        Kokkos::Experimental::require(DryBuoyancyPolicy(execution, {0, 0, 0}, {1, 1, 1}),
            Kokkos::Experimental::WorkItemProperty::HintLightWeight);

    Kokkos::parallel_for("PrepareRegularLatLonDryBuoyancy",
        preparation_policy,
        preparation_functor);

    require_cuda_success(cudaGetLastError(), "launch backend preparation");

    execution.fence("Complete RegularLatLonDryBuoyancy "
                    "backend preparation");

    require_cuda_success(cudaGetLastError(), "complete backend preparation");
#endif
}

void
RegularLatLonDryBuoyancy::validate_volume(
    const Core::Field<3>& field, const int nz, const char* role) const {

    const auto& data = field.get_device_data();

    if (static_cast<int>(data.extent(0)) != nz ||
        static_cast<int>(data.extent(1)) != layout_.local_total_ny() ||
        static_cast<int>(data.extent(2)) != layout_.local_total_nx()) {

        throw std::invalid_argument(std::string("RegularLatLonDryBuoyancy: "
                                                "incorrect volume extents for ") +
                                    role + ".");
    }
}

void
RegularLatLonDryBuoyancy::add_xi_tendency(const Core::Field<3>& th,
    const Core::Field<1>& thbar,
    const Kokkos::View<VVM::Real>& gravity,
    Core::Field<3>& out_tendency,
    const int k_begin,
    const int k_end) const {

    add_tendency(th, thbar, gravity, out_tendency, k_begin, k_end, true);
}

void
RegularLatLonDryBuoyancy::add_eta_tendency(const Core::Field<3>& th,
    const Core::Field<1>& thbar,
    const Kokkos::View<VVM::Real>& gravity,
    Core::Field<3>& out_tendency,
    const int k_begin,
    const int k_end) const {

    add_tendency(th, thbar, gravity, out_tendency, k_begin, k_end, false);
}

void
RegularLatLonDryBuoyancy::add_tendency(const Core::Field<3>& th,
    const Core::Field<1>& thbar,
    const Kokkos::View<VVM::Real>& gravity,
    Core::Field<3>& out_tendency,
    const int k_begin,
    const int k_end,
    const bool xi_component,
    const Core::Field<3>* qv,
    const Core::Field<3>* qp,
    const Core::Field<3>* face_mask,
    const int max_topo_idx) const {

    const int nz = static_cast<int>(th.get_device_data().extent(0));

    if (k_begin < 0 || k_end <= k_begin || k_end >= nz) {

        throw std::invalid_argument("RegularLatLonDryBuoyancy requires "
                                    "a valid half-open vertical range whose "
                                    "upper adjacent T level exists.");
    }

    validate_volume(th, nz, "th");
    validate_volume(out_tendency, nz, "out_tendency");

    if (static_cast<int>(thbar.get_device_data().extent(0)) <= k_end) {

        throw std::invalid_argument("RegularLatLonDryBuoyancy received "
                                    "insufficient thbar entries.");
    }

    if (gravity.extent(0) != 1) {
        throw std::invalid_argument("RegularLatLonDryBuoyancy requires "
                                    "an allocated gravity scalar.");
    }

    auto th_data = th.get_device_data();
    auto thbar_data = thbar.get_device_data();
    auto output_data = out_tendency.get_mutable_device_data();

    const std::array<const VVM::Real*, 3> inputs = {th_data.data(),
        thbar_data.data(),
        gravity.data()};

    for (const VVM::Real* input : inputs) {
        if (input == output_data.data()) {
            throw std::invalid_argument("RegularLatLonDryBuoyancy requires "
                                        "distinct input and output storage.");
        }
    }

    DryBuoyancyFunctor functor;

    functor.operator_view = operator_;
    functor.th = th_data;
    functor.thbar = thbar_data;
    functor.gravity = gravity;
    functor.output = output_data;
    functor.xi_component = xi_component;
    functor.preparation_only = false;
    if (qv != nullptr) {
        for (const auto* field : {qv, qp, face_mask}) {
            validate_volume(*field, nz, "moist buoyancy input");
            if (field->get_device_data().data() == output_data.data()) {
                throw std::invalid_argument("Moist buoyancy requires distinct input/output storage.");
            }
        }
        functor.qv = qv->get_device_data();
        functor.qp = qp->get_device_data();
        functor.face_mask = face_mask->get_device_data();
        functor.moist = true;
        functor.max_topo_idx = max_topo_idx;
    }

    const int h = layout_.halo;
    const int ny = layout_.local_total_ny();
    const int nx = layout_.local_total_nx();

    const auto policy =
        Kokkos::Experimental::require(DryBuoyancyPolicy({k_begin, h, h}, {k_end, ny - h, nx - h}),
            Kokkos::Experimental::WorkItemProperty::HintLightWeight);

    Kokkos::parallel_for(xi_component ? "RegularLatLonDryBuoyancyXi"
                                      : "RegularLatLonDryBuoyancyEta",
        policy,
        functor);
}

void RegularLatLonDryBuoyancy::add_moist_tendency(const Core::Field<3>& th,
    const Core::Field<1>& thbar, const Kokkos::View<VVM::Real>& gravity,
    const Core::Field<3>& qv, const Core::Field<3>& qp,
    const Core::Field<3>& face_mask, Core::Field<3>& output,
    int k_begin, int k_end, int max_topo_idx, bool xi_component) const {
    add_tendency(th, thbar, gravity, output, k_begin, k_end, xi_component,
        &qv, &qp, &face_mask, max_topo_idx);
}

} // namespace Operators
} // namespace Dynamics
} // namespace VVM
