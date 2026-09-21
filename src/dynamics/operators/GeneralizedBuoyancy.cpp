#include "dynamics/operators/GeneralizedBuoyancy.hpp"

#include <array>
#include <cmath>
#include <stdexcept>
#include <string>
#include <type_traits>

#if defined(KOKKOS_ENABLE_CUDA)
#include <cuda_runtime.h>
#endif

namespace VVM::Dynamics::Operators {
namespace {

using Volume = Core::Field<3>::ViewType;
using Profile = Core::Field<1>::ViewType;

using Policy = Kokkos::MDRangePolicy<Kokkos::DefaultExecutionSpace, Kokkos::Rank<3>>;

struct BuoyancyFunctor {
    GeneralizedBuoyancyDeviceView operation;

    Volume th;
    Profile thbar;
    Kokkos::View<Real> gravity;

    Volume output;
    Volume qv;
    Volume qp;
    Volume face_mask;

    int max_topo_idx = -1;
    bool moist = false;
    bool xi_component = true;
    bool preparation_only = false;

    KOKKOS_INLINE_FUNCTION Real
    difference(int k, int j, int i, int dj, int di) const noexcept {
        // Difference before normalization, as in CVVM BUOYF_3D. thbar is
        // horizontally uniform. Keep moisture separate from O(1) th/thbar
        // to avoid cancelling a small moist signal against the base state.
        Real value = (th(k, j + dj, i + di) - th(k, j, i)) / thbar(k);

        if (moist) {
            value += real(0.608) * (qv(k, j + dj, i + di) - qv(k, j, i)) -
                     (qp(k, j + dj, i + di) - qp(k, j, i));
        }

        return value;
    }

    KOKKOS_INLINE_FUNCTION void
    operator()(int k, int j, int i) const {
        if (preparation_only) {
            return;
        }

        const int dj = xi_component ? 1 : 0;
        const int di = xi_component ? 0 : 1;

        const Real lower = difference(k, j, i, dj, di);
        const Real upper = difference(k + 1, j, i, dj, di);

        const Real canonical =
            xi_component ? operation.calculate_omega1_at_v(j, i, lower, upper, gravity())
                         : -operation.calculate_omega2_at_u(j, i, lower, upper, gravity());

        output(k, j, i) += canonical;

        if (moist && k <= max_topo_idx && face_mask(k, j, i) == real(0.0)) {
            output(k, j, i) = real(0.0);
        }
    }
};

#if defined(KOKKOS_ENABLE_CUDA)

void
require_cuda_success(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        throw std::runtime_error(
            std::string("GeneralizedBuoyancy: ") + operation + ": " + cudaGetErrorString(status));
    }
}

#endif

} // namespace

GeneralizedBuoyancyDeviceView
make_generalized_buoyancy_device_view(const Core::Geometry::HorizontalGeometry& geometry) {
    using Core::Geometry::HorizontalLocation;

    if (!(geometry.dq1() > real(0.0)) || !(geometry.dq2() > real(0.0)) ||
        !std::isfinite(geometry.dq1()) || !std::isfinite(geometry.dq2())) {
        throw std::invalid_argument("GeneralizedBuoyancy requires finite positive spacing.");
    }

    GeneralizedBuoyancyDeviceView result;

    result.inverse_jacobian_u = geometry.device_view(HorizontalLocation::U).inv_sqrt_g;

    result.inverse_jacobian_v = geometry.device_view(HorizontalLocation::V).inv_sqrt_g;

    result.dq1 = geometry.dq1();
    result.dq2 = geometry.dq2();

    return result;
}

GeneralizedBuoyancy::GeneralizedBuoyancy(const Core::Geometry::HorizontalGeometry& geometry)
    : layout_(geometry.layout()), operator_(make_generalized_buoyancy_device_view(geometry)) {
    if (layout_.halo < 1 || layout_.local_physical_nx < 1 || layout_.local_physical_ny < 1) {
        throw std::invalid_argument("GeneralizedBuoyancy requires one horizontal halo "
                                    "and a nonempty domain.");
    }
}

void
GeneralizedBuoyancy::prepare_execution() {
#if defined(KOKKOS_ENABLE_CUDA)
    if (!Kokkos::is_initialized()) {
        throw std::logic_error("GeneralizedBuoyancy preparation requires initialized Kokkos.");
    }

    static_assert(std::is_same<Kokkos::DefaultExecutionSpace, Kokkos::Cuda>::value,
        "CUDA preparation requires CUDA as the default execution space.");

    const Kokkos::Cuda execution;

    cudaStreamCaptureStatus status = cudaStreamCaptureStatusNone;

    require_cuda_success(cudaStreamIsCapturing(execution.cuda_stream(), &status),
        "query capture status");

    if (status != cudaStreamCaptureStatusNone) {
        throw std::logic_error("GeneralizedBuoyancy preparation must precede graph capture.");
    }

    require_cuda_success(cudaGetLastError(), "CUDA error before preparation");

    BuoyancyFunctor preparation{};
    preparation.preparation_only = true;

    Kokkos::parallel_for("PrepareGeneralizedBuoyancy",
        Kokkos::Experimental::require(Policy(execution, {0, 0, 0}, {1, 1, 1}),
            Kokkos::Experimental::WorkItemProperty::HintLightWeight),
        preparation);

    require_cuda_success(cudaGetLastError(), "launch preparation");

    execution.fence("Complete GeneralizedBuoyancy preparation");

    require_cuda_success(cudaGetLastError(), "complete preparation");
#endif
}

void
GeneralizedBuoyancy::validate_volume(const Core::Field<3>& field, int nz, const char* role) const {
    const auto& data = field.get_device_data();

    if (static_cast<int>(data.extent(0)) != nz ||
        static_cast<int>(data.extent(1)) != layout_.local_total_ny() ||
        static_cast<int>(data.extent(2)) != layout_.local_total_nx()) {
        throw std::invalid_argument(
            std::string("GeneralizedBuoyancy: incorrect volume extents for ") + role + ".");
    }
}

void
GeneralizedBuoyancy::add_xi_tendency(const Core::Field<3>& th,
    const Core::Field<1>& thbar,
    const Kokkos::View<Real>& gravity,
    Core::Field<3>& output,
    int k_begin,
    int k_end) const {

    add_tendency(th, thbar, gravity, output, k_begin, k_end, true);
}

void
GeneralizedBuoyancy::add_eta_tendency(const Core::Field<3>& th,
    const Core::Field<1>& thbar,
    const Kokkos::View<Real>& gravity,
    Core::Field<3>& output,
    int k_begin,
    int k_end) const {

    add_tendency(th, thbar, gravity, output, k_begin, k_end, false);
}

void
GeneralizedBuoyancy::add_moist_tendency(const Core::Field<3>& th,
    const Core::Field<1>& thbar,
    const Kokkos::View<Real>& gravity,
    const Core::Field<3>& qv,
    const Core::Field<3>& qp,
    const Core::Field<3>& face_mask,
    Core::Field<3>& output,
    int k_begin,
    int k_end,
    int max_topo_idx,
    bool xi_component) const {

    add_tendency(th,
        thbar,
        gravity,
        output,
        k_begin,
        k_end,
        xi_component,
        &qv,
        &qp,
        &face_mask,
        max_topo_idx);
}

void
GeneralizedBuoyancy::add_tendency(const Core::Field<3>& th,
    const Core::Field<1>& thbar,
    const Kokkos::View<Real>& gravity,
    Core::Field<3>& output,
    int k_begin,
    int k_end,
    bool xi_component,
    const Core::Field<3>* qv,
    const Core::Field<3>* qp,
    const Core::Field<3>* face_mask,
    int max_topo_idx) const {
    const int nz = static_cast<int>(th.get_device_data().extent(0));

    if (k_begin < 0 || k_end <= k_begin || k_end >= nz) {
        throw std::invalid_argument("GeneralizedBuoyancy requires a valid range "
                                    "with an upper adjacent T level.");
    }

    validate_volume(th, nz, "th");
    validate_volume(output, nz, "output");

    if (static_cast<int>(thbar.get_device_data().extent(0)) <= k_end) {
        throw std::invalid_argument("GeneralizedBuoyancy received insufficient thbar entries.");
    }

    if (gravity.data() == nullptr) {
        throw std::invalid_argument("GeneralizedBuoyancy requires an allocated gravity scalar.");
    }

    const auto th_data = th.get_device_data();
    const auto thbar_data = thbar.get_device_data();
    const auto output_data = output.get_mutable_device_data();

    const std::array<const Real*, 3> inputs = {th_data.data(), thbar_data.data(), gravity.data()};

    for (const Real* input : inputs) {
        if (input == output_data.data()) {
            throw std::invalid_argument("GeneralizedBuoyancy requires "
                                        "distinct input/output storage.");
        }
    }

    BuoyancyFunctor functor{};

    functor.operation = operator_;
    functor.th = th_data;
    functor.thbar = thbar_data;
    functor.gravity = gravity;
    functor.output = output_data;
    functor.xi_component = xi_component;

    if (qv != nullptr) {
        for (const auto* field : {qv, qp, face_mask}) {
            validate_volume(*field, nz, "moist buoyancy input");

            if (field->get_device_data().data() == output_data.data()) {
                throw std::invalid_argument("GeneralizedBuoyancy requires "
                                            "distinct moist input/output storage.");
            }
        }

        functor.qv = qv->get_device_data();
        functor.qp = qp->get_device_data();
        functor.face_mask = face_mask->get_device_data();
        functor.moist = true;
        functor.max_topo_idx = max_topo_idx;
    }

    const int h = layout_.halo;

    Kokkos::parallel_for(xi_component ? "GeneralizedBuoyancyXi" : "GeneralizedBuoyancyEta",
        Kokkos::Experimental::require(
            Policy({k_begin, h, h},
                {k_end, layout_.local_total_ny() - h, layout_.local_total_nx() - h}),
            Kokkos::Experimental::WorkItemProperty::HintLightWeight),
        functor);
}

} // namespace VVM::Dynamics::Operators
