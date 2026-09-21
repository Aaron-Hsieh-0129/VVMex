#include "dynamics/operators/GeneralizedScalarTransport.hpp"

#include <array>
#include <cmath>
#include <stdexcept>
#include <string>
#include <type_traits>

#include <Kokkos_Core.hpp>

#if defined(KOKKOS_ENABLE_CUDA)
#include <cuda_runtime.h>
#endif

namespace VVM {
namespace Dynamics {
namespace Operators {

namespace {

using VolumeView = Core::Field<3>::ViewType;
using SpacingView = Core::Field<1>::ViewType;

using ScalarTransportPolicy = Kokkos::MDRangePolicy<Kokkos::DefaultExecutionSpace, Kokkos::Rank<3>>;

struct ScalarTransportFunctor {
    Kokkos::View<const TakacsScalarTransportDeviceView> transport;

    VolumeView scalar;

    // These fields contain rho*u^1 and rho*u^2.
    // Horizontal Jacobian weighting remains exclusively inside
    // HorizontalFluxDivergence.
    VolumeView contravariant_mass_flux_q1;
    VolumeView contravariant_mass_flux_q2;

    VolumeView vertical_mass_flux;
    SpacingView vertical_cell_spacing;
    VolumeView output;

    int k_begin = 0;
    int k_end = 0;

    KOKKOS_INLINE_FUNCTION
    void
    operator()(const int k, const int j, const int i) const {
        const VVM::Real horizontal = transport().calculate_horizontal_flux_convergence_at_t(scalar,
            contravariant_mass_flux_q1,
            contravariant_mass_flux_q2,
            k,
            j,
            i);

        const VVM::Real vertical = transport().calculate_vertical_flux_convergence_at_t(scalar,
            vertical_mass_flux,
            k,
            j,
            i,
            k_begin,
            k_end,
            VVM::real(1.0) / vertical_cell_spacing(k));

        output(k, j, i) += horizontal + vertical;
    }
};

#if defined(KOKKOS_ENABLE_CUDA)

void
require_cuda_success(const cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string("GeneralizedScalarTransport: ") + operation + ": " +
                                 cudaGetErrorString(status));
    }
}

#endif

} // namespace

GeneralizedScalarTransport::GeneralizedScalarTransport(
    const Core::Geometry::HorizontalGeometry& geometry, const VVM::Real alpha)
    : layout_(geometry.layout()),
      transport_(make_takacs_scalar_transport_device_view(geometry, alpha)),
      device_transport_("generalized_scalar_transport_device") {
    if (!(geometry.dq1() > VVM::real(0.0)) || !(geometry.dq2() > VVM::real(0.0)) ||
        !std::isfinite(geometry.dq1()) || !std::isfinite(geometry.dq2()) || !std::isfinite(alpha)) {
        throw std::invalid_argument("GeneralizedScalarTransport requires "
                                    "finite positive horizontal spacing "
                                    "and a finite Takacs coefficient.");
    }

    if (layout_.halo < 2) {
        throw std::invalid_argument("GeneralizedScalarTransport requires "
                                    "at least two horizontal halo cells.");
    }

    if (layout_.local_physical_nx < 1 || layout_.local_physical_ny < 1) {
        throw std::invalid_argument("GeneralizedScalarTransport requires "
                                    "a nonempty physical horizontal domain.");
    }

    Kokkos::deep_copy(device_transport_, transport_);
}

void
GeneralizedScalarTransport::prepare_execution() {
#if defined(KOKKOS_ENABLE_CUDA)
    if (!Kokkos::is_initialized()) {
        throw std::logic_error("GeneralizedScalarTransport::prepare_execution "
                               "requires initialized Kokkos.");
    }

    static_assert(std::is_same<Kokkos::DefaultExecutionSpace, Kokkos::Cuda>::value,
        "CUDA preparation requires CUDA as the default execution space.");

    const Kokkos::Cuda execution;

    cudaStreamCaptureStatus capture_status = cudaStreamCaptureStatusNone;

    require_cuda_success(cudaStreamIsCapturing(execution.cuda_stream(), &capture_status),
        "query capture status");

    if (capture_status != cudaStreamCaptureStatusNone) {
        throw std::logic_error("GeneralizedScalarTransport::prepare_execution "
                               "must run before CUDA graph capture.");
    }

    require_cuda_success(cudaGetLastError(), "CUDA error before backend preparation");

    const auto preparation_policy =
        Kokkos::Experimental::require(Kokkos::RangePolicy<Kokkos::Cuda>(execution, 0, 1),
            Kokkos::Experimental::WorkItemProperty::HintLightWeight);

    // Initialize lazy CUDA symbols owned by this compilation unit without
    // sharing runtime state with the captured production functor.
    Kokkos::parallel_for("PrepareGeneralizedScalarTransport",
        preparation_policy,
        KOKKOS_LAMBDA(const int){});

    require_cuda_success(cudaGetLastError(), "launch backend preparation");

    execution.fence("Complete GeneralizedScalarTransport backend preparation");

    require_cuda_success(cudaGetLastError(), "complete backend preparation");
#endif
}

void
GeneralizedScalarTransport::validate_volume(
    const Core::Field<3>& field, const int nz, const char* role) const {
    const auto& data = field.get_device_data();

    if (static_cast<int>(data.extent(0)) != nz ||
        static_cast<int>(data.extent(1)) != layout_.local_total_ny() ||
        static_cast<int>(data.extent(2)) != layout_.local_total_nx()) {
        throw std::invalid_argument(
            std::string("GeneralizedScalarTransport: incorrect volume extents for ") + role + ".");
    }
}

void
GeneralizedScalarTransport::add_flux_convergence(const Core::Field<3>& scalar_q,
    const Core::Field<3>& contravariant_mass_flux_q1,
    const Core::Field<3>& contravariant_mass_flux_q2,
    const Core::Field<3>& vertical_mass_flux,
    const Core::Field<1>& vertical_cell_spacing,
    Core::Field<3>& out_flux_convergence,
    const int k_begin,
    const int k_end) const {
    const int nz = static_cast<int>(scalar_q.get_device_data().extent(0));

    if (k_begin < 0 || k_end <= k_begin || k_end > nz || k_end - k_begin < 3) {
        throw std::invalid_argument("GeneralizedScalarTransport requires "
                                    "a valid half-open vertical range containing "
                                    "at least three scalar levels.");
    }

    validate_volume(scalar_q, nz, "scalar_q");

    validate_volume(contravariant_mass_flux_q1, nz, "contravariant_mass_flux_q1");

    validate_volume(contravariant_mass_flux_q2, nz, "contravariant_mass_flux_q2");

    validate_volume(vertical_mass_flux, nz, "vertical_mass_flux");

    validate_volume(out_flux_convergence, nz, "out_flux_convergence");

    if (static_cast<int>(vertical_cell_spacing.get_device_data().extent(0)) < k_end) {
        throw std::invalid_argument("GeneralizedScalarTransport received "
                                    "insufficient vertical-cell spacing entries.");
    }

    auto scalar_data = scalar_q.get_device_data();

    auto contravariant_mass_flux_q1_data = contravariant_mass_flux_q1.get_device_data();

    auto contravariant_mass_flux_q2_data = contravariant_mass_flux_q2.get_device_data();

    auto vertical_mass_flux_data = vertical_mass_flux.get_device_data();

    auto vertical_cell_spacing_data = vertical_cell_spacing.get_device_data();

    auto output_data = out_flux_convergence.get_mutable_device_data();

    const std::array<const VVM::Real*, 5> input_storage = {scalar_data.data(),
        contravariant_mass_flux_q1_data.data(),
        contravariant_mass_flux_q2_data.data(),
        vertical_mass_flux_data.data(),
        vertical_cell_spacing_data.data()};

    for (const VVM::Real* input : input_storage) {
        if (input == output_data.data()) {
            throw std::invalid_argument("GeneralizedScalarTransport requires "
                                        "distinct input and output storage.");
        }
    }

    ScalarTransportFunctor functor;

    functor.transport = device_transport_;
    functor.scalar = scalar_data;

    functor.contravariant_mass_flux_q1 = contravariant_mass_flux_q1_data;

    functor.contravariant_mass_flux_q2 = contravariant_mass_flux_q2_data;

    functor.vertical_mass_flux = vertical_mass_flux_data;
    functor.vertical_cell_spacing = vertical_cell_spacing_data;
    functor.output = output_data;
    functor.k_begin = k_begin;
    functor.k_end = k_end;

    const int h = layout_.halo;
    const int ny = layout_.local_total_ny();
    const int nx = layout_.local_total_nx();

    const auto policy = Kokkos::Experimental::require(
        ScalarTransportPolicy({k_begin, h, h}, {k_end, ny - h, nx - h}),
        Kokkos::Experimental::WorkItemProperty::HintLightWeight);

    Kokkos::parallel_for("GeneralizedTakacsScalarFluxConvergence", policy, functor);
}

} // namespace Operators
} // namespace Dynamics
} // namespace VVM
