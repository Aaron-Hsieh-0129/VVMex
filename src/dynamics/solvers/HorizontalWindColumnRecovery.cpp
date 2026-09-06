#include "dynamics/solvers/HorizontalWindColumnRecovery.hpp"

#include <stdexcept>
#include <string>

#if defined(KOKKOS_ENABLE_CUDA)
#include <cuda_runtime.h>
#endif

namespace VVM {
namespace Dynamics {

#if defined(KOKKOS_ENABLE_CUDA)
namespace {

void require_cuda_success(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string("HorizontalWindColumnRecovery: ") + operation + ": " + cudaGetErrorString(status));
    }
}

} // namespace
#endif

HorizontalWindColumnRecovery::HorizontalWindColumnRecovery(const Core::Geometry::HorizontalGeometry& geometry)
    : layout_(geometry.layout()),
      reconstruction_(Operators::make_horizontal_wind_reconstruction_device_view(geometry)),
      vorticity_(Operators::make_horizontal_vorticity_device_view(geometry)) {

    if (layout_.halo < 1 || layout_.local_physical_nx < 1 || layout_.local_physical_ny < 1) {
        throw std::invalid_argument("HorizontalWindColumnRecovery requires physical horizontal cells and at least one halo cell.");
    }
}

void HorizontalWindColumnRecovery::prepare_execution() {
#if defined(KOKKOS_ENABLE_CUDA)
    if (!Kokkos::is_initialized()) {
        throw std::logic_error("HorizontalWindColumnRecovery::prepare_execution requires initialized Kokkos.");
    }

    const Kokkos::Cuda execution;
    cudaStreamCaptureStatus capture_status = cudaStreamCaptureStatusNone;

    require_cuda_success(cudaStreamIsCapturing(execution.cuda_stream(), &capture_status), "query capture status");

    if (capture_status != cudaStreamCaptureStatusNone) {
        throw std::logic_error("HorizontalWindColumnRecovery::prepare_execution must run before CUDA graph capture.");
    }

    require_cuda_success(cudaGetLastError(), "CUDA error before backend preparation");

    // Keep this launch in the same compilation unit as recover().
    // Some Kokkos/desul builds lazily copy lock-array pointers to device symbols
    // on the first launch from a compilation unit. Those synchronous copies
    // must finish before manual stream capture begins.
    //
    // A nonempty range is required to enter the backend launch path.
    // The kernel itself does not access any model or scratch fields.
    Kokkos::parallel_for("PrepareHorizontalWindColumnRecovery",
        Kokkos::RangePolicy<Kokkos::Cuda>(execution, 0, 1),
        KOKKOS_LAMBDA(const int) {}
    );

    require_cuda_success(cudaGetLastError(), "launch backend preparation");
    execution.fence("Complete HorizontalWindColumnRecovery backend preparation");
#endif
}

void HorizontalWindColumnRecovery::validate_horizontal_field(const Core::Field<2>& field, const char* role) const {
    const auto& data = field.get_device_data();

    if (static_cast<int>(data.extent(0)) != layout_.local_total_ny()
        || static_cast<int>(data.extent(1)) != layout_.local_total_nx()) {
        throw std::invalid_argument(std::string("HorizontalWindColumnRecovery: incorrect horizontal extents for ") + role + ".");
    }
}

void HorizontalWindColumnRecovery::validate_volume(const Core::Field<3>& field, int nz, const char* role) const {
    const auto& data = field.get_device_data();

    if (static_cast<int>(data.extent(0)) != nz
        || static_cast<int>(data.extent(1)) != layout_.local_total_ny()
        || static_cast<int>(data.extent(2)) != layout_.local_total_nx()) {
        throw std::invalid_argument(std::string("HorizontalWindColumnRecovery: incorrect volume extents for ") + role + ".");
    }
}

void HorizontalWindColumnRecovery::recover(const Core::Field<2>& psi, const Core::Field<2>& chi,
    const Core::Field<3>& w, const Core::Field<3>& omega1, const Core::Field<3>& omega2,
    const Core::Field<1>& spacing, Core::Field<3>& output1, Core::Field<3>& output2,
    int bottom_level, int top_level) const {

    const int nz = static_cast<int>(w.get_device_data().extent(0));

    if (bottom_level < 0 || top_level < bottom_level || top_level >= nz) {
        throw std::invalid_argument("HorizontalWindColumnRecovery: invalid bottom/top level range.");
    }

    validate_horizontal_field(psi, "psi");
    validate_horizontal_field(chi, "chi");
    validate_volume(w, nz, "w");
    validate_volume(omega1, nz, "omega1");
    validate_volume(omega2, nz, "omega2");
    validate_volume(output1, nz, "output1");
    validate_volume(output2, nz, "output2");

    // The highest spacing entry used is top_level-1.
    if (static_cast<int>(spacing.get_device_data().extent(0)) < top_level) {
        throw std::invalid_argument("HorizontalWindColumnRecovery: insufficient vertical spacing entries.");
    }

    const auto psi_data = psi.get_device_data();
    const auto chi_data = chi.get_device_data();
    const auto w_data = w.get_device_data();
    const auto omega1_data = omega1.get_device_data();
    const auto omega2_data = omega2.get_device_data();
    const auto spacing_data = spacing.get_device_data();
    const auto output1_data = output1.get_mutable_device_data();
    const auto output2_data = output2.get_mutable_device_data();

    if (output1_data.data() == output2_data.data()) {
        throw std::invalid_argument("HorizontalWindColumnRecovery requires distinct output storage.");
    }

    const Real* input_data[] = {
        psi_data.data(), chi_data.data(), w_data.data(),
        omega1_data.data(), omega2_data.data(), spacing_data.data()
    };

    for (const Real* input : input_data) {
        if (input == output1_data.data() || input == output2_data.data()) {
            throw std::invalid_argument("HorizontalWindColumnRecovery requires distinct input and output storage.");
        }
    }

    const int h = layout_.halo;
    const int nx = layout_.local_total_nx();
    const int ny = layout_.local_total_ny();

    const auto policy = Kokkos::MDRangePolicy<Kokkos::Rank<2>>({h, h}, {ny - h, nx - h});
    const auto compact_policy = Kokkos::Experimental::require(
        policy, Kokkos::Experimental::WorkItemProperty::HintLightWeight);

    const auto reconstruction = reconstruction_;

    Kokkos::parallel_for("ReconstructCovariantTopWind", compact_policy,
        KOKKOS_LAMBDA(const int j, const int i) {
            output1_data(top_level, j, i) = reconstruction.calculate_covariant_q1_at_u(psi_data, chi_data, j, i);
            output2_data(top_level, j, i) = reconstruction.calculate_covariant_q2_at_v(psi_data, chi_data, j, i);
        }
    );

    const auto vorticity = vorticity_;

    Kokkos::parallel_for("RecoverCovariantWindColumn", compact_policy,
        KOKKOS_LAMBDA(const int j, const int i) {
            for (int k = top_level - 1; k >= bottom_level; --k) {
                const Real du1_dz = vorticity.calculate_covariant_q1_vertical_shear_at_u(w_data, omega2_data, k, j, i);
                const Real du2_dz = vorticity.calculate_covariant_q2_vertical_shear_at_v(w_data, omega1_data, k, j, i);

                output1_data(k, j, i) = output1_data(k + 1, j, i) - du1_dz * spacing_data(k);
                output2_data(k, j, i) = output2_data(k + 1, j, i) - du2_dz * spacing_data(k);
            }
        }
    );
}

} // namespace Dynamics
} // namespace VVM
