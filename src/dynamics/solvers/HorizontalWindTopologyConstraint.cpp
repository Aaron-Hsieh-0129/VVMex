#include "dynamics/solvers/HorizontalWindTopologyConstraint.hpp"

#include "core/geometry/GeometryKind.hpp"
#include "core/geometry/HorizontalLocation.hpp"
#include "dynamics/operators/HorizontalVectorConversion.hpp"

#include <array>
#include <mpi.h>
#include <stdexcept>
#include <string>

#if defined(ENABLE_NCCL)
#include <cuda_runtime.h>
#include <nccl.h>
#endif

namespace VVM::Dynamics {
namespace {

enum class RegularLatLonConstraintMode { PeriodicCycles, BoundedQ2Wall };

#if defined(ENABLE_NCCL)

void
require_cuda(cudaError_t result, const char* operation) {
    if (result != cudaSuccess) {
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(result));
    }
}

void
require_nccl(ncclResult_t result, const char* operation) {
    if (result != ncclSuccess) {
        throw std::runtime_error(std::string(operation) + ": " + ncclGetErrorString(result));
    }
}

#endif

RegularLatLonConstraintMode
select_mode(const Core::Grid& grid) {
    using Core::Geometry::GeometryKind;

    if (grid.geometry().kind() != GeometryKind::RegularLatLon) {
        throw std::invalid_argument("RegularLatLon circulation constraint "
                                    "requires RLL geometry.");
    }

    const auto& topology = grid.horizontal_specification().topology;

    if (topology.q1 != Core::HorizontalEdgeTopology::Periodic) {
        throw std::invalid_argument("RegularLatLon circulation constraint "
                                    "requires periodic q1.");
    }

    if (topology.q2 == Core::HorizontalEdgeTopology::Periodic) {
        return RegularLatLonConstraintMode::PeriodicCycles;
    }

    if (topology.q2 == Core::HorizontalEdgeTopology::Bounded) {
        return RegularLatLonConstraintMode::BoundedQ2Wall;
    }

    throw std::invalid_argument("Unsupported q2 topology for RegularLatLon "
                                "circulation constraint.");
}

class RegularLatLonCirculationConstraint final : public HorizontalWindTopologyConstraint {
public:
    RegularLatLonCirculationConstraint(const Core::Grid& grid,
        Core::State& state,
        Core::Field<3>& covariant_q1_wind,
        Core::Field<3>& covariant_q2_wind)
        : grid_(grid), state_(state), covariant_q1_wind_(covariant_q1_wind),
          covariant_q2_wind_(covariant_q2_wind), mode_(select_mode(grid)) {

        const int gx = grid_.get_global_points_x();

        const int gy = grid_.get_global_points_y();

        if (gx <= 0 || gy <= 0) {
            throw std::invalid_argument("RegularLatLon circulation constraint "
                                        "requires a nonempty global domain.");
        }

        const int count = mode_ == RegularLatLonConstraintMode::PeriodicCycles ? gx + 2 * gy : gx;

        contributions_ = std::make_unique<Core::Field<1>>("RLL topology constraint contributions",
            std::array<int, 1>{count});

#if defined(ENABLE_NCCL)

        targets_device_ = Kokkos::View<Real*, Kokkos::DefaultExecutionSpace::memory_space>(
            "RLL topology constraint targets",
            2);

        measurements_device_ = Kokkos::View<Real*, Kokkos::DefaultExecutionSpace::memory_space>(
            "RLL topology constraint measurements",
            3);

        Kokkos::deep_copy(targets_device_, real(0.0));

        Kokkos::deep_copy(measurements_device_, real(0.0));

#endif
    }

    void
    before_recovery(bool initial) override {
        if (!initial) {
            return;
        }

        // The periodic-cycle target belongs to the incoming model state.
        // At this point the generalized diagnostic has not yet produced its
        // covariant scratch wind, so capture the target from the physical
        // compatibility representation.
        if (mode_ == RegularLatLonConstraintMode::PeriodicCycles) {
            measure_periodic_cycles_from_physical();
            capture_target();
        }
    }

    void
    after_recovery(bool initial) override {
        measure_recovered_covariant_wind();

        // For bounded q2, the target is the first diagnosed wind.
        if (mode_ == RegularLatLonConstraintMode::BoundedQ2Wall && initial) {

            capture_target();
        }

        if (!has_target_) {
            throw std::logic_error("Horizontal wind topology constraint "
                                   "target has not been initialized.");
        }

        apply_correction();
    }

    void
    seed_from_physical_wind() override {
        if (mode_ == RegularLatLonConstraintMode::PeriodicCycles) {
            measure_periodic_cycles_from_physical();
        }
        else {
            measure_bounded_q2_wall_from_physical();
        }
        capture_target();
    }

    void
    measure_periodic_cycles_from_covariant() {
        using Core::Geometry::HorizontalLocation;

        const int h = grid_.get_halo_cells();
        const int nz = grid_.get_local_total_points_z();
        const int ny = grid_.get_local_total_points_y();
        const int nx = grid_.get_local_total_points_x();

        const int gx = grid_.get_global_points_x();
        const int gy = grid_.get_global_points_y();

        const int si = grid_.get_local_physical_start_x();
        const int sj = grid_.get_local_physical_start_y();

        const int top = nz - h - 1;

        const auto q1_cov = covariant_q1_wind_.get_device_data();
        const auto q2_cov = covariant_q2_wind_.get_device_data();

        const auto h1_at_v =
            grid_.geometry().device_view(HorizontalLocation::V).contravariant_to_physical.a11;

        const Real radius = grid_.horizontal_specification().geometry.regular_lat_lon.radius;

        const auto contributions = contributions_->get_mutable_device_data();

        Kokkos::deep_copy(contributions, real(0.0));

        Kokkos::parallel_for("RLLPeriodicCovariantCycleIntegrals",
            Kokkos::MDRangePolicy<Kokkos::Rank<2>>({h, h}, {ny - h, nx - h}),
            KOKKOS_LAMBDA(const int j, const int i) {
                if (sj == 0 && j == h) {
                    contributions(si + i - h) = q1_cov(top, j, i);
                }

                if (si == 0 && i == h) {
                    contributions(gx + sj + j - h) = q2_cov(top, j, i);
                    contributions(gx + gy + sj + j - h) = radius / h1_at_v(j, i);
                }
            });

        reduce_periodic_cycles(gx, gy);
    }

private:
    struct Targets {
        Real q1 = real(0.0);
        Real q2 = real(0.0);
    };

    struct Measurements {
        Real q1 = real(0.0);
        Real q2 = real(0.0);
        Real weight = real(0.0);
    };

    void
    measure_recovered_covariant_wind() {
        if (mode_ == RegularLatLonConstraintMode::PeriodicCycles) {
            measure_periodic_cycles_from_covariant();
        }
        else {
            measure_bounded_q2_wall_from_covariant();
        }
    }

public:
    // IMPORTANT:
    //
    // NVHPC/CUDA extended host-device lambdas require
    // the enclosing member function to have public
    // access. These functions contain KOKKOS_LAMBDA
    // launches.
    //
    // The concrete class itself remains inside this
    // translation unit's anonymous namespace, so
    // making these helpers public does NOT expose them
    // as part of VVMex's external API.

    void
    capture_target() {

#if defined(ENABLE_NCCL)

        const auto targets = targets_device_;

        const auto measurements = measurements_device_;

        const bool periodic = mode_ == RegularLatLonConstraintMode::PeriodicCycles;

        Kokkos::parallel_for("CaptureHorizontalTopologyConstraintTarget",
            Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(0, 1),
            KOKKOS_LAMBDA(const int) {
                targets(0) = measurements(0);

                if (periodic) {
                    targets(1) = measurements(1);
                }
            });

#else

        targets_.q1 = measurements_.q1;

        if (mode_ == RegularLatLonConstraintMode::PeriodicCycles) {

            targets_.q2 = measurements_.q2;
        }

#endif

        has_target_ = true;
    }

    void
    apply_correction() {
        if (mode_ == RegularLatLonConstraintMode::PeriodicCycles) {

            apply_periodic_correction();
        }
        else {
            apply_bounded_q2_wall_correction();
        }
    }

    void
    measure_bounded_q2_wall_from_covariant() {
        const int h = grid_.get_halo_cells();

        const int nz = grid_.get_local_total_points_z();

        const int ny = grid_.get_local_total_points_y();

        const int nx = grid_.get_local_total_points_x();

        const int gx = grid_.get_global_points_x();

        const int top = nz - h - 1;

        const bool owns_q2_minus = grid_.get_local_physical_start_y() == 0;

        const int start_i = grid_.get_local_physical_start_x();

        const auto q1_cov = covariant_q1_wind_.get_device_data();

        const auto contributions = contributions_->get_mutable_device_data();

        Kokkos::deep_copy(contributions, real(0.0));

        Kokkos::parallel_for("RLLBoundedQ2CovariantCirculation",
            Kokkos::MDRangePolicy<Kokkos::Rank<2>>({h, h}, {ny - h, nx - h}),
            KOKKOS_LAMBDA(const int j, const int i) {
                if (owns_q2_minus && j == h) {

                    contributions(start_i + i - h) = q1_cov(top, j, i);
                }
            });

        reduce_bounded_q2_wall(gx);
    }

    void
    measure_bounded_q2_wall_from_physical() {
        const int h = grid_.get_halo_cells();
        const int top = grid_.get_local_total_points_z() - h - 1;
        const int ny = grid_.get_local_total_points_y();
        const int nx = grid_.get_local_total_points_x();
        const int gx = grid_.get_global_points_x();
        const int start_i = grid_.get_local_physical_start_x();
        const bool owns_south = grid_.get_local_physical_start_y() == 0;

        const auto u = state_.get_field<3>("u").get_device_data();
        const auto h1 = grid_.geometry()
                            .device_view(Core::Geometry::HorizontalLocation::U)
                            .contravariant_to_physical.a11;
        const auto contributions = contributions_->get_mutable_device_data();
        Kokkos::deep_copy(contributions, real(0.0));
        Kokkos::parallel_for("RLLRestartSouthWallCirculation",
            Kokkos::MDRangePolicy<Kokkos::Rank<2>>({h, h}, {ny - h, nx - h}),
            KOKKOS_LAMBDA(const int j, const int i) {
                if (owns_south && j == h) {
                    contributions(start_i + i - h) = u(top, j, i) * h1(j, i);
                }
            });
        reduce_bounded_q2_wall(gx);
    }

    void
    reduce_bounded_q2_wall(int gx) {
        const auto contributions = contributions_->get_mutable_device_data();

#if defined(ENABLE_NCCL)

        reduce_device_contributions(gx,
            "RLL channel circulation NCCL "
            "all-reduce");

        const auto measurements = measurements_device_;

        Kokkos::parallel_for("RLLSouthWallCirculationFinalize",
            Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(0, 1),
            KOKKOS_LAMBDA(const int) {
                Real current = real(0.0);

                for (int i = 0; i < gx; ++i) {

                    current += contributions(i);
                }

                measurements(0) = current / static_cast<Real>(gx);
            });

#else

        auto values = contributions_->get_host_data();

        const int mpi_result =
            MPI_Allreduce(MPI_IN_PLACE, values.data(), gx, VVM_MPI_REAL, MPI_SUM, grid_.get_comm());

        if (mpi_result != MPI_SUCCESS) {
            throw std::runtime_error("RLL channel circulation "
                                     "MPI_Allreduce failed.");
        }

        Real current = real(0.0);

        for (int i = 0; i < gx; ++i) {

            current += values(i);
        }

        measurements_.q1 = current / static_cast<Real>(gx);

#endif
    }

    void
    apply_bounded_q2_wall_correction() {
        const int h = grid_.get_halo_cells();

        const int nz = grid_.get_local_total_points_z();

        const int ny = grid_.get_local_total_points_y();

        const int nx = grid_.get_local_total_points_x();

        const int bottom = h - 1;

        const int top = nz - h - 1;

        auto q1_cov = covariant_q1_wind_.get_mutable_device_data();

#if defined(ENABLE_NCCL)

        const auto targets = targets_device_;

        const auto measurements = measurements_device_;

        Kokkos::parallel_for("PreserveBoundedQ2CovariantCirculation",
            Kokkos::MDRangePolicy<Kokkos::Rank<3>>({bottom, h, h}, {top + 1, ny - h, nx - h}),
            KOKKOS_LAMBDA(const int k, const int j, const int i) {
                const Real correction = targets(0) - measurements(0);

                q1_cov(k, j, i) += correction;
            });

#else

        const Real correction = targets_.q1 - measurements_.q1;

        Kokkos::parallel_for("PreserveBoundedQ2CovariantCirculation",
            Kokkos::MDRangePolicy<Kokkos::Rank<3>>({bottom, h, h}, {top + 1, ny - h, nx - h}),
            KOKKOS_LAMBDA(const int k, const int j, const int i) {
                q1_cov(k, j, i) += correction;
            });

#endif
    }

    void
    measure_periodic_cycles_from_physical() {
        using Core::Geometry::HorizontalLocation;
        using Operators::HorizontalVectorConversion;

        const int h = grid_.get_halo_cells();
        const int nz = grid_.get_local_total_points_z();
        const int ny = grid_.get_local_total_points_y();
        const int nx = grid_.get_local_total_points_x();

        const int gx = grid_.get_global_points_x();
        const int gy = grid_.get_global_points_y();

        const int si = grid_.get_local_physical_start_x();
        const int sj = grid_.get_local_physical_start_y();

        const int top = nz - h - 1;

        const auto u = state_.get_field<3>("u").get_device_data();
        const auto v = state_.get_field<3>("v").get_device_data();

        const auto h1_at_u =
            grid_.geometry().device_view(HorizontalLocation::U).contravariant_to_physical.a11;
        const auto h2_at_v =
            grid_.geometry().device_view(HorizontalLocation::V).contravariant_to_physical.a22;
        const auto h1_at_v =
            grid_.geometry().device_view(HorizontalLocation::V).contravariant_to_physical.a11;

        const Real radius = grid_.horizontal_specification().geometry.regular_lat_lon.radius;

        const auto contributions = contributions_->get_mutable_device_data();

        Kokkos::deep_copy(contributions, real(0.0));

        Kokkos::parallel_for("RLLPeriodicCycleIntegrals",
            Kokkos::MDRangePolicy<Kokkos::Rank<2>>({h, h}, {ny - h, nx - h}),
            KOKKOS_LAMBDA(const int j, const int i) {
                if (sj == 0 && j == h) {

                    contributions(si + i - h) =
                        HorizontalVectorConversion::physical_to_covariant(u(top, j, i),
                            h1_at_u(j, i));
                }

                if (si == 0 && i == h) {

                    contributions(gx + sj + j - h) =
                        HorizontalVectorConversion::physical_to_covariant(v(top, j, i),
                            h2_at_v(j, i));

                    contributions(gx + gy + sj + j - h) = radius / h1_at_v(j, i);
                }
            });

        reduce_periodic_cycles(gx, gy);
    }

    void
    reduce_periodic_cycles(int gx, int gy) {

        const int count = gx + 2 * gy;

        const auto contributions = contributions_->get_mutable_device_data();

#if defined(ENABLE_NCCL)

        reduce_device_contributions(count,
            "RLL periodic circulation NCCL "
            "all-reduce");

        const auto measurements = measurements_device_;

        Kokkos::parallel_for("RLLPeriodicCycleIntegralsFinalize",
            Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(0, 1),
            KOKKOS_LAMBDA(const int) {
                Real zonal = real(0.0);

                Real meridional = real(0.0);

                Real weight = real(0.0);

                for (int i = 0; i < gx; ++i) {

                    zonal += contributions(i);
                }

                for (int j = 0; j < gy; ++j) {

                    meridional += contributions(gx + j);

                    weight += contributions(gx + gy + j);
                }

                measurements(0) = zonal / static_cast<Real>(gx);

                measurements(1) = meridional;

                measurements(2) = weight;
            });

#else

        auto values = contributions_->get_host_data();

        const int mpi_result = MPI_Allreduce(MPI_IN_PLACE,
            values.data(),
            count,
            VVM_MPI_REAL,
            MPI_SUM,
            grid_.get_comm());

        if (mpi_result != MPI_SUCCESS) {
            throw std::runtime_error("RLL periodic circulation "
                                     "MPI_Allreduce failed.");
        }

        Real zonal = real(0.0);

        Real meridional = real(0.0);

        Real weight = real(0.0);

        for (int i = 0; i < gx; ++i) {

            zonal += values(i);
        }

        for (int j = 0; j < gy; ++j) {

            meridional += values(gx + j);

            weight += values(gx + gy + j);
        }

        measurements_.q1 = zonal / static_cast<Real>(gx);

        measurements_.q2 = meridional;

        measurements_.weight = weight;

#endif
    }

    void
    apply_periodic_correction() {
        using Core::Geometry::HorizontalLocation;

        const int h = grid_.get_halo_cells();
        const int nz = grid_.get_local_total_points_z();
        const int ny = grid_.get_local_total_points_y();
        const int nx = grid_.get_local_total_points_x();

        const int bottom = h - 1;
        const int top = nz - h - 1;

        auto q1_cov = covariant_q1_wind_.get_mutable_device_data();
        auto q2_cov = covariant_q2_wind_.get_mutable_device_data();

        const auto h1_at_v =
            grid_.geometry().device_view(HorizontalLocation::V).contravariant_to_physical.a11;
        const auto h2_at_v =
            grid_.geometry().device_view(HorizontalLocation::V).contravariant_to_physical.a22;

#if defined(ENABLE_NCCL)
        const auto targets = targets_device_;

        const auto measurements = measurements_device_;

        Kokkos::parallel_for("PreserveRLLPeriodicCovariantCirculation",
            Kokkos::MDRangePolicy<Kokkos::Rank<3>>({bottom, h, h}, {top + 1, ny - h, nx - h}),
            KOKKOS_LAMBDA(const int k, const int j, const int i) {
                const Real dq1 = targets(0) - measurements(0);

                const Real harmonic_q2 = (targets(1) - measurements(1)) / measurements(2);

                q1_cov(k, j, i) += dq1;
                q2_cov(k, j, i) += (harmonic_q2 / h1_at_v(j, i)) * h2_at_v(j, i);
            });

#else

        const Real dq1 = targets_.q1 - measurements_.q1;
        const Real harmonic_q2 = (targets_.q2 - measurements_.q2) / measurements_.weight;

        Kokkos::parallel_for("PreserveRLLPeriodicCovariantCirculation",
            Kokkos::MDRangePolicy<Kokkos::Rank<3>>({bottom, h, h}, {top + 1, ny - h, nx - h}),
            KOKKOS_LAMBDA(const int k, const int j, const int i) {
                q1_cov(k, j, i) += dq1;
                q2_cov(k, j, i) += (harmonic_q2 / h1_at_v(j, i)) * h2_at_v(j, i);
            });

#endif
    }

private:
#if defined(ENABLE_NCCL)

    void
    reduce_device_contributions(int count, const char* operation) {

        const auto contributions = contributions_->get_mutable_device_data();

        const cudaStream_t nccl_stream = state_.get_cuda_stream();

        const cudaStream_t kokkos_stream = Kokkos::DefaultExecutionSpace().cuda_stream();

        const bool separate_stream = nccl_stream != kokkos_stream;

        if (separate_stream) {
            Kokkos::DefaultExecutionSpace().fence("Horizontal topology constraint "
                                                  "producer ready");
        }

        require_nccl(ncclAllReduce(contributions.data(),
                         contributions.data(),
                         static_cast<size_t>(count),
                         VVM_NCCL_REAL,
                         ncclSum,
                         state_.get_nccl_comm(),
                         nccl_stream),
            operation);

        if (separate_stream) {
            require_cuda(cudaStreamSynchronize(nccl_stream),
                "Horizontal topology constraint "
                "NCCL synchronization");
        }
    }

#endif

    const Core::Grid& grid_;
    Core::State& state_;

    Core::Field<3>& covariant_q1_wind_;
    Core::Field<3>& covariant_q2_wind_;

    RegularLatLonConstraintMode mode_;

    std::unique_ptr<Core::Field<1>> contributions_;

    Targets targets_;
    Measurements measurements_;

    bool has_target_ = false;

#if defined(ENABLE_NCCL)

    Kokkos::View<Real*, Kokkos::DefaultExecutionSpace::memory_space> targets_device_;

    Kokkos::View<Real*, Kokkos::DefaultExecutionSpace::memory_space> measurements_device_;

#endif
};

} // namespace

std::unique_ptr<HorizontalWindTopologyConstraint>
make_regular_lat_lon_circulation_constraint(const Core::Grid& grid,
    Core::State& state,
    Core::Field<3>& covariant_q1_wind,
    Core::Field<3>& covariant_q2_wind) {

    return std::make_unique<RegularLatLonCirculationConstraint>(grid,
        state,
        covariant_q1_wind,
        covariant_q2_wind);
}

} // namespace VVM::Dynamics
