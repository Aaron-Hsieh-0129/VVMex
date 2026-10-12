#include "dynamics/solvers/HorizontalWindTopologyConstraint.hpp"
#include "core/StartupCollectives.hpp"

#include "core/geometry/GeometryKind.hpp"
#include "core/geometry/HorizontalLocation.hpp"
#include "core/geometry/StaggeredVectorTransform.hpp"
#include "dynamics/solvers/HorizontalEllipticSolver.hpp"

#include <array>
#include <cmath>
#include <mpi.h>
#include <stdexcept>
#include <string>

#if defined(ENABLE_NCCL)
#include <cuda_runtime.h>
#include <nccl.h>
#endif

namespace VVM::Dynamics {
namespace {

enum class GeneralizedConstraintMode { PeriodicCycles, BoundedQ2Wall };

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

GeneralizedConstraintMode
select_mode(const Core::Grid& grid) {
    using Core::Geometry::GeometryKind;

    const auto& topology = grid.horizontal_specification().topology;

    if (topology.q1 != Core::HorizontalEdgeTopology::Periodic) {
        throw std::invalid_argument("Generalized circulation constraint "
                                    "requires periodic q1.");
    }

    if (topology.q2 == Core::HorizontalEdgeTopology::Periodic) {
        return GeneralizedConstraintMode::PeriodicCycles;
    }

    if (topology.q2 == Core::HorizontalEdgeTopology::Bounded) {
        return GeneralizedConstraintMode::BoundedQ2Wall;
    }

    throw std::invalid_argument("Unsupported q2 topology for Generalized "
                                "circulation constraint.");
}

class GeneralizedCirculationConstraint final : public HorizontalWindTopologyConstraint {
public:
    GeneralizedCirculationConstraint(const Core::Grid& grid,
        Core::State& state,
        Core::Field<3>& covariant_q1_wind,
        Core::Field<3>& covariant_q2_wind,
        Core::HaloExchanger* halo = nullptr,
        HorizontalEllipticSolver* solver = nullptr,
        int iterations = 0)
        : grid_(grid), state_(state), covariant_q1_wind_(covariant_q1_wind),
          covariant_q2_wind_(covariant_q2_wind), mode_(select_mode(grid)) {

        const int gx = grid_.get_global_points_x();

        const int gy = grid_.get_global_points_y();

        if (gx <= 0 || gy <= 0) {
            throw std::invalid_argument("Generalized circulation constraint "
                                        "requires a nonempty global domain.");
        }

        const int count = mode_ == GeneralizedConstraintMode::PeriodicCycles ? gx + 2 * gy : gx;

        contributions_ =
            std::make_unique<Core::Field<1>>("Mapped topology constraint contributions",
                std::array<int, 1>{count});
        initialize_harmonic_basis(halo, solver, iterations);
        if (canonical_correction_q1_) {
            prepare_canonical_harmonic_correction(*halo);
        }

#if defined(ENABLE_NCCL)

        targets_device_ = Kokkos::View<Real*, Kokkos::DefaultExecutionSpace::memory_space>(
            "Mapped topology constraint targets",
            2);

        measurements_device_ = Kokkos::View<Real*, Kokkos::DefaultExecutionSpace::memory_space>(
            "Mapped topology constraint measurements",
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
        if (mode_ == GeneralizedConstraintMode::PeriodicCycles) {
            measure_periodic_cycles_from_physical();
            capture_target();
        }
    }

    void
    after_recovery(bool initial) override {
        measure_recovered_covariant_wind();

        // Prescribed analytic channels use the first diagnosed wind unless an
        // incoming-state target has already been seeded.
        if (mode_ == GeneralizedConstraintMode::BoundedQ2Wall && initial && !has_target_) {

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
        if (mode_ == GeneralizedConstraintMode::PeriodicCycles) {
            measure_periodic_cycles_from_physical();
        }
        else {
            measure_bounded_q2_wall_from_physical();
        }
        capture_target();
    }

    void
    seed_from_contravariant_wind() override {
        using Core::Geometry::HorizontalLocation;
        const auto u = grid_.geometry().device_view(HorizontalLocation::U).g_cov;
        const auto v = grid_.geometry().device_view(HorizontalLocation::V).g_cov;
        Core::Geometry::transform_staggered_vector<true>(
            state_.get_field<3>("u_con").get_device_data(),
            state_.get_field<3>("v_con").get_device_data(),
            covariant_q1_wind_.get_mutable_device_data(),
            covariant_q2_wind_.get_mutable_device_data(),
            Core::Geometry::Matrix2DeviceView{u.a11, u.a12, u.a12, u.a22},
            Core::Geometry::Matrix2DeviceView{v.a11, v.a12, v.a12, v.a22});
        measure_recovered_covariant_wind();
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

        const auto basis_q2 = basis_22_->get_device_data();

        const auto contributions = contributions_->get_mutable_device_data();

        Kokkos::deep_copy(contributions, real(0.0));

        Kokkos::parallel_for("MappedPeriodicCovariantCycleIntegrals",
            Kokkos::MDRangePolicy<Kokkos::Rank<2>>({h, h}, {ny - h, nx - h}),
            KOKKOS_LAMBDA(const int j, const int i) {
                if (sj == 0 && j == h) {
                    contributions(si + i - h) = q1_cov(top, j, i);
                }

                if (si == 0 && i == h) {
                    contributions(gx + sj + j - h) = q2_cov(top, j, i);
                    contributions(gx + gy + sj + j - h) = basis_q2(j, i);
                }
            });

        reduce_periodic_cycles(gx, gy);
    }

    struct PlaneVolume {
        Core::Field<2>::ViewType plane;
        KOKKOS_INLINE_FUNCTION int
        extent(int d) const {
            return d == 0 ? 1 : plane.extent(d - 1);
        }
        KOKKOS_INLINE_FUNCTION Real
        operator()(int, int j, int i) const {
            return plane(j, i);
        }
    };

    void
    prepare_canonical_harmonic_correction(Core::HaloExchanger& halo) {
        const int ny = grid_.get_local_total_points_y(), nx = grid_.get_local_total_points_x();
        const int h = grid_.get_halo_cells();
        halo.exchange_multiple_halos(
            std::vector<Core::Field<2>*>{basis_11_.get(), basis_12_.get()});
        Core::Boundary::HorizontalBoundaryStencils boundary(grid_);
        boundary.fill_centered_q2_neumann_halos(*basis_11_);
        boundary.fill_centered_q2_neumann_halos(*basis_12_);
        Core::Field<3> first("raised harmonic first", {1, ny, nx});
        Core::Field<3> second("raised harmonic second", {1, ny, nx});
        Core::Geometry::raise_staggered_wind(PlaneVolume{basis_11_->get_device_data()},
            PlaneVolume{basis_12_->get_device_data()},
            first.get_mutable_device_data(),
            second.get_mutable_device_data(),
            grid_.geometry());
        const auto q1 = first.get_device_data(), q2 = second.get_device_data();
        const auto c1 = canonical_correction_q1_->get_mutable_device_data();
        const auto c2 = canonical_correction_q2_->get_mutable_device_data();
        // These planes initially contain the directly reconstructed basis.
        Kokkos::parallel_for("CanonicalHarmonicCorrection",
            Kokkos::MDRangePolicy<Kokkos::Rank<2>>({h, h}, {ny - h, nx - h}),
            KOKKOS_LAMBDA(int j, int i) {
                c1(j, i) -= q1(0, j, i);
                c2(j, i) -= q2(0, j, i);
            });
    }

    void
    correct_canonical_recovered_wind() override {
        if (!canonical_correction_q1_) {
            return;
        }
        const int h = grid_.get_halo_cells(), nz = grid_.get_local_total_points_z();
        const int ny = grid_.get_local_total_points_y(), nx = grid_.get_local_total_points_x();
        const auto c1 = canonical_correction_q1_->get_device_data();
        const auto c2 = canonical_correction_q2_->get_device_data();
        const auto q1 = state_.get_field<3>("u_con").get_mutable_device_data();
        const auto q2 = state_.get_field<3>("v_con").get_mutable_device_data();
#if defined(ENABLE_NCCL)
        const auto targets = targets_device_, measurements = measurements_device_;
#else
        const Real amplitude = targets_.q1 - measurements_.q1;
#endif
        Kokkos::parallel_for("PreserveCanonicalChannelHarmonic",
            Kokkos::MDRangePolicy<Kokkos::Rank<3>>({h - 1, h, h}, {nz - h, ny - h, nx - h}),
            KOKKOS_LAMBDA(int k, int j, int i) {
#if defined(ENABLE_NCCL)
                const Real amplitude = targets(0) - measurements(0);
#endif
                q1(k, j, i) += amplitude * c1(j, i);
                q2(k, j, i) += amplitude * c2(j, i);
            });
    }

    void
    prepare_physical_mean_constraint(Core::HaloExchanger& halo) override {
        using Core::Geometry::HorizontalLocation;
        const int ny = grid_.get_local_total_points_y(), nx = grid_.get_local_total_points_x();
        const auto gu = grid_.geometry().device_view(HorizontalLocation::U);
        const auto gv = grid_.geometry().device_view(HorizontalLocation::V);
        const std::array<int, 2> shape{ny, nx};
        Core::Field<2> physical_u("harmonic physical u",
            shape,
            {Core::GridStaggering::StaggeredX, "m s-1", "harmonic physical u"});
        Core::Field<2> physical_v("harmonic physical v",
            shape,
            {Core::GridStaggering::StaggeredY, "m s-1", "harmonic physical v"});
        Core::Field<3> first("harmonic canonical first", {1, ny, nx}),
            second("harmonic canonical second", {1, ny, nx});
        Core::Field<3> u("harmonic physical first scratch", {1, ny, nx}),
            v("harmonic physical second scratch", {1, ny, nx});
        std::unique_ptr<Core::Boundary::HorizontalBoundaryStencils> boundary;
        if (mode_ == GeneralizedConstraintMode::BoundedQ2Wall) {
            boundary = std::make_unique<Core::Boundary::HorizontalBoundaryStencils>(grid_);
        }
        for (int n = 0; n < 2; ++n) {
            auto& cov1 = n == 0 ? *basis_11_ : *basis_21_;
            auto& cov2 = n == 0 ? *basis_12_ : *basis_22_;
            halo.exchange_multiple_halos(std::vector<Core::Field<2>*>{&cov1, &cov2});
            if (boundary) {
                boundary->fill_centered_q2_neumann_halos(cov1);
                boundary->fill_centered_q2_neumann_halos(cov2);
            }
            Core::Geometry::raise_staggered_wind(PlaneVolume{cov1.get_device_data()},
                PlaneVolume{cov2.get_device_data()},
                first.get_mutable_device_data(),
                second.get_mutable_device_data(),
                grid_.geometry());
            if (n == 0 && canonical_correction_q1_) {
                const int h = grid_.get_halo_cells();
                const auto c1 = canonical_correction_q1_->get_device_data();
                const auto c2 = canonical_correction_q2_->get_device_data();
                const auto q1 = first.get_mutable_device_data(),
                           q2 = second.get_mutable_device_data();
                Kokkos::parallel_for("DirectCanonicalMeanBasis",
                    Kokkos::MDRangePolicy<Kokkos::Rank<2>>({h, h}, {ny - h, nx - h}),
                    KOKKOS_LAMBDA(int j, int i) {
                        q1(0, j, i) += c1(j, i);
                        q2(0, j, i) += c2(j, i);
                    });
            }
            if (boundary) {
                boundary->fill_contravariant_wind_halos(first, second);
            }
            mean_basis_[2 * n] =
                std::make_unique<Core::Field<2>>("mean canonical first basis", shape);
            mean_basis_[2 * n + 1] =
                std::make_unique<Core::Field<2>>("mean canonical second basis", shape);
            const auto c1 = first.get_device_data(), c2 = second.get_device_data();
            const auto out1 = mean_basis_[2 * n]->get_mutable_device_data(),
                       out2 = mean_basis_[2 * n + 1]->get_mutable_device_data();
            Kokkos::parallel_for("StoreCanonicalMeanBasis",
                Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {ny, nx}),
                KOKKOS_LAMBDA(int j, int i) {
                    out1(j, i) = c1(0, j, i);
                    out2(j, i) = c2(0, j, i);
                });
            Core::Geometry::transform_staggered_vector<true>(c1,
                c2,
                u.get_mutable_device_data(),
                v.get_mutable_device_data(),
                gu.contravariant_to_physical,
                gv.contravariant_to_physical);
            const auto ud = u.get_device_data(), vd = v.get_device_data();
            const auto pu = physical_u.get_mutable_device_data(),
                       pv = physical_v.get_mutable_device_data();
            Kokkos::parallel_for("PhysicalMeanBasis",
                Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {ny, nx}),
                KOKKOS_LAMBDA(int j, int i) {
                    pu(j, i) = ud(0, j, i);
                    pv(j, i) = vd(0, j, i);
                });
            state_.calculate_horizontal_area_weighted_mean(physical_u, mean_u_);
            state_.calculate_horizontal_area_weighted_mean(physical_v, mean_v_);
            Kokkos::deep_copy(mean_response_[2 * n], mean_u_);
            Kokkos::deep_copy(mean_response_[2 * n + 1], mean_v_);
        }
        const Real determinant =
            mean_response_[0] * mean_response_[3] - mean_response_[1] * mean_response_[2];
        if (mode_ == GeneralizedConstraintMode::PeriodicCycles &&
            (!std::isfinite(determinant) || determinant == real(0.0))) {
            throw std::runtime_error(
                "Degenerate physical mean response of periodic harmonic winds.");
        }
        const Real norm =
            mean_response_[0] * mean_response_[0] + mean_response_[1] * mean_response_[1];
        if (mode_ == GeneralizedConstraintMode::BoundedQ2Wall &&
            (!std::isfinite(norm) || norm == real(0.0))) {
            throw std::runtime_error("Degenerate physical mean response of channel harmonic wind.");
        }
    }

    void
    nudge_physical_top_mean(
        const Core::Field<0>& target_u, const Core::Field<0>& target_v, Real fraction) override {
        if (!mean_basis_[0]) {
            throw std::logic_error("Physical mean wind constraint was not prepared.");
        }
        state_.calculate_horizontal_area_weighted_mean(state_.get_field<3>("u"), mean_u_);
        state_.calculate_horizontal_area_weighted_mean(state_.get_field<3>("v"), mean_v_);
        const auto current_u = mean_u_, current_v = mean_v_;
        const auto desired_u = target_u.get_device_data(), desired_v = target_v.get_device_data();
        const auto response = mean_response_;
        const bool periodic = mode_ == GeneralizedConstraintMode::PeriodicCycles;
        const auto b11 = mean_basis_[0]->get_device_data(), b12 = mean_basis_[1]->get_device_data();
        const auto b21 = mean_basis_[2]->get_device_data(), b22 = mean_basis_[3]->get_device_data();
        const auto u1 = state_.get_field<3>("u_con").get_mutable_device_data();
        const auto u2 = state_.get_field<3>("v_con").get_mutable_device_data();
        const int nz = grid_.get_local_total_points_z(), ny = grid_.get_local_total_points_y(),
                  nx = grid_.get_local_total_points_x();
#if defined(ENABLE_NCCL)
        const auto targets = targets_device_, measurements = measurements_device_;
#else
        Real cu, cv, du, dv;
        Kokkos::deep_copy(cu, current_u);
        Kokkos::deep_copy(cv, current_v);
        Kokkos::deep_copy(du, desired_u);
        Kokkos::deep_copy(dv, desired_v);
        du = fraction * (du - cu);
        dv = fraction * (dv - cv);
        const Real determinant = response[0] * response[3] - response[1] * response[2];
        const Real a1 = periodic ? (du * response[3] - dv * response[2]) / determinant
                                 : (du * response[0] + dv * response[1]) /
                                       (response[0] * response[0] + response[1] * response[1]);
        const Real a2 = periodic ? (dv * response[0] - du * response[1]) / determinant : real(0.0);
        targets_.q1 += a1;
        if (periodic) {
            targets_.q2 += a2 * measurements_.weight;
        }
#endif
        Kokkos::parallel_for("NudgeMappedPhysicalTopMean",
            Kokkos::MDRangePolicy<Kokkos::Rank<3>>({0, 0, 0}, {nz, ny, nx}),
            KOKKOS_LAMBDA(int k, int j, int i) {
                const Real du = fraction * (desired_u() - current_u()),
                           dv = fraction * (desired_v() - current_v());
                const Real determinant = response[0] * response[3] - response[1] * response[2];
                const Real a1 = periodic
                                    ? (du * response[3] - dv * response[2]) / determinant
                                    : (du * response[0] + dv * response[1]) /
                                          (response[0] * response[0] + response[1] * response[1]);
                const Real a2 =
                    periodic ? (dv * response[0] - du * response[1]) / determinant : real(0.0);
                u1(k, j, i) += a1 * b11(j, i) + a2 * b21(j, i);
                u2(k, j, i) += a1 * b12(j, i) + a2 * b22(j, i);
#if defined(ENABLE_NCCL)
                if (k == 0 && j == 0 && i == 0) {
                    targets(0) += a1;
                    if (periodic)
                        targets(1) += a2 * measurements(2);
                }
#endif
            });
    }

    struct AffinePotential {
        Core::Field<2>::ViewType periodic;
        int direction;
        Real dq1, dq2;
        KOKKOS_INLINE_FUNCTION Real
        operator()(int j, int i) const {
            return periodic(j, i) + (direction == 0 ? Real(i) * dq1 : Real(j) * dq2);
        }
    };

    void
    initialize_harmonic_basis(
        Core::HaloExchanger* halo, HorizontalEllipticSolver* solver, int iterations) {
        using Core::Geometry::GeometryFieldLayout;
        using Core::Geometry::HorizontalLocation;
        const int ny = grid_.get_local_total_points_y(), nx = grid_.get_local_total_points_x();
        const int h = grid_.get_halo_cells();
        const std::array<int, 2> shape{ny, nx};
        basis_11_ = std::make_unique<Core::Field<2>>("harmonic 1 covariant q1", shape);
        basis_12_ = std::make_unique<Core::Field<2>>("harmonic 1 covariant q2", shape);
        basis_21_ = std::make_unique<Core::Field<2>>("harmonic 2 covariant q1", shape);
        basis_22_ = std::make_unique<Core::Field<2>>("harmonic 2 covariant q2", shape);
        const auto gu = grid_.geometry().device_view(HorizontalLocation::U);
        const auto gv = grid_.geometry().device_view(HorizontalLocation::V);
        const auto zero = [](const auto& a) {
            return a.layout == GeometryFieldLayout::Constant && a.constant == real(0.0);
        };
        const auto row = [](const auto& a) {
            return a.layout == GeometryFieldLayout::Constant ||
                   a.layout == GeometryFieldLayout::VaryingJ;
        };
        const auto b11 = basis_11_->get_mutable_device_data(),
                   b12 = basis_12_->get_mutable_device_data();
        const auto b21 = basis_21_->get_mutable_device_data(),
                   b22 = basis_22_->get_mutable_device_data();
        // Analytic harmonic covectors for any orthogonal metric depending
        // only on q2. This also retains the inexpensive spherical channel path.
        if (zero(gu.sqrt_g_g_contra.a12) && zero(gv.sqrt_g_g_contra.a12) &&
            row(gu.sqrt_g_g_contra.a11) && row(gv.sqrt_g_g_contra.a22)) {
            Kokkos::deep_copy(b11, real(1.0));
            const auto weighted22 = gv.sqrt_g_g_contra.a22;
            Kokkos::parallel_for("SeparableMetricHarmonicBasis",
                Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {ny, nx}),
                KOKKOS_LAMBDA(int j, int i) { b22(j, i) = real(1.0) / weighted22(j, i); });
            return;
        }
        if (!halo || !solver || iterations <= 0) {
            throw std::invalid_argument("A general metric circulation constraint requires a halo "
                                        "exchanger and harmonic solve iterations.");
        }
        Core::Field<2> rhs("harmonic divergence rhs", shape),
            potential("harmonic potential", shape);
        Core::Field<2> zero_rhs("harmonic zero rhs", shape),
            zero_potential("harmonic zero potential", shape);
        HorizontalEllipticSolver::Options options;
        options.iterations = iterations;
        auto operation = Kokkos::View<Operators::GeneralizedHorizontalEllipticDeviceView>(
            "harmonic metric operator");
        auto host = Kokkos::create_mirror_view(operation);
        host() = Operators::make_generalized_horizontal_elliptic_device_view(grid_.geometry());
        Kokkos::deep_copy(operation, host);
        const auto pd = potential.get_mutable_device_data(), rd = rhs.get_mutable_device_data();
        const Real dq1 = grid_.geometry().dq1(), dq2 = grid_.geometry().dq2();
        if (mode_ == GeneralizedConstraintMode::PeriodicCycles) {
            // e_a+d(phi_a) is curl-free, with phi_a chosen so divergence
            // vanishes. Periodic gradients have zero circulation, therefore
            // these bases retain independent unit cycles on both directions.
            for (int direction = 0; direction < 2; ++direction) {
                Kokkos::deep_copy(pd, real(0.0));
                Kokkos::parallel_for("HarmonicPotentialRhs",
                    Kokkos::MDRangePolicy<Kokkos::Rank<2>>({h, h}, {ny - h, nx - h}),
                    KOKKOS_LAMBDA(int j, int i) {
                        rd(j, i) = -operation().calculate_jacobian_weighted_at_t(
                                       AffinePotential{pd, direction, dq1, dq2},
                                       j,
                                       i) /
                                   operation().jacobian_t(j, i);
                    });
                solver->solve_at_t(rhs, potential, options);
                const auto first = direction == 0 ? b11 : b21;
                const auto second = direction == 0 ? b12 : b22;
                Kokkos::parallel_for("PeriodicMetricHarmonicBasis",
                    Kokkos::MDRangePolicy<Kokkos::Rank<2>>({h, h}, {ny - h, nx - h}),
                    KOKKOS_LAMBDA(int j, int i) {
                        first(j, i) = (direction == 0 ? real(1.0) : real(0.0)) +
                                      (pd(j, i + 1) - pd(j, i)) / dq1;
                        second(j, i) = (direction == 1 ? real(1.0) : real(0.0)) +
                                       (pd(j + 1, i) - pd(j, i)) / dq2;
                    });
            }
        }
        else {
            // A unit wall-to-wall streamfunction difference gives a
            // divergence-free channel wind. L_Z psi=0 also removes its curl.
            canonical_correction_q1_ =
                std::make_unique<Core::Field<2>>("canonical channel harmonic q1", shape);
            canonical_correction_q2_ =
                std::make_unique<Core::Field<2>>("canonical channel harmonic q2", shape);
            const auto direct1 = canonical_correction_q1_->get_mutable_device_data();
            const auto direct2 = canonical_correction_q2_->get_mutable_device_data();
            options.psi_q2_plus = real(1.0);
            const int start_j = grid_.get_local_physical_start_y(),
                      global_ny = grid_.get_global_points_y();
            Kokkos::parallel_for("InitialChannelHarmonicPotential",
                Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {ny, nx}),
                KOKKOS_LAMBDA(int j, int i) {
                    pd(j, i) = Real(start_j + j - h + 1) / Real(global_ny);
                });
            solver->solve_bounded_q2_free_slip_at_z_and_t(zero_rhs,
                potential,
                zero_rhs,
                zero_potential,
                options);
            Kokkos::parallel_for("ChannelMetricHarmonicBasis",
                Kokkos::MDRangePolicy<Kokkos::Rank<2>>({h, h}, {ny - h, nx - h}),
                KOKKOS_LAMBDA(int j, int i) {
                    const Operators::GeneralizedHorizontalEllipticDeviceView::ZeroPotential zero{};
                    b11(j, i) =
                        operation().reconstruction.calculate_covariant_q1_at_u(pd, zero, j, i);
                    b12(j, i) =
                        operation().reconstruction.calculate_covariant_q2_at_v(pd, zero, j, i);
                    direct1(j, i) =
                        operation().reconstruction.calculate_contravariant_q1_at_u(pd, zero, j, i);
                    direct2(j, i) =
                        operation().reconstruction.calculate_contravariant_q2_at_v(pd, zero, j, i);
                });
            Real local = real(0.0), total = real(0.0);
            if (grid_.get_local_physical_start_y() == 0) {
                Kokkos::parallel_reduce("NormalizeChannelHarmonicBasis",
                    Kokkos::RangePolicy<>(h, nx - h),
                    KOKKOS_LAMBDA(int i, Real& value) { value += b11(h, i); },
                    local);
            }
            Core::StartupCollectives::sum(grid_, state_, &local, &total, 1);
            const Real scale = Real(grid_.get_global_points_x()) / total;
            if (!std::isfinite(scale)) {
                throw std::runtime_error("Degenerate channel harmonic basis.");
            }
            Kokkos::parallel_for("ScaleChannelHarmonicBasis",
                Kokkos::MDRangePolicy<Kokkos::Rank<2>>({h, h}, {ny - h, nx - h}),
                KOKKOS_LAMBDA(int j, int i) {
                    b11(j, i) *= scale;
                    b12(j, i) *= scale;
                    direct1(j, i) *= scale;
                    direct2(j, i) *= scale;
                });
        }
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
        if (mode_ == GeneralizedConstraintMode::PeriodicCycles) {
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

        const bool periodic = mode_ == GeneralizedConstraintMode::PeriodicCycles;

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

        if (mode_ == GeneralizedConstraintMode::PeriodicCycles) {

            targets_.q2 = measurements_.q2;
        }

#endif

        has_target_ = true;
    }

    void
    apply_correction() {
        if (mode_ == GeneralizedConstraintMode::PeriodicCycles) {

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

        Kokkos::parallel_for("MappedBoundedQ2CovariantCirculation",
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
        const auto gu = grid_.geometry().device_view(Core::Geometry::HorizontalLocation::U);
        const auto gv = grid_.geometry().device_view(Core::Geometry::HorizontalLocation::V);
        const auto h1 = gu.contravariant_to_physical.a11, cross = gv.contravariant_to_physical.a21;
        const auto v = state_.get_field<3>("v").get_device_data();
        const auto contributions = contributions_->get_mutable_device_data();
        Kokkos::deep_copy(contributions, real(0.0));
        Kokkos::parallel_for("MappedRestartSouthWallCirculation",
            Kokkos::MDRangePolicy<Kokkos::Rank<2>>({h, h}, {ny - h, nx - h}),
            KOKKOS_LAMBDA(const int j, const int i) {
                if (owns_south && j == h) {
                    contributions(start_i + i - h) =
                        u(top, j, i) * h1(j, i) +
                        Core::Geometry::weighted_cross<true>(v, cross, top, j, i);
                }
            });
        reduce_bounded_q2_wall(gx);
    }

    void
    reduce_bounded_q2_wall(int gx) {
        const auto contributions = contributions_->get_mutable_device_data();

#if defined(ENABLE_NCCL)

        reduce_device_contributions(gx,
            "Mapped channel circulation NCCL "
            "all-reduce");

        const auto measurements = measurements_device_;

        Kokkos::parallel_for("MappedSouthWallCirculationFinalize",
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
            throw std::runtime_error("Mapped channel circulation "
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
        auto q2_cov = covariant_q2_wind_.get_mutable_device_data();
        const auto b1 = basis_11_->get_device_data(), b2 = basis_12_->get_device_data();

#if defined(ENABLE_NCCL)

        const auto targets = targets_device_;

        const auto measurements = measurements_device_;

        Kokkos::parallel_for("PreserveBoundedQ2CovariantCirculation",
            Kokkos::MDRangePolicy<Kokkos::Rank<3>>({bottom, h, h}, {top + 1, ny - h, nx - h}),
            KOKKOS_LAMBDA(const int k, const int j, const int i) {
                const Real correction = targets(0) - measurements(0);

                q1_cov(k, j, i) += correction * b1(j, i);
                q2_cov(k, j, i) += correction * b2(j, i);
            });

#else

        const Real correction = targets_.q1 - measurements_.q1;

        Kokkos::parallel_for("PreserveBoundedQ2CovariantCirculation",
            Kokkos::MDRangePolicy<Kokkos::Rank<3>>({bottom, h, h}, {top + 1, ny - h, nx - h}),
            KOKKOS_LAMBDA(const int k, const int j, const int i) {
                q1_cov(k, j, i) += correction * b1(j, i);
                q2_cov(k, j, i) += correction * b2(j, i);
            });

#endif
    }

    void
    measure_periodic_cycles_from_physical() {
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

        const auto u = state_.get_field<3>("u").get_device_data();
        const auto v = state_.get_field<3>("v").get_device_data();

        const auto gu = grid_.geometry().device_view(HorizontalLocation::U);
        const auto gv = grid_.geometry().device_view(HorizontalLocation::V);
        const auto b11 = gu.contravariant_to_physical.a11, b21 = gv.contravariant_to_physical.a21;
        const auto b22 = gv.contravariant_to_physical.a22, b12 = gu.contravariant_to_physical.a12;
        const auto basis_q2 = basis_22_->get_device_data();

        const auto contributions = contributions_->get_mutable_device_data();

        Kokkos::deep_copy(contributions, real(0.0));

        Kokkos::parallel_for("MappedPeriodicCycleIntegrals",
            Kokkos::MDRangePolicy<Kokkos::Rank<2>>({h, h}, {ny - h, nx - h}),
            KOKKOS_LAMBDA(const int j, const int i) {
                if (sj == 0 && j == h) {

                    contributions(si + i - h) =
                        b11(j, i) * u(top, j, i) +
                        Core::Geometry::weighted_cross<true>(v, b21, top, j, i);
                }

                if (si == 0 && i == h) {

                    contributions(gx + sj + j - h) =
                        b22(j, i) * v(top, j, i) +
                        Core::Geometry::weighted_cross<false>(u, b12, top, j, i);

                    contributions(gx + gy + sj + j - h) = basis_q2(j, i);
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
            "Mapped periodic circulation NCCL "
            "all-reduce");

        const auto measurements = measurements_device_;

        Kokkos::parallel_for("MappedPeriodicCycleIntegralsFinalize",
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
            throw std::runtime_error("Mapped periodic circulation "
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

        const auto b11 = basis_11_->get_device_data(), b12 = basis_12_->get_device_data();
        const auto b21 = basis_21_->get_device_data(), b22 = basis_22_->get_device_data();

#if defined(ENABLE_NCCL)
        const auto targets = targets_device_;

        const auto measurements = measurements_device_;

        Kokkos::parallel_for("PreserveMappedPeriodicCovariantCirculation",
            Kokkos::MDRangePolicy<Kokkos::Rank<3>>({bottom, h, h}, {top + 1, ny - h, nx - h}),
            KOKKOS_LAMBDA(const int k, const int j, const int i) {
                const Real dq1 = targets(0) - measurements(0);

                const Real harmonic_q2 = (targets(1) - measurements(1)) / measurements(2);

                q1_cov(k, j, i) += dq1 * b11(j, i) + harmonic_q2 * b21(j, i);
                q2_cov(k, j, i) += dq1 * b12(j, i) + harmonic_q2 * b22(j, i);
            });

#else

        const Real dq1 = targets_.q1 - measurements_.q1;
        const Real harmonic_q2 = (targets_.q2 - measurements_.q2) / measurements_.weight;

        Kokkos::parallel_for("PreserveMappedPeriodicCovariantCirculation",
            Kokkos::MDRangePolicy<Kokkos::Rank<3>>({bottom, h, h}, {top + 1, ny - h, nx - h}),
            KOKKOS_LAMBDA(const int k, const int j, const int i) {
                q1_cov(k, j, i) += dq1 * b11(j, i) + harmonic_q2 * b21(j, i);
                q2_cov(k, j, i) += dq1 * b12(j, i) + harmonic_q2 * b22(j, i);
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

    GeneralizedConstraintMode mode_;

    std::unique_ptr<Core::Field<1>> contributions_;
    std::unique_ptr<Core::Field<2>> basis_11_, basis_12_, basis_21_, basis_22_;
    std::unique_ptr<Core::Field<2>> canonical_correction_q1_, canonical_correction_q2_;

    Targets targets_;
    Measurements measurements_;

    bool has_target_ = false;
    std::array<std::unique_ptr<Core::Field<2>>, 4> mean_basis_;
    Kokkos::Array<Real, 4> mean_response_{};
    Core::ScalarView mean_u_{"harmonic physical mean u"}, mean_v_{"harmonic physical mean v"};

#if defined(ENABLE_NCCL)

    Kokkos::View<Real*, Kokkos::DefaultExecutionSpace::memory_space> targets_device_;

    Kokkos::View<Real*, Kokkos::DefaultExecutionSpace::memory_space> measurements_device_;

#endif
};

} // namespace

std::unique_ptr<HorizontalWindTopologyConstraint>
make_generalized_circulation_constraint(const Core::Grid& grid,
    Core::State& state,
    Core::Field<3>& first,
    Core::Field<3>& second,
    Core::HaloExchanger& halo,
    HorizontalEllipticSolver& solver,
    int iterations) {
    return std::make_unique<GeneralizedCirculationConstraint>(grid,
        state,
        first,
        second,
        &halo,
        &solver,
        iterations);
}

std::unique_ptr<HorizontalWindTopologyConstraint>
make_regular_lat_lon_circulation_constraint(
    const Core::Grid& grid, Core::State& state, Core::Field<3>& first, Core::Field<3>& second) {
    return std::make_unique<GeneralizedCirculationConstraint>(grid, state, first, second);
}

} // namespace VVM::Dynamics
