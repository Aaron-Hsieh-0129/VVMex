#include "GeneralizedWindRecoveryTestGeometry.hpp"
#include "dynamics/solvers/VerticalWindDiagnostic.hpp"

#include <Kokkos_Core.hpp>
#include <mpi.h>

#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>

namespace {

using VVM::Real;
using VVM::real;
using WindRecoveryTest::Geometry;

void
require(bool condition, const std::string& message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

Real
tolerance() {
    return sizeof(Real) == sizeof(double) ? real(2e-11) : real(8e-4);
}

void
require_close(Real actual, Real expected, const std::string& message) {
    const Real scale = std::max({real(1.0), std::abs(actual), std::abs(expected)});
    require(std::isfinite(actual) && std::isfinite(expected) &&
                std::abs(actual - expected) <= tolerance() * scale,
        message + " actual=" + std::to_string(double(actual)) +
            " expected=" + std::to_string(double(expected)));
}

struct Profile {
    Kokkos::View<Real*> values;

    KOKKOS_INLINE_FUNCTION Real
    operator()(int k) const noexcept {
        return values(k);
    }
};

template <typename View>
struct NegatedView3D {
    View view;

    KOKKOS_INLINE_FUNCTION Real
    operator()(int k, int j, int i) const noexcept {
        return -view(k, j, i);
    }
};

Real
omega1_value(const Geometry& geometry, int k, int j, int i) {
    const Real p = geometry.p(2, i);
    const Real q = geometry.q(2, j);
    return real(0.13) + real(0.017) * k + real(0.021) * p - real(0.014) * q + real(0.009) * p * q;
}

Real
omega2_value(const Geometry& geometry, int k, int j, int i) {
    const Real p = geometry.p(1, i);
    const Real q = geometry.q(1, j);
    return -real(0.09) + real(0.011) * k - real(0.018) * p + real(0.025) * q - real(0.007) * p * q;
}

Real
previous_value(const Geometry& geometry, int k, int j, int i) {
    const Real p = geometry.p(0, i);
    const Real q = geometry.q(0, j);
    return real(0.4) + real(0.03) * k + real(0.17) * p * p - real(0.11) * q * q +
           real(0.08) * p * q + real(0.025) * p * p * q;
}

Real
weighted_rhs_reference(const Geometry& geometry,
    const Kokkos::View<Real***>::HostMirror& xi,
    const Kokkos::View<Real***>::HostMirror& eta,
    int k,
    int j,
    int i) {
    const Real dq1 = geometry.dq1();
    const Real dq2 = geometry.dq2();

    const auto u_e = geometry.metric(1, j, i);
    const auto u_w = geometry.metric(1, j, i - 1);
    const auto v_n = geometry.metric(2, j, i);
    const auto v_s = geometry.metric(2, j - 1, i);

    const Real direct_q1 = (u_e.g22 * (-eta(k, j, i)) - u_w.g22 * (-eta(k, j, i - 1))) / dq1;

    const Real direct_q2 = -(v_n.g11 * xi(k, j, i) - v_s.g11 * xi(k, j - 1, i)) / dq2;

    const auto gg12_v = [&](int jj, int ii) {
        return -geometry.metric(2, jj, ii).g12;
    };
    const auto gg12_u = [&](int jj, int ii) {
        return -geometry.metric(1, jj, ii).g12;
    };

    const Real cross_q1 =
        -(gg12_v(j, i + 1) * xi(k, j, i + 1) + gg12_v(j - 1, i + 1) * xi(k, j - 1, i + 1) -
            gg12_v(j, i - 1) * xi(k, j, i - 1) - gg12_v(j - 1, i - 1) * xi(k, j - 1, i - 1)) /
        (real(4.0) * dq1);

    const Real cross_q2 =
        (gg12_u(j + 1, i) * (-eta(k, j + 1, i)) + gg12_u(j + 1, i - 1) * (-eta(k, j + 1, i - 1)) -
            gg12_u(j - 1, i) * (-eta(k, j - 1, i)) -
            gg12_u(j - 1, i - 1) * (-eta(k, j - 1, i - 1))) /
        (real(4.0) * dq2);

    return direct_q1 + direct_q2 + cross_q1 + cross_q2;
}

Real
divergence_reference(const Geometry& geometry,
    const Kokkos::View<Real***>::HostMirror& xi,
    const Kokkos::View<Real***>::HostMirror& eta,
    int k,
    int j,
    int i) {
    const Real omega1_e = xi(k, j, i + 1);
    const Real omega1_w = xi(k, j, i);
    const Real omega2_n = -eta(k, j + 1, i);
    const Real omega2_s = -eta(k, j, i);

    const Real flux1 =
        (geometry.metric(2, j, i + 1).J * omega1_e - geometry.metric(2, j, i).J * omega1_w) /
        geometry.dq1();

    const Real flux2 =
        (geometry.metric(1, j + 1, i).J * omega2_n - geometry.metric(1, j, i).J * omega2_s) /
        geometry.dq2();

    return (flux1 + flux2) / geometry.metric(3, j, i).J;
}

Real
horizontal_neighbors_reference(const Geometry& geometry,
    const Kokkos::View<Real***>::HostMirror& previous,
    int k,
    int j,
    int i) {
    const Real dq1 = geometry.dq1();
    const Real dq2 = geometry.dq2();

    const auto rgg_u = [&](int component, int jj, int ii) {
        const auto m = geometry.metric(1, jj, ii);
        return m.J * (component == 1 ? m.c11 : m.c12);
    };
    const auto rgg_v = [&](int component, int jj, int ii) {
        const auto m = geometry.metric(2, jj, ii);
        return m.J * (component == 2 ? m.c12 : m.c22);
    };

    const Real direct =
        (rgg_u(1, j, i) * previous(k, j, i + 1) + rgg_u(1, j, i - 1) * previous(k, j, i - 1)) /
            (dq1 * dq1) +
        (rgg_v(3, j, i) * previous(k, j + 1, i) + rgg_v(3, j - 1, i) * previous(k, j - 1, i)) /
            (dq2 * dq2);

    const Real cross =
        (rgg_v(2, j, i + 1) * (previous(k, j + 1, i + 1) - previous(k, j, i + 1)) +
            rgg_v(2, j - 1, i + 1) * (previous(k, j, i + 1) - previous(k, j - 1, i + 1)) -
            rgg_v(2, j, i - 1) * (previous(k, j + 1, i - 1) - previous(k, j, i - 1)) -
            rgg_v(2, j - 1, i - 1) * (previous(k, j, i - 1) - previous(k, j - 1, i - 1)) +
            rgg_u(2, j + 1, i) * (previous(k, j + 1, i + 1) - previous(k, j + 1, i)) +
            rgg_u(2, j + 1, i - 1) * (previous(k, j + 1, i) - previous(k, j + 1, i - 1)) -
            rgg_u(2, j - 1, i) * (previous(k, j - 1, i + 1) - previous(k, j - 1, i)) -
            rgg_u(2, j - 1, i - 1) * (previous(k, j - 1, i) - previous(k, j - 1, i - 1))) /
        (real(4.0) * dq1 * dq2);

    return direct + cross;
}

void
run_case(Real shear, bool varying) {
    Geometry geometry(shear, varying, 2);
    const auto operation = VVM::Dynamics::make_vertical_wind_diagnostic_device_view(geometry);

    constexpr int nz = 7;
    const int ny = geometry.ny();
    const int nx = geometry.nx();
    const int h = geometry.h();

    Kokkos::View<Real***> xi("generalized_vertical_xi", nz, ny, nx);
    Kokkos::View<Real***> eta("generalized_vertical_eta", nz, ny, nx);
    Kokkos::View<Real***> previous("generalized_vertical_previous", nz, ny, nx);

    auto hxi = Kokkos::create_mirror_view(xi);
    auto heta = Kokkos::create_mirror_view(eta);
    auto hp = Kokkos::create_mirror_view(previous);

    for (int k = 0; k < nz; ++k) {
        for (int j = 0; j < ny; ++j) {
            for (int i = 0; i < nx; ++i) {
                hxi(k, j, i) = omega1_value(geometry, k, j, i);
                heta(k, j, i) = -omega2_value(geometry, k, j, i);
                hp(k, j, i) = previous_value(geometry, k, j, i);
            }
        }
    }

    Kokkos::deep_copy(xi, hxi);
    Kokkos::deep_copy(eta, heta);
    Kokkos::deep_copy(previous, hp);

    Kokkos::View<Real*> rho("generalized_vertical_rho", nz);
    Kokkos::View<Real*> rho_up("generalized_vertical_rho_up", nz);
    Kokkos::View<Real*> mid("generalized_vertical_mid", nz);
    Kokkos::View<Real*> up("generalized_vertical_up", nz);

    auto hrho = Kokkos::create_mirror_view(rho);
    auto hrho_up = Kokkos::create_mirror_view(rho_up);
    auto hmid = Kokkos::create_mirror_view(mid);
    auto hup = Kokkos::create_mirror_view(up);

    for (int k = 0; k < nz; ++k) {
        hrho(k) = real(1.2) - real(0.035) * k;
        hrho_up(k) = real(1.17) - real(0.033) * k;
        hmid(k) = real(1.04) + real(0.025) * k;
        hup(k) = real(0.93) + real(0.031) * k;
    }

    Kokkos::deep_copy(rho, hrho);
    Kokkos::deep_copy(rho_up, hrho_up);
    Kokkos::deep_copy(mid, hmid);
    Kokkos::deep_copy(up, hup);

    Kokkos::View<Real***> outputs("generalized_vertical_outputs", 7, ny, nx);
    const Real inverse_dz = real(0.013);
    const Real shift = real(0.27);
    const int k = 3;

    Kokkos::parallel_for("GeneralizedVerticalWindDiagnosticCheck",
        Kokkos::MDRangePolicy<Kokkos::Rank<2>>({h, h}, {ny - h, nx - h}),
        KOKKOS_LAMBDA(const int j, const int i) {
            const auto row = operation.calculate_row_at_t(Profile{rho},
                Profile{rho_up},
                Profile{mid},
                Profile{up},
                inverse_dz,
                shift,
                k,
                j,
                i);

            outputs(0, j, i) =
                operation.calculate_weighted_rhs_from_vvm_contravariant_at_t(xi, eta, k, j, i);

            outputs(1, j, i) = operation.calculate_vorticity_divergence_from_contravariant_at_z(xi,
                NegatedView3D<decltype(eta)>{eta},
                k,
                j,
                i);

            outputs(2, j, i) = operation.calculate_horizontal_neighbors_at_t(previous, k, j, i);
            outputs(3, j, i) = row.lower;
            outputs(4, j, i) = row.diagonal;
            outputs(5, j, i) = row.upper;
            outputs(6, j, i) = row.east + row.west + row.north + row.south;
        });

    Kokkos::fence();
    const auto actual = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), outputs);

    Real largest_cross_effect = real(0.0);

    for (int j = h; j < ny - h; ++j) {
        for (int i = h; i < nx - h; ++i) {
            require_close(actual(0, j, i),
                weighted_rhs_reference(geometry, hxi, heta, k, j, i),
                "generalized vertical weighted RHS mismatch");

            require_close(actual(1, j, i),
                divergence_reference(geometry, hxi, heta, k, j, i),
                "generalized vorticity divergence mismatch");

            const Real expected_neighbors = horizontal_neighbors_reference(geometry, hp, k, j, i);

            require_close(actual(2, j, i),
                expected_neighbors,
                "generalized vertical horizontal-neighbor stencil mismatch");

            const auto mt = geometry.metric(0, j, i);
            const auto mu_e = geometry.metric(1, j, i);
            const auto mu_w = geometry.metric(1, j, i - 1);
            const auto mv_n = geometry.metric(2, j, i);
            const auto mv_s = geometry.metric(2, j - 1, i);

            const Real east = mu_e.J * mu_e.c11 / (geometry.dq1() * geometry.dq1());
            const Real west = mu_w.J * mu_w.c11 / (geometry.dq1() * geometry.dq1());
            const Real north = mv_n.J * mv_n.c22 / (geometry.dq2() * geometry.dq2());
            const Real south = mv_s.J * mv_s.c22 / (geometry.dq2() * geometry.dq2());

            const Real vertical = mt.J * hup(k) * inverse_dz * inverse_dz;
            const Real lower = -vertical * hmid(k) / hrho(k);
            const Real upper = -vertical * hmid(k + 1) / hrho(k + 1);
            const Real diagonal =
                (shift + east + west + north + south) / hrho_up(k) - lower - upper;

            require_close(actual(3, j, i), lower, "vertical lower coefficient mismatch");
            require_close(actual(4, j, i), diagonal, "vertical diagonal coefficient mismatch");
            require_close(actual(5, j, i), upper, "vertical upper coefficient mismatch");
            require_close(actual(6, j, i),
                east + west + north + south,
                "vertical direct horizontal coefficient mismatch");

            const Real direct_only = east * hp(k, j, i + 1) + west * hp(k, j, i - 1) +
                                     north * hp(k, j + 1, i) + south * hp(k, j - 1, i);

            largest_cross_effect =
                std::max(largest_cross_effect, std::abs(expected_neighbors - direct_only));
        }
    }

    require(std::abs(shear) < real(1e-12) || largest_cross_effect > real(1e-8),
        "nonorthogonal fixture did not exercise the mixed-metric stencil");
}

} // namespace

int
main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    Kokkos::initialize(argc, argv);

    int result = 0;

    try {
        for (const Real shear : {real(0.37), real(-0.29)}) {
            run_case(shear, false);
            run_case(shear, true);
        }

        std::cout << "PASS: generalized vertical wind diagnostic uses full 2-D metric, "
                     "CVVM mixed terms, and canonical contravariant vorticity\n";
    }
    catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        result = 1;
    }

    Kokkos::finalize();
    MPI_Finalize();
    return result;
}
