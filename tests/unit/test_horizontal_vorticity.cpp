#include "core/geometry/CartesianGeometry.hpp"
#include "core/geometry/RegularLatLonGeometry.hpp"
#include "dynamics/operators/HorizontalVorticity.hpp"
#include "dynamics/operators/RegularLatLonHorizontalDeformation.hpp"

#include <Kokkos_Core.hpp>
#include <mpi.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

#if defined(KOKKOS_ENABLE_CUDA)
#include <cuda_runtime.h>
#endif

namespace {

using VVM::Real;
using VVM::real;
using VVM::Core::Geometry::CartesianGeometry;
using VVM::Core::Geometry::HorizontalDomainLayout;
using VVM::Core::Geometry::HorizontalGeometry;
using VVM::Core::Geometry::RegularLatLonGeometry;
using VVM::Dynamics::Operators::make_horizontal_vorticity_device_view;
using VVM::Dynamics::Operators::make_regular_lat_lon_horizontal_deformation_device_view;
using VVM::Dynamics::Operators::RegularLatLonHorizontalDeformationDeviceView;
using VVM::Dynamics::Operators::RegularLatLonHorizontalDeformationFields;

int failures = 0;

HorizontalDomainLayout make_layout() {
    HorizontalDomainLayout layout;
    layout.global_nx = 12;
    layout.global_ny = 10;
    layout.local_physical_nx = 12;
    layout.local_physical_ny = 10;
    layout.halo = 2;
    return layout;
}

template<typename Layout>
void test_vorticity(const HorizontalGeometry& geometry, bool spherical, Real radius, Real south_edge, const char* layout_name) {
    using View3D = Kokkos::View<Real***, Layout>;

    const auto layout = geometry.layout();
    const int nx = layout.local_total_nx();
    const int ny = layout.local_total_ny();
    const int h = layout.halo;
    constexpr int nz = 5;

    const Real dq1 = geometry.dq1();
    const Real dq2 = geometry.dq2();
    const auto operation = make_horizontal_vorticity_device_view(geometry);

    View3D w("horizontal_vorticity_w", nz, ny, nx);
    View3D u1("horizontal_vorticity_covariant_q1", nz, ny, nx);
    View3D u2("horizontal_vorticity_covariant_q2", nz, ny, nx);
    View3D omega1("horizontal_vorticity_q1", nz, ny, nx);
    View3D omega2("horizontal_vorticity_q2", nz, ny, nx);
    View3D recovered1("horizontal_vorticity_recovered_q1", nz, ny, nx);
    View3D recovered2("horizontal_vorticity_recovered_q2", nz, ny, nx);

    Kokkos::View<Real*> spacing("horizontal_vorticity_spacing", nz - 1);
    Kokkos::View<Real*> inverse_spacing("horizontal_vorticity_inverse_spacing", nz - 1);

    auto w_host = Kokkos::create_mirror_view(w);
    auto u1_host = Kokkos::create_mirror_view(u1);
    auto u2_host = Kokkos::create_mirror_view(u2);
    auto spacing_host = Kokkos::create_mirror_view(spacing);
    auto inverse_spacing_host = Kokkos::create_mirror_view(inverse_spacing);

    const Real tolerance = sizeof(Real) == sizeof(double) ? real(1e-10) : real(5e-4);

    for (bool stretched : {false, true}) {
        const std::array<Real, nz> z = stretched
            ? std::array<Real, nz>{real(0), real(40), real(120), real(240), real(400)}
            : std::array<Real, nz>{real(0), real(100), real(200), real(300), real(400)};

        for (int k = 0; k < nz - 1; ++k) {
            spacing_host(k) = z[k + 1] - z[k];
            inverse_spacing_host(k) = real(1.0) / spacing_host(k);
        }

        Kokkos::deep_copy(spacing, spacing_host);
        Kokkos::deep_copy(inverse_spacing, inverse_spacing_host);

        for (int mode = 0; mode < 4; ++mode) {
            const Real shear_e = (mode & 1) ? real(3e-4) : real(0.0);
            const Real shear_n = (mode & 1) ? real(-4e-4) : real(0.0);
            const Real coordinate_scale = spherical ? radius : real(1.0);
            const Real a = (mode & 2) ? coordinate_scale * real(1e-4) : real(0.0);
            const Real b = (mode & 2) ? coordinate_scale * real(-2e-4) : real(0.0);

            // Manufacture physical winds:
            //   u_E = 12 + shear_e*z
            //   u_N = -8 + shear_n*z
            // and physical vertical velocity:
            //   w = a*q1 + b*q2.
            //
            // Convert the horizontal winds to covariant components using
            // analytic Cartesian/RLL scale factors, independently of the operator.
            for (int j = 0; j < ny; ++j) {
                const Real global_j = static_cast<Real>(layout.global_start_j + j - h);
                const Real q2 = south_edge + (global_j + real(0.5)) * dq2;
                const Real h1_at_u = spherical ? radius * std::cos(q2) : real(1.0);
                const Real h2 = spherical ? radius : real(1.0);

                for (int i = 0; i < nx; ++i) {
                    const Real global_i = static_cast<Real>(layout.global_start_i + i - h);
                    const Real q1 = (global_i + real(0.5)) * dq1;

                    for (int k = 0; k < nz; ++k) {
                        w_host(k, j, i) = a * q1 + b * q2;
                        u1_host(k, j, i) = h1_at_u * (real(12.0) + shear_e * z[k]);
                        u2_host(k, j, i) = h2 * (real(-8.0) + shear_n * z[k]);
                    }
                }
            }

            Kokkos::deep_copy(w, w_host);
            Kokkos::deep_copy(u1, u1_host);
            Kokkos::deep_copy(u2, u2_host);

            const auto policy = Kokkos::MDRangePolicy<Kokkos::Rank<2>>({h, h}, {ny - h, nx - h});
            const auto compact_policy = Kokkos::Experimental::require(
                policy, Kokkos::Experimental::WorkItemProperty::HintLightWeight);

            Kokkos::parallel_for("TestHorizontalVorticity", compact_policy,
                KOKKOS_LAMBDA(const int j, const int i) {
                    for (int k = 0; k < nz - 1; ++k) {
                        omega1(k, j, i) = operation.calculate_contravariant_q1_at_v(w, u2, k, j, i, inverse_spacing(k));
                        omega2(k, j, i) = operation.calculate_contravariant_q2_at_u(w, u1, k, j, i, inverse_spacing(k));
                    }
                }
            );

            Kokkos::parallel_for("TestCovariantWindDownwardRecovery", compact_policy,
                KOKKOS_LAMBDA(const int j, const int i) {
                    recovered1(nz - 1, j, i) = u1(nz - 1, j, i);
                    recovered2(nz - 1, j, i) = u2(nz - 1, j, i);

                    for (int k = nz - 2; k >= 0; --k) {
                        const Real du1_dz = operation.calculate_covariant_q1_vertical_shear_at_u(w, omega2, k, j, i);
                        const Real du2_dz = operation.calculate_covariant_q2_vertical_shear_at_v(w, omega1, k, j, i);

                        recovered1(k, j, i) = recovered1(k + 1, j, i) - du1_dz * spacing(k);
                        recovered2(k, j, i) = recovered2(k + 1, j, i) - du2_dz * spacing(k);
                    }
                }
            );

            Kokkos::fence();

            const auto omega1_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), omega1);
            const auto omega2_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), omega2);
            const auto recovered1_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), recovered1);
            const auto recovered2_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), recovered2);

            Real vorticity_error = real(0.0);
            Real recovery_error = real(0.0);
            bool finite = true;
            bool zero_case_correct = true;

            for (int j = h; j < ny - h; ++j) {
                const Real global_j = static_cast<Real>(layout.global_start_j + j - h);
                const Real phi_u = south_edge + (global_j + real(0.5)) * dq2;
                const Real phi_v = south_edge + (global_j + real(1.0)) * dq2;

                const Real h1_at_u = spherical ? radius * std::cos(phi_u) : real(1.0);
                const Real h1_at_v = spherical ? radius * std::cos(phi_v) : real(1.0);
                const Real h2 = spherical ? radius : real(1.0);

                const Real expected_e = b / h2 - shear_n;
                const Real expected_n = shear_e - a / h1_at_u;

                for (int i = h; i < nx - h; ++i) {
                    for (int k = 0; k < nz - 1; ++k) {
                        const Real actual_e = h1_at_v * omega1_host(k, j, i);
                        const Real actual_n = h2 * omega2_host(k, j, i);

                        if (!std::isfinite(actual_e) || !std::isfinite(actual_n)) {
                            finite = false;
                        } else {
                            vorticity_error = std::max(vorticity_error, std::abs(actual_e - expected_e) / real(1e-3));
                            vorticity_error = std::max(vorticity_error, std::abs(actual_n - expected_n) / real(1e-3));
                        }

                        if (mode == 0) {
                            zero_case_correct = zero_case_correct
                                && omega1_host(k, j, i) == real(0.0)
                                && omega2_host(k, j, i) == real(0.0);
                        }
                    }

                    for (int k = 0; k < nz; ++k) {
                        const Real expected_u = real(12.0) + shear_e * z[k];
                        const Real expected_v = real(-8.0) + shear_n * z[k];
                        const Real actual_u = recovered1_host(k, j, i) / h1_at_u;
                        const Real actual_v = recovered2_host(k, j, i) / h2;

                        if (!std::isfinite(actual_u) || !std::isfinite(actual_v)) {
                            finite = false;
                        } else {
                            recovery_error = std::max(recovery_error,
                                std::abs(actual_u - expected_u) / std::max(real(1.0), std::abs(expected_u)));
                            recovery_error = std::max(recovery_error,
                                std::abs(actual_v - expected_v) / std::max(real(1.0), std::abs(expected_v)));
                        }
                    }
                }
            }

            const bool passed = finite && zero_case_correct
                && vorticity_error <= tolerance && recovery_error <= tolerance;

            std::printf("%s %s stretched=%d mode=%d vorticity_error=%.3e recovery_error=%.3e %s\n",
                geometry.name(), layout_name, static_cast<int>(stretched), mode,
                static_cast<double>(vorticity_error), static_cast<double>(recovery_error),
                passed ? "PASS" : "FAIL");

            if (!passed) ++failures;
        }
    }
}

template<typename View>
std::vector<Real> snapshot(const View& view) {
    const auto host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), view);
    return std::vector<Real>(host.data(), host.data() + host.size());
}

#if defined(KOKKOS_ENABLE_CUDA)
void require_cuda(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(status));
    }
}

struct CapturedDeformation {
    cudaGraph_t graph = nullptr;
    cudaGraphExec_t executable = nullptr;

    ~CapturedDeformation() {
        if (executable) cudaGraphExecDestroy(executable);
        if (graph) cudaGraphDestroy(graph);
    }
};
#endif

template<typename Layout>
struct DeformationFunctor {
    using Volume = Kokkos::View<Real***, Layout>;
    using Profile = Kokkos::View<Real*>;
    using Plane = Kokkos::View<Real**, Layout>;

    RegularLatLonHorizontalDeformationDeviceView operation;
    RegularLatLonHorizontalDeformationFields<Volume, Profile, Plane> fields;
    Kokkos::View<Real****, Layout> output;
    bool preparation_only = false;

    KOKKOS_INLINE_FUNCTION
    void operator()(int k, int j, int i) const {
        if (preparation_only) return;

        const auto xi = operation.calculate_xi_at_v(fields, k, j, i);
        const auto eta = operation.calculate_eta_at_u(fields, k, j, i);

        output(0, k, j, i) = xi.stretching;
        output(1, k, j, i) = xi.twisting;
        output(2, k, j, i) = xi.planetary;
        output(3, k, j, i) = eta.stretching;
        output(4, k, j, i) = eta.twisting;
        output(5, k, j, i) = eta.planetary;
    }
};

template<typename Layout>
void test_deformation(const HorizontalGeometry& geometry, Real radius, Real south_edge, const char* layout_name) {
    using Functor = DeformationFunctor<Layout>;
    using Volume = typename Functor::Volume;
    using Profile = typename Functor::Profile;
    using Plane = typename Functor::Plane;

    const auto layout = geometry.layout();
    const int h = layout.halo;
    const int nx = layout.local_total_nx();
    const int ny = layout.local_total_ny();
    const int nz = 9;
    const Real sentinel = real(-731.0);
    const Real tolerance = sizeof(Real) == sizeof(double) ? real(2e-9) : real(2e-3);
    const Real dq1 = geometry.dq1();
    const Real dq2 = geometry.dq2();

    Functor functor;
    functor.operation = make_regular_lat_lon_horizontal_deformation_device_view(geometry);
    auto& fields = functor.fields;
    fields.u = Volume("deformation_u", nz, ny, nx);
    fields.v = Volume("deformation_v", nz, ny, nx);
    fields.xi = Volume("deformation_xi", nz, ny, nx);
    fields.eta = Volume("deformation_eta", nz, ny, nx);
    fields.zeta = Volume("deformation_zeta", nz, ny, nx);
    fields.f_at_z = Plane("deformation_f_at_z", ny, nx);
    fields.rho = Profile("deformation_rho", nz);
    fields.rho_up = Profile("deformation_rho_up", nz);
    fields.fn1 = Profile("deformation_fn1", nz);
    fields.fn2 = Profile("deformation_fn2", nz);
    fields.inverse_spacing = Profile("deformation_inverse_spacing", nz);
    functor.output = Kokkos::View<Real****, Layout>("deformation_output", 6, nz, ny, nx);

    auto u = Kokkos::create_mirror_view(fields.u);
    auto v = Kokkos::create_mirror_view(fields.v);
    auto xi = Kokkos::create_mirror_view(fields.xi);
    auto eta = Kokkos::create_mirror_view(fields.eta);
    auto zeta = Kokkos::create_mirror_view(fields.zeta);
    auto f = Kokkos::create_mirror_view(fields.f_at_z);
    auto rho = Kokkos::create_mirror_view(fields.rho);
    auto rho_up = Kokkos::create_mirror_view(fields.rho_up);
    auto fn1 = Kokkos::create_mirror_view(fields.fn1);
    auto fn2 = Kokkos::create_mirror_view(fields.fn2);
    auto inverse_spacing = Kokkos::create_mirror_view(fields.inverse_spacing);

    const auto input_snapshot = [&]() {
        std::vector<Real> result;
        const auto append = [&](const auto& view) {
            const auto values = snapshot(view);
            result.insert(result.end(), values.begin(), values.end());
        };
        append(fields.u);
        append(fields.v);
        append(fields.xi);
        append(fields.eta);
        append(fields.zeta);
        append(fields.f_at_z);
        append(fields.rho);
        append(fields.rho_up);
        append(fields.fn1);
        append(fields.fn2);
        append(fields.inverse_spacing);
        return result;
    };

    const auto policy = Kokkos::Experimental::require(
        Kokkos::MDRangePolicy<Kokkos::Rank<3>>({h, h, h}, {nz - h - 1, ny - h, nx - h}),
        Kokkos::Experimental::WorkItemProperty::HintLightWeight);

#if defined(KOKKOS_ENABLE_CUDA)
    static_assert(std::is_same<Kokkos::DefaultExecutionSpace, Kokkos::Cuda>::value,
        "CUDA capture test requires CUDA as the default execution space.");
    const Kokkos::Cuda execution;
    CapturedDeformation capture;

    // Prepare the exact test launch functor without evaluating any fields.
    auto preparation = functor;
    preparation.preparation_only = true;
    Kokkos::parallel_for("PrepareHorizontalDeformation", policy, preparation);
    execution.fence();
    require_cuda(cudaGetLastError(), "Prepare horizontal deformation");
#endif

    for (bool stretched : {false, true}) {
        std::vector<Real> height(nz, real(0.0));
        for (int k = 1; k < nz; ++k) {
            height[k] = height[k - 1]
                + (stretched ? real(40.0) + real(15.0) * k : real(100.0));
        }

        for (int k = 0; k < nz; ++k) {
            rho(k) = real(1.0) + real(0.1) * k;
            rho_up(k) = real(1.05) + real(0.1) * k;
            fn1(k) = real(0.8) + real(0.03) * k;
            fn2(k) = real(1.1) - real(0.04) * k;
            inverse_spacing(k) = k + 1 < nz
                ? real(1.0) / (height[k + 1] - height[k]) : real(0.0);
        }

        Kokkos::deep_copy(fields.rho, rho);
        Kokkos::deep_copy(fields.rho_up, rho_up);
        Kokkos::deep_copy(fields.fn1, fn1);
        Kokkos::deep_copy(fields.fn2, fn2);
        Kokkos::deep_copy(fields.inverse_spacing, inverse_spacing);

        // Rest, stretching, cross twisting, vertical twisting/planetary,
        // mixed, and final zero reset. Reuse all captured allocations.
        for (int mode = 0; mode < 6; ++mode) {
            const bool along = mode == 1 || mode == 4;
            const bool cross = mode == 2 || mode == 4;
            const bool vertical = mode == 3 || mode == 4;
            const Real a = along ? real(1e-6) : real(0.0);
            const Real b = cross ? real(-2e-6) : real(0.0);
            const Real c = vertical ? real(3e-9) : real(0.0);
            const Real d = cross ? real(4e-6) : real(0.0);
            const Real e = along ? real(-5e-6) : real(0.0);
            const Real g = vertical ? real(-6e-9) : real(0.0);
            const Real x = real(2e-10);
            const Real y = real(-3e-10);
            const Real z = real(4e-4);

            for (int j = 0; j < ny; ++j) {
                const Real phi_u = south_edge + (j - h + real(0.5)) * dq2;
                const Real phi_v = south_edge + (j - h + real(1.0)) * dq2;
                for (int i = 0; i < nx; ++i) {
                    const Real lambda_u = (i - h + real(1.0)) * dq1;
                    const Real lambda_v = (i - h + real(0.5)) * dq1;
                    f(j, i) = real(1e-4) + real(2e-5) * phi_v;
                    for (int k = 0; k < nz; ++k) {
                        // Affine contravariant winds, converted analytically
                        // to physical fields at their native locations.
                        u(k, j, i) = radius * std::cos(phi_u)
                            * (a * lambda_u + b * phi_u + c * height[k]);
                        v(k, j, i) = radius
                            * (d * lambda_v + e * phi_v + g * height[k]);
                        xi(k, j, i) = radius * std::cos(phi_v) * x;
                        eta(k, j, i) = -radius * y;
                        zeta(k, j, i) = z;
                    }
                }
            }

            Kokkos::deep_copy(fields.u, u);
            Kokkos::deep_copy(fields.v, v);
            Kokkos::deep_copy(fields.xi, xi);
            Kokkos::deep_copy(fields.eta, eta);
            Kokkos::deep_copy(fields.zeta, zeta);
            Kokkos::deep_copy(fields.f_at_z, f);
            const auto before = input_snapshot();
            Kokkos::deep_copy(functor.output, sentinel);

#if defined(KOKKOS_ENABLE_CUDA)
            if (!capture.executable) {
                execution.fence();
                require_cuda(cudaStreamBeginCapture(execution.cuda_stream(), cudaStreamCaptureModeGlobal),
                    "Begin horizontal deformation capture");
                Kokkos::parallel_for("HorizontalDeformation", policy, functor);
                require_cuda(cudaStreamEndCapture(execution.cuda_stream(), &capture.graph),
                    "End horizontal deformation capture");
                require_cuda(cudaGraphInstantiate(&capture.executable, capture.graph, nullptr, nullptr, 0),
                    "Instantiate horizontal deformation graph");
            }
            require_cuda(cudaGraphLaunch(capture.executable, execution.cuda_stream()),
                "Replay horizontal deformation graph");
#else
            Kokkos::parallel_for("HorizontalDeformation", policy, functor);
#endif
            Kokkos::fence();
            const auto first = snapshot(functor.output);

            Kokkos::deep_copy(functor.output, sentinel);
            Kokkos::parallel_for("HorizontalDeformation", policy, functor);
            Kokkos::fence();

            const bool repeat_equal = first == snapshot(functor.output);
            const bool inputs_preserved = before == input_snapshot();
            const auto actual = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), functor.output);
            bool regions_preserved = true;
            bool finite = true;
            Real error = real(0.0);

            for (int k = 0; k < nz; ++k) {
                for (int j = 0; j < ny; ++j) {
                    const Real phi_u = south_edge + (j - h + real(0.5)) * dq2;
                    const Real phi_v = south_edge + (j - h + real(1.0)) * dq2;
                    for (int i = 0; i < nx; ++i) {
                        const bool active = k >= h && k < nz - h - 1
                            && j >= h && j < ny - h && i >= h && i < nx - h;

                        if (!active) {
                            for (int n = 0; n < 6; ++n) {
                                regions_preserved = regions_preserved
                                    && actual(n, k, j, i) == sentinel;
                            }
                            continue;
                        }

                        const Real weight = real(0.5) * (fn1(k) + fn2(k));
                        const Real density_factor = real(0.5) * rho_up(k)
                            * (real(1.0) / rho(k) + real(1.0) / rho(k + 1));
                        const Real east_scale = radius * std::cos(phi_v);
                        const Real f_v = real(1e-4) + real(2e-5) * phi_v;
                        const Real f_u = real(1e-4) + real(2e-5) * phi_u;

                        const std::array<Real, 6> expected = {
                            east_scale * weight * x * a,
                            east_scale * (weight * y * b + rho_up(k) * z * c),
                            east_scale * density_factor * f_v * c,
                            -radius * weight * y * e,
                            -radius * (weight * x * d + rho_up(k) * z * g),
                            -radius * density_factor * f_u * g
                        };

                        for (int n = 0; n < 6; ++n) {
                            const Real value = actual(n, k, j, i);
                            finite = finite && std::isfinite(value);
                            error = std::max(error,
                                std::abs(value - expected[n])
                                / std::max(real(1e-8), std::abs(expected[n])));
                        }
                    }
                }
            }

            const bool passed = finite && repeat_equal && inputs_preserved
                && regions_preserved && error <= tolerance;

            std::printf("RLL deformation %s stretched=%d mode=%d error=%.3e repeat=%d inputs=%d regions=%d %s\n",
                layout_name, static_cast<int>(stretched), mode, static_cast<double>(error),
                static_cast<int>(repeat_equal), static_cast<int>(inputs_preserved),
                static_cast<int>(regions_preserved), passed ? "PASS" : "FAIL");
            if (!passed) ++failures;
        }
    }

#if defined(KOKKOS_ENABLE_CUDA)
    std::puts("RLL deformation execution: prepared CUDA capture/replay");
#else
    std::puts("RLL deformation execution: ordinary repeated execution");
#endif
}

} // namespace

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);
    Kokkos::initialize(argc, argv);

    try {
        int size = 0;
        MPI_Comm_size(MPI_COMM_WORLD, &size);
        if (size != 1) throw std::runtime_error("This test requires one MPI rank.");

        const auto layout = make_layout();
        const CartesianGeometry cartesian(layout, real(2.0), real(3.0));

        const Real radius = real(6371220.0);
        const Real south_edge = real(-0.3);
        const RegularLatLonGeometry rll(layout, real(0.08), real(0.04), real(0.0), south_edge, radius);

        test_vorticity<Kokkos::LayoutLeft>(cartesian, false, real(1.0), real(0.0), "LayoutLeft");
        test_vorticity<Kokkos::LayoutRight>(cartesian, false, real(1.0), real(0.0), "LayoutRight");
        test_vorticity<Kokkos::LayoutLeft>(rll, true, radius, south_edge, "LayoutLeft");
        test_vorticity<Kokkos::LayoutRight>(rll, true, radius, south_edge, "LayoutRight");

        bool rejected_cartesian = false;
        try {
            (void)make_regular_lat_lon_horizontal_deformation_device_view(cartesian);
        } catch (const std::invalid_argument&) {
            rejected_cartesian = true;
        }
        if (!rejected_cartesian) {
            ++failures;
            std::fputs("RLL deformation failed to reject Cartesian geometry\n", stderr);
        }

        test_deformation<Kokkos::LayoutLeft>(rll, radius, south_edge, "LayoutLeft");
        test_deformation<Kokkos::LayoutRight>(rll, radius, south_edge, "LayoutRight");
    } catch (const std::exception& error) {
        ++failures;
        std::fprintf(stderr, "test_horizontal_vorticity: %s\n", error.what());
    }

    int global_failures = 0;
    MPI_Allreduce(&failures, &global_failures, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

    if (global_failures == 0) std::puts("test_horizontal_vorticity: PASS");

    Kokkos::finalize();
    MPI_Finalize();
    return global_failures == 0 ? 0 : 1;
}
