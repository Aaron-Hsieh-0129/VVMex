#include "core/geometry/CartesianGeometry.hpp"
#include "core/geometry/RegularLatLonGeometry.hpp"
#include "dynamics/operators/GeneralizedHorizontalDeformation.hpp"
#include "dynamics/operators/GeneralizedHorizontalVorticityTransport.hpp"
#include "dynamics/operators/GeneralizedTopDeformation.hpp"

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
using VVM::Dynamics::Operators::GeneralizedHorizontalDeformationDeviceView;
using VVM::Dynamics::Operators::GeneralizedHorizontalVorticityTransportDeviceView;
using VVM::Dynamics::Operators::GeneralizedTopDeformationDeviceView;
using VVM::Dynamics::Operators::make_generalized_horizontal_deformation_device_view;
using VVM::Dynamics::Operators::make_generalized_horizontal_vorticity_transport_device_view;
using VVM::Dynamics::Operators::make_generalized_top_deformation_device_view;

int failures = 0;

HorizontalDomainLayout
make_layout() {
    HorizontalDomainLayout layout;
    layout.global_nx = 12;
    layout.global_ny = 10;
    layout.local_physical_nx = 12;
    layout.local_physical_ny = 10;
    layout.halo = 2;
    return layout;
}

template <typename View>
std::vector<Real>
snapshot(const View& view) {
    const auto host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), view);
    return std::vector<Real>(host.data(), host.data() + host.size());
}

#if defined(KOKKOS_ENABLE_CUDA)
void
require_cuda(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(status));
    }
}

struct CapturedDeformation {
    cudaGraph_t graph = nullptr;
    cudaGraphExec_t executable = nullptr;

    ~CapturedDeformation() {
        if (executable) {
            cudaGraphExecDestroy(executable);
        }
        if (graph) {
            cudaGraphDestroy(graph);
        }
    }
};
#endif

template <typename Volume, typename Profile, typename Plane>
struct HorizontalComponentFields {
    Volume u1, u2;
    Volume omega1_over_rho, omega2_over_rho, omega3_over_rho;
    Plane f3_at_z;
    Profile rho, rho_up, fn1, fn2, inverse_spacing;
};

template <typename Volume, typename Profile, typename Plane>
struct TopComponentFields {
    Volume w;
    Volume omega1_over_rho, omega2_over_rho, omega3_over_rho;
    Plane f3_at_z;
    Profile rho, rho_up, inverse_spacing_mid, inverse_spacing_up;
};

template <typename Layout>
struct DeformationFunctor {
    using Volume = Kokkos::View<Real***, Layout>;
    using Profile = Kokkos::View<Real*>;
    using Plane = Kokkos::View<Real**, Layout>;

    GeneralizedHorizontalVorticityTransportDeviceView transport;
    GeneralizedHorizontalDeformationDeviceView deformation;
    HorizontalComponentFields<Volume, Profile, Plane> fields;
    GeneralizedTopDeformationDeviceView top_operation;
    TopComponentFields<Volume, Profile, Plane> top_fields;
    Kokkos::View<Real****, Layout> output;
    int k_begin = 0;
    int k_top = 0;
    bool preparation_only = false;

    KOKKOS_INLINE_FUNCTION
    void
    operator()(int k, int j, int i) const {
        if (preparation_only) {
            return;
        }

        if (k == k_top) {
            const auto top = top_operation.calculate_at_z(top_fields, k, j, i);
            output(6, k, j, i) = top.stretching;
            output(7, k, j, i) = top.twisting;
            output(8, k, j, i) = top.planetary;
            return;
        }

        const auto omega1 = deformation.calculate_omega1_at_v(fields, k, j, i);
        const auto omega2 = deformation.calculate_omega2_at_u(fields, k, j, i);
        const auto transport1 =
            transport.calculate_omega1_at_v(fields, top_fields.w, k, j, i, k_begin, k_top);
        const auto transport2 =
            transport.calculate_omega2_at_u(fields, top_fields.w, k, j, i, k_begin, k_top);

        output(0, k, j, i) = omega1.stretching;
        output(1, k, j, i) = omega1.twisting;
        output(2, k, j, i) = omega1.planetary;
        output(3, k, j, i) = omega2.stretching;
        output(4, k, j, i) = omega2.twisting;
        output(5, k, j, i) = omega2.planetary;
        output(9, k, j, i) = transport1.q1;
        output(10, k, j, i) = transport1.q2;
        output(11, k, j, i) = transport1.vertical;
        output(12, k, j, i) = transport2.q1;
        output(13, k, j, i) = transport2.q2;
        output(14, k, j, i) = transport2.vertical;
    }
};

Real
wave(Real longitude) {
    return std::sin(longitude) + real(0.4) * std::cos(real(2.0) * longitude);
}

// Independent constant-velocity reduction of the Takacs face formula.
Real
wave_face(Real left_longitude, Real spacing, Real velocity) {
    const Real left = wave(left_longitude);
    const Real right = wave(left_longitude + spacing);
    const Real centered = real(0.5) * velocity * (left + right);

    if (velocity >= real(0.0)) {
        return centered -
               velocity * (right - real(2.0) * left + wave(left_longitude - spacing)) / real(6.0);
    }

    return centered - velocity *
                          (left - real(2.0) * right + wave(left_longitude + real(2.0) * spacing)) /
                          real(6.0);
}

template <typename Layout>
void
test_deformation(
    const HorizontalGeometry& geometry, Real radius, Real south_edge, const char* layout_name) {
    using Functor = DeformationFunctor<Layout>;
    using Volume = typename Functor::Volume;
    using Profile = typename Functor::Profile;
    using Plane = typename Functor::Plane;

    const auto layout = geometry.layout();
    const int h = layout.halo;
    const int nx = layout.local_total_nx();
    const int ny = layout.local_total_ny();
    const int nz = 9;
    const int k_top = nz - h - 1;
    const Real sentinel = real(-731.0);
    const Real tolerance = sizeof(Real) == sizeof(double) ? real(2e-9) : real(2e-3);
    const Real dq1 = geometry.dq1();
    const Real dq2 = geometry.dq2();

    Functor functor;
    functor.transport = make_generalized_horizontal_vorticity_transport_device_view(geometry);
    functor.deformation = make_generalized_horizontal_deformation_device_view(geometry);
    functor.top_operation = make_generalized_top_deformation_device_view(geometry);
    functor.k_begin = h;
    functor.k_top = k_top;

    auto& fields = functor.fields;
    fields.u1 = Volume("deformation_u1", nz, ny, nx);
    fields.u2 = Volume("deformation_u2", nz, ny, nx);
    fields.omega1_over_rho = Volume("deformation_omega1_over_rho", nz, ny, nx);
    fields.omega2_over_rho = Volume("deformation_omega2_over_rho", nz, ny, nx);
    fields.omega3_over_rho = Volume("deformation_omega3_over_rho", nz, ny, nx);
    fields.f3_at_z = Plane("deformation_f3_at_z", ny, nx);
    fields.rho = Profile("deformation_rho", nz);
    fields.rho_up = Profile("deformation_rho_up", nz);
    fields.fn1 = Profile("deformation_fn1", nz);
    fields.fn2 = Profile("deformation_fn2", nz);
    fields.inverse_spacing = Profile("deformation_inverse_spacing", nz);

    auto& top_fields = functor.top_fields;
    top_fields.w = Volume("deformation_w", nz, ny, nx);
    top_fields.omega1_over_rho = fields.omega1_over_rho;
    top_fields.omega2_over_rho = fields.omega2_over_rho;
    top_fields.omega3_over_rho = fields.omega3_over_rho;
    top_fields.f3_at_z = fields.f3_at_z;
    top_fields.rho = fields.rho;
    top_fields.rho_up = fields.rho_up;
    top_fields.inverse_spacing_up = fields.inverse_spacing;
    top_fields.inverse_spacing_mid = Profile("deformation_inverse_spacing_mid", nz);

    functor.output = Kokkos::View<Real****, Layout>("deformation_output", 15, nz, ny, nx);

    auto u1 = Kokkos::create_mirror_view(fields.u1);
    auto u2 = Kokkos::create_mirror_view(fields.u2);
    auto omega1 = Kokkos::create_mirror_view(fields.omega1_over_rho);
    auto omega2 = Kokkos::create_mirror_view(fields.omega2_over_rho);
    auto omega3 = Kokkos::create_mirror_view(fields.omega3_over_rho);
    auto f = Kokkos::create_mirror_view(fields.f3_at_z);
    auto rho = Kokkos::create_mirror_view(fields.rho);
    auto rho_up = Kokkos::create_mirror_view(fields.rho_up);
    auto fn1 = Kokkos::create_mirror_view(fields.fn1);
    auto fn2 = Kokkos::create_mirror_view(fields.fn2);
    auto inverse_spacing = Kokkos::create_mirror_view(fields.inverse_spacing);
    auto w = Kokkos::create_mirror_view(top_fields.w);
    auto inverse_mid = Kokkos::create_mirror_view(top_fields.inverse_spacing_mid);

    const auto input_snapshot = [&]() {
        std::vector<Real> result;
        const auto append = [&](const auto& view) {
            const auto values = snapshot(view);
            result.insert(result.end(), values.begin(), values.end());
        };
        append(fields.u1);
        append(fields.u2);
        append(fields.omega1_over_rho);
        append(fields.omega2_over_rho);
        append(fields.omega3_over_rho);
        append(fields.f3_at_z);
        append(fields.rho);
        append(fields.rho_up);
        append(fields.fn1);
        append(fields.fn2);
        append(fields.inverse_spacing);
        append(top_fields.w);
        append(top_fields.inverse_spacing_mid);
        return result;
    };

    const auto policy = Kokkos::Experimental::require(
        Kokkos::MDRangePolicy<Kokkos::Rank<3>>({h, h, h}, {k_top + 1, ny - h, nx - h}),
        Kokkos::Experimental::WorkItemProperty::HintLightWeight);

#if defined(KOKKOS_ENABLE_CUDA)
    static_assert(std::is_same<Kokkos::DefaultExecutionSpace, Kokkos::Cuda>::value,
        "CUDA capture test requires CUDA as the default execution space.");
    const Kokkos::Cuda execution;
    CapturedDeformation capture;

    // Prepare the exact combined launch functor without evaluating fields.
    auto preparation = functor;
    preparation.preparation_only = true;
    Kokkos::parallel_for("PrepareVorticityComponents", policy, preparation);
    execution.fence();
    require_cuda(cudaGetLastError(), "Prepare vorticity components");
#endif

    for (bool stretched : {false, true}) {
        std::vector<Real> height(nz, real(0.0));
        for (int k = 1; k < nz; ++k) {
            height[k] = height[k - 1] + (stretched ? real(40.0) + real(15.0) * k : real(100.0));
        }

        for (int k = 0; k < nz; ++k) {
            rho(k) = real(1.0) + real(0.1) * k;
            rho_up(k) = real(1.05) + real(0.1) * k;
            fn1(k) = real(0.8) + real(0.03) * k;
            fn2(k) = real(1.1) - real(0.04) * k;
            inverse_spacing(k) = k + 1 < nz ? real(1.0) / (height[k + 1] - height[k]) : real(0.0);
            inverse_mid(k) =
                stretched ? real(1.0) / (real(30.0) + real(17.0) * k) : real(1.0) / real(100.0);
        }

        Kokkos::deep_copy(fields.rho, rho);
        Kokkos::deep_copy(fields.rho_up, rho_up);
        Kokkos::deep_copy(fields.fn1, fn1);
        Kokkos::deep_copy(fields.fn2, fn2);
        Kokkos::deep_copy(fields.inverse_spacing, inverse_spacing);
        Kokkos::deep_copy(top_fields.inverse_spacing_mid, inverse_mid);

        // Retain the six deformation cases, then exercise positive and
        // negative transport of a nonconstant wave and a final zero reset.
        for (int mode = 0; mode < 9; ++mode) {
            const bool along = mode == 1 || mode == 4;
            const bool cross = mode == 2 || mode == 4;
            const bool vertical = mode == 3 || mode == 4;
            const bool wave_case = mode == 6 || mode == 7;
            const Real wave_velocity = mode == 6 ? real(1e-6) : mode == 7 ? real(-1e-6) : real(0.0);
            const Real a = along ? real(1e-6) : real(0.0);
            const Real b = cross ? real(-2e-6) : real(0.0);
            const Real c = vertical ? real(3e-9) : real(0.0);
            const Real d = cross ? real(4e-6) : real(0.0);
            const Real e = along ? real(-5e-6) : real(0.0);
            const Real g = vertical ? real(-6e-9) : real(0.0);
            const Real x = real(2e-10);
            const Real y = real(-3e-10);
            const Real z = real(4e-4);
            const Real p = cross ? real(0.2) : real(0.0);
            const Real q = cross ? real(-0.3) : real(0.0);
            const Real offset = along ? real(0.1) : real(0.0);

            for (int j = 0; j < ny; ++j) {
                const Real phi_u = south_edge + (j - h + real(0.5)) * dq2;
                const Real phi_v = south_edge + (j - h + real(1.0)) * dq2;
                for (int i = 0; i < nx; ++i) {
                    const Real lambda_u = (i - h + real(1.0)) * dq1;
                    const Real lambda_v = (i - h + real(0.5)) * dq1;
                    f(j, i) = real(1e-4) + real(2e-5) * phi_v;
                    for (int k = 0; k < nz; ++k) {
                        u1(k, j, i) = a * lambda_u + b * phi_u + c * height[k] + wave_velocity;
                        u2(k, j, i) = d * lambda_v + e * phi_v + g * height[k];
                        omega1(k, j, i) = x * (wave_case ? wave(lambda_v) : real(1.0));
                        omega2(k, j, i) = y * (wave_case ? wave(lambda_u) : real(1.0));
                        omega3(k, j, i) = z;
                        w(k, j, i) =
                            (height[k_top] - height[k]) * (offset + p * lambda_v + q * phi_u);
                    }
                }
            }

            Kokkos::deep_copy(fields.u1, u1);
            Kokkos::deep_copy(fields.u2, u2);
            Kokkos::deep_copy(fields.omega1_over_rho, omega1);
            Kokkos::deep_copy(fields.omega2_over_rho, omega2);
            Kokkos::deep_copy(fields.omega3_over_rho, omega3);
            Kokkos::deep_copy(fields.f3_at_z, f);
            Kokkos::deep_copy(top_fields.w, w);
            const auto before = input_snapshot();
            Kokkos::deep_copy(functor.output, sentinel);

#if defined(KOKKOS_ENABLE_CUDA)
            if (!capture.executable) {
                execution.fence();
                require_cuda(
                    cudaStreamBeginCapture(execution.cuda_stream(), cudaStreamCaptureModeGlobal),
                    "Begin vorticity component capture");
                Kokkos::parallel_for("VorticityComponents", policy, functor);
                require_cuda(cudaStreamEndCapture(execution.cuda_stream(), &capture.graph),
                    "End vorticity component capture");
                require_cuda(
                    cudaGraphInstantiate(&capture.executable, capture.graph, nullptr, nullptr, 0),
                    "Instantiate vorticity component graph");
            }
            require_cuda(cudaGraphLaunch(capture.executable, execution.cuda_stream()),
                "Replay vorticity component graph");
#else
            Kokkos::parallel_for("VorticityComponents", policy, functor);
#endif
            Kokkos::fence();
            const auto first = snapshot(functor.output);

            Kokkos::deep_copy(functor.output, sentinel);
            Kokkos::parallel_for("VorticityComponents", policy, functor);
            Kokkos::fence();

            const bool repeat_equal = first == snapshot(functor.output);
            const bool inputs_preserved = before == input_snapshot();
            const auto actual =
                Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), functor.output);
            bool regions_preserved = true;
            bool finite = true;
            Real error = real(0.0);
            Real top_error = real(0.0);
            Real transport_error = real(0.0);

            for (int k = 0; k < nz; ++k) {
                for (int j = 0; j < ny; ++j) {
                    const Real phi_u = south_edge + (j - h + real(0.5)) * dq2;
                    const Real phi_v = south_edge + (j - h + real(1.0)) * dq2;
                    for (int i = 0; i < nx; ++i) {
                        const Real lambda_u = (i - h + real(1.0)) * dq1;
                        const Real lambda_v = (i - h + real(0.5)) * dq1;
                        const bool horizontal = j >= h && j < ny - h && i >= h && i < nx - h;
                        const bool active = horizontal && k >= h && k < k_top;
                        const bool top_active = horizontal && k == k_top;

                        for (int n = 0; n < 15; ++n) {
                            const bool used = n >= 6 && n <= 8 ? top_active : active;
                            if (!used) {
                                regions_preserved =
                                    regions_preserved && actual(n, k, j, i) == sentinel;
                            }
                            else {
                                finite = finite && std::isfinite(actual(n, k, j, i));
                            }
                        }

                        if (active) {
                            const Real weight = real(0.5) * (fn1(k) + fn2(k));
                            const Real density_factor =
                                real(0.5) * rho_up(k) *
                                (real(1.0) / rho(k) + real(1.0) / rho(k + 1));
                            const Real east_scale = radius * std::cos(phi_v);
                            const Real f_v = real(1e-4) + real(2e-5) * phi_v;
                            const Real f_u = real(1e-4) + real(2e-5) * phi_u;

                            // Wave cases have constant canonical horizontal
                            // wind and zero vertical wind, so deformation is zero.
                            const std::array<Real, 6> expected = {east_scale * weight * x * a,
                                east_scale * (weight * y * b + rho_up(k) * z * c),
                                east_scale * density_factor * f_v * c,
                                -radius * weight * y * e,
                                -radius * (weight * x * d + rho_up(k) * z * g),
                                -radius * density_factor * f_u * g};

                            for (int n = 0; n < 6; ++n) {
                                const Real physical =
                                    (n < 3 ? east_scale : -radius) * actual(n, k, j, i);
                                error = std::max(error,
                                    std::abs(physical - expected[n]) /
                                        std::max(real(1e-8), std::abs(expected[n])));
                            }

                            std::array<Real, 6> expected_transport{};
                            if (wave_case) {
                                const Real velocity = weight * wave_velocity;
                                expected_transport[0] =
                                    -east_scale * x *
                                    (wave_face(lambda_v, dq1, velocity) -
                                        wave_face(lambda_v - dq1, dq1, velocity)) /
                                    dq1;
                                expected_transport[3] =
                                    radius * y *
                                    (wave_face(lambda_u, dq1, velocity) -
                                        wave_face(lambda_u - dq1, dq1, velocity)) /
                                    dq1;
                            }
                            else {
                                const Real weighted_height =
                                    real(0.5) * (fn1(k) * height[k + 1] + fn2(k) * height[k]);

                                // The radius-squared factors cancel between
                                // the face Jacobians and destination Jacobian.
                                const auto meridional_flux = [&](Real longitude, Real latitude) {
                                    return std::cos(latitude) *
                                           (weight * (d * longitude + e * latitude) +
                                               g * weighted_height);
                                };

                                const Real xi_q2 = (meridional_flux(lambda_v, phi_u + dq2) -
                                                       meridional_flux(lambda_v, phi_u)) /
                                                   (std::cos(phi_v) * dq2);
                                const Real eta_q2 = (meridional_flux(lambda_u, phi_v) -
                                                        meridional_flux(lambda_u, phi_v - dq2)) /
                                                    (std::cos(phi_u) * dq2);

                                const auto vertical_profile_flux = [&](int face) {
                                    return real(0.5) *
                                           (rho_up(face) * (height[k_top] - height[face]) +
                                               rho_up(face + 1) *
                                                   (height[k_top] - height[face + 1]));
                                };
                                const Real vertical_difference =
                                    (vertical_profile_flux(k) - vertical_profile_flux(k - 1)) *
                                    inverse_spacing(k);

                                expected_transport = {-east_scale * x * weight * a,
                                    -east_scale * x * xi_q2,
                                    -east_scale * x * vertical_difference *
                                        (offset + p * lambda_v + q * phi_v),
                                    radius * y * weight * a,
                                    radius * y * eta_q2,
                                    radius * y * vertical_difference *
                                        (offset + p * lambda_u + q * phi_u)};
                            }

                            for (int n = 0; n < 6; ++n) {
                                const Real physical =
                                    (n < 3 ? east_scale : -radius) * actual(n + 9, k, j, i);
                                transport_error = std::max(transport_error,
                                    std::abs(physical - expected_transport[n]) /
                                        std::max(real(1e-12), std::abs(expected_transport[n])));
                            }
                        }

                        if (top_active) {
                            const Real distance = height[k_top] - height[k_top - 1];
                            const Real w_at_z = distance * (offset + p * lambda_u + q * phi_v);
                            const Real f_at_z = real(1e-4) + real(2e-5) * phi_v;
                            const Real lower_factor =
                                rho_up(k_top - 1) * inverse_mid(k_top) / inverse_spacing(k_top - 1);

                            const std::array<Real, 3> expected = {-rho(k_top) * inverse_mid(k_top) *
                                                                      z * w_at_z,
                                real(0.5) * lower_factor * distance * (x * p + y * q),
                                -inverse_mid(k_top) * f_at_z * w_at_z};

                            for (int n = 0; n < 3; ++n) {
                                top_error = std::max(top_error,
                                    std::abs(actual(n + 6, k, j, i) - expected[n]) /
                                        std::max(real(1e-12), std::abs(expected[n])));
                            }
                        }
                    }
                }
            }

            const bool passed = finite && repeat_equal && inputs_preserved && regions_preserved &&
                                error <= tolerance && top_error <= tolerance &&
                                transport_error <= tolerance;

            std::printf("RLL components %s stretched=%d mode=%d deformation=%.3e top=%.3e "
                        "transport=%.3e repeat=%d inputs=%d regions=%d %s\n",
                layout_name,
                static_cast<int>(stretched),
                mode,
                static_cast<double>(error),
                static_cast<double>(top_error),
                static_cast<double>(transport_error),
                static_cast<int>(repeat_equal),
                static_cast<int>(inputs_preserved),
                static_cast<int>(regions_preserved),
                passed ? "PASS" : "FAIL");
            if (!passed) {
                ++failures;
            }
        }
    }

#if defined(KOKKOS_ENABLE_CUDA)
    std::puts("RLL component execution: prepared CUDA capture/replay");
#else
    std::puts("RLL component execution: ordinary repeated execution");
#endif
}

} // namespace

int
main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);
    Kokkos::initialize(argc, argv);

    try {
        int size = 0;
        MPI_Comm_size(MPI_COMM_WORLD, &size);
        if (size != 1) {
            throw std::runtime_error("This test requires one MPI rank.");
        }

        const auto layout = make_layout();
        const CartesianGeometry cartesian(layout, real(2.0), real(3.0));

        const Real radius = real(6371220.0);
        const Real south_edge = real(-0.3);
        const RegularLatLonGeometry rll(layout,
            real(0.08),
            real(0.04),
            real(0.0),
            south_edge,
            radius);

        // Generalized component operators are geometry-driven and may also be
        // constructed for Cartesian geometry. Existing Cartesian production
        // remains on its exact-regression path; this only checks the operator API.
        (void)make_generalized_horizontal_deformation_device_view(cartesian);
        (void)make_generalized_top_deformation_device_view(cartesian);
        (void)make_generalized_horizontal_vorticity_transport_device_view(cartesian);

        auto narrow_layout = layout;
        narrow_layout.halo = 1;
        const RegularLatLonGeometry narrow(narrow_layout,
            real(0.08),
            real(0.04),
            real(0.0),
            south_edge,
            radius);
        bool rejected_narrow_halo = false;
        try {
            (void)make_generalized_horizontal_vorticity_transport_device_view(narrow);
        }
        catch (const std::invalid_argument&) {
            rejected_narrow_halo = true;
        }
        if (!rejected_narrow_halo) {
            ++failures;
            std::fputs("Generalized vorticity transport failed to reject a one-cell halo\n",
                stderr);
        }

        test_deformation<Kokkos::LayoutLeft>(rll, radius, south_edge, "LayoutLeft");
        test_deformation<Kokkos::LayoutRight>(rll, radius, south_edge, "LayoutRight");
    }
    catch (const std::exception& error) {
        ++failures;
        std::fprintf(stderr, "test_horizontal_vorticity: %s\n", error.what());
    }

    int global_failures = 0;
    MPI_Allreduce(&failures, &global_failures, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

    if (global_failures == 0) {
        std::puts("test_horizontal_vorticity: PASS");
    }

    Kokkos::finalize();
    MPI_Finalize();
    return global_failures == 0 ? 0 : 1;
}
