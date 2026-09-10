#include "core/Field.hpp"
#include "core/geometry/CartesianGeometry.hpp"
#include "core/geometry/RegularLatLonGeometry.hpp"
#include "dynamics/operators/OrthogonalAnelasticMassFlux.hpp"
#include "dynamics/operators/RegularLatLonScalarTransport.hpp"
#include "dynamics/operators/TakacsScalarTransport.hpp"

#include <Kokkos_Core.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstring>
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
using VVM::Core::Field;
using VVM::Core::Geometry::CartesianGeometry;
using VVM::Core::Geometry::HorizontalDomainLayout;
using VVM::Core::Geometry::RegularLatLonGeometry;
using VVM::Dynamics::Operators::make_orthogonal_contravariant_mass_flux_q1_device_view;
using VVM::Dynamics::Operators::make_orthogonal_contravariant_mass_flux_q2_device_view;
using VVM::Dynamics::Operators::make_takacs_scalar_transport_device_view;
using VVM::Dynamics::Operators::RegularLatLonScalarTransport;

int failures = 0;

void
check(const bool condition, const char* message) {

    if (!condition) {
        ++failures;
        std::fprintf(stderr, "FAIL: %s\n", message);
    }
}

bool
close(const Real actual, const Real expected, const Real tolerance) {

    return std::abs(actual - expected) <=
           tolerance * std::max(real(1.0), std::max(std::abs(actual), std::abs(expected)));
}

template <std::size_t Dim>
std::vector<Real>
snapshot(const Field<Dim>& field) {

    const auto host = field.get_host_data();

    std::vector<Real> result;
    result.reserve(host.size());

    if constexpr (Dim == 1) {
        for (std::size_t k = 0; k < host.extent(0); ++k) {

            result.push_back(host(k));
        }
    }
    else if constexpr (Dim == 3) {
        for (std::size_t k = 0; k < host.extent(0); ++k) {

            for (std::size_t j = 0; j < host.extent(1); ++j) {

                for (std::size_t i = 0; i < host.extent(2); ++i) {

                    result.push_back(host(k, j, i));
                }
            }
        }
    }
    else {
        static_assert(Dim == 1 || Dim == 3, "Unsupported snapshot dimension.");
    }

    return result;
}

bool
same_bits(const std::vector<Real>& first, const std::vector<Real>& second) {

    if (first.size() != second.size()) {
        return false;
    }

    return first.empty() ||
           std::memcmp(first.data(), second.data(), first.size() * sizeof(Real)) == 0;
}

HorizontalDomainLayout
make_layout() {
    HorizontalDomainLayout layout;

    layout.global_nx = 20;
    layout.global_ny = 12;
    layout.local_physical_nx = 20;
    layout.local_physical_ny = 12;
    layout.global_start_i = 0;
    layout.global_start_j = 0;
    layout.halo = 2;
    layout.panel_id = -1;

    return layout;
}

#if defined(KOKKOS_ENABLE_CUDA)

void
require_cuda(const cudaError_t status, const char* operation) {

    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(status));
    }
}

class TestGraph {
public:
    explicit TestGraph(const cudaStream_t stream) : stream_(stream) {}

    TestGraph(const TestGraph&) = delete;
    TestGraph& operator=(const TestGraph&) = delete;

    ~TestGraph() {
        if (capturing_) {
            cudaGraph_t abandoned = nullptr;

            cudaStreamEndCapture(stream_, &abandoned);

            if (abandoned != nullptr) {
                cudaGraphDestroy(abandoned);
            }
        }

        cudaStreamSynchronize(stream_);

        if (executable_ != nullptr) {
            cudaGraphExecDestroy(executable_);
        }

        if (graph_ != nullptr) {
            cudaGraphDestroy(graph_);
        }
    }

    void
    begin() {
        require_cuda(cudaStreamBeginCapture(stream_, cudaStreamCaptureModeGlobal),
            "begin scalar-transport capture");

        capturing_ = true;
    }

    void
    finish() {
        const cudaError_t status = cudaStreamEndCapture(stream_, &graph_);

        capturing_ = false;

        require_cuda(status, "end scalar-transport capture");

        if (graph_ == nullptr) {
            throw std::runtime_error("Scalar-transport capture returned "
                                     "a null graph.");
        }

        std::size_t node_count = 0;

        require_cuda(cudaGraphGetNodes(graph_, nullptr, &node_count),
            "read scalar-transport graph nodes");

        if (node_count == 0) {
            throw std::runtime_error("Scalar-transport capture recorded "
                                     "no nodes.");
        }

        require_cuda(cudaGraphInstantiate(&executable_, graph_, nullptr, nullptr, 0),
            "instantiate scalar-transport graph");
    }

    void
    launch() {
        require_cuda(cudaGraphLaunch(executable_, stream_), "launch scalar-transport graph");

        require_cuda(cudaStreamSynchronize(stream_), "complete scalar-transport graph");
    }

private:
    cudaStream_t stream_ = nullptr;
    cudaGraph_t graph_ = nullptr;
    cudaGraphExec_t executable_ = nullptr;
    bool capturing_ = false;
};

#endif

void
test_validation(const RegularLatLonGeometry& geometry) {

    const HorizontalDomainLayout layout = make_layout();

    const CartesianGeometry cartesian(layout, real(1000.0), real(1000.0));

    bool rejected_cartesian = false;

    try {
        const RegularLatLonScalarTransport invalid(cartesian);

        (void)invalid;
    }
    catch (const std::invalid_argument&) {
        rejected_cartesian = true;
    }

    check(rejected_cartesian,
        "RLL scalar transport must reject "
        "Cartesian geometry");

    const int nz = 8;
    const int ny = layout.local_total_ny();
    const int nx = layout.local_total_nx();

    Field<3> scalar("validation_scalar", {nz, ny, nx});
    Field<3> mass_flux_q1("validation_mass_flux_q1", {nz, ny, nx});
    Field<3> mass_flux_q2("validation_mass_flux_q2", {nz, ny, nx});
    Field<3> mass_flux_vertical("validation_mass_flux_vertical", {nz, ny, nx});
    Field<1> spacing("validation_spacing", {nz});

    const RegularLatLonScalarTransport transport(geometry);

    bool rejected_alias = false;

    try {
        transport.add_flux_convergence(scalar,
            mass_flux_q1,
            mass_flux_q2,
            mass_flux_vertical,
            spacing,
            scalar,
            2,
            6);
    }
    catch (const std::invalid_argument&) {
        rejected_alias = true;
    }

    check(rejected_alias,
        "RLL scalar transport must reject "
        "overlapping input and output storage");

    bool rejected_short_range = false;

    try {
        transport.add_flux_convergence(scalar,
            mass_flux_q1,
            mass_flux_q2,
            mass_flux_vertical,
            spacing,
            mass_flux_q1,
            2,
            4);
    }
    catch (const std::invalid_argument&) {
        rejected_short_range = true;
    }

    check(rejected_short_range,
        "RLL scalar transport must reject "
        "a vertical range shorter than three cells");
}

void
test_transport_and_replay(const RegularLatLonGeometry& geometry) {

    const HorizontalDomainLayout layout = geometry.layout();

    const int halo = layout.halo;
    const int ny = layout.local_total_ny();
    const int nx = layout.local_total_nx();
    const int nz = 9;
    const int k_begin = 2;
    const int k_end = 7;

    const Real sentinel = real(-17.25);

    Field<3> scalar("stage_scalar", {nz, ny, nx});
    Field<3> physical_mass_flux_q1("stage_physical_mass_flux_q1", {nz, ny, nx});
    Field<3> physical_mass_flux_q2("stage_physical_mass_flux_q2", {nz, ny, nx});
    Field<3> vertical_mass_flux("stage_vertical_mass_flux", {nz, ny, nx});
    Field<1> vertical_spacing("stage_vertical_spacing", {nz});

    Field<3> expected("stage_expected", {nz, ny, nx});
    Field<3> direct("stage_direct", {nz, ny, nx});
    Field<3> replay("stage_replay", {nz, ny, nx});

    auto scalar_data = scalar.get_mutable_device_data();
    auto physical_q1_data = physical_mass_flux_q1.get_mutable_device_data();
    auto physical_q2_data = physical_mass_flux_q2.get_mutable_device_data();
    auto vertical_flux_data = vertical_mass_flux.get_mutable_device_data();
    auto spacing_data = vertical_spacing.get_mutable_device_data();

    Kokkos::parallel_for("InitializeRllScalarTransportVolumes",
        Kokkos::MDRangePolicy<Kokkos::Rank<3>>({0, 0, 0}, {nz, ny, nx}),
        KOKKOS_LAMBDA(const int k, const int j, const int i) {
            const Real level = static_cast<Real>(k);
            const Real latitude_index = static_cast<Real>(j - halo);
            const Real longitude_index = static_cast<Real>(i - halo);

            scalar_data(k, j, i) = real(2.0) + real(0.03) * longitude_index -
                                   real(0.02) * latitude_index + real(0.04) * level +
                                   real(0.001) * longitude_index * latitude_index;

            physical_q1_data(k, j, i) = real(1.1) + real(0.025) * longitude_index +
                                        real(0.015) * latitude_index + real(0.01) * level;

            physical_q2_data(k, j, i) = real(-0.6) + real(0.012) * longitude_index -
                                        real(0.02) * latitude_index + real(0.008) * level;

            vertical_flux_data(k, j, i) = real(0.08) + real(0.004) * longitude_index -
                                          real(0.003) * latitude_index + real(0.02) * level;
        });

    Kokkos::parallel_for("InitializeRllScalarTransportSpacing",
        Kokkos::RangePolicy<>(0, nz),
        KOKKOS_LAMBDA(
            const int k) { spacing_data(k) = real(80.0) + real(7.0) * static_cast<Real>(k); });

    Kokkos::deep_copy(expected.get_mutable_device_data(), sentinel);
    Kokkos::deep_copy(direct.get_mutable_device_data(), sentinel);
    Kokkos::deep_copy(replay.get_mutable_device_data(), sentinel);

    Kokkos::fence();

    const auto inputs_before = [&]() {
        std::vector<Real> values;

        const auto append = [&](const auto& field) {
            const auto part = snapshot(field);

            values.insert(values.end(), part.begin(), part.end());
        };

        append(scalar);
        append(physical_mass_flux_q1);
        append(physical_mass_flux_q2);
        append(vertical_mass_flux);
        append(vertical_spacing);

        return values;
    }();

    const auto scalar_reference = scalar.get_device_data();
    const auto physical_q1_reference = physical_mass_flux_q1.get_device_data();
    const auto physical_q2_reference = physical_mass_flux_q2.get_device_data();
    const auto vertical_flux_reference = vertical_mass_flux.get_device_data();
    const auto spacing_reference = vertical_spacing.get_device_data();
    const auto expected_data = expected.get_mutable_device_data();

    const auto operator_view = make_takacs_scalar_transport_device_view(geometry);
    const auto adapted_q1 =
        make_orthogonal_contravariant_mass_flux_q1_device_view(geometry, physical_q1_reference);
    const auto adapted_q2 =
        make_orthogonal_contravariant_mass_flux_q2_device_view(geometry, physical_q2_reference);

    Kokkos::parallel_for("EvaluateExplicitRllScalarTransportReference",
        Kokkos::MDRangePolicy<Kokkos::Rank<3>>({k_begin, halo, halo},
            {k_end, ny - halo, nx - halo}),
        KOKKOS_LAMBDA(const int k, const int j, const int i) {
            expected_data(k, j, i) +=
                operator_view.calculate_horizontal_flux_convergence_at_t(scalar_reference,
                    adapted_q1,
                    adapted_q2,
                    k,
                    j,
                    i) +
                operator_view.calculate_vertical_flux_convergence_at_t(scalar_reference,
                    vertical_flux_reference,
                    k,
                    j,
                    i,
                    k_begin,
                    k_end,
                    real(1.0) / spacing_reference(k));
        });

    Kokkos::fence();

    const RegularLatLonScalarTransport transport(geometry);

    RegularLatLonScalarTransport::prepare_execution();

#if defined(KOKKOS_ENABLE_CUDA)
    static_assert(std::is_same<Kokkos::DefaultExecutionSpace, Kokkos::Cuda>::value,
        "The scalar-transport CUDA graph test "
        "requires CUDA as the default execution space.");

    // Capture the first real stage launch after explicit preparation.
    TestGraph graph(Kokkos::Cuda().cuda_stream());

    graph.begin();

    transport.add_flux_convergence(scalar,
        physical_mass_flux_q1,
        physical_mass_flux_q2,
        vertical_mass_flux,
        vertical_spacing,
        replay,
        k_begin,
        k_end);

    graph.finish();
    graph.launch();
#else
    transport.add_flux_convergence(scalar,
        physical_mass_flux_q1,
        physical_mass_flux_q2,
        vertical_mass_flux,
        vertical_spacing,
        replay,
        k_begin,
        k_end);
#endif

    transport.add_flux_convergence(scalar,
        physical_mass_flux_q1,
        physical_mass_flux_q2,
        vertical_mass_flux,
        vertical_spacing,
        direct,
        k_begin,
        k_end);

    Kokkos::fence();

    const auto expected_host = expected.get_host_data();
    const auto direct_host = direct.get_host_data();
    const auto replay_host = replay.get_host_data();

    const Real tolerance = sizeof(Real) == sizeof(double) ? real(5.0e-12) : real(5.0e-5);

    bool reference_match = true;
    bool replay_match = true;
    bool finite = true;
    bool untouched = true;
    bool changed = false;

    for (int k = 0; k < nz; ++k) {
        for (int j = 0; j < ny; ++j) {
            for (int i = 0; i < nx; ++i) {
                const bool written = k >= k_begin && k < k_end && j >= halo && j < ny - halo &&
                                     i >= halo && i < nx - halo;

                if (written) {
                    reference_match =
                        reference_match &&
                        close(direct_host(k, j, i), expected_host(k, j, i), tolerance);

                    replay_match = replay_match &&
                                   close(replay_host(k, j, i), direct_host(k, j, i), tolerance);

                    finite = finite && std::isfinite(expected_host(k, j, i)) &&
                             std::isfinite(direct_host(k, j, i)) &&
                             std::isfinite(replay_host(k, j, i));

                    changed = changed || direct_host(k, j, i) != sentinel;
                }
                else {
                    untouched = untouched && expected_host(k, j, i) == sentinel &&
                                direct_host(k, j, i) == sentinel &&
                                replay_host(k, j, i) == sentinel;
                }
            }
        }
    }

    check(reference_match,
        "Field-level RLL scalar transport must "
        "match explicit operator composition");

    check(replay_match,
        "RLL scalar transport replay must match "
        "ordinary execution");

    check(finite,
        "RLL scalar transport must produce "
        "finite values");

    check(changed,
        "RLL scalar transport must update "
        "the requested physical cells");

    check(untouched,
        "RLL scalar transport must preserve "
        "halos and levels outside its write range");

    const auto inputs_after = [&]() {
        std::vector<Real> values;

        const auto append = [&](const auto& field) {
            const auto part = snapshot(field);

            values.insert(values.end(), part.begin(), part.end());
        };

        append(scalar);
        append(physical_mass_flux_q1);
        append(physical_mass_flux_q2);
        append(vertical_mass_flux);
        append(vertical_spacing);

        return values;
    }();

    check(same_bits(inputs_before, inputs_after),
        "RLL scalar transport must not modify "
        "its input fields");
}

} // namespace

int
main(int argc, char** argv) {

    Kokkos::initialize(argc, argv);

    {
        try {
            const HorizontalDomainLayout layout = make_layout();

            const Real pi = std::acos(real(-1.0));

            const RegularLatLonGeometry geometry(layout,
                real(2.0) * pi / static_cast<Real>(layout.global_nx),
                (pi / real(3.0)) / static_cast<Real>(layout.global_ny),
                -pi,
                -pi / real(6.0),
                real(6371220.0));

            test_validation(geometry);
            test_transport_and_replay(geometry);
        }
        catch (const std::exception& error) {
            ++failures;

            std::fprintf(stderr, "Unexpected exception: %s\n", error.what());
        }
    }

    Kokkos::finalize();

    if (failures == 0) {
        std::fprintf(stdout, "test_regular_latlon_scalar_transport: PASS\n");
    }
    else {
        std::fprintf(stderr,
            "test_regular_latlon_scalar_transport: "
            "%d failure(s)\n",
            failures);
    }

    return failures == 0 ? 0 : 1;
}
