#include "core/Grid.hpp"
#include "core/Parameters.hpp"
#include "core/State.hpp"
#include "io/bp5/Bp5HistoryWriter.hpp"
#include "utils/ConfigurationManager.hpp"

#include <Kokkos_Core.hpp>
#include <adios2.h>
#include <mpi.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

#include "../../externals/json/json.hpp"

namespace {

constexpr int kNx = 8;
constexpr int kNy = 7; // Deliberately uneven for two- and four-rank decompositions.
constexpr int kNz = 6;
constexpr int kSteps = 3;

int g_rank = 0;
int g_failures = 0;

void check(bool condition, const std::string& message) {
    if (condition) return;
    ++g_failures;
    std::fprintf(stderr, "[rank %d] FAIL: %s\n", g_rank, message.c_str());
}

KOKKOS_INLINE_FUNCTION VVM::Real expected_1d(int step, int k) {
    return VVM::real(1000000 * step + k);
}

KOKKOS_INLINE_FUNCTION VVM::Real expected_2d(int step, int j, int i) {
    return VVM::real(1000000 * step + 100 * j + i);
}

KOKKOS_INLINE_FUNCTION VVM::Real expected_3d(int step, int k, int j, int i) {
    return VVM::real(1000000 * step + 10000 * k + 100 * j + i);
}

KOKKOS_INLINE_FUNCTION VVM::Real expected_4d(int step, int c, int k, int j, int i) {
    return VVM::real(10000000 * c) + expected_3d(step, k, j, i);
}

template <typename View>
void fill_junk(View view) {
    Kokkos::deep_copy(view, VVM::real(-987654321));
}

void fill_fields(VVM::Core::State& state, const VVM::Core::Grid& grid, int step) {
    const int h = grid.get_halo_cells();
    const int nx = grid.get_local_total_points_x();
    const int ny = grid.get_local_total_points_y();
    const int nz = grid.get_local_total_points_z();
    const int i0 = grid.get_local_physical_start_x();
    const int j0 = grid.get_local_physical_start_y();

    auto thbar = state.get_field<1>("thbar").get_mutable_device_data();
    auto topo = state.get_field<2>("topo").get_mutable_device_data();
    auto u = state.get_field<3>("u").get_mutable_device_data();
    auto four = state.get_field<4>("bp5_test_4d").get_mutable_device_data();
    fill_junk(thbar);
    fill_junk(topo);
    fill_junk(u);
    fill_junk(four);

    Kokkos::parallel_for(
        "bp5_test_fill_1d", Kokkos::RangePolicy<>(h, nz - h),
        KOKKOS_LAMBDA(const int k) { thbar(k) = expected_1d(step, k - h); });
    Kokkos::parallel_for(
        "bp5_test_fill_2d",
        Kokkos::MDRangePolicy<Kokkos::Rank<2>>({h, h}, {ny - h, nx - h}),
        KOKKOS_LAMBDA(const int j, const int i) {
            topo(j, i) = expected_2d(step, j0 + j - h, i0 + i - h);
        });
    Kokkos::parallel_for(
        "bp5_test_fill_3d",
        Kokkos::MDRangePolicy<Kokkos::Rank<3>>({h, h, h}, {nz - h, ny - h, nx - h}),
        KOKKOS_LAMBDA(const int k, const int j, const int i) {
            u(k, j, i) = expected_3d(step, k - h, j0 + j - h, i0 + i - h);
            for (int c = 0; c < 2; ++c) {
                four(c, k, j, i) =
                    expected_4d(step, c, k - h, j0 + j - h, i0 + i - h);
            }
        });
    Kokkos::fence("bp5_test_fill_complete");
}

void fill_coordinates(VVM::Core::Parameters& parameters, const VVM::Core::Grid& grid) {
    const int h = grid.get_halo_cells();
    const int nz = grid.get_local_total_points_z();
    auto z = parameters.z_mid.get_mutable_device_data();
    fill_junk(z);
    Kokkos::parallel_for(
        "bp5_test_fill_z", Kokkos::RangePolicy<>(h, nz - h),
        KOKKOS_LAMBDA(const int k) { z(k) = VVM::real(50 + 25 * (k - h)); });
    Kokkos::fence("bp5_test_coordinates_complete");
}

nlohmann::json make_config(
    const std::filesystem::path& base_config,
    const std::filesystem::path& output_dir,
    const std::string& mode,
    bool async_write,
    const std::string& precision,
    bool empty_rank_case,
    int comm_size) {
    std::ifstream stream(base_config);
    if (!stream) throw std::runtime_error("Cannot read base test configuration.");
    nlohmann::json config;
    stream >> config;
    config["grid"]["nx"] = kNx;
    config["grid"]["ny"] = kNy;
    config["grid"]["nz"] = kNz;
    config["grid"]["n_halo_cells"] = 2;
    config["grid"]["dx"] = 125.0;
    config["grid"]["dy"] = 250.0;
    config["output"]["engine"] = "BP5";
    config["output"]["output_dir"] = output_dir.string();
    config["output"]["output_filename_prefix"] = "history";
    config["output"]["fields_to_output"] =
        nlohmann::json::array({"thbar", "topo", "u", "bp5_test_4d"});
    config["output"]["output_grid"] = {
        {"x_start", 0}, {"x_end", empty_rank_case ? 1 : 6},
        {"y_start", 0}, {"y_end", empty_rank_case ? 1 : 5},
        {"z_start", 1}, {"z_end", 4}};
    config["output"]["bp5"] = {
        {"aggregation_type", "TwoLevelShm"},
        {"num_subfiles", std::min(comm_size, 2)},
        {"stats_level", 0},
        {"async_write", async_write},
        {"buffer_mode", mode},
        {"precision", precision},
        {"overwrite", true}};
    return config;
}

template <typename T>
bool same(const T& a, const T& b) {
    return a == b;
}

void check_shape(const adios2::Dims& got, const adios2::Dims& want,
                 const std::string& name) {
    check(got == want, name + " global shape");
}

// Elem is the on-disk field element type, which is output.bp5.precision and not
// necessarily VVM::Real. Coordinates and clocks are deliberately excluded: the
// writer keeps those at full precision regardless, and this asserts that.
template <typename Elem>
void read_and_check(
    const std::filesystem::path& dataset,
    int x_end,
    int y_end) {
    using Other = std::conditional_t<std::is_same_v<Elem, float>, double, float>;
    adios2::ADIOS adios(MPI_COMM_SELF);
    adios2::IO io = adios.DeclareIO("VVM_BP5_TEST_READER");
    adios2::Engine reader = io.Open(dataset.string(), adios2::Mode::Read, MPI_COMM_SELF);
    int observed_steps = 0;
    while (reader.BeginStep() == adios2::StepStatus::OK) {
        auto time_var = io.InquireVariable<VVM::Real>("time");
        auto model_time_var = io.InquireVariable<double>("model_time_s");
        auto model_step_var = io.InquireVariable<std::int64_t>("model_step");
        auto x_var = io.InquireVariable<VVM::Real>("coordinates/x");
        auto y_var = io.InquireVariable<VVM::Real>("coordinates/y");
        auto z_var = io.InquireVariable<VVM::Real>("coordinates/z_mid");
        auto one_var = io.InquireVariable<Elem>("thbar");
        auto two_var = io.InquireVariable<Elem>("topo");
        auto three_var = io.InquireVariable<Elem>("u");
        auto four_var = io.InquireVariable<Elem>("bp5_test_4d");

        // The whole point of the precision option: the fields really are the
        // requested type on disk, and really are not the other one. Without the
        // negative half, a writer that ignored the setting would still pass.
        check(!io.InquireVariable<Other>("thbar"), "thbar is not the other float type");
        check(!io.InquireVariable<Other>("topo"), "topo is not the other float type");
        check(!io.InquireVariable<Other>("u"), "u is not the other float type");
        check(!io.InquireVariable<Other>("bp5_test_4d"),
              "bp5_test_4d is not the other float type");

        check(time_var && model_time_var && model_step_var, "clock variables exist");
        check(x_var && y_var && z_var, "coordinate variables exist");
        check(one_var && two_var && three_var && four_var, "all configured fields exist");
        if (!(time_var && model_time_var && model_step_var && x_var && y_var && z_var &&
              one_var && two_var && three_var && four_var)) {
            reader.EndStep();
            ++observed_steps;
            continue;
        }

        check_shape(x_var.Shape(), {kNx}, "coordinates/x");
        check_shape(y_var.Shape(), {kNy}, "coordinates/y");
        check_shape(z_var.Shape(), {kNz}, "coordinates/z_mid");
        check_shape(one_var.Shape(), {kNz}, "thbar");
        check_shape(two_var.Shape(), {kNy, kNx}, "topo");
        check_shape(three_var.Shape(), {kNz, kNy, kNx}, "u");
        check_shape(four_var.Shape(), {2, kNz, kNy, kNx}, "bp5_test_4d");

        VVM::Real time = VVM::real(-1);
        double model_time = -1;
        std::int64_t model_step = -1;
        reader.Get(time_var, time, adios2::Mode::Sync);
        reader.Get(model_time_var, model_time, adios2::Mode::Sync);
        reader.Get(model_step_var, model_step, adios2::Mode::Sync);
        check(same(time, VVM::real(observed_steps * 2.5)), "time value");
        check(same(model_time, observed_steps * 2.5), "model_time_s value");
        check(model_step == 10 + observed_steps, "model_step value");

        std::vector<VVM::Real> x(kNx), y(kNy), z(kNz);
        reader.Get(x_var, x.data(), adios2::Mode::Sync);
        reader.Get(y_var, y.data(), adios2::Mode::Sync);
        reader.Get(z_var, z.data(), adios2::Mode::Sync);
        for (int i = 0; i < kNx; ++i) check(x[i] == VVM::real(125 * i), "x coordinate value");
        for (int j = 0; j < kNy; ++j) check(y[j] == VVM::real(250 * j), "y coordinate value");
        for (int k = 0; k < kNz; ++k) check(z[k] == VVM::real(50 + 25 * k), "z coordinate value");

        constexpr int z_start = 1;
        constexpr int z_count = 4;
        const int x_count = x_end + 1;
        const int y_count = y_end + 1;
        one_var.SetSelection({{z_start}, {z_count}});
        two_var.SetSelection({{0, 0}, {static_cast<std::size_t>(y_count),
                                       static_cast<std::size_t>(x_count)}});
        three_var.SetSelection({{z_start, 0, 0},
                                {z_count, static_cast<std::size_t>(y_count),
                                 static_cast<std::size_t>(x_count)}});
        four_var.SetSelection({{0, z_start, 0, 0},
                               {2, z_count, static_cast<std::size_t>(y_count),
                                static_cast<std::size_t>(x_count)}});
        std::vector<Elem> one(z_count);
        std::vector<Elem> two(y_count * x_count);
        std::vector<Elem> three(z_count * y_count * x_count);
        std::vector<Elem> four(2 * z_count * y_count * x_count);
        reader.Get(one_var, one.data(), adios2::Mode::Sync);
        reader.Get(two_var, two.data(), adios2::Mode::Sync);
        reader.Get(three_var, three.data(), adios2::Mode::Sync);
        reader.Get(four_var, four.data(), adios2::Mode::Sync);
        // Exact equality against the cast of the model value. Conversion is a
        // plain static_cast, so there is no tolerance to tune even at float32:
        // a different rounding would be a real defect, not noise.
        for (int k = 0; k < z_count; ++k)
            check(one[k] == static_cast<Elem>(expected_1d(observed_steps, z_start + k)),
                  "1-D field value");
        for (int j = 0; j < y_count; ++j) {
            for (int i = 0; i < x_count; ++i) {
                check(two[j * x_count + i] ==
                          static_cast<Elem>(expected_2d(observed_steps, j, i)),
                      "2-D field value");
                for (int k = 0; k < z_count; ++k) {
                    const std::size_t p = (k * y_count + j) * x_count + i;
                    check(three[p] == static_cast<Elem>(
                                          expected_3d(observed_steps, z_start + k, j, i)),
                          "3-D field value");
                    for (int c = 0; c < 2; ++c) {
                        const std::size_t q = ((c * z_count + k) * y_count + j) * x_count + i;
                        check(four[q] == static_cast<Elem>(
                                             expected_4d(observed_steps, c, z_start + k, j, i)),
                              "4-D field value");
                    }
                }
            }
        }

        if (observed_steps == 0) {
            const auto units = io.InquireAttribute<std::string>("units", "u");
            const auto long_name = io.InquireAttribute<std::string>("long_name", "u");
            const auto standard_name = io.InquireAttribute<std::string>("standard_name", "u");
            const auto staggering = io.InquireAttribute<std::string>("grid_staggering", "u");
            const auto schema_version = io.InquireAttribute<std::string>("vvm_schema_version");
            check(units && units.Data().at(0) == "m s-1", "field units metadata");
            check(long_name && long_name.Data().at(0) == "x wind", "field long_name metadata");
            check(!standard_name,
                  "empty standard_name is omitted");
            check(staggering && staggering.Data().at(0) == "staggered_x",
                  "field staggering metadata");
            check(schema_version && schema_version.Data().at(0) == "1",
                  "dataset schema metadata");

            // A reader has to be able to tell a lossless file from a narrowed
            // one without inspecting variable types by hand.
            const auto field_precision =
                io.InquireAttribute<std::string>("vvm_field_precision");
            const auto real_precision =
                io.InquireAttribute<std::string>("vvm_real_precision");
            const std::string want_field =
                std::is_same_v<Elem, float> ? "float32" : "float64";
            const std::string want_real =
                sizeof(VVM::Real) == 8 ? "float64" : "float32";
            check(field_precision && field_precision.Data().at(0) == want_field,
                  "vvm_field_precision records the on-disk field type");
            check(real_precision && real_precision.Data().at(0) == want_real,
                  "vvm_real_precision still records the model's own precision");
        }
        reader.EndStep();
        ++observed_steps;
    }
    reader.Close();
    check(observed_steps == kSteps, "three BP5 steps were readable");
}

} // namespace

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &g_rank);
    int comm_size = 1;
    MPI_Comm_size(MPI_COMM_WORLD, &comm_size);
    if (argc != 6 && argc != 7) {
        if (g_rank == 0) {
            std::fprintf(stderr,
                         "usage: test_bp5_history_mpi BASE_CONFIG WORK_ROOT direct|pack "
                         "sync|async EXPECTED_RANKS [native|float32|float64]\n");
        }
        MPI_Finalize();
        return 2;
    }
    const int expected_ranks = std::stoi(argv[5]);
    if (comm_size != expected_ranks) {
        if (g_rank == 0) {
            std::fprintf(stderr,
                         "MPI launcher mismatch: expected %d ranks, MPI_COMM_WORLD has %d. "
                         "Use the mpiexec from the same MPI installation as the compiler.\n",
                         expected_ranks, comm_size);
        }
        MPI_Finalize();
        return 2;
    }

    const std::string mode = argv[3];
    const bool async_write = std::string(argv[4]) == "async";
    // Omitted precision means 'native', which is the pre-existing behaviour, so
    // the tests registered before this option existed keep their exact meaning.
    const std::string precision = (argc == 7) ? argv[6] : "native";
    const bool empty_rank_case = comm_size >= 4;
    const std::string tag = mode + "_" + argv[4] + "_" + precision + "_r" +
                            std::to_string(comm_size);
    const std::filesystem::path case_dir =
        std::filesystem::path(argv[2]) / ("bp5_history_" + tag);
    const std::filesystem::path config_path = case_dir.parent_path() / ("bp5_" + tag + ".json");

    try {
        if (g_rank == 0) {
            std::filesystem::create_directories(case_dir.parent_path());
            const auto json = make_config(argv[1], case_dir, mode, async_write,
                                          precision, empty_rank_case, comm_size);
            std::ofstream output(config_path);
            output << json.dump(2) << '\n';
        }
        MPI_Barrier(MPI_COMM_WORLD);

        Kokkos::initialize(Kokkos::InitializationSettings().set_device_id(0));
        {
            VVM::Utils::ConfigurationManager config(config_path.string());
            VVM::Core::Grid grid(config, MPI_COMM_WORLD);
            VVM::Core::Parameters parameters(config, grid);
            fill_coordinates(parameters, grid);
#if defined(ENABLE_NCCL)
            ncclUniqueId id;
            if (g_rank == 0) ncclGetUniqueId(&id);
            MPI_Bcast(&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD);
            ncclComm_t nccl_comm;
            ncclCommInitRank(&nccl_comm, comm_size, id, g_rank);
            cudaStream_t stream = Kokkos::Cuda().cuda_stream();
            VVM::Core::State state(
                config, parameters, grid, nccl_comm, stream);
#else
            VVM::Core::State state(config, parameters, grid);
#endif
            state.add_field<4>(
                "bp5_test_4d",
                {2, grid.get_local_total_points_z(), grid.get_local_total_points_y(),
                 grid.get_local_total_points_x()},
                VVM::Core::FieldMetadata{VVM::Core::GridStaggering::Centered,
                                         "test_unit", "BP5 four-dimensional test field",
                                         "bp5_test_4d", "integration test"});
            VVM::IO::BP5::Bp5HistoryWriter writer(
                config, grid, parameters, state, MPI_COMM_WORLD);
            for (int step = 0; step < kSteps; ++step) {
                fill_fields(state, grid, step);
                writer.write(10 + step, VVM::real(step * 2.5));
            }
            writer.close();
            MPI_Barrier(MPI_COMM_WORLD);
            if (g_rank == 0) {
                const int x_end = empty_rank_case ? 1 : 6;
                const int y_end = empty_rank_case ? 1 : 5;
                const bool on_disk_is_float32 =
                    (precision == "float32") ||
                    (precision == "native" && sizeof(VVM::Real) == sizeof(float));
                if (on_disk_is_float32) {
                    read_and_check<float>(case_dir / "history.bp", x_end, y_end);
                } else {
                    read_and_check<double>(case_dir / "history.bp", x_end, y_end);
                }
            }
            MPI_Barrier(MPI_COMM_WORLD);
#if defined(ENABLE_NCCL)
            ncclCommDestroy(nccl_comm);
#endif
        }
        Kokkos::finalize();
    } catch (const std::exception& e) {
        std::fprintf(stderr, "[rank %d] exception: %s\n", g_rank, e.what());
        ++g_failures;
        if (Kokkos::is_initialized()) Kokkos::finalize();
    }

    int global_failures = 0;
    MPI_Allreduce(&g_failures, &global_failures, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    if (g_rank == 0 && global_failures == 0) {
        std::printf("test_bp5_history_mpi (%s): PASS\n", tag.c_str());
    }
    MPI_Finalize();
    return global_failures == 0 ? 0 : 1;
}
