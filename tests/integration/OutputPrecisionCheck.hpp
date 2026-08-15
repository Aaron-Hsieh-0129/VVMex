// Shared by test_hdf5_precision and test_sst_precision: build a small case with
// a chosen output.precision, fill known values, and check what the resulting
// HDF5 file actually contains.
#ifndef VVM_TESTS_OUTPUT_PRECISION_CHECK_HPP
#define VVM_TESTS_OUTPUT_PRECISION_CHECK_HPP

#include "core/Grid.hpp"
#include "core/Parameters.hpp"
#include "core/State.hpp"

#include <Kokkos_Core.hpp>
#include <hdf5.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "../../externals/json/json.hpp"

namespace VVMTest {

constexpr int kNx = 8;
constexpr int kNy = 6;
constexpr int kNz = 5;

inline int failures = 0;

inline void check(bool condition, const std::string& message) {
    if (condition) return;
    ++failures;
    std::fprintf(stderr, "FAIL: %s\n", message.c_str());
}

// Deliberately not representable in float32, so a narrowed file has to differ
// from the model value while staying close to it.
inline double expected_3d(int k, int j, int i) {
    return 300.0 + 0.0001 * (10000 * k + 100 * j + i);
}

inline void fill_fields(VVM::Core::State& state, const VVM::Core::Grid& grid) {
    const int h = grid.get_halo_cells();
    const int nx = grid.get_local_total_points_x();
    const int ny = grid.get_local_total_points_y();
    const int nz = grid.get_local_total_points_z();
    const int i0 = grid.get_local_physical_start_x();
    const int j0 = grid.get_local_physical_start_y();

    auto thbar = state.get_field<1>("thbar").get_mutable_device_data();
    auto topo = state.get_field<2>("topo").get_mutable_device_data();
    auto u = state.get_field<3>("u").get_mutable_device_data();

    Kokkos::parallel_for(
        "fill_1d", Kokkos::RangePolicy<>(h, nz - h),
        KOKKOS_LAMBDA(const int k) { thbar(k) = VVM::real(k - h); });
    Kokkos::parallel_for(
        "fill_2d", Kokkos::MDRangePolicy<Kokkos::Rank<2>>({h, h}, {ny - h, nx - h}),
        KOKKOS_LAMBDA(const int j, const int i) {
            topo(j, i) = VVM::real(100 * (j0 + j - h) + (i0 + i - h));
        });
    Kokkos::parallel_for(
        "fill_3d",
        Kokkos::MDRangePolicy<Kokkos::Rank<3>>({h, h, h}, {nz - h, ny - h, nx - h}),
        KOKKOS_LAMBDA(const int k, const int j, const int i) {
            u(k, j, i) = VVM::real(300.0) +
                         VVM::real(0.0001) * VVM::real(10000 * (k - h) +
                                                       100 * (j0 + j - h) +
                                                       (i0 + i - h));
        });
    Kokkos::fence("fill_complete");
}

inline void fill_coordinates(VVM::Core::Parameters& parameters, const VVM::Core::Grid& grid) {
    const int h = grid.get_halo_cells();
    const int nz = grid.get_local_total_points_z();
    auto z = parameters.z_mid.get_mutable_device_data();
    Kokkos::parallel_for(
        "fill_z", Kokkos::RangePolicy<>(h, nz - h),
        KOKKOS_LAMBDA(const int k) { z(k) = VVM::real(50 + 25 * (k - h)); });
    Kokkos::fence("coordinates_complete");
}

inline nlohmann::json make_config(
    const std::filesystem::path& base_config,
    const std::filesystem::path& output_dir,
    const std::string& engine,
    const std::string& precision) {
    std::ifstream stream(base_config);
    if (!stream) throw std::runtime_error("Cannot read base test configuration.");
    nlohmann::json config;
    stream >> config;
    config["grid"]["nx"] = kNx;
    config["grid"]["ny"] = kNy;
    config["grid"]["nz"] = kNz;
    config["grid"]["n_halo_cells"] = 2;
    config["output"]["engine"] = engine;
    config["output"]["output_dir"] = output_dir.string();
    config["output"]["output_filename_prefix"] = "history";
    config["output"]["fields_to_output"] =
        nlohmann::json::array({"thbar", "topo", "u"});
    config["output"]["output_grid"] = {
        {"x_start", 0}, {"x_end", kNx - 1},
        {"y_start", 0}, {"y_end", kNy - 1},
        {"z_start", 0}, {"z_end", kNz - 1}};
    // "unset" leaves the key out entirely, which is the pre-existing behaviour.
    if (precision != "unset") config["output"]["precision"] = precision;
    return config;
}

inline std::size_t dataset_element_size(hid_t file, const std::string& path) {
    hid_t dataset = H5Dopen2(file, path.c_str(), H5P_DEFAULT);
    if (dataset < 0) return 0;
    hid_t type = H5Dget_type(dataset);
    const std::size_t size = H5Tget_size(type);
    H5Tclose(type);
    H5Dclose(dataset);
    return size;
}

inline std::vector<double> read_dataset(hid_t file, const std::string& path) {
    std::vector<double> values;
    hid_t dataset = H5Dopen2(file, path.c_str(), H5P_DEFAULT);
    if (dataset < 0) return values;
    hid_t space = H5Dget_space(dataset);
    const hssize_t elements = H5Sget_simple_extent_npoints(space);
    if (elements > 0) {
        values.resize(static_cast<std::size_t>(elements));
        // HDF5 converts on read, so the buffer type need not match the file --
        // which is also why a restart from a narrowed file still loads.
        H5Dread(dataset, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, values.data());
    }
    H5Sclose(space);
    H5Dclose(dataset);
    return values;
}

inline bool narrowed_on_disk(const std::string& precision) {
    if (precision == "float32") return true;
    if (precision == "float64") return false;
    return sizeof(VVM::Real) == sizeof(float);
}

inline void inspect(const std::filesystem::path& file_path, const std::string& precision) {
    const std::size_t want_field = narrowed_on_disk(precision) ? 4 : 8;

    hid_t file = H5Fopen(file_path.string().c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
    check(file >= 0, "the HDF5 output file opens: " + file_path.string());
    if (file < 0) return;

    for (const char* name : {"u", "topo", "thbar"}) {
        const std::size_t size = dataset_element_size(file, std::string("/Step0/") + name);
        check(size == want_field,
              std::string("field '") + name + "' is " + std::to_string(want_field) +
                  " bytes on disk, found " + std::to_string(size));
    }

    // Clocks and coordinates are kilobytes against gigabytes of field data, so
    // they keep VVM::Real whatever the fields do.
    for (const char* name : {"time", "coordinates/x", "coordinates/z_mid"}) {
        const std::size_t size = dataset_element_size(file, std::string("/Step0/") + name);
        check(size == sizeof(VVM::Real),
              std::string("'") + name + "' keeps VVM::Real, found " + std::to_string(size));
    }
    check(dataset_element_size(file, "/Step0/model_step") == sizeof(std::int64_t),
          "model_step stays an exact int64");

    const auto u = read_dataset(file, "/Step0/u");
    check(u.size() == static_cast<std::size_t>(kNx) * kNy * kNz,
          "the 3-D field has the full global shape");
    // float32 keeps ~7 significant digits, so values near 300 land within 1e-4.
    const double tolerance = narrowed_on_disk(precision) ? 1e-4 : 1e-12;
    bool values_match = u.size() == static_cast<std::size_t>(kNx) * kNy * kNz;
    for (int k = 0; k < kNz && values_match; ++k) {
        for (int j = 0; j < kNy && values_match; ++j) {
            for (int i = 0; i < kNx; ++i) {
                const std::size_t index =
                    (static_cast<std::size_t>(k) * kNy + j) * kNx + i;
                if (std::fabs(u[index] - expected_3d(k, j, i)) > tolerance) {
                    std::fprintf(stderr, "  u(%d,%d,%d): got %.9g want %.9g\n",
                                 k, j, i, u[index], expected_3d(k, j, i));
                    values_match = false;
                    break;
                }
            }
        }
    }
    check(values_match, "field values survive the conversion");

    H5Fclose(file);
}

} // namespace VVMTest

#endif
