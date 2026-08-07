#include "core/vvm_types.hpp"

#include <adios2.h>
#include <hdf5.h>
#include <mpi.h>

#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <numeric>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace {
int failures = 0;

void check(bool condition, const std::string& message) {
    if (condition) return;
    ++failures;
    std::fprintf(stderr, "FAIL: %s\n", message.c_str());
}

std::size_t elements(const adios2::Dims& shape) {
    return std::accumulate(shape.begin(), shape.end(), std::size_t{1},
                           std::multiplies<std::size_t>());
}

template <typename T>
hid_t hdf5_type() {
    if constexpr (std::is_same_v<T, double>) {
        return H5T_NATIVE_DOUBLE;
    } else if constexpr (std::is_same_v<T, float>) {
        return H5T_NATIVE_FLOAT;
    } else if constexpr (std::is_same_v<T, std::int64_t>) {
        return H5T_NATIVE_INT64;
    } else {
        static_assert(!sizeof(T), "unsupported HDF5 comparison type");
    }
}

template <typename T>
T read_hdf5_scalar(hid_t file, const std::string& name) {
    const std::string path = "/Step0/" + name;
    const hid_t dataset = H5Dopen2(file, path.c_str(), H5P_DEFAULT);
    if (dataset < 0) throw std::runtime_error("Cannot open HDF5 dataset " + path);
    T value{};
    const herr_t result = H5Dread(dataset, hdf5_type<T>(), H5S_ALL, H5S_ALL,
                                  H5P_DEFAULT, &value);
    H5Dclose(dataset);
    if (result < 0) throw std::runtime_error("Cannot read HDF5 dataset " + path);
    return value;
}

template <typename T>
std::pair<adios2::Dims, std::vector<T>> read_hdf5_array(
    hid_t file, const std::string& name) {
    const std::string path = "/Step0/" + name;
    const hid_t dataset = H5Dopen2(file, path.c_str(), H5P_DEFAULT);
    if (dataset < 0) throw std::runtime_error("Cannot open HDF5 dataset " + path);
    const hid_t space = H5Dget_space(dataset);
    const int rank = H5Sget_simple_extent_ndims(space);
    std::vector<hsize_t> h5_shape(static_cast<std::size_t>(rank));
    H5Sget_simple_extent_dims(space, h5_shape.data(), nullptr);
    adios2::Dims shape(h5_shape.begin(), h5_shape.end());
    std::vector<T> values(elements(shape));
    const herr_t result = H5Dread(dataset, hdf5_type<T>(), H5S_ALL, H5S_ALL,
                                  H5P_DEFAULT, values.data());
    H5Sclose(space);
    H5Dclose(dataset);
    if (result < 0) throw std::runtime_error("Cannot read HDF5 dataset " + path);
    return {shape, values};
}

std::string read_hdf5_attribute(hid_t file, const std::string& name) {
    const hid_t attribute = H5Aopen(file, name.c_str(), H5P_DEFAULT);
    if (attribute < 0) return {};
    const hid_t type = H5Aget_type(attribute);
    std::vector<char> value(H5Tget_size(type) + 1, '\0');
    const herr_t result = H5Aread(attribute, type, value.data());
    H5Tclose(type);
    H5Aclose(attribute);
    if (result < 0) throw std::runtime_error("Cannot read HDF5 attribute " + name);
    return value.data();
}

template <typename T>
void compare_scalar(adios2::IO& bp_io, adios2::Engine& bp_reader,
                    hid_t h5_file, const std::string& name) {
    auto bp_var = bp_io.InquireVariable<T>(name);
    check(static_cast<bool>(bp_var), name + " exists in BP5");
    if (!bp_var) return;
    T bp_value{};
    bp_reader.Get(bp_var, bp_value, adios2::Mode::Sync);
    const T h5_value = read_hdf5_scalar<T>(h5_file, name);
    check(bp_value == h5_value, name + " scalar value is identical");
}

template <typename T>
void compare_array(adios2::IO& bp_io, adios2::Engine& bp_reader,
                   hid_t h5_file, const std::string& name) {
    auto bp_var = bp_io.InquireVariable<T>(name);
    check(static_cast<bool>(bp_var), name + " exists in BP5");
    if (!bp_var) return;
    const auto bp_shape = bp_var.Shape();
    const auto [h5_shape, h5_values] = read_hdf5_array<T>(h5_file, name);
    check(bp_shape == h5_shape, name + " global shape is identical");
    if (bp_shape != h5_shape) return;
    std::vector<T> bp_values(elements(bp_shape));
    bp_reader.Get(bp_var, bp_values.data(), adios2::Mode::Sync);
    check(bp_values == h5_values, name + " values are bit-for-bit identical");
}

void compare_attribute(adios2::IO& bp_io, hid_t h5_file,
                       const std::string& variable, const std::string& name) {
    const auto bp_attr = bp_io.InquireAttribute<std::string>(name, variable);
    const std::string h5_attr = read_hdf5_attribute(h5_file, variable + "/" + name);
    check(bp_attr && !h5_attr.empty(), variable + "/" + name + " exists in both formats");
    if (bp_attr && !h5_attr.empty()) {
        check(bp_attr.Data().at(0) == h5_attr,
              variable + "/" + name + " metadata is identical");
    }
}
} // namespace

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    if (argc != 3) {
        std::fprintf(stderr, "usage: test_bp5_hdf5_compat BP5_DATASET HDF5_DIRECTORY\n");
        MPI_Finalize();
        return 2;
    }
    try {
        adios2::ADIOS adios(MPI_COMM_SELF);
        auto bp_io = adios.DeclareIO("VVM_BP5_COMPAT_READER");
        auto bp_reader = bp_io.Open(argv[1], adios2::Mode::Read, MPI_COMM_SELF);
        int step = 0;
        while (bp_reader.BeginStep() == adios2::StepStatus::OK) {
            char filename[64];
            std::snprintf(filename, sizeof(filename), "history_%06d.h5", step);
            const auto h5_path = std::filesystem::path(argv[2]) / filename;
            const hid_t h5_file = H5Fopen(h5_path.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
            if (h5_file < 0) throw std::runtime_error("Cannot open " + h5_path.string());

            compare_scalar<VVM::Real>(bp_io, bp_reader, h5_file, "time");
            compare_scalar<double>(bp_io, bp_reader, h5_file, "model_time_s");
            compare_scalar<std::int64_t>(bp_io, bp_reader, h5_file, "model_step");
            for (const std::string name : {
                     "coordinates/x", "coordinates/y", "coordinates/z_mid",
                     "thbar", "topo", "u"}) {
                compare_array<VVM::Real>(bp_io, bp_reader, h5_file, name);
            }
            for (const std::string attribute : {"units", "long_name", "grid_staggering"}) {
                compare_attribute(bp_io, h5_file, "u", attribute);
            }

            H5Fclose(h5_file);
            bp_reader.EndStep();
            ++step;
        }
        bp_reader.Close();
        check(step == 11, "all eleven history times were compared");
    } catch (const std::exception& e) {
        std::fprintf(stderr, "exception: %s\n", e.what());
        ++failures;
    }
    if (failures == 0) std::puts("test_bp5_hdf5_compat: PASS");
    MPI_Finalize();
    return failures == 0 ? 0 : 1;
}
