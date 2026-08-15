// The same run written by every engine must agree bit-for-bit, in values and in
// metadata. BP5 is compared against each HDF5 tree (the HDF5 engine's own, and
// the one an SST run relays through the I/O server), which also settles the two
// HDF5 trees against each other; their attributes are then compared directly,
// because that is the part that can drift without any value changing.
#include "core/vvm_types.hpp"

#include <adios2.h>
#include <hdf5.h>

#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <map>
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
std::filesystem::path step_file(const std::string& directory, int step) {
    char filename[64];
    std::snprintf(filename, sizeof(filename), "history_%06d.h5", step);
    return std::filesystem::path(directory) / filename;
}

hid_t open_step(const std::string& directory, int step) {
    const auto path = step_file(directory, step);
    const hid_t file = H5Fopen(path.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
    if (file < 0) throw std::runtime_error("Cannot open " + path.string());
    return file;
}

herr_t collect_attribute(hid_t, const char* name, const H5A_info_t*, void* data) {
    static_cast<std::vector<std::string>*>(data)->emplace_back(name);
    return 0;
}

// Every attribute on the file root, which is where ADIOS2 records the flat
// "variable/attribute" metadata a reader looks up.
std::map<std::string, std::string> root_attributes(hid_t file) {
    std::vector<std::string> names;
    hsize_t index = 0;
    H5Aiterate2(file, H5_INDEX_NAME, H5_ITER_INC, &index, collect_attribute, &names);

    std::map<std::string, std::string> attributes;
    for (const auto& name : names) {
        const hid_t attribute = H5Aopen(file, name.c_str(), H5P_DEFAULT);
        if (attribute < 0) continue;
        const hid_t type = H5Aget_type(attribute);
        if (H5Tget_class(type) == H5T_STRING) {
            std::vector<char> value(H5Tget_size(type) + 1, '\0');
            if (H5Aread(attribute, type, value.data()) >= 0) {
                attributes[name] = value.data();
            }
        }
        H5Tclose(type);
        H5Aclose(attribute);
    }
    return attributes;
}

void compare_bp5_against_hdf5(const std::string& bp5_dataset,
                              const std::string& hdf5_directory,
                              const std::string& label,
                              int& steps_compared) {
    adios2::ADIOS adios;
    auto bp_io = adios.DeclareIO("VVM_COMPAT_READER_" + label);
    auto bp_reader = bp_io.Open(bp5_dataset, adios2::Mode::Read);
    int step = 0;
    while (bp_reader.BeginStep() == adios2::StepStatus::OK) {
        const hid_t h5_file = open_step(hdf5_directory, step);

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
    steps_compared = step;
}

// Values are settled by comparing both trees against BP5; this catches the
// metadata an engine can silently omit while every number still matches.
void compare_hdf5_metadata(const std::string& a, const std::string& b, int steps) {
    for (int step = 0; step < steps; ++step) {
        const hid_t file_a = open_step(a, step);
        const hid_t file_b = open_step(b, step);
        const auto attributes_a = root_attributes(file_a);
        const auto attributes_b = root_attributes(file_b);
        check(attributes_a == attributes_b,
              "step " + std::to_string(step) +
                  ": HDF5 and SST files carry identical attributes");
        if (attributes_a != attributes_b) {
            for (const auto& entry : attributes_a) {
                const auto found = attributes_b.find(entry.first);
                if (found == attributes_b.end()) {
                    std::fprintf(stderr, "  only in %s: %s\n", a.c_str(),
                                 entry.first.c_str());
                } else if (found->second != entry.second) {
                    std::fprintf(stderr, "  %s differs: '%s' vs '%s'\n",
                                 entry.first.c_str(), entry.second.c_str(),
                                 found->second.c_str());
                }
            }
            for (const auto& entry : attributes_b) {
                if (attributes_a.count(entry.first) == 0) {
                    std::fprintf(stderr, "  only in %s: %s\n", b.c_str(),
                                 entry.first.c_str());
                }
            }
        }
        H5Fclose(file_a);
        H5Fclose(file_b);
    }
}
} // namespace

int main(int argc, char** argv) {
    if (argc != 3 && argc != 4) {
        std::fprintf(stderr,
                     "usage: test_engine_output_compat BP5_DATASET HDF5_DIRECTORY "
                     "[SST_HDF5_DIRECTORY]\n");
        return 2;
    }
    try {
        int hdf5_steps = 0;
        compare_bp5_against_hdf5(argv[1], argv[2], "hdf5", hdf5_steps);
        check(hdf5_steps == 11, "all eleven history times were compared against HDF5");

        if (argc == 4) {
            int sst_steps = 0;
            compare_bp5_against_hdf5(argv[1], argv[3], "sst", sst_steps);
            check(sst_steps == hdf5_steps,
                  "the SST relay wrote the same number of history times");
            compare_hdf5_metadata(argv[2], argv[3], std::min(hdf5_steps, sst_steps));
        }
    } catch (const std::exception& e) {
        std::fprintf(stderr, "exception: %s\n", e.what());
        ++failures;
    }
    if (failures == 0) std::puts("test_engine_output_compat: PASS");
    return failures == 0 ? 0 : 1;
}
