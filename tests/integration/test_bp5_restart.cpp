// A run restarted from a BP5 dataset must come up holding exactly what that
// step of the dataset holds. The resumed run is configured to take no further
// steps, so its initial output is the loaded state itself: comparing that file
// against the source step checks the reader's field selection, its per-rank
// slabs, and the clock it recovers.
#include "core/vvm_types.hpp"

#include <adios2.h>
#include <hdf5.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <numeric>
#include <stdexcept>
#include <string>
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

hid_t open_hdf5_dataset(hid_t file, const std::string& name) {
    const std::string path = "/Step0/" + name;
    hid_t dataset = -1;
    H5E_BEGIN_TRY { dataset = H5Dopen2(file, path.c_str(), H5P_DEFAULT); } H5E_END_TRY;
    if (dataset < 0) throw std::runtime_error("Cannot open HDF5 dataset " + path);
    return dataset;
}

template <typename T>
hid_t hdf5_type() {
    if constexpr (std::is_same_v<T, double>) return H5T_NATIVE_DOUBLE;
    else if constexpr (std::is_same_v<T, float>) return H5T_NATIVE_FLOAT;
    else return H5T_NATIVE_INT64;
}

template <typename T>
std::vector<T> read_hdf5(hid_t file, const std::string& name, adios2::Dims& shape) {
    const hid_t dataset = open_hdf5_dataset(file, name);
    const hid_t space = H5Dget_space(dataset);
    const int rank = H5Sget_simple_extent_ndims(space);
    std::vector<hsize_t> dims(static_cast<std::size_t>(std::max(rank, 0)));
    if (rank > 0) H5Sget_simple_extent_dims(space, dims.data(), nullptr);
    shape.assign(dims.begin(), dims.end());
    std::vector<T> values(rank > 0 ? elements(shape) : 1);
    const herr_t status =
        H5Dread(dataset, hdf5_type<T>(), H5S_ALL, H5S_ALL, H5P_DEFAULT, values.data());
    H5Sclose(space);
    H5Dclose(dataset);
    if (status < 0) throw std::runtime_error("Cannot read HDF5 dataset " + name);
    return values;
}

// The dataset may hold float32 fields while the model is float64. Comparison is
// done in the file's own type, so a narrowed history is compared as written.
template <typename T>
void compare_field(adios2::IO& io, adios2::Engine& reader, std::size_t step,
                   hid_t h5_file, const std::string& name) {
    auto variable = io.InquireVariable<T>(name);
    check(static_cast<bool>(variable), name + " exists in the BP5 dataset");
    if (!variable) return;

    variable.SetStepSelection({step, 1});
    std::vector<T> bp_values(elements(variable.Shape()));
    reader.Get(variable, bp_values.data(), adios2::Mode::Sync);

    adios2::Dims h5_shape;
    const std::vector<T> h5_values = read_hdf5<T>(h5_file, name, h5_shape);
    check(h5_shape == variable.Shape(), name + " has the same shape after restart");
    if (h5_shape != variable.Shape()) return;
    check(bp_values == h5_values, name + " is bit-for-bit identical after restart");
}

void compare_any_field(adios2::IO& io, adios2::Engine& reader, std::size_t step,
                       hid_t h5_file, const std::string& name) {
    const std::string type = io.VariableType(name);
    if (type == "float") compare_field<float>(io, reader, step, h5_file, name);
    else if (type == "double") compare_field<double>(io, reader, step, h5_file, name);
    else check(false, name + " has a comparable float type in the BP5 dataset (got '" + type + "')");
}
} // namespace

int main(int argc, char** argv) {
    if (argc < 4) {
        std::fprintf(stderr,
                     "usage: test_bp5_restart BP5_DATASET STEP RESUMED_HDF5 [field ...]\n");
        return 2;
    }
    const std::string dataset_path = argv[1];
    const auto step = static_cast<std::size_t>(std::stoul(argv[2]));
    const std::string h5_path = argv[3];

    try {
        adios2::ADIOS adios;
        auto io = adios.DeclareIO("VVM_BP5_RESTART_CHECK");
        io.SetEngine("BP5");
        auto reader = io.Open(dataset_path, adios2::Mode::ReadRandomAccess);

        const hid_t h5_file = H5Fopen(h5_path.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
        if (h5_file < 0) throw std::runtime_error("Cannot open " + h5_path);

        // The resumed run must come up on the source step's clock.
        auto model_time = io.InquireVariable<double>("model_time_s");
        auto model_step = io.InquireVariable<std::int64_t>("model_step");
        check(static_cast<bool>(model_time) && static_cast<bool>(model_step),
              "the dataset stores model_time_s and model_step");
        if (model_time && model_step) {
            model_time.SetStepSelection({step, 1});
            model_step.SetStepSelection({step, 1});
            double bp_time = 0.0;
            std::int64_t bp_step = 0;
            reader.Get(model_time, &bp_time, adios2::Mode::Sync);
            reader.Get(model_step, &bp_step, adios2::Mode::Sync);

            adios2::Dims shape;
            const double h5_time = read_hdf5<double>(h5_file, "model_time_s", shape).at(0);
            const std::int64_t h5_step =
                read_hdf5<std::int64_t>(h5_file, "model_step", shape).at(0);
            check(bp_time == h5_time,
                  "recovered model_time_s matches the source step (" +
                      std::to_string(h5_time) + " vs " + std::to_string(bp_time) + ")");
            check(bp_step == h5_step,
                  "recovered model_step matches the source step (" +
                      std::to_string(h5_step) + " vs " + std::to_string(bp_step) + ")");
        }

        for (int i = 4; i < argc; ++i) {
            compare_any_field(io, reader, step, h5_file, argv[i]);
        }

        H5Fclose(h5_file);
        reader.Close();
    } catch (const std::exception& e) {
        std::fprintf(stderr, "exception: %s\n", e.what());
        ++failures;
    }

    if (failures == 0) std::puts("test_bp5_restart: PASS");
    return failures == 0 ? 0 : 1;
}
