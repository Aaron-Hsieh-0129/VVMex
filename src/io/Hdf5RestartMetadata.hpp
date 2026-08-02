#ifndef VVM_IO_HDF5_RESTART_METADATA_HPP
#define VVM_IO_HDF5_RESTART_METADATA_HPP

#include "utils/RestartTime.hpp"

#include <hdf5.h>
#include <string>
#include <vector>

namespace VVM {
namespace IO {

// Reading the restart clock out of an HDF5 output file.
inline hid_t open_restart_scalar_dataset(hid_t file_id, const std::string& name) {
    hid_t dataset = -1;
    H5E_BEGIN_TRY {
        dataset = H5Dopen2(file_id, ("/Step0/" + name).c_str(), H5P_DEFAULT);
    } H5E_END_TRY;
    if (dataset >= 0) return dataset;

    H5E_BEGIN_TRY {
        dataset = H5Dopen2(file_id, ("/" + name).c_str(), H5P_DEFAULT);
    } H5E_END_TRY;
    return dataset;
}

// Read a one-element dataset into `out`, converting to mem_type. A scalar
// dataspace and a length-1 array are both accepted; anything else is not a
// restart scalar and is reported as absent.
inline bool read_restart_scalar(hid_t file_id, const std::string& name,
                                hid_t mem_type, void* out) {
    hid_t dataset = open_restart_scalar_dataset(file_id, name);
    if (dataset < 0) return false;

    bool ok = false;
    hid_t space = H5Dget_space(dataset);
    if (space >= 0) {
        if (H5Sget_simple_extent_npoints(space) == 1) {
            H5E_BEGIN_TRY {
                ok = H5Dread(dataset, mem_type, H5S_ALL, H5S_ALL, H5P_DEFAULT, out) >= 0;
            } H5E_END_TRY;
        }
        H5Sclose(space);
    }
    H5Dclose(dataset);
    return ok;
}

inline bool read_restart_scalar_double(hid_t file_id, const std::string& name, double& out) {
    return read_restart_scalar(file_id, name, H5T_NATIVE_DOUBLE, &out);
}

inline bool read_restart_scalar_int64(hid_t file_id, const std::string& name, long long& out) {
    return read_restart_scalar(file_id, name, H5T_NATIVE_LLONG, &out);
}

// Names searched for the elapsed-seconds scalar, most specific first.
inline const std::vector<std::string>& hdf5_restart_time_variable_names() {
    static const std::vector<std::string> names = {"model_time_s", "time"};
    return names;
}

inline const std::vector<std::string>& hdf5_restart_step_variable_names() {
    static const std::vector<std::string> names = {"model_step"};
    return names;
}

// Collect whatever restart metadata an already-open file carries. Never throws
// and never guesses: a file with none of these datasets simply reports nothing.
inline VVM::Utils::RestartFileMetadata read_hdf5_restart_metadata(hid_t file_id) {
    VVM::Utils::RestartFileMetadata metadata;

    for (const auto& name : hdf5_restart_time_variable_names()) {
        double value = 0.0;
        if (read_restart_scalar_double(file_id, name, value)) {
            metadata.has_time = true;
            metadata.time_s = value;
            break;
        }
    }

    for (const auto& name : hdf5_restart_step_variable_names()) {
        long long value = 0;
        if (read_restart_scalar_int64(file_id, name, value)) {
            metadata.has_step = true;
            metadata.step = value;
            break;
        }
    }

    return metadata;
}

inline bool read_hdf5_restart_metadata(const std::string& path,
                                       VVM::Utils::RestartFileMetadata& metadata) {
    hid_t file_id = -1;
    H5E_BEGIN_TRY {
        file_id = H5Fopen(path.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
    } H5E_END_TRY;
    if (file_id < 0) return false;

    metadata = read_hdf5_restart_metadata(file_id);
    H5Fclose(file_id);
    return true;
}

} // namespace IO
} // namespace VVM

#endif // VVM_IO_HDF5_RESTART_METADATA_HPP
