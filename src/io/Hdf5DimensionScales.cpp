#include "Hdf5DimensionScales.hpp"

#include <hdf5_hl.h>
#include <array>

namespace VVM {
namespace IO {

namespace {
void mark_dimension_scale(hid_t file, const char* dataset_path, const char* scale_name) {
    hid_t scale = H5Dopen2(file, dataset_path, H5P_DEFAULT);
    if (scale < 0) return;

    if (H5DSis_scale(scale) <= 0) {
        H5DSset_scale(scale, scale_name);
    }
    H5Dclose(scale);
}

hid_t open_dimension_scale(hid_t file, const char* source_path, const char* scale_path, const char* scale_name) {
    if (H5Lexists(file, scale_path, H5P_DEFAULT) <= 0) {
        if (H5Ocopy(file, source_path, file,
                    scale_path, H5P_DEFAULT, H5P_DEFAULT) < 0) {
            return -1;
        }
    }

    hid_t scale = H5Dopen2(file, scale_path, H5P_DEFAULT);
    if (scale >= 0 && H5DSis_scale(scale) <= 0) {
        H5DSset_scale(scale, scale_name);
    }
    return scale;
}

hid_t create_component_scale(hid_t file, hsize_t size) {
    constexpr const char* scale_path = "/Step0/component";
    hid_t scale = -1;

    if (H5Lexists(file, scale_path, H5P_DEFAULT) > 0) {
        scale = H5Dopen2(file, scale_path, H5P_DEFAULT);
    } 
    else {
        hid_t dataspace = H5Screate_simple(1, &size, nullptr);
        scale = H5Dcreate2(file, scale_path, H5T_STD_U64LE, dataspace,
                           H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

        if (scale >= 0) {
            std::vector<unsigned long long> values(size);
            for (hsize_t index = 0; index < size; ++index) {
                values[index] = static_cast<unsigned long long>(index);
            }
            H5Dwrite(scale, H5T_NATIVE_ULLONG, H5S_ALL,
                     H5S_ALL, H5P_DEFAULT, values.data());
        }

        H5Sclose(dataspace);
    }

    if (scale >= 0 && H5DSis_scale(scale) <= 0) {
        H5DSset_scale(scale, "component");
    }
    return scale;
}

void attach_scale(hid_t dataset, hid_t scale, unsigned int dimension, const char* label) {
    if (scale < 0) return;

    if (H5DSis_attached(dataset, scale, dimension) <= 0) {
        H5DSattach_scale(dataset, scale, dimension);
    }
    H5DSset_label(dataset, dimension, label);
}
} // namespace

void attach_hdf5_dimension_scales(hid_t file, const std::vector<std::string>& field_names) {
    const std::array<hid_t, 3> scales = {
        open_dimension_scale(file, "/Step0/coordinates/x", "/Step0/x", "x"),
        open_dimension_scale(file, "/Step0/coordinates/y", "/Step0/y", "y"),
        open_dimension_scale(file, "/Step0/coordinates/z_mid", "/Step0/z", "z")
    };

    mark_dimension_scale(file, "/Step0/coordinates/x", "x");
    mark_dimension_scale(file, "/Step0/coordinates/y", "y");
    mark_dimension_scale(file, "/Step0/coordinates/z_mid", "z_mid");

    hid_t component_scale = -1;
    hsize_t component_size = 0;

    for (const auto& field_name : field_names) {
        const std::string dataset_path = "/Step0/" + field_name;
        hid_t dataset = H5Dopen2(file, dataset_path.c_str(), H5P_DEFAULT);
        if (dataset < 0) continue;

        hid_t dataspace = H5Dget_space(dataset);
        const int rank = H5Sget_simple_extent_ndims(dataspace);

        if (rank == 1) {
            attach_scale(dataset, scales[2], 0, "z");
        } 
        else if (rank == 2) {
            attach_scale(dataset, scales[1], 0, "y");
            attach_scale(dataset, scales[0], 1, "x");
        } 
        else if (rank == 3) {
            attach_scale(dataset, scales[2], 0, "z");
            attach_scale(dataset, scales[1], 1, "y");
            attach_scale(dataset, scales[0], 2, "x");
        } 
        else if (rank == 4) {
            hsize_t dimensions[4];
            H5Sget_simple_extent_dims(dataspace, dimensions, nullptr);
            if (component_scale < 0) {
                component_scale = create_component_scale(file, dimensions[0]);
                component_size = dimensions[0];
            }
            if (component_size == dimensions[0]) {
                attach_scale(dataset, component_scale, 0, "component");
            }
            attach_scale(dataset, scales[2], 1, "z");
            attach_scale(dataset, scales[1], 2, "y");
            attach_scale(dataset, scales[0], 3, "x");
        }

        H5Sclose(dataspace);
        H5Dclose(dataset);
    }

    for (const auto scale : scales) {
        if (scale >= 0) H5Dclose(scale);
    }
    if (component_scale >= 0) H5Dclose(component_scale);
}

} // namespace IO
} // namespace VVM
