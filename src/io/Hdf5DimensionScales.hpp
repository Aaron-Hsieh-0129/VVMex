#pragma once

#include <hdf5.h>
#include <string>
#include <vector>

namespace VVM {
namespace IO {

void attach_hdf5_dimension_scales(
    hid_t file,
    const std::vector<std::string>& field_names);

} // namespace IO
} // namespace VVM
