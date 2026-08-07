#ifndef VVM_IO_BP5_FIELD_SCHEMA_HPP
#define VVM_IO_BP5_FIELD_SCHEMA_HPP

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include <adios2.h>

#include "core/Grid.hpp"
#include "utils/ConfigurationManager.hpp"

namespace VVM::IO::BP5 {

struct InclusiveBounds {
    std::size_t x_start = 0;
    std::size_t x_end = 0;
    std::size_t y_start = 0;
    std::size_t y_end = 0;
    std::size_t z_start = 0;
    std::size_t z_end = 0;

    std::size_t nx() const noexcept { return x_end - x_start + 1; }
    std::size_t ny() const noexcept { return y_end - y_start + 1; }
    std::size_t nz() const noexcept { return z_end - z_start + 1; }
};

struct GridRegion {
    std::size_t global_nx = 0;
    std::size_t global_ny = 0;
    std::size_t global_nz = 0;
    std::size_t local_x_start = 0;
    std::size_t local_y_start = 0;
    std::size_t local_z_start = 0;
    std::size_t local_nx = 0;
    std::size_t local_ny = 0;
    std::size_t local_nz = 0;
    std::size_t halo = 0;
};

struct FieldSelection {
    std::size_t dimensions = 0;
    adios2::Dims shape;
    adios2::Dims start;
    adios2::Dims count;
    adios2::Dims memory_start;
    adios2::Dims memory_count;

    bool empty() const noexcept;
    std::size_t elements() const noexcept;
};

class Bp5FieldSchema {
public:
    Bp5FieldSchema(InclusiveBounds bounds, GridRegion grid, int rank);

    static InclusiveBounds parse_bounds(
        const Utils::ConfigurationManager& config,
        std::size_t nx,
        std::size_t ny,
        std::size_t nz);

    static GridRegion from_grid(const Core::Grid& grid);

    FieldSelection selection(std::size_t dimensions,
                             std::size_t components = 1) const;

    const InclusiveBounds& bounds() const noexcept { return bounds_; }
    const GridRegion& grid() const noexcept { return grid_; }

private:
    InclusiveBounds bounds_;
    GridRegion grid_;
    int rank_ = 0;
};

} // namespace VVM::IO::BP5

#endif
