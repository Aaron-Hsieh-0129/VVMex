#include "Bp5FieldSchema.hpp"

#include <algorithm>
#include <limits>
#include <numeric>
#include <stdexcept>

namespace VVM::IO::BP5 {
namespace {

struct AxisIntersection {
    std::size_t start = 0;
    std::size_t count = 0;
    std::size_t memory_start = 0;
};

std::size_t checked_size(int value, const char* name) {
    if (value < 0) {
        throw std::invalid_argument(std::string(name) + " cannot be negative.");
    }
    return static_cast<std::size_t>(value);
}

std::pair<std::size_t, std::size_t> normalize_axis(
    std::int64_t start,
    std::int64_t end,
    std::size_t global_size,
    const char* axis) {
    if (global_size == 0) {
        throw std::invalid_argument(std::string("Global ") + axis + " size must be positive.");
    }
    if (end == -1) end = static_cast<std::int64_t>(global_size) - 1;
    if (start < 0 || end < 0 || start > end ||
        static_cast<std::uint64_t>(end) >= global_size) {
        throw std::invalid_argument(
            std::string("Invalid output.output_grid ") + axis + " bounds [" +
            std::to_string(start) + ", " + std::to_string(end) +
            "] for global size " + std::to_string(global_size) + ".");
    }
    return {static_cast<std::size_t>(start), static_cast<std::size_t>(end)};
}

AxisIntersection intersect_axis(
    std::size_t rank_start,
    std::size_t rank_count,
    std::size_t output_start,
    std::size_t output_end,
    std::size_t halo) {
    AxisIntersection result;
    if (rank_count == 0) return result;
    const std::size_t rank_end = rank_start + rank_count - 1;
    const std::size_t first = std::max(rank_start, output_start);
    const std::size_t last = std::min(rank_end, output_end);
    if (last < first) return result;
    result.start = first;
    result.count = last - first + 1;
    result.memory_start = first - rank_start + halo;
    return result;
}

} // namespace

bool FieldSelection::empty() const noexcept {
    return count.empty() ||
           std::any_of(count.begin(), count.end(), [](std::size_t n) { return n == 0; });
}

std::size_t FieldSelection::elements() const noexcept {
    if (empty()) return 0;
    return std::accumulate(count.begin(), count.end(), std::size_t{1},
                           std::multiplies<std::size_t>());
}

Bp5FieldSchema::Bp5FieldSchema(InclusiveBounds bounds, GridRegion grid, int rank)
    : bounds_(bounds), grid_(grid), rank_(rank) {
    if (rank_ < 0) throw std::invalid_argument("MPI rank cannot be negative.");
    if (grid_.global_nx == 0 || grid_.global_ny == 0 || grid_.global_nz == 0) {
        throw std::invalid_argument("Global grid dimensions must be positive.");
    }
    if (bounds_.x_start > bounds_.x_end || bounds_.x_end >= grid_.global_nx ||
        bounds_.y_start > bounds_.y_end || bounds_.y_end >= grid_.global_ny ||
        bounds_.z_start > bounds_.z_end || bounds_.z_end >= grid_.global_nz) {
        throw std::invalid_argument("BP5 output bounds are outside the global grid.");
    }
}

InclusiveBounds Bp5FieldSchema::parse_bounds(
    const Utils::ConfigurationManager& config,
    std::size_t nx,
    std::size_t ny,
    std::size_t nz) {
    const auto x = normalize_axis(
        config.get_value<std::int64_t>("output.output_grid.x_start", 0),
        config.get_value<std::int64_t>("output.output_grid.x_end", -1), nx, "x");
    const auto y = normalize_axis(
        config.get_value<std::int64_t>("output.output_grid.y_start", 0),
        config.get_value<std::int64_t>("output.output_grid.y_end", -1), ny, "y");
    const auto z = normalize_axis(
        config.get_value<std::int64_t>("output.output_grid.z_start", 0),
        config.get_value<std::int64_t>("output.output_grid.z_end", -1), nz, "z");
    return {x.first, x.second, y.first, y.second, z.first, z.second};
}

GridRegion Bp5FieldSchema::from_grid(const Core::Grid& grid) {
    GridRegion result;
    result.global_nx = checked_size(grid.get_global_points_x(), "global nx");
    result.global_ny = checked_size(grid.get_global_points_y(), "global ny");
    result.global_nz = checked_size(grid.get_global_points_z(), "global nz");
    result.local_x_start = checked_size(grid.get_local_physical_start_x(), "local x start");
    result.local_y_start = checked_size(grid.get_local_physical_start_y(), "local y start");
    result.local_z_start = checked_size(grid.get_local_physical_start_z(), "local z start");
    result.local_nx = checked_size(grid.get_local_physical_points_x(), "local nx");
    result.local_ny = checked_size(grid.get_local_physical_points_y(), "local ny");
    result.local_nz = checked_size(grid.get_local_physical_points_z(), "local nz");
    result.halo = checked_size(grid.get_halo_cells(), "halo");
    return result;
}

FieldSelection Bp5FieldSchema::selection(
    std::size_t dimensions,
    std::size_t components) const {
    if (dimensions < 1 || dimensions > 4) {
        throw std::invalid_argument("BP5 fields must have 1 through 4 dimensions.");
    }
    if (dimensions == 4 && components == 0) {
        throw std::invalid_argument("A 4-D BP5 field must have at least one component.");
    }

    const auto x = intersect_axis(grid_.local_x_start, grid_.local_nx,
                                  bounds_.x_start, bounds_.x_end, grid_.halo);
    const auto y = intersect_axis(grid_.local_y_start, grid_.local_ny,
                                  bounds_.y_start, bounds_.y_end, grid_.halo);
    const auto z = intersect_axis(grid_.local_z_start, grid_.local_nz,
                                  bounds_.z_start, bounds_.z_end, grid_.halo);

    FieldSelection result;
    result.dimensions = dimensions;
    if (dimensions == 1) {
        result.shape = {grid_.global_nz};
        result.start = {z.start};
        result.count = {rank_ == 0 ? z.count : 0};
        result.memory_start = {z.memory_start};
        result.memory_count = {grid_.local_nz + 2 * grid_.halo};
    } else if (dimensions == 2) {
        result.shape = {grid_.global_ny, grid_.global_nx};
        result.start = {y.start, x.start};
        result.count = {y.count, x.count};
        result.memory_start = {y.memory_start, x.memory_start};
        result.memory_count = {
            grid_.local_ny + 2 * grid_.halo,
            grid_.local_nx + 2 * grid_.halo};
    } else if (dimensions == 3) {
        result.shape = {grid_.global_nz, grid_.global_ny, grid_.global_nx};
        result.start = {z.start, y.start, x.start};
        result.count = {z.count, y.count, x.count};
        result.memory_start = {z.memory_start, y.memory_start, x.memory_start};
        result.memory_count = {
            grid_.local_nz + 2 * grid_.halo,
            grid_.local_ny + 2 * grid_.halo,
            grid_.local_nx + 2 * grid_.halo};
    } else {
        result.shape = {components, grid_.global_nz, grid_.global_ny, grid_.global_nx};
        result.start = {0, z.start, y.start, x.start};
        result.count = {components, z.count, y.count, x.count};
        result.memory_start = {0, z.memory_start, y.memory_start, x.memory_start};
        result.memory_count = {
            components,
            grid_.local_nz + 2 * grid_.halo,
            grid_.local_ny + 2 * grid_.halo,
            grid_.local_nx + 2 * grid_.halo};
    }
    return result;
}

} // namespace VVM::IO::BP5
