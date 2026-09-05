#include "core/boundary/HorizontalBoundaryStencils.hpp"

#include <stdexcept>

#include <Kokkos_Core.hpp>

namespace VVM {
namespace Core {
namespace Boundary {

HorizontalBoundaryStencils::HorizontalBoundaryStencils(const Grid& grid)
    : grid_(grid) {

    const auto& horizontal = grid_.horizontal_specification();

    if (horizontal.ny > 1 && horizontal.topology.q2 != HorizontalEdgeTopology::Bounded) {
        throw std::invalid_argument("HorizontalBoundaryStencils requires bounded q2 topology.");
    }
}

void HorizontalBoundaryStencils::fill_constant_q2_halos(Field<2>& field) const {
    const int halo = grid_.get_halo_cells();
    if (halo == 0 || grid_.get_global_points_y() == 1) {
        return;
    }

    const bool is_south_boundary = grid_.get_local_physical_start_y() == 0;
    const bool is_north_boundary = grid_.get_local_physical_end_y() == grid_.get_global_points_y() - 1;

    auto data = field.get_mutable_device_data();
    const int ny = static_cast<int>(data.extent(0));
    const int nx = static_cast<int>(data.extent(1));

    if (is_south_boundary) {
        Kokkos::parallel_for("fill_constant_q2_south_2d",
            Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {halo, nx}),
            KOKKOS_LAMBDA(const int j_halo, const int i) {
                data(j_halo, i) = data(halo, i);
            }
        );
    }

    if (is_north_boundary) {
        const int last_physical_j = ny - halo - 1;
        const int first_north_halo_j = ny - halo;

        Kokkos::parallel_for("fill_constant_q2_north_2d",
            Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {halo, nx}),
            KOKKOS_LAMBDA(const int j_halo, const int i) {
                data(first_north_halo_j + j_halo, i) =
                    data(last_physical_j, i);
            }
        );
    }
}

void HorizontalBoundaryStencils::fill_constant_q2_halos(Field<3>& field) const {
    const int halo = grid_.get_halo_cells();
    if (halo == 0 || grid_.get_global_points_y() == 1) {
        return;
    }

    const bool is_south_boundary = grid_.get_local_physical_start_y() == 0;
    const bool is_north_boundary = grid_.get_local_physical_end_y() == grid_.get_global_points_y() - 1;

    auto data = field.get_mutable_device_data();
    const int nz = static_cast<int>(data.extent(0));
    const int ny = static_cast<int>(data.extent(1));
    const int nx = static_cast<int>(data.extent(2));

    if (is_south_boundary) {
        Kokkos::parallel_for("fill_constant_q2_south_3d",
            Kokkos::MDRangePolicy<Kokkos::Rank<3>>({0, 0, 0}, {nz, halo, nx}),
            KOKKOS_LAMBDA(const int k, const int j_halo, const int i) {
                data(k, j_halo, i) = data(k, halo, i);
            }
        );
    }

    if (is_north_boundary) {
        const int last_physical_j = ny - halo - 1;
        const int first_north_halo_j = ny - halo;

        Kokkos::parallel_for("fill_constant_q2_north_3d",
            Kokkos::MDRangePolicy<Kokkos::Rank<3>>({0, 0, 0}, {nz, halo, nx}),
            KOKKOS_LAMBDA(const int k, const int j_halo, const int i) {
                data(k, first_north_halo_j + j_halo, i) =
                    data(k, last_physical_j, i);
            }
        );
    }
}

} // namespace Boundary
} // namespace Core
} // namespace VVM
