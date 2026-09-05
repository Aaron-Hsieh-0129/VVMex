#ifndef VVM_CORE_BOUNDARY_HORIZONTAL_BOUNDARY_STENCILS_HPP
#define VVM_CORE_BOUNDARY_HORIZONTAL_BOUNDARY_STENCILS_HPP

#include <cstddef>
#include <Kokkos_Core.hpp>

#include "core/Field.hpp"
#include "core/Grid.hpp"

namespace VVM {
namespace Core {
namespace Boundary {

class HorizontalBoundaryStencils {
public:
    explicit HorizontalBoundaryStencils(const Grid& grid);

    // Constantly extrapolate the nearest physical q2 row into every exterior
    // q2 halo row. This reproduces the MODE=2 normal-boundary stencil used by
    // CVVM's BOUND_NORMAL routine.
    //
    // This is a low-level stencil, not a complete free-slip boundary policy.
    template<size_t Dim, typename Layout>
    void fill_constant_q2_halos(Field<Dim, Layout>& field) const {
        static_assert(Dim == 2 || Dim == 3, "Constant q2 halo filling supports only two- and three-dimensional fields.");
        const int halo = grid_.get_halo_cells();

        if (halo == 0 || grid_.get_global_points_y() == 1) {
            return;
        }

        const bool is_south_boundary = grid_.get_local_physical_start_y() == 0;
        const bool is_north_boundary = grid_.get_local_physical_end_y() == grid_.get_global_points_y() - 1;

        auto data = field.get_mutable_device_data();

        if constexpr (Dim == 2) {
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

        if constexpr (Dim == 3) {
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
                        data(k, first_north_halo_j + j_halo, i) = data(k, last_physical_j, i);
                    }
                );
            }
        }
    }

private:
    const Grid& grid_;
};

} // namespace Boundary
} // namespace Core
} // namespace VVM

#endif // VVM_CORE_BOUNDARY_HORIZONTAL_BOUNDARY_STENCILS_HPP
