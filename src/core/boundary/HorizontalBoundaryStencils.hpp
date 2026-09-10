#ifndef VVM_CORE_BOUNDARY_HORIZONTAL_BOUNDARY_STENCILS_HPP
#define VVM_CORE_BOUNDARY_HORIZONTAL_BOUNDARY_STENCILS_HPP

#include <cstddef>
#include <stdexcept>
#include <Kokkos_Core.hpp>

#include "core/Field.hpp"
#include "core/Grid.hpp"
#include "core/geometry/GeometryKind.hpp"
#include "core/geometry/HorizontalLocation.hpp"

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
    template <size_t Dim, typename Layout>
    void
    fill_constant_q2_halos(Field<Dim, Layout>& field) const {
        static_assert(Dim == 2 || Dim == 3,
            "Constant q2 halo filling supports only two- and three-dimensional fields.");
        const int halo = grid_.get_halo_cells();

        if (halo == 0 || grid_.get_global_points_y() == 1) {
            return;
        }

        const bool is_south_boundary = grid_.get_local_physical_start_y() == 0;
        const bool is_north_boundary =
            grid_.get_local_physical_end_y() == grid_.get_global_points_y() - 1;

        auto data = field.get_mutable_device_data();

        if constexpr (Dim == 2) {
            const int ny = static_cast<int>(data.extent(0));
            const int nx = static_cast<int>(data.extent(1));

            if (is_south_boundary) {
                Kokkos::parallel_for("fill_constant_q2_south_2d",
                    Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {halo, nx}),
                    KOKKOS_LAMBDA(const int j_halo, const int i) {
                        data(j_halo, i) = data(halo, i);
                    });
            }

            if (is_north_boundary) {
                const int last_physical_j = ny - halo - 1;
                const int first_north_halo_j = ny - halo;

                Kokkos::parallel_for("fill_constant_q2_north_2d",
                    Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {halo, nx}),
                    KOKKOS_LAMBDA(const int j_halo, const int i) {
                        data(first_north_halo_j + j_halo, i) = data(last_physical_j, i);
                    });
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
                    });
            }

            if (is_north_boundary) {
                const int last_physical_j = ny - halo - 1;
                const int first_north_halo_j = ny - halo;

                Kokkos::parallel_for("fill_constant_q2_north_3d",
                    Kokkos::MDRangePolicy<Kokkos::Rank<3>>({0, 0, 0}, {nz, halo, nx}),
                    KOKKOS_LAMBDA(const int k, const int j_halo, const int i) {
                        data(k, first_north_halo_j + j_halo, i) = data(k, last_physical_j, i);
                    });
            }
        }
    }

    // Even reflection about a physical q2 wall located halfway between
    // centered T/U rows. This is the discrete homogeneous Neumann condition
    // used by chi, w and other centered scalar-like fields.
    template <size_t Dim, typename Layout>
    void
    fill_centered_q2_neumann_halos(Field<Dim, Layout>& field) const {
        static_assert(Dim == 2 || Dim == 3,
            "Centered q2 Neumann filling supports only two- and three-dimensional fields.");

        const int halo = grid_.get_halo_cells();

        if (halo == 0 || grid_.get_global_points_y() == 1) {
            return;
        }

        const bool is_south_boundary = grid_.get_local_physical_start_y() == 0;
        const bool is_north_boundary =
            grid_.get_local_physical_end_y() == grid_.get_global_points_y() - 1;

        auto data = field.get_mutable_device_data();

        if constexpr (Dim == 2) {
            const int ny = static_cast<int>(data.extent(0));
            const int nx = static_cast<int>(data.extent(1));

            if (is_south_boundary) {
                Kokkos::parallel_for("FillCenteredQ2NeumannSouth2D",
                    Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {halo, nx}),
                    KOKKOS_LAMBDA(const int distance, const int i) {
                        const int exterior_j = halo - 1 - distance;
                        const int interior_j = halo + distance;
                        data(exterior_j, i) = data(interior_j, i);
                    });
            }

            if (is_north_boundary) {
                const int first_north_halo_j = ny - halo;
                const int last_physical_j = first_north_halo_j - 1;

                Kokkos::parallel_for("FillCenteredQ2NeumannNorth2D",
                    Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {halo, nx}),
                    KOKKOS_LAMBDA(const int distance, const int i) {
                        data(first_north_halo_j + distance, i) =
                            data(last_physical_j - distance, i);
                    });
            }
        }

        if constexpr (Dim == 3) {
            const int nz = static_cast<int>(data.extent(0));
            const int ny = static_cast<int>(data.extent(1));
            const int nx = static_cast<int>(data.extent(2));

            if (is_south_boundary) {
                Kokkos::parallel_for("FillCenteredQ2NeumannSouth3D",
                    Kokkos::MDRangePolicy<Kokkos::Rank<3>>({0, 0, 0}, {nz, halo, nx}),
                    KOKKOS_LAMBDA(const int k, const int distance, const int i) {
                        const int exterior_j = halo - 1 - distance;
                        const int interior_j = halo + distance;
                        data(k, exterior_j, i) = data(k, interior_j, i);
                    });
            }

            if (is_north_boundary) {
                const int first_north_halo_j = ny - halo;
                const int last_physical_j = first_north_halo_j - 1;

                Kokkos::parallel_for("FillCenteredQ2NeumannNorth3D",
                    Kokkos::MDRangePolicy<Kokkos::Rank<3>>({0, 0, 0}, {nz, halo, nx}),
                    KOKKOS_LAMBDA(const int k, const int distance, const int i) {
                        data(k, first_north_halo_j + distance, i) =
                            data(k, last_physical_j - distance, i);
                    });
            }
        }
    }

    // Homogeneous Dirichlet data on positive-face V/Z placement. The south
    // wall is row halo-1; the north wall is the final physical row. Exterior
    // values are odd reflections about the wall.
    //
    // This is used for psi under a zero-wall-gauge channel policy and for
    // quantities that vanish at a free-slip wall, such as normal wind and
    // vertical vorticity.
    template <size_t Dim, typename Layout>
    void
    fill_positive_face_q2_homogeneous_dirichlet_halos(Field<Dim, Layout>& field) const {
        static_assert(Dim == 2 || Dim == 3,
            "Positive-face q2 Dirichlet filling supports only two- and three-dimensional fields.");

        const int halo = grid_.get_halo_cells();

        if (halo == 0 || grid_.get_global_points_y() == 1) {
            return;
        }

        const bool is_south_boundary = grid_.get_local_physical_start_y() == 0;
        const bool is_north_boundary =
            grid_.get_local_physical_end_y() == grid_.get_global_points_y() - 1;

        auto data = field.get_mutable_device_data();

        if constexpr (Dim == 2) {
            const int ny = static_cast<int>(data.extent(0));
            const int nx = static_cast<int>(data.extent(1));

            if (is_south_boundary) {
                const int wall_j = halo - 1;

                Kokkos::parallel_for("FillPositiveFaceQ2DirichletSouth2D",
                    Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {halo, nx}),
                    KOKKOS_LAMBDA(const int distance, const int i) {
                        const int target_j = wall_j - distance;

                        if (distance == 0) {
                            data(target_j, i) = VVM::real(0.0);
                        }
                        else {
                            data(target_j, i) = -data(wall_j + distance, i);
                        }
                    });
            }

            if (is_north_boundary) {
                const int wall_j = ny - halo - 1;

                Kokkos::parallel_for("FillPositiveFaceQ2DirichletNorth2D",
                    Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {halo + 1, nx}),
                    KOKKOS_LAMBDA(const int distance, const int i) {
                        const int target_j = wall_j + distance;

                        if (distance == 0) {
                            data(target_j, i) = VVM::real(0.0);
                        }
                        else {
                            data(target_j, i) = -data(wall_j - distance, i);
                        }
                    });
            }
        }

        if constexpr (Dim == 3) {
            const int nz = static_cast<int>(data.extent(0));
            const int ny = static_cast<int>(data.extent(1));
            const int nx = static_cast<int>(data.extent(2));

            if (is_south_boundary) {
                const int wall_j = halo - 1;

                Kokkos::parallel_for("FillPositiveFaceQ2DirichletSouth3D",
                    Kokkos::MDRangePolicy<Kokkos::Rank<3>>({0, 0, 0}, {nz, halo, nx}),
                    KOKKOS_LAMBDA(const int k, const int distance, const int i) {
                        const int target_j = wall_j - distance;

                        if (distance == 0) {
                            data(k, target_j, i) = VVM::real(0.0);
                        }
                        else {
                            data(k, target_j, i) = -data(k, wall_j + distance, i);
                        }
                    });
            }

            if (is_north_boundary) {
                const int wall_j = ny - halo - 1;

                Kokkos::parallel_for("FillPositiveFaceQ2DirichletNorth3D",
                    Kokkos::MDRangePolicy<Kokkos::Rank<3>>({0, 0, 0}, {nz, halo + 1, nx}),
                    KOKKOS_LAMBDA(const int k, const int distance, const int i) {
                        const int target_j = wall_j + distance;

                        if (distance == 0) {
                            data(k, target_j, i) = VVM::real(0.0);
                        }
                        else {
                            data(k, target_j, i) = -data(k, wall_j - distance, i);
                        }
                    });
            }
        }
    }

    // Apply the RLL physical free-slip wall rule to physical eastward and
    // northward wind:
    //
    //   v_N = 0
    //   d(u_1)/dphi = 0, where u_1 = R cos(phi) u_E.
    //
    // Normal wind uses positive-face odd reflection. Eastward physical wind
    // uses centered reflection of its covariant component, rather than
    // reflection of physical u_E itself.
    template <typename ULayout, typename VLayout>
    void
    fill_regular_lat_lon_free_slip_physical_wind_halos(Field<3, ULayout>& u,
        Field<3, VLayout>& v) const {

        if (grid_.geometry().kind() != Geometry::GeometryKind::RegularLatLon) {
            throw std::invalid_argument("RLL free-slip physical wind boundaries require regular "
                                        "latitude-longitude geometry.");
        }

        const int halo = grid_.get_halo_cells();

        if (halo == 0 || grid_.get_global_points_y() == 1) {
            return;
        }

        auto u_data = u.get_mutable_device_data();
        auto v_data = v.get_mutable_device_data();

        if (u_data.extent(0) != v_data.extent(0) || u_data.extent(1) != v_data.extent(1) ||
            u_data.extent(2) != v_data.extent(2)) {

            throw std::invalid_argument(
                "RLL free-slip physical u and v fields must have matching extents.");
        }

        const int nz = static_cast<int>(u_data.extent(0));
        const int ny = static_cast<int>(u_data.extent(1));
        const int nx = static_cast<int>(u_data.extent(2));

        const bool is_south_boundary = grid_.get_local_physical_start_y() == 0;
        const bool is_north_boundary =
            grid_.get_local_physical_end_y() == grid_.get_global_points_y() - 1;

        const auto h1_at_u = grid_.geometry()
                                 .device_view(Geometry::HorizontalLocation::U)
                                 .contravariant_to_physical.a11;

        if (is_south_boundary) {
            Kokkos::parallel_for("FillRLLFreeSlipUAtSouth",
                Kokkos::MDRangePolicy<Kokkos::Rank<3>>({0, 0, 0}, {nz, halo, nx}),
                KOKKOS_LAMBDA(const int k, const int distance, const int i) {
                    const int exterior_j = halo - 1 - distance;
                    const int interior_j = halo + distance;

                    u_data(k, exterior_j, i) =
                        u_data(k, interior_j, i) * h1_at_u(interior_j, i) / h1_at_u(exterior_j, i);
                });
        }

        if (is_north_boundary) {
            const int first_north_halo_j = ny - halo;
            const int last_physical_j = first_north_halo_j - 1;

            Kokkos::parallel_for("FillRLLFreeSlipUAtNorth",
                Kokkos::MDRangePolicy<Kokkos::Rank<3>>({0, 0, 0}, {nz, halo, nx}),
                KOKKOS_LAMBDA(const int k, const int distance, const int i) {
                    const int exterior_j = first_north_halo_j + distance;
                    const int interior_j = last_physical_j - distance;

                    u_data(k, exterior_j, i) =
                        u_data(k, interior_j, i) * h1_at_u(interior_j, i) / h1_at_u(exterior_j, i);
                });
        }

        fill_positive_face_q2_homogeneous_dirichlet_halos(v);
    }

    // Constant wall values for the total streamfunction. The wall difference
    // retains the prescribed channel transport; homogeneous callers are unchanged.
    void
    fill_positive_face_q2_dirichlet_halos(Field<2>& field, Real south, Real north) const {
        fill_positive_face_q2_homogeneous_dirichlet_halos(field);
        const int h = grid_.get_halo_cells();
        const int ny = grid_.get_local_total_points_y();
        const int nx = grid_.get_local_total_points_x();
        const auto data = field.get_mutable_device_data();
        if (south != real(0.0) && grid_.get_local_physical_start_y() == 0) {
            Kokkos::parallel_for("RLLSouthStreamfunctionValue",
                Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {h, nx}),
                KOKKOS_LAMBDA(int distance, int i) {
                    data(h - 1 - distance, i) += (distance == 0 ? real(1.) : real(2.)) * south;
                });
        }
        if (north != real(0.0) &&
            grid_.get_local_physical_end_y() == grid_.get_global_points_y() - 1) {
            Kokkos::parallel_for("RLLNorthStreamfunctionValue",
                Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {h + 1, nx}),
                KOKKOS_LAMBDA(int distance, int i) {
                    data(ny - h - 1 + distance, i) += (distance == 0 ? real(1.) : real(2.)) * north;
                });
        }
    }

private:
    const Grid& grid_;
};

} // namespace Boundary
} // namespace Core
} // namespace VVM

#endif // VVM_CORE_BOUNDARY_HORIZONTAL_BOUNDARY_STENCILS_HPP
