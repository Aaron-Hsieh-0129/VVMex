#ifndef VVM_CORE_BOUNDARY_HORIZONTAL_BOUNDARY_STENCILS_HPP
#define VVM_CORE_BOUNDARY_HORIZONTAL_BOUNDARY_STENCILS_HPP

#include <cstddef>
#include <memory>
#include "core/geometry/StaggeredVectorTransform.hpp"
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

    // Free-slip chart wall: covariant tangential velocity is even; the
    // contravariant normal mass flux is odd and zero on the wall face.
    // Scalar transport instead reflects physical tangential flux evenly.
    template <typename Q1Layout, typename Q2Layout>
    void
    fill_contravariant_wind_halos(
        Field<3, Q1Layout>& first, Field<3, Q2Layout>& second, bool scalar_flux = false) const {
        const int h = grid_.get_halo_cells();
        if (h == 0 || grid_.get_global_points_y() == 1) {
            return;
        }
        const auto q1 = first.get_mutable_device_data();
        const auto q2 = second.get_mutable_device_data();
        const int nz = q1.extent(0), ny = q1.extent(1), nx = q1.extent(2);
        if (q2.extent(0) != q1.extent(0) || q2.extent(1) != q1.extent(1) ||
            q2.extent(2) != q1.extent(2)) {
            throw std::invalid_argument("Horizontal vector fields must have matching extents.");
        }
        const auto gu = grid_.geometry().device_view(Geometry::HorizontalLocation::U);
        const auto gv = grid_.geometry().device_view(Geometry::HorizontalLocation::V);
        const auto g11 = gu.g_cov.a11, g12 = gv.g_cov.a12;
        const auto normal_g11 = gv.g_cov.a11, normal_inv_j = gv.inv_sqrt_g;
        const bool south = grid_.get_local_physical_start_y() == 0;
        const bool north = grid_.get_local_physical_end_y() == grid_.get_global_points_y() - 1;
        for (int side = 0; side < 2; ++side) {
            if ((side == 0 && !south) || (side == 1 && !north)) {
                continue;
            }
            const int wall = side == 0 ? h - 1 : ny - h - 1;
            const int direction = side == 0 ? -1 : 1;
            const int count = side == 0 ? h : h + 1;
            Kokkos::parallel_for("ReflectContravariantNormalFlux",
                Kokkos::Experimental::require(
                    Kokkos::MDRangePolicy<Kokkos::Rank<3>>({0, 0, 0}, {nz, count, nx}),
                    Kokkos::Experimental::WorkItemProperty::HintLightWeight),
                KOKKOS_LAMBDA(int k, int distance, int i) {
                    const int exterior = wall + direction * distance;
                    const int interior = wall - direction * distance;
                    q2(k, exterior, i) =
                        distance == 0
                            ? real(0.0)
                            : -q2(k, interior, i) *
                                  Kokkos::sqrt(normal_g11(exterior, i) / normal_g11(interior, i)) *
                                  normal_inv_j(exterior, i) / normal_inv_j(interior, i);
                });
            Kokkos::parallel_for("ReflectCovariantTangentialWind",
                Kokkos::Experimental::require(
                    Kokkos::MDRangePolicy<Kokkos::Rank<3>>({0, 0, 0}, {nz, h, nx}),
                    Kokkos::Experimental::WorkItemProperty::HintLightWeight),
                KOKKOS_LAMBDA(int k, int distance, int i) {
                    const int exterior = side == 0 ? h - 1 - distance : ny - h + distance;
                    const int interior = side == 0 ? h + distance : ny - h - 1 - distance;
                    Real covariant = g11(interior, i) * q1(k, interior, i) +
                                     Geometry::weighted_cross<true>(q2, g12, k, interior, i);
                    if (scalar_flux) {
                        covariant *= Kokkos::sqrt(g11(exterior, i) / g11(interior, i));
                    }
                    q1(k, exterior, i) =
                        (covariant - Geometry::weighted_cross<true>(q2, g12, k, exterior, i)) /
                        g11(exterior, i);
                });
        }
    }

    template <typename Q1Layout, typename Q2Layout>
    void
    fill_contravariant_scalar_mass_flux_halos(Field<3, Q1Layout>& q1,
        Field<3, Q2Layout>& q2) const {
        fill_contravariant_wind_halos(q1, q2, true);
    }

    // VVM stores xi=omega^1@V and eta=-omega^2@U. At a free-slip
    // wall the tangential covector component vanishes; the normal component
    // is reflected evenly. Treat the pair together on a skew chart.
    void
    fill_contravariant_vorticity_halos(Field<3>& xi, Field<3>& eta) const {
        const int h = grid_.get_halo_cells();
        if (h == 0 || grid_.get_global_points_y() == 1) {
            return;
        }
        const auto x = xi.get_mutable_device_data(), e = eta.get_mutable_device_data();
        const int nz = x.extent(0), ny = x.extent(1), nx = x.extent(2);
        const auto gv = grid_.geometry().device_view(Geometry::HorizontalLocation::V);
        const auto gu = grid_.geometry().device_view(Geometry::HorizontalLocation::U);
        const auto g11 = gv.g_cov.a11, g12 = gu.g_cov.a12;
        const auto normal_g11 = gu.g_cov.a11, normal_inv_j = gu.inv_sqrt_g;
        for (int side = 0; side < 2; ++side) {
            if (side == 0 && grid_.get_local_physical_start_y() != 0) {
                continue;
            }
            if (side == 1 && grid_.get_local_physical_end_y() != grid_.get_global_points_y() - 1) {
                continue;
            }
            const int wall = side == 0 ? h - 1 : ny - h - 1;
            const int direction = side == 0 ? -1 : 1;
            const int count = side == 0 ? h : h + 1;
            Kokkos::parallel_for("ReflectNormalVorticityComponent",
                Kokkos::Experimental::require(
                    Kokkos::MDRangePolicy<Kokkos::Rank<3>>({0, 0, 0}, {nz, h, nx}),
                    Kokkos::Experimental::WorkItemProperty::HintLightWeight),
                KOKKOS_LAMBDA(int k, int distance, int i) {
                    const int exterior = side == 0 ? h - 1 - distance : ny - h + distance;
                    const int interior = side == 0 ? h + distance : ny - h - 1 - distance;
                    e(k, exterior, i) =
                        e(k, interior, i) *
                        Kokkos::sqrt(normal_g11(exterior, i) / normal_g11(interior, i)) *
                        normal_inv_j(exterior, i) / normal_inv_j(interior, i);
                });
            Kokkos::parallel_for("ReflectHorizontalVorticityPair",
                Kokkos::Experimental::require(
                    Kokkos::MDRangePolicy<Kokkos::Rank<3>>({0, 0, 0}, {nz, count, nx}),
                    Kokkos::Experimental::WorkItemProperty::HintLightWeight),
                KOKKOS_LAMBDA(int k, int distance, int i) {
                    const int exterior = wall + direction * distance;
                    const int interior = wall - direction * distance;
                    const Real cov =
                        distance == 0
                            ? real(0.0)
                            : -(g11(interior, i) * x(k, interior, i) -
                                  Geometry::weighted_cross<false>(e, g12, k, interior, i)) *
                                  Kokkos::sqrt(g11(exterior, i) / g11(interior, i));
                    x(k, exterior, i) =
                        (cov + Geometry::weighted_cross<false>(e, g12, k, exterior, i)) /
                        g11(exterior, i);
                });
        }
    }

    // Oblique physical Neumann condition g^2i d_i(s)=0 on q2 walls.
    // Interior tangential differences avoid simultaneous ghost-row reads.
    template <size_t Dim, typename Layout>
    void
    fill_physical_q2_neumann_halos(Field<Dim, Layout>& field) const {
        static_assert(Dim == 2 || Dim == 3);
        const int h = grid_.get_halo_cells();
        if (h == 0 || grid_.get_global_points_y() == 1) {
            return;
        }
        auto data = field.get_mutable_device_data();
        const int ny = grid_.get_local_total_points_y(), nx = grid_.get_local_total_points_x();
        const int nz = Dim == 3 ? grid_.get_local_total_points_z() : 1;
        const auto gv = grid_.geometry().device_view(Geometry::HorizontalLocation::V);
        const auto c12 = gv.sqrt_g_g_contra.a12, c22 = gv.sqrt_g_g_contra.a22;
        const Real spacing = grid_.geometry().dq2() / (real(2.0) * grid_.geometry().dq1());
        for (int side = 0; side < 2; ++side) {
            if (side == 0 && grid_.get_local_physical_start_y() != 0) {
                continue;
            }
            if (side == 1 && grid_.get_local_physical_end_y() != grid_.get_global_points_y() - 1) {
                continue;
            }
            const int wall = side == 0 ? h - 1 : ny - h - 1;
            Kokkos::parallel_for("PhysicalQ2NeumannHalos",
                Kokkos::Experimental::require(
                    Kokkos::MDRangePolicy<Kokkos::Rank<3>>({0, 0, 0}, {nz, h, nx}),
                    Kokkos::Experimental::WorkItemProperty::HintLightWeight),
                KOKKOS_LAMBDA(int k, int d, int i) {
                    // Capture the view before constexpr branching for NVCC.
                    const auto values = data;
                    const int exterior = side == 0 ? h - 1 - d : ny - h + d;
                    const int interior = side == 0 ? h + d : ny - h - 1 - d;
                    const int im = i > 0 ? i - 1 : i, ip = i + 1 < nx ? i + 1 : i;
                    const Real scale = (side == 0 ? real(1.0) : real(-1.0)) * real(2 * d + 1) *
                                       spacing * c12(wall, i) / c22(wall, i);
                    if constexpr (Dim == 2) {
                        values(exterior, i) =
                            values(interior, i) + scale * (values(interior, ip) - values(interior, im));
                    }
                    else {
                        values(k, exterior, i) =
                            values(k, interior, i) +
                            scale * (values(k, interior, ip) - values(k, interior, im));
                    }
                });
        }
    }

    // Physical compatibility halos are exported from a canonical pair. The
    // original physical interior is retained, avoiding a staggered round trip.
    template <typename ULayout, typename VLayout>
    void
    fill_free_slip_physical_wind_halos(Field<3, ULayout>& u, Field<3, VLayout>& v) const {
        const int h = grid_.get_halo_cells();
        if (h == 0 || grid_.get_global_points_y() == 1) {
            return;
        }
        const auto ud = u.get_mutable_device_data(), vd = v.get_mutable_device_data();
        if (ud.extent(0) != vd.extent(0) || ud.extent(1) != vd.extent(1) ||
            ud.extent(2) != vd.extent(2)) {
            throw std::invalid_argument("Physical wind fields must have matching extents.");
        }
        const auto gu = grid_.geometry().device_view(Geometry::HorizontalLocation::U);
        const auto gv = grid_.geometry().device_view(Geometry::HorizontalLocation::V);
        const auto zero = [](const auto& coefficient) {
            return coefficient.layout == Geometry::GeometryFieldLayout::Constant &&
                   coefficient.constant == real(0.0);
        };
        if (zero(gu.physical_to_contravariant.a12) &&
            zero(gv.physical_to_contravariant.a21)) {
            // On a diagonal chart, reflect physical normal wind directly.
            // A physical/canonical round trip would lose exact odd symmetry.
            const auto h1 = gu.contravariant_to_physical.a11;
            const int nz = ud.extent(0), ny = ud.extent(1), nx = ud.extent(2);
            for (int side = 0; side < 2; ++side) {
                if ((side == 0 && grid_.get_local_physical_start_y() != 0) ||
                    (side == 1 && grid_.get_local_physical_end_y() !=
                                      grid_.get_global_points_y() - 1)) {
                    continue;
                }
                Kokkos::parallel_for("ReflectDiagonalPhysicalTangentialWind",
                    Kokkos::Experimental::require(
                        Kokkos::MDRangePolicy<Kokkos::Rank<3>>({0, 0, 0}, {nz, h, nx}),
                        Kokkos::Experimental::WorkItemProperty::HintLightWeight),
                    KOKKOS_LAMBDA(int k, int distance, int i) {
                        const int exterior = side == 0 ? h - 1 - distance : ny - h + distance;
                        const int interior = side == 0 ? h + distance : ny - h - 1 - distance;
                        ud(k, exterior, i) =
                            ud(k, interior, i) * h1(interior, i) / h1(exterior, i);
                    });
            }
            fill_positive_face_q2_homogeneous_dirichlet_halos(v);
            return;
        }
        if (!wind_q1_) {
            const std::array<int, 3> shape{grid_.get_local_total_points_z(),
                grid_.get_local_total_points_y(),
                grid_.get_local_total_points_x()};
            wind_q1_ = std::make_unique<Field<3>>("wall canonical q1 scratch", shape);
            wind_q2_ = std::make_unique<Field<3>>("wall canonical q2 scratch", shape);
        }
        auto q1 = wind_q1_->get_mutable_device_data(), q2 = wind_q2_->get_mutable_device_data();
        Geometry::transform_staggered_vector<true>(u.get_device_data(),
            v.get_device_data(),
            q1,
            q2,
            gu.physical_to_contravariant,
            gv.physical_to_contravariant);
        fill_contravariant_wind_halos(*wind_q1_, *wind_q2_);
        const auto b11 = gu.contravariant_to_physical.a11, b12 = gv.contravariant_to_physical.a12;
        const auto b21 = gu.contravariant_to_physical.a21, b22 = gv.contravariant_to_physical.a22;
        const int nz = ud.extent(0), ny = ud.extent(1), nx = ud.extent(2);
        const bool south = grid_.get_local_physical_start_y() == 0;
        const bool north = grid_.get_local_physical_end_y() == grid_.get_global_points_y() - 1;
        Kokkos::parallel_for("ExportPhysicalWallWindHalos",
            Kokkos::Experimental::require(
                    Kokkos::MDRangePolicy<Kokkos::Rank<3>>({0, 0, 0}, {nz, ny, nx}),
                    Kokkos::Experimental::WorkItemProperty::HintLightWeight),
            KOKKOS_LAMBDA(int k, int j, int i) {
                if ((south && j < h) || (north && j >= ny - h)) {
                    ud(k, j, i) =
                        b11(j, i) * q1(k, j, i) + Geometry::weighted_cross<true>(q2, b12, k, j, i);
                }
                if ((south && j < h) || (north && j >= ny - h - 1)) {
                    vd(k, j, i) =
                        b22(j, i) * q2(k, j, i) + Geometry::weighted_cross<false>(q1, b21, k, j, i);
                }
            });
    }

    // Compatibility names for downstream callers of the former RLL API.
    template <typename A, typename B>
    void
    fill_regular_lat_lon_free_slip_physical_wind_halos(Field<3, A>& u, Field<3, B>& v) const {
        fill_free_slip_physical_wind_halos(u, v);
    }
    template <typename A, typename B>
    void
    fill_regular_lat_lon_contravariant_scalar_mass_flux_halos(Field<3, A>& u,
        Field<3, B>& v) const {
        fill_contravariant_scalar_mass_flux_halos(u, v);
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
    mutable std::unique_ptr<Field<3>> wind_q1_, wind_q2_;
};

} // namespace Boundary
} // namespace Core
} // namespace VVM

#endif // VVM_CORE_BOUNDARY_HORIZONTAL_BOUNDARY_STENCILS_HPP
