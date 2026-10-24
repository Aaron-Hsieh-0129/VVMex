#ifndef VVM_CORE_GEOMETRY_HORIZONTAL_BOUNDARY_DISTANCES_HPP
#define VVM_CORE_GEOMETRY_HORIZONTAL_BOUNDARY_DISTANCES_HPP

#include <array>
#include <algorithm>
#include <vector>
#include <mpi.h>
#include "core/grid/Grid.hpp"
#include "core/state/Field.hpp"
#include "core/initialization/StartupCollectives.hpp"

namespace VVM::Core::Geometry {

// Integrate physical normal spacing along each coordinate family. Prefix
// sums exchange row and column totals, never a global plane.
// This startup utility is used for metre-valued lateral forcing distances.
struct HorizontalBoundaryDistances {
    Field<2> west, east, south, north;
    HorizontalBoundaryDistances(const Grid& grid, const State& state)
        : west("mapped west distance",
              {grid.get_local_total_points_y(), grid.get_local_total_points_x()}),
          east("mapped east distance",
              {grid.get_local_total_points_y(), grid.get_local_total_points_x()}),
          south("mapped south distance",
              {grid.get_local_total_points_y(), grid.get_local_total_points_x()}),
          north("mapped north distance",
              {grid.get_local_total_points_y(), grid.get_local_total_points_x()}) {
        initialize(grid, state);
    }

    // CUDA kernel template arguments require a public functor type.
    struct PhysicalNormalSpacing {
        Field<2>::ViewType wx, sy;
        GeometryField2D inv_j, c11, c22;
        Real dq1, dq2;

        KOKKOS_INLINE_FUNCTION
        void
        operator()(int j, int i) const {
            wx(j, i) = dq1 / Kokkos::sqrt(inv_j(j, i) * c11(j, i));
            sy(j, i) = dq2 / Kokkos::sqrt(inv_j(j, i) * c22(j, i));
        }
    };

private:
    void
    initialize(const Grid& grid, const State& state) {
        const int h = grid.get_halo_cells(), ny = grid.get_local_physical_points_y(),
                  nx = grid.get_local_physical_points_x();
        const auto metric = grid.geometry().device_view(HorizontalLocation::T);
        const auto inv_j = metric.inv_sqrt_g, c11 = metric.sqrt_g_g_contra.a11,
                   c22 = metric.sqrt_g_g_contra.a22;
        const Real dq1 = grid.geometry().dq1(), dq2 = grid.geometry().dq2();
        auto wx = west.get_mutable_device_data(), sy = south.get_mutable_device_data();
        Kokkos::parallel_for("MappedPhysicalNormalSpacing",
            Kokkos::MDRangePolicy<Kokkos::Rank<2>>({h, h}, {h + ny, h + nx}),
            PhysicalNormalSpacing{wx, sy, inv_j, c11, c22, dq1, dq2});
        auto x = west.get_host_data(), y = south.get_host_data();
        auto xe = east.get_host_data(), yn = north.get_host_data();
        std::vector<Real> row(ny, real(0.0)), column(nx, real(0.0));
        std::vector<Real> row_offset(ny, real(0.0)), column_offset(nx, real(0.0)), row_total(ny),
            column_total(nx);
        for (int j = 0; j < ny; ++j) {
            for (int i = 0; i < nx; ++i) {
                row[j] += x(j + h, i + h);
                column[i] += y(j + h, i + h);
            }
        }
        int rank = 0, coords[2];
        MPI_Comm_rank(grid.get_cart_comm(), &rank);
        MPI_Cart_coords(grid.get_cart_comm(), rank, 2, coords);
#if defined(ENABLE_NCCL)
        // Gather padded row/column totals, carrying Cartesian coordinates so
        // communicator rank reordering and uneven decompositions are harmless.
        // MPI is used only to query topology; NCCL exchanges the numerical data.
        int dimensions[2], periods[2], cart_coords[2];
        MPI_Cart_get(grid.get_cart_comm(), 2, dimensions, periods, cart_coords);
        const int rows = (grid.get_global_points_y() + dimensions[0] - 1) / dimensions[0];
        const int columns = (grid.get_global_points_x() + dimensions[1] - 1) / dimensions[1];
        const size_t packet_size = 2 + rows + columns;
        std::vector<Real> packet(packet_size, real(0.0));
        packet[0] = Real(coords[0]);
        packet[1] = Real(coords[1]);
        std::copy(row.begin(), row.end(), packet.begin() + 2);
        std::copy(column.begin(), column.end(), packet.begin() + 2 + rows);
        const auto gathered = StartupCollectives::all_gather(state, packet);
        std::fill(row_total.begin(), row_total.end(), real(0.0));
        std::fill(column_total.begin(), column_total.end(), real(0.0));
        for (size_t offset = 0; offset < gathered.size(); offset += packet_size) {
            const int process_j = static_cast<int>(gathered[offset]);
            const int process_i = static_cast<int>(gathered[offset + 1]);
            if (process_j == coords[0]) {
                for (int j = 0; j < ny; ++j) {
                    const Real value = gathered[offset + 2 + j];
                    row_total[j] += value;
                    if (process_i < coords[1]) {
                        row_offset[j] += value;
                    }
                }
            }
            if (process_i == coords[1]) {
                for (int i = 0; i < nx; ++i) {
                    const Real value = gathered[offset + 2 + rows + i];
                    column_total[i] += value;
                    if (process_j < coords[0]) {
                        column_offset[i] += value;
                    }
                }
            }
        }
#else
        (void)state;
        MPI_Comm along_q1, along_q2;
        MPI_Comm_split(grid.get_cart_comm(), coords[0], coords[1], &along_q1);
        MPI_Comm_split(grid.get_cart_comm(), coords[1], coords[0], &along_q2);
        MPI_Exscan(row.data(), row_offset.data(), ny, VVM_MPI_REAL, MPI_SUM, along_q1);
        MPI_Exscan(column.data(), column_offset.data(), nx, VVM_MPI_REAL, MPI_SUM, along_q2);
        // MPI leaves rank-zero Exscan receive buffers undefined.
        if (coords[1] == 0) {
            std::fill(row_offset.begin(), row_offset.end(), real(0.0));
        }
        if (coords[0] == 0) {
            std::fill(column_offset.begin(), column_offset.end(), real(0.0));
        }
        MPI_Allreduce(row.data(), row_total.data(), ny, VVM_MPI_REAL, MPI_SUM, along_q1);
        MPI_Allreduce(column.data(), column_total.data(), nx, VVM_MPI_REAL, MPI_SUM, along_q2);
        MPI_Comm_free(&along_q1);
        MPI_Comm_free(&along_q2);
#endif
        for (int j = 0; j < ny; ++j) {
            Real prefix = row_offset[j];
            for (int i = 0; i < nx; ++i) {
                const Real width = x(j + h, i + h);
                x(j + h, i + h) = prefix + real(0.5) * width;
                xe(j + h, i + h) = row_total[j] - x(j + h, i + h);
                prefix += width;
            }
        }
        for (int i = 0; i < nx; ++i) {
            Real prefix = column_offset[i];
            for (int j = 0; j < ny; ++j) {
                const Real width = y(j + h, i + h);
                y(j + h, i + h) = prefix + real(0.5) * width;
                yn(j + h, i + h) = column_total[i] - y(j + h, i + h);
                prefix += width;
            }
        }
        Kokkos::deep_copy(wx, x);
        Kokkos::deep_copy(east.get_mutable_device_data(), xe);
        Kokkos::deep_copy(sy, y);
        Kokkos::deep_copy(north.get_mutable_device_data(), yn);
    }
};
}
#endif
