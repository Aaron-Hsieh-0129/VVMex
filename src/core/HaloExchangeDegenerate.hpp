#pragma once

#include "Field.hpp"

#include <Kokkos_Core.hpp>

namespace VVM::Core::Detail {

// A singleton horizontal axis represents a reduced-dimensional model. Its
// derivatives must see the sole physical plane on both sides, so every halo
// layer is filled from that plane without involving MPI or NCCL.
template<typename ExecSpace, typename FieldT>
void fill_periodic_x_halo(
    const ExecSpace& exec_space,
    FieldT& field,
    int halo_offset,
    int physical_size,
    int depth) {
    constexpr size_t Dim = FieldT::DimValue;
    auto data = field.get_mutable_device_data();

    if constexpr (Dim == 2) {
        const int ny = data.extent(0);
        Kokkos::parallel_for(
            "fill_singleton_x_2d",
            Kokkos::MDRangePolicy<Kokkos::Rank<2>, ExecSpace>(
                exec_space, {0, 0}, {ny, depth}),
            KOKKOS_LAMBDA(int j, int i_h) {
                data(j, halo_offset - depth + i_h) =
                    data(j, halo_offset + (physical_size - depth + i_h) % physical_size);
                data(j, halo_offset + physical_size + i_h) =
                    data(j, halo_offset + i_h % physical_size);
            });
    }
    else if constexpr (Dim == 3) {
        const int nz = data.extent(0);
        const int ny = data.extent(1);
        Kokkos::parallel_for(
            "fill_singleton_x_3d",
            Kokkos::MDRangePolicy<Kokkos::Rank<3>, ExecSpace>(
                exec_space, {0, 0, 0}, {nz, ny, depth}),
            KOKKOS_LAMBDA(int k, int j, int i_h) {
                data(k, j, halo_offset - depth + i_h) =
                    data(k, j, halo_offset + (physical_size - depth + i_h) % physical_size);
                data(k, j, halo_offset + physical_size + i_h) =
                    data(k, j, halo_offset + i_h % physical_size);
            });
    }
    else if constexpr (Dim == 4) {
        const int nw = data.extent(0);
        const int nz = data.extent(1);
        const int ny = data.extent(2);
        Kokkos::parallel_for(
            "fill_singleton_x_4d",
            Kokkos::MDRangePolicy<Kokkos::Rank<4>, ExecSpace>(
                exec_space, {0, 0, 0, 0}, {nw, nz, ny, depth}),
            KOKKOS_LAMBDA(int w, int k, int j, int i_h) {
                data(w, k, j, halo_offset - depth + i_h) =
                    data(w, k, j, halo_offset + (physical_size - depth + i_h) % physical_size);
                data(w, k, j, halo_offset + physical_size + i_h) =
                    data(w, k, j, halo_offset + i_h % physical_size);
            });
    }
}

template<typename ExecSpace, typename FieldT>
void fill_singleton_x_halo(
    const ExecSpace& exec_space,
    FieldT& field,
    int halo_offset,
    int depth) {
    fill_periodic_x_halo(exec_space, field, halo_offset, 1, depth);
}

template<typename ExecSpace, typename FieldT>
void fill_periodic_y_halo(
    const ExecSpace& exec_space,
    FieldT& field,
    int halo_offset,
    int physical_size,
    int depth) {
    constexpr size_t Dim = FieldT::DimValue;
    auto data = field.get_mutable_device_data();

    if constexpr (Dim == 2) {
        const int nx = data.extent(1);
        Kokkos::parallel_for(
            "fill_singleton_y_2d",
            Kokkos::MDRangePolicy<Kokkos::Rank<2>, ExecSpace>(
                exec_space, {0, 0}, {nx, depth}),
            KOKKOS_LAMBDA(int i, int j_h) {
                data(halo_offset - depth + j_h, i) =
                    data(halo_offset + (physical_size - depth + j_h) % physical_size, i);
                data(halo_offset + physical_size + j_h, i) =
                    data(halo_offset + j_h % physical_size, i);
            });
    }
    else if constexpr (Dim == 3) {
        const int nz = data.extent(0);
        const int nx = data.extent(2);
        Kokkos::parallel_for(
            "fill_singleton_y_3d",
            Kokkos::MDRangePolicy<Kokkos::Rank<3>, ExecSpace>(
                exec_space, {0, 0, 0}, {nz, nx, depth}),
            KOKKOS_LAMBDA(int k, int i, int j_h) {
                data(k, halo_offset - depth + j_h, i) =
                    data(k, halo_offset + (physical_size - depth + j_h) % physical_size, i);
                data(k, halo_offset + physical_size + j_h, i) =
                    data(k, halo_offset + j_h % physical_size, i);
            });
    }
    else if constexpr (Dim == 4) {
        const int nw = data.extent(0);
        const int nz = data.extent(1);
        const int nx = data.extent(3);
        Kokkos::parallel_for(
            "fill_singleton_y_4d",
            Kokkos::MDRangePolicy<Kokkos::Rank<4>, ExecSpace>(
                exec_space, {0, 0, 0, 0}, {nw, nz, nx, depth}),
            KOKKOS_LAMBDA(int w, int k, int i, int j_h) {
                data(w, k, halo_offset - depth + j_h, i) =
                    data(w, k, halo_offset + (physical_size - depth + j_h) % physical_size, i);
                data(w, k, halo_offset + physical_size + j_h, i) =
                    data(w, k, halo_offset + j_h % physical_size, i);
            });
    }
}

template<typename ExecSpace, typename FieldT>
void fill_singleton_y_halo(
    const ExecSpace& exec_space,
    FieldT& field,
    int halo_offset,
    int depth) {
    fill_periodic_y_halo(exec_space, field, halo_offset, 1, depth);
}

template<typename ExecSpace>
void fill_periodic_x_slice(
    const ExecSpace& exec_space,
    Field<3>& field,
    int k_layer,
    int halo_offset,
    int physical_size,
    int depth) {
    auto data = field.get_mutable_device_data();
    const int ny = data.extent(1);
    Kokkos::parallel_for(
        "fill_singleton_x_slice",
        Kokkos::MDRangePolicy<Kokkos::Rank<2>, ExecSpace>(
            exec_space, {0, 0}, {ny, depth}),
        KOKKOS_LAMBDA(int j, int i_h) {
            data(k_layer, j, halo_offset - depth + i_h) =
                data(k_layer, j, halo_offset + (physical_size - depth + i_h) % physical_size);
            data(k_layer, j, halo_offset + physical_size + i_h) =
                data(k_layer, j, halo_offset + i_h % physical_size);
        });
}

template<typename ExecSpace>
void fill_singleton_x_slice(
    const ExecSpace& exec_space,
    Field<3>& field,
    int k_layer,
    int halo_offset,
    int depth) {
    fill_periodic_x_slice(
        exec_space, field, k_layer, halo_offset, 1, depth);
}

template<typename ExecSpace>
void fill_periodic_y_slice(
    const ExecSpace& exec_space,
    Field<3>& field,
    int k_layer,
    int halo_offset,
    int physical_size,
    int depth) {
    auto data = field.get_mutable_device_data();
    const int nx = data.extent(2);
    Kokkos::parallel_for(
        "fill_singleton_y_slice",
        Kokkos::MDRangePolicy<Kokkos::Rank<2>, ExecSpace>(
            exec_space, {0, 0}, {nx, depth}),
        KOKKOS_LAMBDA(int i, int j_h) {
            data(k_layer, halo_offset - depth + j_h, i) =
                data(k_layer, halo_offset + (physical_size - depth + j_h) % physical_size, i);
            data(k_layer, halo_offset + physical_size + j_h, i) =
                data(k_layer, halo_offset + j_h % physical_size, i);
        });
}

template<typename ExecSpace>
void fill_singleton_y_slice(
    const ExecSpace& exec_space,
    Field<3>& field,
    int k_layer,
    int halo_offset,
    int depth) {
    fill_periodic_y_slice(
        exec_space, field, k_layer, halo_offset, 1, depth);
}

} // namespace VVM::Core::Detail
