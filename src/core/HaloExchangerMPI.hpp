#pragma once

#include "Grid.hpp"
#include "HaloExchangeDegenerate.hpp"
#include "Field.hpp"
#include "State.hpp"
#include "vvm_types.hpp"
#include <algorithm>
#include <vector>
#include <Kokkos_Core.hpp>

#include <stdexcept>
#include <string>

namespace VVM {
namespace Core {

enum class HaloExchangeTags {
    SEND_TO_RIGHT = 10, // X-direction
    SEND_TO_LEFT  = 11, // X-direction
    SEND_TO_TOP    = 20, // Y-direction
    SEND_TO_BOTTOM = 21,  // Y-direction
    SLICE_SEND_TO_RIGHT = 30,
    SLICE_SEND_TO_LEFT = 31,
    SLICE_SEND_TO_TOP = 40,
    SLICE_SEND_TO_BOTTOM = 41
};

// Struct to hold MPI requests for non-blocking communication
struct HaloExchangeRequests {
    std::vector<MPI_Request> requests;
    int count = 0;
};


class HaloExchanger {
public:
    explicit HaloExchanger(const Grid& grid);

    void exchange_halos(State& state) const;

    template<typename FieldT>
    void exchange_halos(FieldT& field, int depth = -1) const {
        static constexpr size_t Dim = FieldT::DimValue;
        if constexpr (Dim >= 2) {
            auto reqs_y = post_exchange_halo_y(field, depth);
            wait_exchange_halo_y(field, reqs_y, depth);

            auto reqs_x = post_exchange_halo_x(field, depth);
            wait_exchange_halo_x(field, reqs_x, depth);
        }
    }

    // Same name-based batched exchange the NCCL implementation offers; the
    // definition follows below.
    void exchange_multiple_halos(const std::vector<std::string>& field_names, State& state) const;

    void exchange_multiple_halos(const std::vector<Field<3>*>& fields) const;

    // Interface parity with the NCCL implementation. Not batched here: without NCCL
    // groups there is nothing to gain from packing the fields together.
    void exchange_multiple_halos(const std::vector<Field<2>*>& fields, int depth = -1) const {
        for (Field<2>* field : fields) {
            if (field) exchange_halos(*field, depth);
        }
    }

    // --- Asynchronous Halo Exchange Functions ---
    template<typename FieldT>
    HaloExchangeRequests post_exchange_halo_x(FieldT& field, int depth = -1) const;

    template<typename FieldT>
    void wait_exchange_halo_x(FieldT& field, HaloExchangeRequests& reqs, int depth = -1) const;

    template<typename FieldT>
    HaloExchangeRequests post_exchange_halo_y(FieldT& field, int depth = -1) const;

    template<typename FieldT>
    void wait_exchange_halo_y(FieldT& field, HaloExchangeRequests& reqs, int depth = -1) const;

    void exchange_halos_slice(Field<3>& field, int k_layer) const;
    void exchange_halos_top_slice(Field<3>& field) const {
        const int nz = grid_ref_.get_local_total_points_z();
        const int h = grid_ref_.get_halo_cells();
        exchange_halos_slice(field, nz - h - 1);
    }

private:
    mutable std::vector<Field<3>*> batch_fields_;

    const Grid& grid_ref_;
    MPI_Comm cart_comm_;
    int neighbor_left_, neighbor_right_;
    int neighbor_bottom_, neighbor_top_;

    mutable Kokkos::View<VVM::Real*> send_x_left_, recv_x_left_;
    mutable Kokkos::View<VVM::Real*> send_x_right_, recv_x_right_;
    mutable Kokkos::View<VVM::Real*> send_y_bottom_, recv_y_bottom_;
    mutable Kokkos::View<VVM::Real*> send_y_top_, recv_y_top_;

    // mutable Kokkos::View<VVM::Real*>::HostMirror send_x_left_h_, recv_x_left_h_;
    // mutable Kokkos::View<VVM::Real*>::HostMirror send_x_right_h_, recv_x_right_h_;
    // mutable Kokkos::View<VVM::Real*>::HostMirror send_y_bottom_h_, recv_y_bottom_h_;
    // mutable Kokkos::View<VVM::Real*>::HostMirror send_y_top_h_, recv_y_top_h_;

    size_t buffer_size_x_2d_, buffer_size_y_2d_;
    size_t buffer_size_x_3d_, buffer_size_y_3d_;
    size_t buffer_size_x_4d_, buffer_size_y_4d_;
    size_t buffer_size_slice_x_, buffer_size_slice_y_;
};


inline HaloExchanger::HaloExchanger(const Grid& grid)
    : grid_ref_(grid), cart_comm_(grid.get_cart_comm()) {
    int rank;
    MPI_Comm_rank(grid_ref_.get_comm(), &rank);

    if (cart_comm_ == MPI_COMM_NULL) {
        std::cerr << "Rank " << rank << ": HaloExchanger initialized with NULL communicator!" << std::endl;
        MPI_Abort(grid_ref_.get_comm(), -1);
    }
    // Dim 1 is X (left-right), Dim 0 is y (up-down)
    MPI_Cart_shift(cart_comm_, 1, 1, &neighbor_left_, &neighbor_right_);
    MPI_Cart_shift(cart_comm_, 0, 1, &neighbor_bottom_, &neighbor_top_);

    const int h = grid_ref_.get_halo_cells();
    if (h > 0) {
        const int nx_total = grid.get_local_total_points_x();
        const int ny_total = grid.get_local_total_points_y();
        const int nz_total = grid.get_local_total_points_z();
        const int nw_dummy = 16; // Pre-sized to avoid resize after CUDA graph capture

        buffer_size_x_2d_ = static_cast<size_t>(h) * ny_total;
        buffer_size_y_2d_ = static_cast<size_t>(h) * nx_total;
        buffer_size_x_3d_ = static_cast<size_t>(h) * ny_total * nz_total;
        buffer_size_y_3d_ = static_cast<size_t>(h) * nx_total * nz_total;
        buffer_size_x_4d_ = static_cast<size_t>(h) * ny_total * nz_total * nw_dummy;
        buffer_size_y_4d_ = static_cast<size_t>(h) * nx_total * nz_total * nw_dummy;

        buffer_size_slice_x_ = static_cast<size_t>(h) * ny_total;
        buffer_size_slice_y_ = static_cast<size_t>(h) * nx_total;

        size_t max_buffer_size_x = std::max({buffer_size_x_2d_, buffer_size_x_3d_, buffer_size_x_4d_, buffer_size_slice_x_});
        size_t max_buffer_size_y = std::max({buffer_size_y_2d_, buffer_size_y_3d_, buffer_size_y_4d_, buffer_size_slice_y_});

        if (max_buffer_size_x > 0) {
            send_x_left_  = Kokkos::View<VVM::Real*>("send_x_left_buf", max_buffer_size_x);
            recv_x_left_  = Kokkos::View<VVM::Real*>("recv_x_left_buf", max_buffer_size_x);
            send_x_right_ = Kokkos::View<VVM::Real*>("send_x_right_buf", max_buffer_size_x);
            recv_x_right_ = Kokkos::View<VVM::Real*>("recv_x_right_buf", max_buffer_size_x);

            // send_x_left_h_  = Kokkos::create_mirror_view(send_x_left_);
            // recv_x_left_h_  = Kokkos::create_mirror_view(recv_x_left_);
            // send_x_right_h_ = Kokkos::create_mirror_view(send_x_right_);
            // recv_x_right_h_ = Kokkos::create_mirror_view(recv_x_right_);
        }
        if (max_buffer_size_y > 0) {
            send_y_bottom_ = Kokkos::View<VVM::Real*>("send_y_bottom_buf", max_buffer_size_y);
            recv_y_bottom_ = Kokkos::View<VVM::Real*>("recv_y_bottom_buf", max_buffer_size_y);
            send_y_top_    = Kokkos::View<VVM::Real*>("send_y_top_buf", max_buffer_size_y);
            recv_y_top_    = Kokkos::View<VVM::Real*>("recv_y_top_buf", max_buffer_size_y);

            // send_y_bottom_h_ = Kokkos::create_mirror_view(send_y_bottom_);
            // recv_y_bottom_h_ = Kokkos::create_mirror_view(recv_y_bottom_);
            // send_y_top_h_    = Kokkos::create_mirror_view(send_y_top_);
            // recv_y_top_h_    = Kokkos::create_mirror_view(recv_y_top_);
        }
    }
}

inline void HaloExchanger::exchange_halos(State& state) const {
    for (auto& field_pair : state) {
        std::visit([this](auto& field) {
            using T = std::decay_t<decltype(field)>;
            if constexpr (!std::is_same_v<T, std::monostate>) {
                this->exchange_halos(field);
            }
        }, field_pair.second);
    }
}

template<typename FieldT>
HaloExchangeRequests HaloExchanger::post_exchange_halo_x(FieldT& field, int depth) const {
    constexpr size_t Dim = FieldT::DimValue;
    const int halo_start_offset = grid_ref_.get_halo_cells();
    // depth == -1 means the whole halo, exactly as in the NCCL implementation.
    // Resolving it to a single layer leaves the outer halo stale, which shows up
    // as a slow blow-up rather than as an obvious failure.
    const int h = (depth == -1) ? grid_ref_.get_halo_cells() : depth;

    if (h == 0) return {};
    if (grid_ref_.is_singleton_x()) {
        Detail::fill_singleton_x_halo(
            Kokkos::DefaultExecutionSpace{}, field, halo_start_offset, h);
        return {};
    }

    auto data = field.get_mutable_device_data();
    const int nx_phys = grid_ref_.get_local_physical_points_x();

    size_t count = 0;
    if constexpr (Dim == 2) count = static_cast<size_t>(h) * data.extent(0);
    else if constexpr (Dim == 3) count = static_cast<size_t>(h) * data.extent(1) * data.extent(0);
    else if constexpr (Dim == 4) count = static_cast<size_t>(h) * data.extent(2) * data.extent(1) * data.extent(0);

    if (count == 0) return {};

    auto send_l = Kokkos::subview(send_x_left_, std::make_pair((size_t)0, count));
    auto recv_l = Kokkos::subview(recv_x_left_, std::make_pair((size_t)0, count));
    auto send_r = Kokkos::subview(send_x_right_, std::make_pair((size_t)0, count));
    auto recv_r = Kokkos::subview(recv_x_right_, std::make_pair((size_t)0, count));

    // auto send_l_h = Kokkos::subview(send_x_left_h_, std::make_pair((size_t)0, count));
    // auto recv_l_h = Kokkos::subview(recv_x_left_h_, std::make_pair((size_t)0, count));
    // auto send_r_h = Kokkos::subview(send_x_right_h_, std::make_pair((size_t)0, count));
    // auto recv_r_h = Kokkos::subview(recv_x_right_h_, std::make_pair((size_t)0, count));

    // Pack data
    if constexpr (Dim == 2) {
        const int ny = data.extent(0);
        Kokkos::parallel_for("pack_x_2d", Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0,0}, {ny, h}),
            KOKKOS_LAMBDA(int j, int i_h) {
                send_l(j * h + i_h) = data(j, halo_start_offset + i_h);
                send_r(j * h + i_h) = data(j, halo_start_offset + nx_phys - h + i_h);
        });
    }
    else if constexpr (Dim == 3) {
        const int nz = data.extent(0);
        const int ny = data.extent(1);
        Kokkos::parallel_for("pack_x_3d", Kokkos::MDRangePolicy<Kokkos::Rank<3>>({0,0,0}, {nz, ny, h}),
            KOKKOS_LAMBDA(int k, int j, int i_h) {
                const size_t idx = k * (ny * h) + j * h + i_h;
                send_l(idx) = data(k, j, halo_start_offset + i_h);
                send_r(idx) = data(k, j, halo_start_offset + nx_phys - h + i_h);
        });
    }
    else if constexpr (Dim == 4) {
        const int nw = data.extent(0);
        const int nz = data.extent(1);
        const int ny = data.extent(2);
        Kokkos::parallel_for("pack_x_4d", Kokkos::MDRangePolicy<Kokkos::Rank<4>>({0,0,0,0}, {nw, nz, ny, h}),
            KOKKOS_LAMBDA(int w, int k, int j, int i_h) {
                size_t idx = w * (nz*ny*h) + k * (ny*h) + j * h + i_h;
                send_l(idx) = data(w, k, j, halo_start_offset + i_h);
                send_r(idx) = data(w, k, j, halo_start_offset + nx_phys - h + i_h);
        });
    }
    // The pack kernels are asynchronous. CUDA-aware MPI reads these device
    // buffers from its own engine, so without this fence it can send whatever
    // the buffer held before the pack finished -- which showed up as run-to-run
    // nondeterminism and, over a few hundred steps, as a blow-up.
    Kokkos::fence();

    // Kokkos::deep_copy(send_l_h, send_l);
    // Kokkos::deep_copy(send_r_h, send_r);

    HaloExchangeRequests req_obj;
    req_obj.requests.resize(4);

    if(neighbor_right_ != MPI_PROC_NULL) MPI_Irecv(recv_r.data(), count, VVM_MPI_REAL, neighbor_right_, static_cast<int>(HaloExchangeTags::SEND_TO_LEFT), cart_comm_, &req_obj.requests[req_obj.count++]);
    if(neighbor_left_  != MPI_PROC_NULL) MPI_Irecv(recv_l.data(), count, VVM_MPI_REAL, neighbor_left_, static_cast<int>(HaloExchangeTags::SEND_TO_RIGHT), cart_comm_, &req_obj.requests[req_obj.count++]);
    if(neighbor_left_  != MPI_PROC_NULL) MPI_Isend(send_l.data(), count, VVM_MPI_REAL, neighbor_left_, static_cast<int>(HaloExchangeTags::SEND_TO_LEFT), cart_comm_, &req_obj.requests[req_obj.count++]);
    if(neighbor_right_ != MPI_PROC_NULL) MPI_Isend(send_r.data(), count, VVM_MPI_REAL, neighbor_right_, static_cast<int>(HaloExchangeTags::SEND_TO_RIGHT), cart_comm_, &req_obj.requests[req_obj.count++]);

    // if(neighbor_right_ != MPI_PROC_NULL) MPI_Irecv(recv_r_h.data(), count, VVM_MPI_REAL, neighbor_right_, static_cast<int>(HaloExchangeTags::SEND_TO_LEFT), cart_comm_, &req_obj.requests[req_obj.count++]);
    // if(neighbor_left_  != MPI_PROC_NULL) MPI_Irecv(recv_l_h.data(), count, VVM_MPI_REAL, neighbor_left_, static_cast<int>(HaloExchangeTags::SEND_TO_RIGHT), cart_comm_, &req_obj.requests[req_obj.count++]);
    // if(neighbor_left_  != MPI_PROC_NULL) MPI_Isend(send_l_h.data(), count, VVM_MPI_REAL, neighbor_left_, static_cast<int>(HaloExchangeTags::SEND_TO_LEFT), cart_comm_, &req_obj.requests[req_obj.count++]);
    // if(neighbor_right_ != MPI_PROC_NULL) MPI_Isend(send_r_h.data(), count, VVM_MPI_REAL, neighbor_right_, static_cast<int>(HaloExchangeTags::SEND_TO_RIGHT), cart_comm_, &req_obj.requests[req_obj.count++]);

    return req_obj;
}

template<typename FieldT>
void HaloExchanger::wait_exchange_halo_x(FieldT& field, HaloExchangeRequests& reqs, int depth) const {
    constexpr size_t Dim = FieldT::DimValue;
    if (reqs.count == 0) return;

    MPI_Waitall(reqs.count, reqs.requests.data(), MPI_STATUSES_IGNORE);
    // Kokkos::fence();

    const int halo_start_offset = grid_ref_.get_halo_cells();
    const int h = (depth == -1) ? grid_ref_.get_halo_cells() : depth;

    auto data = field.get_mutable_device_data();
    const int nx_phys = grid_ref_.get_local_physical_points_x();

    size_t count = 0;
    if constexpr (Dim == 2) count = static_cast<size_t>(h) * data.extent(0);
    else if constexpr (Dim == 3) count = static_cast<size_t>(h) * data.extent(1) * data.extent(0);
    else if constexpr (Dim == 4) count = static_cast<size_t>(h) * data.extent(2) * data.extent(1) * data.extent(0);

    auto recv_l = Kokkos::subview(recv_x_left_, std::make_pair((size_t)0, count));
    auto recv_r = Kokkos::subview(recv_x_right_, std::make_pair((size_t)0, count));

    // auto recv_l_h = Kokkos::subview(recv_x_left_h_, std::make_pair((size_t)0, count));
    // auto recv_r_h = Kokkos::subview(recv_x_right_h_, std::make_pair((size_t)0, count));

    // Kokkos::deep_copy(recv_l, recv_l_h);
    // Kokkos::deep_copy(recv_r, recv_r_h);
    // Kokkos::fence();

    // Unpack data
    const int neighbor_left = neighbor_left_;
    const int neighbor_right = neighbor_right_;
    if constexpr (Dim == 2) {
        const int ny = data.extent(0);
        Kokkos::parallel_for("unpack_x_2d", Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0,0}, {ny, h}),
            KOKKOS_LAMBDA(int j, int i_h) {
                if (neighbor_left != MPI_PROC_NULL) data(j, halo_start_offset - h + i_h) = recv_l(j * h + i_h);
                if (neighbor_right != MPI_PROC_NULL) data(j, halo_start_offset + nx_phys + i_h) = recv_r(j * h + i_h);
        });
    }
    else if constexpr (Dim == 3) {
        const int nz = data.extent(0);
        const int ny = data.extent(1);
        Kokkos::parallel_for("unpack_x_3d", Kokkos::MDRangePolicy<Kokkos::Rank<3>>({0,0,0}, {nz, ny, h}),
            KOKKOS_LAMBDA(int k, int j, int i_h) {
                const size_t idx = k * (ny * h) + j * h + i_h;
                if (neighbor_left != MPI_PROC_NULL) data(k, j, halo_start_offset - h + i_h) = recv_l(idx);
                if (neighbor_right != MPI_PROC_NULL) data(k, j, halo_start_offset + nx_phys + i_h) = recv_r(idx);
        });
    }
    else if constexpr (Dim == 4) {
        const int nw = data.extent(0);
        const int nz = data.extent(1);
        const int ny = data.extent(2);
        Kokkos::parallel_for("unpack_x_4d", Kokkos::MDRangePolicy<Kokkos::Rank<4>>({0,0,0,0}, {nw, nz, ny, h}),
            KOKKOS_LAMBDA(int w, int k, int j, int i_h) {
                size_t idx = w * (nz*ny*h) + k * (ny*h) + j * h + i_h;
                if (neighbor_left != MPI_PROC_NULL) data(w, k, j, halo_start_offset - h + i_h) = recv_l(idx);
                if (neighbor_right != MPI_PROC_NULL) data(w, k, j, halo_start_offset + nx_phys + i_h) = recv_r(idx);
        });
    }
}

template<typename FieldT>
HaloExchangeRequests HaloExchanger::post_exchange_halo_y(FieldT& field, int depth) const {
    constexpr size_t Dim = FieldT::DimValue;
    const int halo_start_offset = grid_ref_.get_halo_cells();
    const int h = (depth == -1) ? grid_ref_.get_halo_cells() : depth;
    if (h == 0) return {};
    if (grid_ref_.is_singleton_y()) {
        Detail::fill_singleton_y_halo(
            Kokkos::DefaultExecutionSpace{}, field, halo_start_offset, h);
        return {};
    }

    auto data = field.get_mutable_device_data();
    const int ny_phys = grid_ref_.get_local_physical_points_y();

    size_t count = 0;
    if constexpr (Dim == 2) count = static_cast<size_t>(h) * data.extent(1);
    else if constexpr (Dim == 3) count = static_cast<size_t>(h) * data.extent(2) * data.extent(0);
    else if constexpr (Dim == 4) count = static_cast<size_t>(h) * data.extent(3) * data.extent(1) * data.extent(0);

    if (count == 0) return {};

    auto send_b = Kokkos::subview(send_y_bottom_, std::make_pair((size_t)0, count));
    auto recv_b = Kokkos::subview(recv_y_bottom_, std::make_pair((size_t)0, count));
    auto send_t = Kokkos::subview(send_y_top_, std::make_pair((size_t)0, count));
    auto recv_t = Kokkos::subview(recv_y_top_, std::make_pair((size_t)0, count));

    // auto send_b_h = Kokkos::subview(send_y_bottom_h_, std::make_pair((size_t)0, count));
    // auto recv_b_h = Kokkos::subview(recv_y_bottom_h_, std::make_pair((size_t)0, count));
    // auto send_t_h = Kokkos::subview(send_y_top_h_, std::make_pair((size_t)0, count));
    // auto recv_t_h = Kokkos::subview(recv_y_top_h_, std::make_pair((size_t)0, count));

    // Pack data
    if constexpr (Dim == 2) {
        const int nx = data.extent(1);
        Kokkos::parallel_for("pack_y_2d", Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0,0}, {nx, h}),
            KOKKOS_LAMBDA(int i, int j_h) {
                const size_t idx = j_h * nx + i;
                send_b(idx) = data(halo_start_offset + j_h, i);
                send_t(idx) = data(halo_start_offset + ny_phys - h + j_h, i);
        });
    }
    else if constexpr (Dim == 3) {
        const int nz = data.extent(0);
        const int nx = data.extent(2);
        Kokkos::parallel_for("pack_y_3d", Kokkos::MDRangePolicy<Kokkos::Rank<3>>({0,0,0}, {nz, nx, h}),
            KOKKOS_LAMBDA(int k, int i, int j_h) {
                const size_t idx = k * (h * nx) + j_h * nx + i;
                send_b(idx) = data(k, halo_start_offset + j_h, i);
                send_t(idx) = data(k, halo_start_offset + ny_phys - h + j_h, i);
        });
    }
    else if constexpr (Dim == 4) {
        const int nw = data.extent(0);
        const int nz = data.extent(1);
        const int nx = data.extent(3);
        Kokkos::parallel_for("pack_y_4d", Kokkos::MDRangePolicy<Kokkos::Rank<4>>({0,0,0,0}, {nw, nz, nx, h}),
            KOKKOS_LAMBDA(int w, int k, int i, int j_h) {
                size_t idx = w * (nz*h*nx) + k * (h*nx) + j_h * nx + i;
                send_b(idx) = data(w, k, halo_start_offset + j_h, i);
                send_t(idx) = data(w, k, halo_start_offset + ny_phys - h + j_h, i);
        });
    }
    // The pack kernels are asynchronous. CUDA-aware MPI reads these device
    // buffers from its own engine, so without this fence it can send whatever
    // the buffer held before the pack finished -- which showed up as run-to-run
    // nondeterminism and, over a few hundred steps, as a blow-up.
    Kokkos::fence();

    // Kokkos::deep_copy(send_b_h, send_b);
    // Kokkos::deep_copy(send_t_h, send_t);

    HaloExchangeRequests req_obj;
    req_obj.requests.resize(4);

    if(neighbor_bottom_ != MPI_PROC_NULL) MPI_Irecv(recv_b.data(), count, VVM_MPI_REAL, neighbor_bottom_, static_cast<int>(HaloExchangeTags::SEND_TO_TOP), cart_comm_, &req_obj.requests[req_obj.count++]);
    if(neighbor_top_    != MPI_PROC_NULL) MPI_Irecv(recv_t.data(), count, VVM_MPI_REAL, neighbor_top_, static_cast<int>(HaloExchangeTags::SEND_TO_BOTTOM), cart_comm_, &req_obj.requests[req_obj.count++]);
    if(neighbor_top_    != MPI_PROC_NULL) MPI_Isend(send_t.data(), count, VVM_MPI_REAL, neighbor_top_, static_cast<int>(HaloExchangeTags::SEND_TO_TOP), cart_comm_, &req_obj.requests[req_obj.count++]);
    if(neighbor_bottom_ != MPI_PROC_NULL) MPI_Isend(send_b.data(), count, VVM_MPI_REAL, neighbor_bottom_, static_cast<int>(HaloExchangeTags::SEND_TO_BOTTOM), cart_comm_, &req_obj.requests[req_obj.count++]);

    // if(neighbor_bottom_ != MPI_PROC_NULL) MPI_Irecv(recv_b_h.data(), count, VVM_MPI_REAL, neighbor_bottom_, static_cast<int>(HaloExchangeTags::SEND_TO_TOP), cart_comm_, &req_obj.requests[req_obj.count++]);
    // if(neighbor_top_    != MPI_PROC_NULL) MPI_Irecv(recv_t_h.data(), count, VVM_MPI_REAL, neighbor_top_, static_cast<int>(HaloExchangeTags::SEND_TO_BOTTOM), cart_comm_, &req_obj.requests[req_obj.count++]);
    // if(neighbor_top_    != MPI_PROC_NULL) MPI_Isend(send_t_h.data(), count, VVM_MPI_REAL, neighbor_top_, static_cast<int>(HaloExchangeTags::SEND_TO_TOP), cart_comm_, &req_obj.requests[req_obj.count++]);
    // if(neighbor_bottom_ != MPI_PROC_NULL) MPI_Isend(send_b_h.data(), count, VVM_MPI_REAL, neighbor_bottom_, static_cast<int>(HaloExchangeTags::SEND_TO_BOTTOM), cart_comm_, &req_obj.requests[req_obj.count++]);

    return req_obj;
}

template<typename FieldT>
void HaloExchanger::wait_exchange_halo_y(FieldT& field, HaloExchangeRequests& reqs, int depth) const {
    constexpr size_t Dim = FieldT::DimValue;
    if (reqs.count == 0) return;

    MPI_Waitall(reqs.count, reqs.requests.data(), MPI_STATUSES_IGNORE);
    // Kokkos::fence();

    const int halo_start_offset = grid_ref_.get_halo_cells();
    const int h = (depth == -1) ? grid_ref_.get_halo_cells() : depth;

    auto data = field.get_mutable_device_data();
    const int ny_phys = grid_ref_.get_local_physical_points_y();

    // Must match what post_exchange_halo_y() actually sent for this depth.
    size_t count = 0;
    if constexpr (Dim == 2) count = static_cast<size_t>(h) * data.extent(1);
    else if constexpr (Dim == 3) count = static_cast<size_t>(h) * data.extent(2) * data.extent(0);
    else if constexpr (Dim == 4) count = static_cast<size_t>(h) * data.extent(3) * data.extent(1) * data.extent(0);

    auto recv_b = Kokkos::subview(recv_y_bottom_, std::make_pair((size_t)0, count));
    auto recv_t = Kokkos::subview(recv_y_top_, std::make_pair((size_t)0, count));

    // auto recv_b_h = Kokkos::subview(recv_y_bottom_h_, std::make_pair((size_t)0, count));
    // auto recv_t_h = Kokkos::subview(recv_y_top_h_, std::make_pair((size_t)0, count));

    // Kokkos::deep_copy(recv_b, recv_b_h);
    // Kokkos::deep_copy(recv_t, recv_t_h);
    // Kokkos::fence();

    // Unpack data
    const int neighbor_bottom = neighbor_bottom_;
    const int neighbor_top = neighbor_top_;
    if constexpr (Dim == 2) {
        const int nx = data.extent(1);
        Kokkos::parallel_for("unpack_y_2d", Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0,0}, {nx, h}),
            KOKKOS_LAMBDA(int i, int j_h) {
                const size_t idx = j_h * nx + i;
                if (neighbor_bottom != MPI_PROC_NULL) data(halo_start_offset - h + j_h, i) = recv_b(idx);
                if (neighbor_top != MPI_PROC_NULL) data(halo_start_offset + ny_phys + j_h, i) = recv_t(idx);
        });
    }
    else if constexpr (Dim == 3) {
        const int nz = data.extent(0);
        const int nx = data.extent(2);
        Kokkos::parallel_for("unpack_y_3d", Kokkos::MDRangePolicy<Kokkos::Rank<3>>({0,0,0}, {nz, nx, h}),
            KOKKOS_LAMBDA(int k, int i, int j_h) {
                const size_t idx = k * (h * nx) + j_h * nx + i;
                if (neighbor_bottom != MPI_PROC_NULL) data(k, halo_start_offset - h + j_h, i) = recv_b(idx);
                if (neighbor_top != MPI_PROC_NULL) data(k, halo_start_offset + ny_phys + j_h, i) = recv_t(idx);
        });
    }
    else if constexpr (Dim == 4) {
        const int nw = data.extent(0);
        const int nz = data.extent(1);
        const int nx = data.extent(3);
        Kokkos::parallel_for("unpack_y_4d", Kokkos::MDRangePolicy<Kokkos::Rank<4>>({0,0,0,0}, {nw, nz, nx, h}),
            KOKKOS_LAMBDA(int w, int k, int i, int j_h) {
                size_t idx = w * (nz*h*nx) + k * (h*nx) + j_h * nx + i;
                if (neighbor_bottom != MPI_PROC_NULL) data(w, k, halo_start_offset - h + j_h, i) = recv_b(idx);
                if (neighbor_top != MPI_PROC_NULL) data(w, k, halo_start_offset + ny_phys + j_h, i) = recv_t(idx);
        });
    }
}

inline void HaloExchanger::exchange_multiple_halos(const std::vector<std::string>& field_names, State& state) const {
    batch_fields_.clear();
    batch_fields_.reserve(field_names.size());
    for (const auto& field_name : field_names) {
        batch_fields_.push_back(&state.get_field<3>(field_name));
    }
    exchange_multiple_halos(batch_fields_);
}

inline void HaloExchanger::exchange_multiple_halos(const std::vector<Field<3>*>& fields) const {
    if (fields.empty()) return;
    const int h = grid_ref_.get_halo_cells();
    if (h == 0) return;

    size_t num_fields = fields.size();
    size_t count_x_total = num_fields * buffer_size_x_3d_;
    size_t count_y_total = num_fields * buffer_size_y_3d_;

    if (send_x_left_.extent(0) < count_x_total) {
        Kokkos::resize(send_x_left_, count_x_total); Kokkos::resize(recv_x_left_, count_x_total);
        Kokkos::resize(send_x_right_, count_x_total); Kokkos::resize(recv_x_right_, count_x_total);
    }
    if (send_y_bottom_.extent(0) < count_y_total) {
        Kokkos::resize(send_y_bottom_, count_y_total); Kokkos::resize(recv_y_bottom_, count_y_total);
        Kokkos::resize(send_y_top_, count_y_total); Kokkos::resize(recv_y_top_, count_y_total);
    }

    const int nx_phys = grid_ref_.get_local_physical_points_x();
    const int ny_phys = grid_ref_.get_local_physical_points_y();
    const int nz = grid_ref_.get_local_total_points_z();
    const int ny = grid_ref_.get_local_total_points_y();
    const int nx = grid_ref_.get_local_total_points_x();
    const int halo_start_offset = h;

    // A reduced axis has no remote neighbor and no distinct boundary planes to
    // pack. Replicate its sole physical plane locally before exchanging the
    // remaining active direction, which also produces correct corner halos.
    if (grid_ref_.is_singleton_y()) {
        for (Field<3>* field : fields) {
            if (field) {
                Detail::fill_singleton_y_halo(
                    Kokkos::DefaultExecutionSpace{}, *field, halo_start_offset, h);
            }
        }
    }
    if (grid_ref_.is_singleton_x()) {
        for (Field<3>* field : fields) {
            if (field) {
                Detail::fill_singleton_x_halo(
                    Kokkos::DefaultExecutionSpace{}, *field, halo_start_offset, h);
            }
        }
    }

    const int neighbor_left = neighbor_left_;
    const int neighbor_right = neighbor_right_;
    const int neighbor_bottom = neighbor_bottom_;
    const int neighbor_top = neighbor_top_;

    // --- X Direction ---
    if (count_x_total > 0 && !grid_ref_.is_singleton_x()) {
        for (size_t f = 0; f < num_fields; ++f) {
            auto data = fields[f]->get_mutable_device_data();
            size_t offset = f * buffer_size_x_3d_;
            auto send_l = Kokkos::subview(send_x_left_, std::make_pair(offset, offset + buffer_size_x_3d_));
            auto send_r = Kokkos::subview(send_x_right_, std::make_pair(offset, offset + buffer_size_x_3d_));

            Kokkos::parallel_for("pack_multi_x", Kokkos::MDRangePolicy<Kokkos::Rank<3>>({0,0,0}, {nz, ny, h}),
                KOKKOS_LAMBDA(int k, int j, int i_h) {
                    const size_t idx = k * (ny * h) + j * h + i_h;
                    send_l(idx) = data(k, j, halo_start_offset + i_h);
                    send_r(idx) = data(k, j, halo_start_offset + nx_phys - h + i_h);
            });
        }
        Kokkos::fence();

        HaloExchangeRequests req_obj;
        req_obj.requests.resize(4);
        auto recv_l = Kokkos::subview(recv_x_left_, std::make_pair((size_t)0, count_x_total));
        auto recv_r = Kokkos::subview(recv_x_right_, std::make_pair((size_t)0, count_x_total));
        auto send_l = Kokkos::subview(send_x_left_, std::make_pair((size_t)0, count_x_total));
        auto send_r = Kokkos::subview(send_x_right_, std::make_pair((size_t)0, count_x_total));

        if(neighbor_right != MPI_PROC_NULL) MPI_Irecv(recv_r.data(), count_x_total, VVM_MPI_REAL, neighbor_right, static_cast<int>(HaloExchangeTags::SEND_TO_LEFT), cart_comm_, &req_obj.requests[req_obj.count++]);
        if(neighbor_left  != MPI_PROC_NULL) MPI_Irecv(recv_l.data(), count_x_total, VVM_MPI_REAL, neighbor_left, static_cast<int>(HaloExchangeTags::SEND_TO_RIGHT), cart_comm_, &req_obj.requests[req_obj.count++]);
        if(neighbor_left  != MPI_PROC_NULL) MPI_Isend(send_l.data(), count_x_total, VVM_MPI_REAL, neighbor_left, static_cast<int>(HaloExchangeTags::SEND_TO_LEFT), cart_comm_, &req_obj.requests[req_obj.count++]);
        if(neighbor_right != MPI_PROC_NULL) MPI_Isend(send_r.data(), count_x_total, VVM_MPI_REAL, neighbor_right, static_cast<int>(HaloExchangeTags::SEND_TO_RIGHT), cart_comm_, &req_obj.requests[req_obj.count++]);

        if (req_obj.count > 0) {
            MPI_Waitall(req_obj.count, req_obj.requests.data(), MPI_STATUSES_IGNORE);
            Kokkos::fence();
        }

        for (size_t f = 0; f < num_fields; ++f) {
            auto data = fields[f]->get_mutable_device_data();
            size_t offset = f * buffer_size_x_3d_;
            auto recv_l_f = Kokkos::subview(recv_x_left_, std::make_pair(offset, offset + buffer_size_x_3d_));
            auto recv_r_f = Kokkos::subview(recv_x_right_, std::make_pair(offset, offset + buffer_size_x_3d_));

            Kokkos::parallel_for("unpack_multi_x", Kokkos::MDRangePolicy<Kokkos::Rank<3>>({0,0,0}, {nz, ny, h}),
                KOKKOS_LAMBDA(int k, int j, int i_h) {
                    const size_t idx = k * (ny * h) + j * h + i_h;
                    if (neighbor_left != MPI_PROC_NULL) data(k, j, halo_start_offset - h + i_h) = recv_l_f(idx);
                    if (neighbor_right != MPI_PROC_NULL) data(k, j, halo_start_offset + nx_phys + i_h) = recv_r_f(idx);
            });
        }
    }

    // --- Y Direction ---
    if (count_y_total > 0 && !grid_ref_.is_singleton_y()) {
        for (size_t f = 0; f < num_fields; ++f) {
            auto data = fields[f]->get_mutable_device_data();
            size_t offset = f * buffer_size_y_3d_;
            auto send_b = Kokkos::subview(send_y_bottom_, std::make_pair(offset, offset + buffer_size_y_3d_));
            auto send_t = Kokkos::subview(send_y_top_, std::make_pair(offset, offset + buffer_size_y_3d_));

            Kokkos::parallel_for("pack_multi_y", Kokkos::MDRangePolicy<Kokkos::Rank<3>>({0,0,0}, {nz, nx, h}),
                KOKKOS_LAMBDA(int k, int i, int j_h) {
                    const size_t idx = k * (h * nx) + j_h * nx + i;
                    send_b(idx) = data(k, halo_start_offset + j_h, i);
                    send_t(idx) = data(k, halo_start_offset + ny_phys - h + j_h, i);
            });
        }
        Kokkos::fence();

        HaloExchangeRequests req_obj;
        req_obj.requests.resize(4);
        auto recv_b = Kokkos::subview(recv_y_bottom_, std::make_pair((size_t)0, count_y_total));
        auto recv_t = Kokkos::subview(recv_y_top_, std::make_pair((size_t)0, count_y_total));
        auto send_b = Kokkos::subview(send_y_bottom_, std::make_pair((size_t)0, count_y_total));
        auto send_t = Kokkos::subview(send_y_top_, std::make_pair((size_t)0, count_y_total));

        if(neighbor_bottom != MPI_PROC_NULL) MPI_Irecv(recv_b.data(), count_y_total, VVM_MPI_REAL, neighbor_bottom, static_cast<int>(HaloExchangeTags::SEND_TO_TOP), cart_comm_, &req_obj.requests[req_obj.count++]);
        if(neighbor_top    != MPI_PROC_NULL) MPI_Irecv(recv_t.data(), count_y_total, VVM_MPI_REAL, neighbor_top, static_cast<int>(HaloExchangeTags::SEND_TO_BOTTOM), cart_comm_, &req_obj.requests[req_obj.count++]);
        if(neighbor_top    != MPI_PROC_NULL) MPI_Isend(send_t.data(), count_y_total, VVM_MPI_REAL, neighbor_top, static_cast<int>(HaloExchangeTags::SEND_TO_TOP), cart_comm_, &req_obj.requests[req_obj.count++]);
        if(neighbor_bottom != MPI_PROC_NULL) MPI_Isend(send_b.data(), count_y_total, VVM_MPI_REAL, neighbor_bottom, static_cast<int>(HaloExchangeTags::SEND_TO_BOTTOM), cart_comm_, &req_obj.requests[req_obj.count++]);

        if (req_obj.count > 0) {
            MPI_Waitall(req_obj.count, req_obj.requests.data(), MPI_STATUSES_IGNORE);
            Kokkos::fence();
        }

        for (size_t f = 0; f < num_fields; ++f) {
            auto data = fields[f]->get_mutable_device_data();
            size_t offset = f * buffer_size_y_3d_;
            auto recv_b_f = Kokkos::subview(recv_y_bottom_, std::make_pair(offset, offset + buffer_size_y_3d_));
            auto recv_t_f = Kokkos::subview(recv_y_top_, std::make_pair(offset, offset + buffer_size_y_3d_));

            Kokkos::parallel_for("unpack_multi_y", Kokkos::MDRangePolicy<Kokkos::Rank<3>>({0,0,0}, {nz, nx, h}),
                KOKKOS_LAMBDA(int k, int i, int j_h) {
                    const size_t idx = k * (h * nx) + j_h * nx + i;
                    if (neighbor_bottom != MPI_PROC_NULL) data(k, halo_start_offset - h + j_h, i) = recv_b_f(idx);
                    if (neighbor_top != MPI_PROC_NULL) data(k, halo_start_offset + ny_phys + j_h, i) = recv_t_f(idx);
            });
        }
    }
}

inline void HaloExchanger::exchange_halos_slice(Field<3>& field, int k_layer) const {
    const int h = grid_ref_.get_halo_cells();
    if (h == 0) return;

    auto data = field.get_mutable_device_data();
    const int ny_phys = grid_ref_.get_local_physical_points_y();
    const int nx_phys = grid_ref_.get_local_physical_points_x();
    const int ny = data.extent(1);
    const int nx = data.extent(2);

    if (grid_ref_.is_singleton_y()) {
        Detail::fill_singleton_y_slice(
            Kokkos::DefaultExecutionSpace{}, field, k_layer, h, h);
    }
    if (grid_ref_.is_singleton_x()) {
        Detail::fill_singleton_x_slice(
            Kokkos::DefaultExecutionSpace{}, field, k_layer, h, h);
    }

    // --- Y-direction exchange for the slice ---
    if (!grid_ref_.is_singleton_y()) {
        size_t count = buffer_size_slice_y_;
        if (count > 0) {
            auto send_b = Kokkos::subview(send_y_bottom_, std::make_pair((size_t)0, count));
            auto recv_b = Kokkos::subview(recv_y_bottom_, std::make_pair((size_t)0, count));
            auto send_t = Kokkos::subview(send_y_top_, std::make_pair((size_t)0, count));
            auto recv_t = Kokkos::subview(recv_y_top_, std::make_pair((size_t)0, count));

            // auto send_b_h = Kokkos::subview(send_y_bottom_h_, std::make_pair((size_t)0, count));
            // auto recv_b_h = Kokkos::subview(recv_y_bottom_h_, std::make_pair((size_t)0, count));
            // auto send_t_h = Kokkos::subview(send_y_top_h_, std::make_pair((size_t)0, count));
            // auto recv_t_h = Kokkos::subview(recv_y_top_h_, std::make_pair((size_t)0, count));

            Kokkos::parallel_for("pack_y_slice", Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {nx, h}),
                KOKKOS_LAMBDA(int i, int j_h) {
                    const size_t idx = static_cast<size_t>(j_h) * nx + i;
                    send_b(idx) = data(k_layer, h + j_h, i);
                    send_t(idx) = data(k_layer, h + ny_phys - h + j_h, i);
            });
            Kokkos::fence();  // pack must land before MPI reads the buffer

            // Kokkos::deep_copy(send_b_h, send_b);
            // Kokkos::deep_copy(send_t_h, send_t);

            MPI_Request reqs[4];
            int req_count = 0;
            if(neighbor_bottom_ != MPI_PROC_NULL) MPI_Irecv(recv_b.data(), count, VVM_MPI_REAL, neighbor_bottom_, static_cast<int>(HaloExchangeTags::SLICE_SEND_TO_TOP), cart_comm_, &reqs[req_count++]);
            if(neighbor_top_    != MPI_PROC_NULL) MPI_Irecv(recv_t.data(), count, VVM_MPI_REAL, neighbor_top_, static_cast<int>(HaloExchangeTags::SLICE_SEND_TO_BOTTOM), cart_comm_, &reqs[req_count++]);
            if(neighbor_top_    != MPI_PROC_NULL) MPI_Isend(send_t.data(), count, VVM_MPI_REAL, neighbor_top_, static_cast<int>(HaloExchangeTags::SLICE_SEND_TO_TOP), cart_comm_, &reqs[req_count++]);
            if(neighbor_bottom_ != MPI_PROC_NULL) MPI_Isend(send_b.data(), count, VVM_MPI_REAL, neighbor_bottom_, static_cast<int>(HaloExchangeTags::SLICE_SEND_TO_BOTTOM), cart_comm_, &reqs[req_count++]);

            // if(neighbor_bottom_ != MPI_PROC_NULL) MPI_Irecv(recv_b_h.data(), count, VVM_MPI_REAL, neighbor_bottom_, static_cast<int>(HaloExchangeTags::SLICE_SEND_TO_TOP), cart_comm_, &reqs[req_count++]);
            // if(neighbor_top_    != MPI_PROC_NULL) MPI_Irecv(recv_t_h.data(), count, VVM_MPI_REAL, neighbor_top_, static_cast<int>(HaloExchangeTags::SLICE_SEND_TO_BOTTOM), cart_comm_, &reqs[req_count++]);
            // if(neighbor_top_    != MPI_PROC_NULL) MPI_Isend(send_t_h.data(), count, VVM_MPI_REAL, neighbor_top_, static_cast<int>(HaloExchangeTags::SLICE_SEND_TO_TOP), cart_comm_, &reqs[req_count++]);
            // if(neighbor_bottom_ != MPI_PROC_NULL) MPI_Isend(send_b_h.data(), count, VVM_MPI_REAL, neighbor_bottom_, static_cast<int>(HaloExchangeTags::SLICE_SEND_TO_BOTTOM), cart_comm_, &reqs[req_count++]);

            if(req_count > 0) MPI_Waitall(req_count, reqs, MPI_STATUSES_IGNORE);
            // Kokkos::fence();

            // Kokkos::deep_copy(recv_b, recv_b_h);
            // Kokkos::deep_copy(recv_t, recv_t_h);
            // Kokkos::fence();

            const int neighbor_bottom = neighbor_bottom_;
            const int neighbor_top = neighbor_top_;
            Kokkos::parallel_for("unpack_y_slice", Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {nx, h}),
                KOKKOS_LAMBDA(int i, int j_h) {
                    const size_t idx = static_cast<size_t>(j_h) * nx + i;
                    if (neighbor_bottom != MPI_PROC_NULL) data(k_layer, j_h, i) = recv_b(idx);
                    if (neighbor_top != MPI_PROC_NULL) data(k_layer, h + ny_phys + j_h, i) = recv_t(idx);
            });
            // Kokkos::fence();
        }
    }

    // --- X-direction exchange for the slice ---
    if (!grid_ref_.is_singleton_x()) {
        size_t count = buffer_size_slice_x_;
        if (count > 0) {
            auto send_l = Kokkos::subview(send_x_left_, std::make_pair((size_t)0, count));
            auto recv_l = Kokkos::subview(recv_x_left_, std::make_pair((size_t)0, count));
            auto send_r = Kokkos::subview(send_x_right_, std::make_pair((size_t)0, count));
            auto recv_r = Kokkos::subview(recv_x_right_, std::make_pair((size_t)0, count));

            // auto send_l_h = Kokkos::subview(send_x_left_h_, std::make_pair((size_t)0, count));
            // auto recv_l_h = Kokkos::subview(recv_x_left_h_, std::make_pair((size_t)0, count));
            // auto send_r_h = Kokkos::subview(send_x_right_h_, std::make_pair((size_t)0, count));
            // auto recv_r_h = Kokkos::subview(recv_x_right_h_, std::make_pair((size_t)0, count));

            Kokkos::parallel_for("pack_x_slice", Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {ny, h}),
                KOKKOS_LAMBDA(int j, int i_h) {
                    const size_t idx = static_cast<size_t>(j) * h + i_h;
                    send_l(idx) = data(k_layer, j, h + i_h);
                    send_r(idx) = data(k_layer, j, h + nx_phys - h + i_h);
            });
            Kokkos::fence();  // pack must land before MPI reads the buffer

            // Kokkos::deep_copy(send_l_h, send_l);
            // Kokkos::deep_copy(send_r_h, send_r);

            MPI_Request reqs[4];
            int req_count = 0;
            if(neighbor_right_ != MPI_PROC_NULL) MPI_Irecv(recv_r.data(), count, VVM_MPI_REAL, neighbor_right_, static_cast<int>(HaloExchangeTags::SLICE_SEND_TO_LEFT), cart_comm_, &reqs[req_count++]);
            if(neighbor_left_  != MPI_PROC_NULL) MPI_Irecv(recv_l.data(), count, VVM_MPI_REAL, neighbor_left_, static_cast<int>(HaloExchangeTags::SLICE_SEND_TO_RIGHT), cart_comm_, &reqs[req_count++]);
            if(neighbor_left_  != MPI_PROC_NULL) MPI_Isend(send_l.data(), count, VVM_MPI_REAL, neighbor_left_, static_cast<int>(HaloExchangeTags::SLICE_SEND_TO_LEFT), cart_comm_, &reqs[req_count++]);
            if(neighbor_right_ != MPI_PROC_NULL) MPI_Isend(send_r.data(), count, VVM_MPI_REAL, neighbor_right_, static_cast<int>(HaloExchangeTags::SLICE_SEND_TO_RIGHT), cart_comm_, &reqs[req_count++]);

            // if(neighbor_right_ != MPI_PROC_NULL) MPI_Irecv(recv_r_h.data(), count, VVM_MPI_REAL, neighbor_right_, static_cast<int>(HaloExchangeTags::SLICE_SEND_TO_LEFT), cart_comm_, &reqs[req_count++]);
            // if(neighbor_left_  != MPI_PROC_NULL) MPI_Irecv(recv_l_h.data(), count, VVM_MPI_REAL, neighbor_left_, static_cast<int>(HaloExchangeTags::SLICE_SEND_TO_RIGHT), cart_comm_, &reqs[req_count++]);
            // if(neighbor_left_  != MPI_PROC_NULL) MPI_Isend(send_l_h.data(), count, VVM_MPI_REAL, neighbor_left_, static_cast<int>(HaloExchangeTags::SLICE_SEND_TO_LEFT), cart_comm_, &reqs[req_count++]);
            // if(neighbor_right_ != MPI_PROC_NULL) MPI_Isend(send_r_h.data(), count, VVM_MPI_REAL, neighbor_right_, static_cast<int>(HaloExchangeTags::SLICE_SEND_TO_RIGHT), cart_comm_, &reqs[req_count++]);

            if(req_count > 0) MPI_Waitall(req_count, reqs, MPI_STATUSES_IGNORE);
            // Kokkos::fence();

            // Kokkos::deep_copy(recv_l, recv_l_h);
            // Kokkos::deep_copy(recv_r, recv_r_h);
            // Kokkos::fence();

            const int neighbor_left = neighbor_left_;
            const int neighbor_right = neighbor_right_;
            Kokkos::parallel_for("unpack_x_slice", Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {ny, h}),
                KOKKOS_LAMBDA(int j, int i_h) {
                    const size_t idx = static_cast<size_t>(j) * h + i_h;
                    if (neighbor_left != MPI_PROC_NULL) data(k_layer, j, i_h) = recv_l(idx);
                    if (neighbor_right != MPI_PROC_NULL) data(k_layer, j, h + nx_phys + i_h) = recv_r(idx);
            });
            // Kokkos::fence();
        }
    }
}


} // namespace Core
} // namespace VVM
