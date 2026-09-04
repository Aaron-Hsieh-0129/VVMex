#pragma once

#include "core/Grid.hpp"
#include "core/haloexchange/HaloExchangeDegenerate.hpp"
#include "core/Field.hpp"
#include "core/State.hpp"
#include "core/vvm_types.hpp"
#include <algorithm>
#include <vector>
#include <Kokkos_Core.hpp>

#include "utils/ConfigurationManager.hpp"
#include <nccl.h>
#include <cuda_runtime.h>
#include <map>
#include <string>
#include <set>
#include <cassert>
#include <stdexcept>

namespace VVM {
namespace Core {

using ExecSpace = Kokkos::Cuda;

class HaloExchanger {
public:
    explicit HaloExchanger(const Utils::ConfigurationManager& config, const Grid& grid, ncclComm_t nccl_comm, cudaStream_t stream);
    ~HaloExchanger();

    HaloExchanger(const HaloExchanger&) = delete;
    HaloExchanger& operator=(const HaloExchanger&) = delete;

    void exchange_halos(State& state);

    template<typename FieldT>
    void exchange_halos(FieldT& field, int depth = -1) const {
        exchange_halos_impl(field, depth);

        cudaStreamCaptureStatus capture_status;
        cudaStreamIsCapturing(stream_, &capture_status);
        if (depth == -1 && capture_status == cudaStreamCaptureStatusNone && grid_ref_.get_mpi_size() > 1) {
             cudaStreamSynchronize(stream_);
        }
    }

    template<typename FieldT>
    void exchange_halos_impl(FieldT& field, int depth = -1) const;

    void exchange_halos_slice(Field<3>& field, int k_layer) const;

    void exchange_halos_top_slice(Field<3>& field) const {
        const int nz = grid_ref_.get_local_total_points_z();
        const int h = grid_ref_.get_halo_cells();
        exchange_halos_slice(field, nz - h - 1);
    }

    void exchange_multiple_halos(const std::vector<std::string>& field_names, State& state) const;

    void exchange_multiple_halos(const std::vector<Field<3>*>& fields) const;

    // Batched exchange for fields not registered in State (e.g. solver-private work
    // arrays). All fields must share the same halo width and extents.
    void exchange_multiple_halos(const std::vector<Field<2>*>& fields, int depth = -1) const;

private:
    mutable std::vector<Field<3>*> batch_fields_;

    const Grid& grid_ref_;
    MPI_Comm cart_comm_;
    ncclComm_t nccl_comm_;
    cudaStream_t stream_;
    ExecSpace exec_space_;

    int neighbor_left_, neighbor_right_;
    int neighbor_bottom_, neighbor_top_;

    bool is_single_rank_;

    std::set<std::string> enabled_graph_vars_;

    std::map<std::string, cudaGraphExec_t> graph_map_;

    mutable Kokkos::View<VVM::Real*, ExecSpace> send_x_left_, recv_x_left_;
    mutable Kokkos::View<VVM::Real*, ExecSpace> send_x_right_, recv_x_right_;
    mutable Kokkos::View<VVM::Real*, ExecSpace> send_y_bottom_, recv_y_bottom_;
    mutable Kokkos::View<VVM::Real*, ExecSpace> send_y_top_, recv_y_top_;

    size_t buffer_size_x_2d_, buffer_size_y_2d_;
    size_t buffer_size_x_3d_, buffer_size_y_3d_;
    size_t buffer_size_x_4d_, buffer_size_y_4d_;
    size_t buffer_size_slice_x_, buffer_size_slice_y_;
};

inline HaloExchanger::HaloExchanger(const Utils::ConfigurationManager& config, const Grid& grid, ncclComm_t nccl_comm, cudaStream_t stream)
    : grid_ref_(grid),
      cart_comm_(grid.get_cart_comm()),
      nccl_comm_(nccl_comm),
      stream_(stream),
      exec_space_(stream)
{
    is_single_rank_ = (grid.get_mpi_size() == 1);

    if (config.has_key("optimization.cuda_graph_halo_exchange")) {
        auto vars = config.get_value<std::vector<std::string>>("optimization.cuda_graph_halo_exchange");
        enabled_graph_vars_.insert(vars.begin(), vars.end());
    }

    if (grid_ref_.get_mpi_rank() == 0 && !enabled_graph_vars_.empty()) {
        std::cout << "HaloExchanger: CUDA Graph enabled for fields: ";
        for (const auto& var : enabled_graph_vars_) std::cout << var << " ";
        std::cout << std::endl;
    }

    if (cart_comm_ != MPI_COMM_NULL) {
        int cart_left, cart_right, cart_bottom, cart_top;
        MPI_Cart_shift(cart_comm_, 1, 1, &cart_left, &cart_right);
        MPI_Cart_shift(cart_comm_, 0, 1, &cart_bottom, &cart_top);

        MPI_Group cart_group, world_group;
        MPI_Comm_group(cart_comm_, &cart_group);
        MPI_Comm_group(grid_ref_.get_comm(), &world_group);

        int cart_ranks[4] = {cart_left, cart_right, cart_bottom, cart_top};
        int world_ranks[4] = {MPI_PROC_NULL, MPI_PROC_NULL, MPI_PROC_NULL, MPI_PROC_NULL};

        for(int i=0; i<4; ++i) {
            if (cart_ranks[i] != MPI_PROC_NULL) {
                MPI_Group_translate_ranks(cart_group, 1, &cart_ranks[i], world_group, &world_ranks[i]);
            }
        }

        neighbor_left_   = world_ranks[0];
        neighbor_right_  = world_ranks[1];
        neighbor_bottom_ = world_ranks[2];
        neighbor_top_    = world_ranks[3];

        MPI_Group_free(&cart_group);
        MPI_Group_free(&world_group);
    }
    else {
        neighbor_left_ = neighbor_right_ = neighbor_bottom_ = neighbor_top_ = MPI_PROC_NULL;
    }

    const int h = grid_ref_.get_halo_cells();
    if (h > 0) {
        const int nx = grid.get_local_total_points_x();
        const int ny = grid.get_local_total_points_y();
        const int nz = grid.get_local_total_points_z();
        // Pre-size to max batch used by exchange_multiple_halos (thermo: 10 fields).
        // Prevents Kokkos::resize from freeing buffer pointers that may be baked
        // into CUDA graphs captured via cudaStreamCaptureModeGlobal.
        const int nw_dummy = 16;

        buffer_size_x_2d_ = static_cast<size_t>(h) * ny;
        buffer_size_y_2d_ = static_cast<size_t>(h) * nx;
        buffer_size_x_3d_ = static_cast<size_t>(h) * ny * nz;
        buffer_size_y_3d_ = static_cast<size_t>(h) * nx * nz;
        buffer_size_x_4d_ = static_cast<size_t>(h) * ny * nz * nw_dummy;
        buffer_size_y_4d_ = static_cast<size_t>(h) * nx * nz * nw_dummy;

        buffer_size_slice_x_ = static_cast<size_t>(h) * ny;
        buffer_size_slice_y_ = static_cast<size_t>(h) * nx;

        // A periodic dimension split over exactly two ranks has one remote rank
        // on both sides. NCCL has no tags, so those directions use a combined
        // ordered payload below.
        size_t max_x = 2 * std::max({buffer_size_x_2d_, buffer_size_x_3d_, buffer_size_x_4d_, buffer_size_slice_x_});
        size_t max_y = 2 * std::max({buffer_size_y_2d_, buffer_size_y_3d_, buffer_size_y_4d_, buffer_size_slice_y_});

        if (max_x > 0) {
            send_x_left_  = Kokkos::View<VVM::Real*, ExecSpace>("sxl", max_x);
            recv_x_left_  = Kokkos::View<VVM::Real*, ExecSpace>("rxl", max_x);
            send_x_right_ = Kokkos::View<VVM::Real*, ExecSpace>("sxr", max_x);
            recv_x_right_ = Kokkos::View<VVM::Real*, ExecSpace>("rxr", max_x);
        }
        if (max_y > 0) {
            send_y_bottom_ = Kokkos::View<VVM::Real*, ExecSpace>("syb", max_y);
            recv_y_bottom_ = Kokkos::View<VVM::Real*, ExecSpace>("ryb", max_y);
            send_y_top_    = Kokkos::View<VVM::Real*, ExecSpace>("syt", max_y);
            recv_y_top_    = Kokkos::View<VVM::Real*, ExecSpace>("ryt", max_y);
        }
    }
}

inline HaloExchanger::~HaloExchanger() {
    for (auto& pair : graph_map_) {
        if (pair.second) {
            cudaGraphExecDestroy(pair.second);
        }
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

    const int my_rank = grid_ref_.get_mpi_rank();

    const int neighbor_left = neighbor_left_;
    const int neighbor_right = neighbor_right_;
    const int neighbor_bottom = neighbor_bottom_;
    const int neighbor_top = neighbor_top_;
    const bool is_single = is_single_rank_;
    const bool same_remote_x =
        neighbor_left != MPI_PROC_NULL && neighbor_left == neighbor_right && neighbor_left != my_rank;
    const bool same_remote_y =
        neighbor_bottom != MPI_PROC_NULL && neighbor_bottom == neighbor_top && neighbor_bottom != my_rank;

    const size_t required_x_total = same_remote_x ? 2 * count_x_total : count_x_total;
    const size_t required_y_total = same_remote_y ? 2 * count_y_total : count_y_total;

    if (send_x_left_.extent(0) < required_x_total) {
        Kokkos::resize(send_x_left_, required_x_total); Kokkos::resize(recv_x_left_, required_x_total);
        Kokkos::resize(send_x_right_, required_x_total); Kokkos::resize(recv_x_right_, required_x_total);
    }
    if (send_y_bottom_.extent(0) < required_y_total) {
        Kokkos::resize(send_y_bottom_, required_y_total); Kokkos::resize(recv_y_bottom_, required_y_total);
        Kokkos::resize(send_y_top_, required_y_total); Kokkos::resize(recv_y_top_, required_y_total);
    }

    const int nx_phys = grid_ref_.get_local_physical_points_x();
    const int ny_phys = grid_ref_.get_local_physical_points_y();
    const int nz = grid_ref_.get_local_total_points_z();
    const int ny = grid_ref_.get_local_total_points_y();
    const int nx = grid_ref_.get_local_total_points_x();
    const int halo_start_offset = h;

    if (grid_ref_.is_singleton_x()) {
        for (Field<3>* field : fields) {
            if (field) {
                Detail::fill_singleton_x_halo(
                    exec_space_, *field, halo_start_offset, h);
            }
        }
    }
    if (grid_ref_.is_singleton_y()) {
        for (Field<3>* field : fields) {
            if (field) {
                Detail::fill_singleton_y_halo(
                    exec_space_, *field, halo_start_offset, h);
            }
        }
    }

    if (count_x_total > 0 && !grid_ref_.is_singleton_x()) {
        for (size_t f = 0; f < num_fields; ++f) {
            auto data = fields[f]->get_mutable_device_data();
            size_t offset = f * buffer_size_x_3d_;
            auto send_l = Kokkos::subview(send_x_left_, std::make_pair(offset, offset + buffer_size_x_3d_));
            auto send_r = Kokkos::subview(send_x_right_, std::make_pair(offset, offset + buffer_size_x_3d_));

            Kokkos::parallel_for("pack_multi_x", Kokkos::MDRangePolicy<Kokkos::Rank<3>, ExecSpace>(exec_space_, {0,0,0}, {nz, ny, h}),
                KOKKOS_LAMBDA(int k, int j, int i_h) {
                    const size_t idx = k * (ny * h) + j * h + i_h;
                    send_l(idx) = data(k, j, halo_start_offset + i_h);
                    send_r(idx) = data(k, j, halo_start_offset + nx_phys - h + i_h);
            });
        }

        if (is_single || (neighbor_left == my_rank && neighbor_right == my_rank)) {
            Kokkos::deep_copy(exec_space_, Kokkos::subview(recv_x_left_, std::make_pair((size_t)0, count_x_total)), Kokkos::subview(send_x_right_, std::make_pair((size_t)0, count_x_total)));
            Kokkos::deep_copy(exec_space_, Kokkos::subview(recv_x_right_, std::make_pair((size_t)0, count_x_total)), Kokkos::subview(send_x_left_, std::make_pair((size_t)0, count_x_total)));
        } else if (same_remote_x) {
            Kokkos::deep_copy(exec_space_,
                Kokkos::subview(send_x_right_, std::make_pair(count_x_total, 2 * count_x_total)),
                Kokkos::subview(send_x_left_, std::make_pair((size_t)0, count_x_total)));
            ncclGroupStart();
            ncclSend(send_x_right_.data(), 2 * count_x_total, VVM_NCCL_REAL, neighbor_right, nccl_comm_, stream_);
            ncclRecv(recv_x_right_.data(), 2 * count_x_total, VVM_NCCL_REAL, neighbor_left, nccl_comm_, stream_);
            ncclGroupEnd();
            Kokkos::deep_copy(exec_space_,
                Kokkos::subview(recv_x_left_, std::make_pair((size_t)0, count_x_total)),
                Kokkos::subview(recv_x_right_, std::make_pair((size_t)0, count_x_total)));
            Kokkos::deep_copy(exec_space_,
                Kokkos::subview(recv_x_right_, std::make_pair((size_t)0, count_x_total)),
                Kokkos::subview(recv_x_right_, std::make_pair(count_x_total, 2 * count_x_total)));
        } else {
            ncclGroupStart();
            if(neighbor_right != MPI_PROC_NULL) ncclSend(send_x_right_.data(), count_x_total, VVM_NCCL_REAL, neighbor_right, nccl_comm_, stream_);
            if(neighbor_left != MPI_PROC_NULL) ncclRecv(recv_x_left_.data(), count_x_total, VVM_NCCL_REAL, neighbor_left, nccl_comm_, stream_);
            if(neighbor_left != MPI_PROC_NULL) ncclSend(send_x_left_.data(), count_x_total, VVM_NCCL_REAL, neighbor_left, nccl_comm_, stream_);
            if(neighbor_right != MPI_PROC_NULL) ncclRecv(recv_x_right_.data(), count_x_total, VVM_NCCL_REAL, neighbor_right, nccl_comm_, stream_);
            ncclGroupEnd();
        }

        for (size_t f = 0; f < num_fields; ++f) {
            auto data = fields[f]->get_mutable_device_data();
            size_t offset = f * buffer_size_x_3d_;
            auto recv_l = Kokkos::subview(recv_x_left_, std::make_pair(offset, offset + buffer_size_x_3d_));
            auto recv_r = Kokkos::subview(recv_x_right_, std::make_pair(offset, offset + buffer_size_x_3d_));

            Kokkos::parallel_for("unpack_multi_x", Kokkos::MDRangePolicy<Kokkos::Rank<3>, ExecSpace>(exec_space_, {0,0,0}, {nz, ny, h}),
                KOKKOS_LAMBDA(int k, int j, int i_h) {
                    const size_t idx = k * (ny * h) + j * h + i_h;
                    if (neighbor_left != MPI_PROC_NULL || is_single) data(k, j, halo_start_offset - h + i_h) = recv_l(idx);
                    if (neighbor_right != MPI_PROC_NULL || is_single) data(k, j, halo_start_offset + nx_phys + i_h) = recv_r(idx);
            });
        }
    }

    if (count_y_total > 0 && !grid_ref_.is_singleton_y()) {
        for (size_t f = 0; f < num_fields; ++f) {
            auto data = fields[f]->get_mutable_device_data();
            size_t offset = f * buffer_size_y_3d_;
            auto send_b = Kokkos::subview(send_y_bottom_, std::make_pair(offset, offset + buffer_size_y_3d_));
            auto send_t = Kokkos::subview(send_y_top_, std::make_pair(offset, offset + buffer_size_y_3d_));

            Kokkos::parallel_for("pack_multi_y", Kokkos::MDRangePolicy<Kokkos::Rank<3>, ExecSpace>(exec_space_, {0,0,0}, {nz, nx, h}),
                KOKKOS_LAMBDA(int k, int i, int j_h) {
                    const size_t idx = k * (h * nx) + j_h * nx + i;
                    send_b(idx) = data(k, halo_start_offset + j_h, i);
                    send_t(idx) = data(k, halo_start_offset + ny_phys - h + j_h, i);
            });
        }

        if (is_single || (neighbor_bottom == my_rank && neighbor_top == my_rank)) {
            Kokkos::deep_copy(exec_space_, Kokkos::subview(recv_y_bottom_, std::make_pair((size_t)0, count_y_total)), Kokkos::subview(send_y_top_, std::make_pair((size_t)0, count_y_total)));
            Kokkos::deep_copy(exec_space_, Kokkos::subview(recv_y_top_, std::make_pair((size_t)0, count_y_total)), Kokkos::subview(send_y_bottom_, std::make_pair((size_t)0, count_y_total)));
        } else if (same_remote_y) {
            Kokkos::deep_copy(exec_space_,
                Kokkos::subview(send_y_top_, std::make_pair(count_y_total, 2 * count_y_total)),
                Kokkos::subview(send_y_bottom_, std::make_pair((size_t)0, count_y_total)));
            ncclGroupStart();
            ncclSend(send_y_top_.data(), 2 * count_y_total, VVM_NCCL_REAL, neighbor_top, nccl_comm_, stream_);
            ncclRecv(recv_y_top_.data(), 2 * count_y_total, VVM_NCCL_REAL, neighbor_bottom, nccl_comm_, stream_);
            ncclGroupEnd();
            Kokkos::deep_copy(exec_space_,
                Kokkos::subview(recv_y_bottom_, std::make_pair((size_t)0, count_y_total)),
                Kokkos::subview(recv_y_top_, std::make_pair((size_t)0, count_y_total)));
            Kokkos::deep_copy(exec_space_,
                Kokkos::subview(recv_y_top_, std::make_pair((size_t)0, count_y_total)),
                Kokkos::subview(recv_y_top_, std::make_pair(count_y_total, 2 * count_y_total)));
        } else {
            ncclGroupStart();
            if(neighbor_top != MPI_PROC_NULL) ncclSend(send_y_top_.data(), count_y_total, VVM_NCCL_REAL, neighbor_top, nccl_comm_, stream_);
            if(neighbor_bottom != MPI_PROC_NULL) ncclRecv(recv_y_bottom_.data(), count_y_total, VVM_NCCL_REAL, neighbor_bottom, nccl_comm_, stream_);
            if(neighbor_bottom != MPI_PROC_NULL) ncclSend(send_y_bottom_.data(), count_y_total, VVM_NCCL_REAL, neighbor_bottom, nccl_comm_, stream_);
            if(neighbor_top != MPI_PROC_NULL) ncclRecv(recv_y_top_.data(), count_y_total, VVM_NCCL_REAL, neighbor_top, nccl_comm_, stream_);
            ncclGroupEnd();
        }

        for (size_t f = 0; f < num_fields; ++f) {
            auto data = fields[f]->get_mutable_device_data();
            size_t offset = f * buffer_size_y_3d_;
            auto recv_b = Kokkos::subview(recv_y_bottom_, std::make_pair(offset, offset + buffer_size_y_3d_));
            auto recv_t = Kokkos::subview(recv_y_top_, std::make_pair(offset, offset + buffer_size_y_3d_));

            Kokkos::parallel_for("unpack_multi_y", Kokkos::MDRangePolicy<Kokkos::Rank<3>, ExecSpace>(exec_space_, {0,0,0}, {nz, nx, h}),
                KOKKOS_LAMBDA(int k, int i, int j_h) {
                    const size_t idx = k * (h * nx) + j_h * nx + i;
                    if (neighbor_bottom != MPI_PROC_NULL || is_single) data(k, halo_start_offset - h + j_h, i) = recv_b(idx);
                    if (neighbor_top != MPI_PROC_NULL || is_single) data(k, halo_start_offset + ny_phys + j_h, i) = recv_t(idx);
            });
        }
    }

    cudaStreamSynchronize(stream_);
}

// Batched exchange of 2-D fields that are not registered in State. Used for the
// psi/chi relaxation, where both fields share an operator and can ride in one NCCL
// group. All fields must share halo width and extents.
inline void HaloExchanger::exchange_multiple_halos(const std::vector<Field<2>*>& fields, int depth) const {
    if (fields.empty()) return;

    const int halo_start_offset = grid_ref_.get_halo_cells();
    const int h = (depth == -1) ? halo_start_offset : depth;
    if (h == 0) return;

    // Keep the reduced-axis path centralized in the single-field exchange.
    // It remains stream-ordered and safe during CUDA graph capture.
    if (grid_ref_.is_singleton_x() || grid_ref_.is_singleton_y()) {
        for (Field<2>* field : fields) {
            if (field) exchange_halos_impl(*field, depth);
        }
        return;
    }

    const size_t num_fields = fields.size();
    const int nx_phys = grid_ref_.get_local_physical_points_x();
    const int ny_phys = grid_ref_.get_local_physical_points_y();
    const int ny = fields[0]->get_mutable_device_data().extent(0);
    const int nx = fields[0]->get_mutable_device_data().extent(1);

    // Strides are the exact packed size, so no padding is sent.
    const size_t stride_x = static_cast<size_t>(h) * ny;
    const size_t stride_y = static_cast<size_t>(h) * nx;
    const size_t count_x_total = num_fields * stride_x;
    const size_t count_y_total = num_fields * stride_y;

    assert(count_x_total * 2 <= send_x_left_.extent(0) && "2-D batch exceeds pre-sized halo buffer");
    assert(count_y_total * 2 <= send_y_bottom_.extent(0) && "2-D batch exceeds pre-sized halo buffer");

    if (is_single_rank_) {
        for (size_t f = 0; f < num_fields; ++f) {
            auto data = fields[f]->get_mutable_device_data();
            Kokkos::parallel_for("local_copy_x_2d_multi", Kokkos::MDRangePolicy<Kokkos::Rank<2>, ExecSpace>(exec_space_, {0,0}, {ny, h}),
                KOKKOS_LAMBDA(int j, int i_h) {
                    data(j, halo_start_offset - h + i_h) = data(j, halo_start_offset + nx_phys - h + i_h);
                    data(j, halo_start_offset + nx_phys + i_h) = data(j, halo_start_offset + i_h);
            });
            Kokkos::parallel_for("local_copy_y_2d_multi", Kokkos::MDRangePolicy<Kokkos::Rank<2>, ExecSpace>(exec_space_, {0,0}, {nx, h}),
                KOKKOS_LAMBDA(int i, int j_h) {
                    data(halo_start_offset - h + j_h, i) = data(halo_start_offset + ny_phys - h + j_h, i);
                    data(halo_start_offset + ny_phys + j_h, i) = data(halo_start_offset + j_h, i);
            });
        }
        return;
    }

    const int my_rank = grid_ref_.get_mpi_rank();
    const int neighbor_left = neighbor_left_;
    const int neighbor_right = neighbor_right_;
    const int neighbor_bottom = neighbor_bottom_;
    const int neighbor_top = neighbor_top_;
    const bool same_remote_x =
        neighbor_left != MPI_PROC_NULL && neighbor_left == neighbor_right && neighbor_left != my_rank;
    const bool same_remote_y =
        neighbor_bottom != MPI_PROC_NULL && neighbor_bottom == neighbor_top && neighbor_bottom != my_rank;

    if (count_x_total > 0) {
        for (size_t f = 0; f < num_fields; ++f) {
            auto data = fields[f]->get_mutable_device_data();
            const size_t offset = f * stride_x;
            auto send_l = Kokkos::subview(send_x_left_, std::make_pair(offset, offset + stride_x));
            auto send_r = Kokkos::subview(send_x_right_, std::make_pair(offset, offset + stride_x));
            Kokkos::parallel_for("pack_multi_x_2d", Kokkos::MDRangePolicy<Kokkos::Rank<2>, ExecSpace>(exec_space_, {0,0}, {ny, h}),
                KOKKOS_LAMBDA(int j, int i_h) {
                    const size_t idx = static_cast<size_t>(j) * h + i_h;
                    send_l(idx) = data(j, halo_start_offset + i_h);
                    send_r(idx) = data(j, halo_start_offset + nx_phys - h + i_h);
            });
        }

        if (neighbor_left == my_rank && neighbor_right == my_rank) {
            Kokkos::deep_copy(exec_space_, Kokkos::subview(recv_x_left_, std::make_pair((size_t)0, count_x_total)), Kokkos::subview(send_x_right_, std::make_pair((size_t)0, count_x_total)));
            Kokkos::deep_copy(exec_space_, Kokkos::subview(recv_x_right_, std::make_pair((size_t)0, count_x_total)), Kokkos::subview(send_x_left_, std::make_pair((size_t)0, count_x_total)));
        } else if (same_remote_x) {
            Kokkos::deep_copy(exec_space_,
                Kokkos::subview(send_x_right_, std::make_pair(count_x_total, 2 * count_x_total)),
                Kokkos::subview(send_x_left_, std::make_pair((size_t)0, count_x_total)));
            ncclGroupStart();
            ncclSend(send_x_right_.data(), 2 * count_x_total, VVM_NCCL_REAL, neighbor_right, nccl_comm_, stream_);
            ncclRecv(recv_x_right_.data(), 2 * count_x_total, VVM_NCCL_REAL, neighbor_left, nccl_comm_, stream_);
            ncclGroupEnd();
            Kokkos::deep_copy(exec_space_,
                Kokkos::subview(recv_x_left_, std::make_pair((size_t)0, count_x_total)),
                Kokkos::subview(recv_x_right_, std::make_pair((size_t)0, count_x_total)));
            Kokkos::deep_copy(exec_space_,
                Kokkos::subview(recv_x_right_, std::make_pair((size_t)0, count_x_total)),
                Kokkos::subview(recv_x_right_, std::make_pair(count_x_total, 2 * count_x_total)));
        } else {
            ncclGroupStart();
            if (neighbor_right != MPI_PROC_NULL) ncclSend(send_x_right_.data(), count_x_total, VVM_NCCL_REAL, neighbor_right, nccl_comm_, stream_);
            if (neighbor_left  != MPI_PROC_NULL) ncclRecv(recv_x_left_.data(),  count_x_total, VVM_NCCL_REAL, neighbor_left,  nccl_comm_, stream_);
            if (neighbor_left  != MPI_PROC_NULL) ncclSend(send_x_left_.data(),  count_x_total, VVM_NCCL_REAL, neighbor_left,  nccl_comm_, stream_);
            if (neighbor_right != MPI_PROC_NULL) ncclRecv(recv_x_right_.data(), count_x_total, VVM_NCCL_REAL, neighbor_right, nccl_comm_, stream_);
            ncclGroupEnd();
        }

        for (size_t f = 0; f < num_fields; ++f) {
            auto data = fields[f]->get_mutable_device_data();
            const size_t offset = f * stride_x;
            auto recv_l = Kokkos::subview(recv_x_left_, std::make_pair(offset, offset + stride_x));
            auto recv_r = Kokkos::subview(recv_x_right_, std::make_pair(offset, offset + stride_x));
            Kokkos::parallel_for("unpack_multi_x_2d", Kokkos::MDRangePolicy<Kokkos::Rank<2>, ExecSpace>(exec_space_, {0,0}, {ny, h}),
                KOKKOS_LAMBDA(int j, int i_h) {
                    const size_t idx = static_cast<size_t>(j) * h + i_h;
                    if (neighbor_left  != MPI_PROC_NULL) data(j, halo_start_offset - h + i_h) = recv_l(idx);
                    if (neighbor_right != MPI_PROC_NULL) data(j, halo_start_offset + nx_phys + i_h) = recv_r(idx);
            });
        }
    }

    if (count_y_total > 0) {
        for (size_t f = 0; f < num_fields; ++f) {
            auto data = fields[f]->get_mutable_device_data();
            const size_t offset = f * stride_y;
            auto send_b = Kokkos::subview(send_y_bottom_, std::make_pair(offset, offset + stride_y));
            auto send_t = Kokkos::subview(send_y_top_,    std::make_pair(offset, offset + stride_y));
            Kokkos::parallel_for("pack_multi_y_2d", Kokkos::MDRangePolicy<Kokkos::Rank<2>, ExecSpace>(exec_space_, {0,0}, {nx, h}),
                KOKKOS_LAMBDA(int i, int j_h) {
                    const size_t idx = static_cast<size_t>(j_h) * nx + i;
                    send_b(idx) = data(halo_start_offset + j_h, i);
                    send_t(idx) = data(halo_start_offset + ny_phys - h + j_h, i);
            });
        }

        if (neighbor_bottom == my_rank && neighbor_top == my_rank) {
            Kokkos::deep_copy(exec_space_, Kokkos::subview(recv_y_bottom_, std::make_pair((size_t)0, count_y_total)), Kokkos::subview(send_y_top_,    std::make_pair((size_t)0, count_y_total)));
            Kokkos::deep_copy(exec_space_, Kokkos::subview(recv_y_top_,    std::make_pair((size_t)0, count_y_total)), Kokkos::subview(send_y_bottom_, std::make_pair((size_t)0, count_y_total)));
        } else if (same_remote_y) {
            Kokkos::deep_copy(exec_space_,
                Kokkos::subview(send_y_top_, std::make_pair(count_y_total, 2 * count_y_total)),
                Kokkos::subview(send_y_bottom_, std::make_pair((size_t)0, count_y_total)));
            ncclGroupStart();
            ncclSend(send_y_top_.data(), 2 * count_y_total, VVM_NCCL_REAL, neighbor_top, nccl_comm_, stream_);
            ncclRecv(recv_y_top_.data(), 2 * count_y_total, VVM_NCCL_REAL, neighbor_bottom, nccl_comm_, stream_);
            ncclGroupEnd();
            Kokkos::deep_copy(exec_space_,
                Kokkos::subview(recv_y_bottom_, std::make_pair((size_t)0, count_y_total)),
                Kokkos::subview(recv_y_top_, std::make_pair((size_t)0, count_y_total)));
            Kokkos::deep_copy(exec_space_,
                Kokkos::subview(recv_y_top_, std::make_pair((size_t)0, count_y_total)),
                Kokkos::subview(recv_y_top_, std::make_pair(count_y_total, 2 * count_y_total)));
        } else {
            ncclGroupStart();
            if (neighbor_top    != MPI_PROC_NULL) ncclSend(send_y_top_.data(),    count_y_total, VVM_NCCL_REAL, neighbor_top,    nccl_comm_, stream_);
            if (neighbor_bottom != MPI_PROC_NULL) ncclRecv(recv_y_bottom_.data(), count_y_total, VVM_NCCL_REAL, neighbor_bottom, nccl_comm_, stream_);
            if (neighbor_bottom != MPI_PROC_NULL) ncclSend(send_y_bottom_.data(), count_y_total, VVM_NCCL_REAL, neighbor_bottom, nccl_comm_, stream_);
            if (neighbor_top    != MPI_PROC_NULL) ncclRecv(recv_y_top_.data(),    count_y_total, VVM_NCCL_REAL, neighbor_top,    nccl_comm_, stream_);
            ncclGroupEnd();
        }

        for (size_t f = 0; f < num_fields; ++f) {
            auto data = fields[f]->get_mutable_device_data();
            const size_t offset = f * stride_y;
            auto recv_b = Kokkos::subview(recv_y_bottom_, std::make_pair(offset, offset + stride_y));
            auto recv_t = Kokkos::subview(recv_y_top_,    std::make_pair(offset, offset + stride_y));
            Kokkos::parallel_for("unpack_multi_y_2d", Kokkos::MDRangePolicy<Kokkos::Rank<2>, ExecSpace>(exec_space_, {0,0}, {nx, h}),
                KOKKOS_LAMBDA(int i, int j_h) {
                    const size_t idx = static_cast<size_t>(j_h) * nx + i;
                    if (neighbor_bottom != MPI_PROC_NULL) data(halo_start_offset - h + j_h, i) = recv_b(idx);
                    if (neighbor_top    != MPI_PROC_NULL) data(halo_start_offset + ny_phys + j_h, i) = recv_t(idx);
            });
        }
    }

    // No stream sync here: this is called from inside CUDA graph capture, where a
    // sync is illegal. Callers outside capture must fence themselves.
}

inline void HaloExchanger::exchange_halos(State& state) {
    // Kokkos::fence();

    for (auto& field_pair : state) {
        std::visit([this](auto& field) {
            using T = std::decay_t<decltype(field)>;
            if constexpr (!std::is_same_v<T, std::monostate>) {
                const std::string name = field.get_name();

                if (enabled_graph_vars_.count(name)) {
                    auto it = graph_map_.find(name);

                    if (it == graph_map_.end()) {
                        cudaStreamBeginCapture(stream_, cudaStreamCaptureModeGlobal);

                        this->exchange_halos_impl(field);

                        cudaGraph_t graph;
                        cudaStreamEndCapture(stream_, &graph);

                        cudaGraphExec_t instance;
                        cudaGraphInstantiate(&instance, graph, nullptr, nullptr, 0);
                        cudaGraphDestroy(graph);

                        graph_map_[name] = instance;
                        it = graph_map_.find(name);
                    }

                    cudaGraphLaunch(it->second, stream_);
                }
                else {
                    this->exchange_halos_impl(field);
                }
            }
        }, field_pair.second);
    }

    cudaStreamCaptureStatus capture_status;
    cudaStreamIsCapturing(stream_, &capture_status);
}

template<typename FieldT>
void HaloExchanger::exchange_halos_impl(FieldT& field, int depth) const {
    constexpr size_t Dim = FieldT::DimValue;
    const int halo_start_offset = grid_ref_.get_halo_cells();
    int h = (depth == -1) ? grid_ref_.get_halo_cells() : depth;
    if (h == 0) return;

    auto data = field.get_mutable_device_data();
    const int nx_phys = grid_ref_.get_local_physical_points_x();
    const int ny_phys = grid_ref_.get_local_physical_points_y();

    if (is_single_rank_ &&
        (grid_ref_.is_singleton_x() || grid_ref_.is_singleton_y())) {
        Detail::fill_periodic_x_halo(
            exec_space_, field, halo_start_offset, nx_phys, h);
        Detail::fill_periodic_y_halo(
            exec_space_, field, halo_start_offset, ny_phys, h);
        return;
    }

    if (grid_ref_.is_singleton_x()) {
        Detail::fill_singleton_x_halo(
            exec_space_, field, halo_start_offset, h);
    }
    if (grid_ref_.is_singleton_y()) {
        Detail::fill_singleton_y_halo(
            exec_space_, field, halo_start_offset, h);
    }

    if (is_single_rank_) {
        // X-Direction Periodic Copy
        // Left Halo (start-h .. start) <== Right Phys (start+nx_phys-h .. start+nx_phys)
        // Right Halo (start+nx_phys .. end) <== Left Phys (start .. start+h)
        if constexpr (Dim == 2) {
            const int ny = data.extent(0);
            Kokkos::parallel_for("local_copy_x_2d", Kokkos::MDRangePolicy<Kokkos::Rank<2>, ExecSpace>(exec_space_, {0,0}, {ny, h}),
                KOKKOS_LAMBDA(int j, int i_h) {
                    // Left Halo gets Right Physical
                    data(j, halo_start_offset - h + i_h) = data(j, halo_start_offset + nx_phys - h + i_h);
                    // Right Halo gets Left Physical
                    data(j, halo_start_offset + nx_phys + i_h) = data(j, halo_start_offset + i_h);
            });
        }
        else if constexpr (Dim == 3) {
            const int nz = data.extent(0);
            const int ny = data.extent(1);
            Kokkos::parallel_for("local_copy_x_3d", Kokkos::MDRangePolicy<Kokkos::Rank<3>, ExecSpace>(exec_space_, {0,0,0}, {nz, ny, h}),
                KOKKOS_LAMBDA(int k, int j, int i_h) {
                    // Left Halo gets Right Physical
                    data(k, j, halo_start_offset - h + i_h) = data(k, j, halo_start_offset + nx_phys - h + i_h);
                    // Right Halo gets Left Physical
                    data(k, j, halo_start_offset + nx_phys + i_h) = data(k, j, halo_start_offset + i_h);
            });
        }
        else if constexpr (Dim == 4) {
            const int nw = data.extent(0);
            const int nz = data.extent(1);
            const int ny = data.extent(2);
            Kokkos::parallel_for("local_copy_x_4d", Kokkos::MDRangePolicy<Kokkos::Rank<4>, ExecSpace>(exec_space_, {0,0,0,0}, {nw, nz, ny, h}),
                KOKKOS_LAMBDA(int w, int k, int j, int i_h) {
                    data(w, k, j, halo_start_offset - h + i_h) = data(w, k, j, halo_start_offset + nx_phys - h + i_h);
                    data(w, k, j, halo_start_offset + nx_phys + i_h) = data(w, k, j, halo_start_offset + i_h);
            });
        }

        // Y-Direction Periodic Copy
        // Bottom Halo <== Top Phys
        // Top Halo <== Bottom Phys
        if constexpr (Dim == 2) {
            const int nx = data.extent(1);
            Kokkos::parallel_for("local_copy_y_2d", Kokkos::MDRangePolicy<Kokkos::Rank<2>, ExecSpace>(exec_space_, {0,0}, {nx, h}),
                KOKKOS_LAMBDA(int i, int j_h) {
                    // Bottom Halo gets Top Physical
                    data(halo_start_offset - h + j_h, i) = data(halo_start_offset + ny_phys - h + j_h, i);
                    // Top Halo gets Bottom Physical
                    data(halo_start_offset + ny_phys + j_h, i) = data(halo_start_offset + j_h, i);
            });
        }
        else if constexpr (Dim == 3) {
            const int nz = data.extent(0);
            const int nx = data.extent(2);
            Kokkos::parallel_for("local_copy_y_3d", Kokkos::MDRangePolicy<Kokkos::Rank<3>, ExecSpace>(exec_space_, {0,0,0}, {nz, nx, h}),
                KOKKOS_LAMBDA(int k, int i, int j_h) {
                    data(k, halo_start_offset - h + j_h, i) = data(k, halo_start_offset + ny_phys - h + j_h, i);
                    data(k, halo_start_offset + ny_phys + j_h, i) = data(k, halo_start_offset + j_h, i);
            });
        }
        else if constexpr (Dim == 4) {
            const int nw = data.extent(0);
            const int nz = data.extent(1);
            const int nx = data.extent(3);
            Kokkos::parallel_for("local_copy_y_4d", Kokkos::MDRangePolicy<Kokkos::Rank<4>, ExecSpace>(exec_space_, {0,0,0,0}, {nw, nz, nx, h}),
                KOKKOS_LAMBDA(int w, int k, int i, int j_h) {
                    data(w, k, halo_start_offset - h + j_h, i) = data(w, k, halo_start_offset + ny_phys - h + j_h, i);
                    data(w, k, halo_start_offset + ny_phys + j_h, i) = data(w, k, halo_start_offset + j_h, i);
            });
        }

        return; // Skip the rest of NCCL/MPI logic
    }

    const int neighbor_left = neighbor_left_;
    const int neighbor_right = neighbor_right_;
    const int neighbor_bottom = neighbor_bottom_;
    const int neighbor_top = neighbor_top_;

    const int my_rank = grid_ref_.get_mpi_rank();
    const bool same_remote_x =
        neighbor_left != MPI_PROC_NULL && neighbor_left == neighbor_right && neighbor_left != my_rank;
    const bool same_remote_y =
        neighbor_bottom != MPI_PROC_NULL && neighbor_bottom == neighbor_top && neighbor_bottom != my_rank;

    size_t count_x = 0;
    size_t count_y = 0;

    if constexpr (Dim == 2) {
        count_x = static_cast<size_t>(h) * data.extent(0);
        count_y = static_cast<size_t>(h) * data.extent(1);
    }
    else if constexpr (Dim == 3) {
        count_x = static_cast<size_t>(h) * data.extent(1) * data.extent(0);
        count_y = static_cast<size_t>(h) * data.extent(2) * data.extent(0);
    }
    else if constexpr (Dim == 4) {
        count_x = static_cast<size_t>(h) * data.extent(2) * data.extent(1) * data.extent(0);
        count_y = static_cast<size_t>(h) * data.extent(3) * data.extent(1) * data.extent(0);
    }

    if (count_x > 0 && !grid_ref_.is_singleton_x()) {
        auto send_l = Kokkos::subview(send_x_left_, std::make_pair((size_t)0, count_x));
        auto recv_l = Kokkos::subview(recv_x_left_, std::make_pair((size_t)0, count_x));
        auto send_r = Kokkos::subview(send_x_right_, std::make_pair((size_t)0, count_x));
        auto recv_r = Kokkos::subview(recv_x_right_, std::make_pair((size_t)0, count_x));

        if constexpr (Dim == 2) {
            const int ny = data.extent(0);
            Kokkos::parallel_for("pack_x_2d", Kokkos::MDRangePolicy<Kokkos::Rank<2>, ExecSpace>(exec_space_, {0,0}, {ny, h}),
                KOKKOS_LAMBDA(int j, int i_h) {
                    send_l(j * h + i_h) = data(j, halo_start_offset + i_h);
                    send_r(j * h + i_h) = data(j, halo_start_offset + nx_phys - h + i_h);
            });
        }
        else if constexpr (Dim == 3) {
            const int nz = data.extent(0);
            const int ny = data.extent(1);
            Kokkos::parallel_for("pack_x_3d", Kokkos::MDRangePolicy<Kokkos::Rank<3>, ExecSpace>(exec_space_, {0,0,0}, {nz, ny, h}),
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
            Kokkos::parallel_for("pack_x_4d", Kokkos::MDRangePolicy<Kokkos::Rank<4>, ExecSpace>(exec_space_, {0,0,0,0}, {nw, nz, ny, h}),
                KOKKOS_LAMBDA(int w, int k, int j, int i_h) {
                    size_t idx = w * (nz*ny*h) + k * (ny*h) + j * h + i_h;
                    send_l(idx) = data(w, k, j, halo_start_offset + i_h);
                    send_r(idx) = data(w, k, j, halo_start_offset + nx_phys - h + i_h);
            });
        }

        if (neighbor_left != MPI_PROC_NULL && neighbor_left == my_rank && neighbor_right == my_rank) {
            Kokkos::deep_copy(exec_space_, recv_l, send_r);
            Kokkos::deep_copy(exec_space_, recv_r, send_l);
        }
        else if (same_remote_x) {
            Kokkos::deep_copy(exec_space_,
                Kokkos::subview(send_x_right_, std::make_pair(count_x, 2 * count_x)),
                send_l);
            ncclGroupStart();
            ncclSend(send_x_right_.data(), 2 * count_x, VVM_NCCL_REAL, neighbor_right, nccl_comm_, stream_);
            ncclRecv(recv_x_right_.data(), 2 * count_x, VVM_NCCL_REAL, neighbor_left, nccl_comm_, stream_);
            ncclGroupEnd();
            Kokkos::deep_copy(exec_space_, recv_l,
                Kokkos::subview(recv_x_right_, std::make_pair((size_t)0, count_x)));
            Kokkos::deep_copy(exec_space_, recv_r,
                Kokkos::subview(recv_x_right_, std::make_pair(count_x, 2 * count_x)));
        }
        else {
            // NCCL Communication
            ncclGroupStart();
            if(neighbor_right != MPI_PROC_NULL)
                ncclSend(send_r.data(), count_x, VVM_NCCL_REAL, neighbor_right, nccl_comm_, stream_);
            if(neighbor_left != MPI_PROC_NULL)
                ncclRecv(recv_l.data(), count_x, VVM_NCCL_REAL, neighbor_left, nccl_comm_, stream_);

            if(neighbor_left != MPI_PROC_NULL)
                ncclSend(send_l.data(), count_x, VVM_NCCL_REAL, neighbor_left, nccl_comm_, stream_);
            if(neighbor_right != MPI_PROC_NULL)
                ncclRecv(recv_r.data(), count_x, VVM_NCCL_REAL, neighbor_right, nccl_comm_, stream_);
            ncclGroupEnd();
        }

        // Unpack Data from Buffers
        if constexpr (Dim == 2) {
            const int ny = data.extent(0);
            Kokkos::parallel_for("unpack_x_2d", Kokkos::MDRangePolicy<Kokkos::Rank<2>, ExecSpace>(exec_space_, {0,0}, {ny, h}),
                KOKKOS_LAMBDA(int j, int i_h) {
                    if (neighbor_left != MPI_PROC_NULL) data(j, halo_start_offset - h + i_h) = recv_l(j * h + i_h);
                    if (neighbor_right != MPI_PROC_NULL) data(j, halo_start_offset + nx_phys + i_h) = recv_r(j * h + i_h);
            });
        }
        else if constexpr (Dim == 3) {
            const int nz = data.extent(0);
            const int ny = data.extent(1);
            Kokkos::parallel_for("unpack_x_3d", Kokkos::MDRangePolicy<Kokkos::Rank<3>, ExecSpace>(exec_space_, {0,0,0}, {nz, ny, h}),
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
            Kokkos::parallel_for("unpack_x_4d", Kokkos::MDRangePolicy<Kokkos::Rank<4>, ExecSpace>(exec_space_, {0,0,0,0}, {nw, nz, ny, h}),
                KOKKOS_LAMBDA(int w, int k, int j, int i_h) {
                    size_t idx = w * (nz*ny*h) + k * (ny*h) + j * h + i_h;
                    if (neighbor_left != MPI_PROC_NULL) data(w, k, j, halo_start_offset - h + i_h) = recv_l(idx);
                    if (neighbor_right != MPI_PROC_NULL) data(w, k, j, halo_start_offset + nx_phys + i_h) = recv_r(idx);
            });
        }
    }

    if (count_y > 0 && !grid_ref_.is_singleton_y()) {
        auto send_b = Kokkos::subview(send_y_bottom_, std::make_pair((size_t)0, count_y));
        auto recv_b = Kokkos::subview(recv_y_bottom_, std::make_pair((size_t)0, count_y));
        auto send_t = Kokkos::subview(send_y_top_, std::make_pair((size_t)0, count_y));
        auto recv_t = Kokkos::subview(recv_y_top_, std::make_pair((size_t)0, count_y));

        // Pack
        if constexpr (Dim == 2) {
            const int nx = data.extent(1);
            Kokkos::parallel_for("pack_y_2d", Kokkos::MDRangePolicy<Kokkos::Rank<2>, ExecSpace>(exec_space_, {0,0}, {nx, h}),
                KOKKOS_LAMBDA(int i, int j_h) {
                    const size_t idx = j_h * nx + i;
                    send_b(idx) = data(halo_start_offset + j_h, i);
                    send_t(idx) = data(halo_start_offset + ny_phys - h + j_h, i);
            });
        } else if constexpr (Dim == 3) {
            const int nz = data.extent(0);
            const int nx = data.extent(2);
            Kokkos::parallel_for("pack_y_3d", Kokkos::MDRangePolicy<Kokkos::Rank<3>, ExecSpace>(exec_space_, {0,0,0}, {nz, nx, h}),
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
            Kokkos::parallel_for("pack_y_4d", Kokkos::MDRangePolicy<Kokkos::Rank<4>, ExecSpace>(exec_space_, {0,0,0,0}, {nw, nz, nx, h}),
                KOKKOS_LAMBDA(int w, int k, int i, int j_h) {
                    size_t idx = w * (nz*h*nx) + k * (h*nx) + j_h * nx + i;
                    send_b(idx) = data(w, k, halo_start_offset + j_h, i);
                    send_t(idx) = data(w, k, halo_start_offset + ny_phys - h + j_h, i);
            });
        }

        if (neighbor_bottom != MPI_PROC_NULL && neighbor_bottom == my_rank && neighbor_top == my_rank) {
             Kokkos::deep_copy(exec_space_, recv_b, send_t);
             Kokkos::deep_copy(exec_space_, recv_t, send_b);
        }
        else if (same_remote_y) {
             Kokkos::deep_copy(exec_space_,
                 Kokkos::subview(send_y_top_, std::make_pair(count_y, 2 * count_y)),
                 send_b);
             ncclGroupStart();
             ncclSend(send_y_top_.data(), 2 * count_y, VVM_NCCL_REAL, neighbor_top, nccl_comm_, stream_);
             ncclRecv(recv_y_top_.data(), 2 * count_y, VVM_NCCL_REAL, neighbor_bottom, nccl_comm_, stream_);
             ncclGroupEnd();
             Kokkos::deep_copy(exec_space_, recv_b,
                 Kokkos::subview(recv_y_top_, std::make_pair((size_t)0, count_y)));
             Kokkos::deep_copy(exec_space_, recv_t,
                 Kokkos::subview(recv_y_top_, std::make_pair(count_y, 2 * count_y)));
        }
        else {
            // NCCL
            ncclGroupStart();
            if(neighbor_top != MPI_PROC_NULL)
                ncclSend(send_t.data(), count_y, VVM_NCCL_REAL, neighbor_top, nccl_comm_, stream_);
            if(neighbor_bottom != MPI_PROC_NULL)
                ncclRecv(recv_b.data(), count_y, VVM_NCCL_REAL, neighbor_bottom, nccl_comm_, stream_);

            if(neighbor_bottom != MPI_PROC_NULL)
                ncclSend(send_b.data(), count_y, VVM_NCCL_REAL, neighbor_bottom, nccl_comm_, stream_);
            if(neighbor_top != MPI_PROC_NULL)
                ncclRecv(recv_t.data(), count_y, VVM_NCCL_REAL, neighbor_top, nccl_comm_, stream_);
            ncclGroupEnd();
        }

        // Unpack
        if constexpr (Dim == 2) {
            const int nx = data.extent(1);
            Kokkos::parallel_for("unpack_y_2d", Kokkos::MDRangePolicy<Kokkos::Rank<2>, ExecSpace>(exec_space_, {0,0}, {nx, h}),
                KOKKOS_LAMBDA(int i, int j_h) {
                    const size_t idx = j_h * nx + i;
                    if (neighbor_bottom != MPI_PROC_NULL) data(halo_start_offset - h + j_h, i) = recv_b(idx);
                    if (neighbor_top != MPI_PROC_NULL) data(halo_start_offset + ny_phys + j_h, i) = recv_t(idx);
            });
        } else if constexpr (Dim == 3) {
            const int nz = data.extent(0);
            const int nx = data.extent(2);
            Kokkos::parallel_for("unpack_y_3d", Kokkos::MDRangePolicy<Kokkos::Rank<3>, ExecSpace>(exec_space_, {0,0,0}, {nz, nx, h}),
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
            Kokkos::parallel_for("unpack_y_4d", Kokkos::MDRangePolicy<Kokkos::Rank<4>, ExecSpace>(exec_space_, {0,0,0,0}, {nw, nz, nx, h}),
                KOKKOS_LAMBDA(int w, int k, int i, int j_h) {
                    size_t idx = w * (nz*h*nx) + k * (h*nx) + j_h * nx + i;
                    if (neighbor_bottom != MPI_PROC_NULL) data(w, k, halo_start_offset - h + j_h, i) = recv_b(idx);
                    if (neighbor_top != MPI_PROC_NULL) data(w, k, halo_start_offset + ny_phys + j_h, i) = recv_t(idx);
            });
        }
    }

    cudaStreamCaptureStatus capture_status;
    cudaStreamIsCapturing(stream_, &capture_status);
    if (depth == -1 && capture_status == cudaStreamCaptureStatusNone && grid_ref_.get_mpi_size() > 1) {
         cudaStreamSynchronize(stream_);
    }
}

inline void HaloExchanger::exchange_halos_slice(Field<3>& field, int k_layer) const {
    const int halo_start_offset = grid_ref_.get_halo_cells();
    const int h = grid_ref_.get_halo_cells();
    if (h == 0) return;

    auto data = field.get_mutable_device_data();
    const int ny_phys = grid_ref_.get_local_physical_points_y();
    const int nx_phys = grid_ref_.get_local_physical_points_x();
    const int ny = data.extent(1);
    const int nx = data.extent(2);

    if (is_single_rank_ &&
        (grid_ref_.is_singleton_x() || grid_ref_.is_singleton_y())) {
        Detail::fill_periodic_x_slice(
            exec_space_, field, k_layer, h, nx_phys, h);
        Detail::fill_periodic_y_slice(
            exec_space_, field, k_layer, h, ny_phys, h);
        cudaStreamSynchronize(stream_);
        return;
    }
    if (grid_ref_.is_singleton_x()) {
        Detail::fill_singleton_x_slice(
            exec_space_, field, k_layer, h, h);
    }
    if (grid_ref_.is_singleton_y()) {
        Detail::fill_singleton_y_slice(
            exec_space_, field, k_layer, h, h);
    }

    const int neighbor_left = neighbor_left_;
    const int neighbor_right = neighbor_right_;
    const int neighbor_bottom = neighbor_bottom_;
    const int neighbor_top = neighbor_top_;
    const int my_rank = grid_ref_.get_mpi_rank();
    const bool same_remote_x =
        neighbor_left != MPI_PROC_NULL && neighbor_left == neighbor_right && neighbor_left != my_rank;
    const bool same_remote_y =
        neighbor_bottom != MPI_PROC_NULL && neighbor_bottom == neighbor_top && neighbor_bottom != my_rank;

    // Kokkos::fence();

    // --- Y-Direction Slice ---
    if (!grid_ref_.is_singleton_y()) {
        size_t count = buffer_size_slice_y_;
        if (count > 0) {
            auto send_b = Kokkos::subview(send_y_bottom_, std::make_pair((size_t)0, count));
            auto recv_b = Kokkos::subview(recv_y_bottom_, std::make_pair((size_t)0, count));
            auto send_t = Kokkos::subview(send_y_top_, std::make_pair((size_t)0, count));
            auto recv_t = Kokkos::subview(recv_y_top_, std::make_pair((size_t)0, count));

            Kokkos::parallel_for("pack_y_slice", Kokkos::MDRangePolicy<Kokkos::Rank<2>, ExecSpace>(exec_space_, {0, 0}, {nx, h}),
                KOKKOS_LAMBDA(int i, int j_h) {
                    const size_t idx = static_cast<size_t>(j_h) * nx + i;
                    send_b(idx) = data(k_layer, halo_start_offset + j_h, i);
                    send_t(idx) = data(k_layer, halo_start_offset + ny_phys - h + j_h, i);
            });

            if (neighbor_bottom == my_rank && neighbor_top == my_rank) {
                Kokkos::deep_copy(exec_space_, recv_b, send_t);
                Kokkos::deep_copy(exec_space_, recv_t, send_b);
            } else if (same_remote_y) {
                Kokkos::deep_copy(exec_space_,
                    Kokkos::subview(send_y_top_, std::make_pair(count, 2 * count)),
                    send_b);
                ncclGroupStart();
                ncclSend(send_y_top_.data(), 2 * count, VVM_NCCL_REAL, neighbor_top, nccl_comm_, stream_);
                ncclRecv(recv_y_top_.data(), 2 * count, VVM_NCCL_REAL, neighbor_bottom, nccl_comm_, stream_);
                ncclGroupEnd();
                Kokkos::deep_copy(exec_space_, recv_b,
                    Kokkos::subview(recv_y_top_, std::make_pair((size_t)0, count)));
                Kokkos::deep_copy(exec_space_, recv_t,
                    Kokkos::subview(recv_y_top_, std::make_pair(count, 2 * count)));
            } else {
                ncclGroupStart();
                if(neighbor_top != MPI_PROC_NULL) ncclSend(send_t.data(), count, VVM_NCCL_REAL, neighbor_top, nccl_comm_, stream_);
                if(neighbor_bottom != MPI_PROC_NULL) ncclRecv(recv_b.data(), count, VVM_NCCL_REAL, neighbor_bottom, nccl_comm_, stream_);
                if(neighbor_bottom != MPI_PROC_NULL) ncclSend(send_b.data(), count, VVM_NCCL_REAL, neighbor_bottom, nccl_comm_, stream_);
                if(neighbor_top != MPI_PROC_NULL) ncclRecv(recv_t.data(), count, VVM_NCCL_REAL, neighbor_top, nccl_comm_, stream_);
                ncclGroupEnd();
            }

            Kokkos::parallel_for("unpack_y_slice", Kokkos::MDRangePolicy<Kokkos::Rank<2>, ExecSpace>(exec_space_, {0, 0}, {nx, h}),
                KOKKOS_LAMBDA(int i, int j_h) {
                    const size_t idx = static_cast<size_t>(j_h) * nx + i;
                    if (neighbor_bottom != MPI_PROC_NULL) data(k_layer, halo_start_offset - h + j_h, i) = recv_b(idx);
                    if (neighbor_top != MPI_PROC_NULL) data(k_layer, halo_start_offset + ny_phys + j_h, i) = recv_t(idx);
            });
        }
    }

    // --- X-Direction Slice ---
    if (!grid_ref_.is_singleton_x()) {
        size_t count = buffer_size_slice_x_;
        if (count > 0) {
            auto send_l = Kokkos::subview(send_x_left_, std::make_pair((size_t)0, count));
            auto recv_l = Kokkos::subview(recv_x_left_, std::make_pair((size_t)0, count));
            auto send_r = Kokkos::subview(send_x_right_, std::make_pair((size_t)0, count));
            auto recv_r = Kokkos::subview(recv_x_right_, std::make_pair((size_t)0, count));

            Kokkos::parallel_for("pack_x_slice", Kokkos::MDRangePolicy<Kokkos::Rank<2>, ExecSpace>(exec_space_, {0, 0}, {ny, h}),
                KOKKOS_LAMBDA(int j, int i_h) {
                    const size_t idx = static_cast<size_t>(j) * h + i_h;
                    send_l(idx) = data(k_layer, j, halo_start_offset + i_h);
                    send_r(idx) = data(k_layer, j, halo_start_offset + nx_phys - h + i_h);
            });

            if (neighbor_left == my_rank && neighbor_right == my_rank) {
                Kokkos::deep_copy(exec_space_, recv_l, send_r);
                Kokkos::deep_copy(exec_space_, recv_r, send_l);
            } else if (same_remote_x) {
                Kokkos::deep_copy(exec_space_,
                    Kokkos::subview(send_x_right_, std::make_pair(count, 2 * count)),
                    send_l);
                ncclGroupStart();
                ncclSend(send_x_right_.data(), 2 * count, VVM_NCCL_REAL, neighbor_right, nccl_comm_, stream_);
                ncclRecv(recv_x_right_.data(), 2 * count, VVM_NCCL_REAL, neighbor_left, nccl_comm_, stream_);
                ncclGroupEnd();
                Kokkos::deep_copy(exec_space_, recv_l,
                    Kokkos::subview(recv_x_right_, std::make_pair((size_t)0, count)));
                Kokkos::deep_copy(exec_space_, recv_r,
                    Kokkos::subview(recv_x_right_, std::make_pair(count, 2 * count)));
            } else {
                ncclGroupStart();
                if(neighbor_right != MPI_PROC_NULL) ncclSend(send_r.data(), count, VVM_NCCL_REAL, neighbor_right, nccl_comm_, stream_);
                if(neighbor_left != MPI_PROC_NULL) ncclRecv(recv_l.data(), count, VVM_NCCL_REAL, neighbor_left, nccl_comm_, stream_);
                if(neighbor_left != MPI_PROC_NULL) ncclSend(send_l.data(), count, VVM_NCCL_REAL, neighbor_left, nccl_comm_, stream_);
                if(neighbor_right != MPI_PROC_NULL) ncclRecv(recv_r.data(), count, VVM_NCCL_REAL, neighbor_right, nccl_comm_, stream_);
                ncclGroupEnd();
            }

            Kokkos::parallel_for("unpack_x_slice", Kokkos::MDRangePolicy<Kokkos::Rank<2>, ExecSpace>(exec_space_, {0, 0}, {ny, h}),
                KOKKOS_LAMBDA(int j, int i_h) {
                    const size_t idx = static_cast<size_t>(j) * h + i_h;
                    if (neighbor_left != MPI_PROC_NULL) data(k_layer, j, halo_start_offset - h + i_h) = recv_l(idx);
                    if (neighbor_right != MPI_PROC_NULL) data(k_layer, j, halo_start_offset + nx_phys + i_h) = recv_r(idx);
            });
        }
    }

    cudaStreamSynchronize(stream_);
}

} // namespace Core
} // namespace VVM
