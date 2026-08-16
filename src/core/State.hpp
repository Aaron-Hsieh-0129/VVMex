// State class integrates various fields in the simulation.

#ifndef VVM_CORE_VVMSTATE_HPP
#define VVM_CORE_VVMSTATE_HPP

#include "Grid.hpp"
#include "Field.hpp"
#include "utils/ConfigurationManager.hpp"
#include "Parameters.hpp"
#include "vvm_types.hpp"
#include <algorithm>
#include <map>
#include <string>
#include <memory>
#include <stdexcept>
#include <variant>
#include <vector>
#if defined(KOKKOS_ENABLE_CUDA)
    #include <cuda_runtime.h>
#endif
#if defined(ENABLE_NCCL)
    #include <nccl.h>
#endif

namespace VVM { namespace Dynamics { class AdamsBashforth2; } }

namespace VVM {
namespace Core {

// Device-resident scalar, the result type of the horizontal-mean reduction.
using ScalarView = Kokkos::View<VVM::Real, Kokkos::DefaultExecutionSpace::memory_space>;

// A variant that can hold a field of any supported dimension
using AnyField = std::variant<
    std::monostate, // default state
    Field<0>,
    Field<1>,
    Field<2>,
    Field<3>,
    Field<4>
>;

class State {
    friend class VVM::Dynamics::AdamsBashforth2;
public:
    // Constructor
#if defined(ENABLE_NCCL)
    State(const Utils::ConfigurationManager& config, const Parameters& params, const Grid& grid, ncclComm_t nccl_comm,
          cudaStream_t nccl_stream);
#else
    State(const Utils::ConfigurationManager& config, const Parameters& params, const Grid& grid);
#endif

#if defined(ENABLE_NCCL)
    ncclComm_t get_nccl_comm() const { return nccl_comm_; }
    cudaStream_t get_cuda_stream() const { return nccl_stream_; }
#endif

    template<size_t Dim>
    void add_field(const std::string& name, std::initializer_list<int> dims_list, FieldMetadata metadata = {}) {
        if (dims_list.size() != Dim) {
            throw std::runtime_error("Dimension mismatch for field '" + name + "'");
        }
        std::array<int, Dim> dims;
        std::copy(dims_list.begin(), dims_list.end(), dims.begin());
        auto [it, inserted] = fields_.try_emplace(name, std::in_place_type_t<Field<Dim>>(), name, dims, std::move(metadata));
        if (inserted) std::get<Field<Dim>>(it->second).set_to_zero();
    }

    template<size_t Dim>
    void add_field(const std::string& name, const std::array<int, Dim>& dims, FieldMetadata metadata = {}) {
        auto [it, inserted] = fields_.try_emplace(name, std::in_place_type_t<Field<Dim>>(), name, dims, std::move(metadata));
        if (inserted) std::get<Field<Dim>>(it->second).set_to_zero();
    }

    // Get a field by name
    template<size_t Dim>
    Field<Dim>& get_field(const std::string& name) {
        try { 
            return std::get<Field<Dim>>(fields_.at(name));
        }
        catch (const std::out_of_range& e) {
            throw std::runtime_error("Field '" + name + "' not found in State.");
        }
        catch (const std::bad_variant_access& e) {
            throw std::runtime_error("Field '" + name + "' has incorrect dimension.");
        }
    }
    
    template<size_t Dim>
    const Field<Dim>& get_field(const std::string& name) const {
        try { 
            return std::get<Field<Dim>>(fields_.at(name));
        }
        catch (const std::out_of_range& e) {
            throw std::runtime_error("Field '" + name + "' not found in State.");
        }
        catch (const std::bad_variant_access& e) {
            throw std::runtime_error("Field '" + name + "' has incorrect dimension.");
        }
    }


    // Global horizontal mean over the physical (halo-free) cells of every rank:
    //
    //     mean = sum(phi over all physical horizontal cells) / (gnx * gny)
    //
    // One call-site API for both communication backends. Only the global sum
    // differs: an NCCL all-reduce on the device, or a host MPI_Allreduce. The
    // local reduction, the level convention and the normalization are shared.
    //
    // For 3D fields, k_level = -1 (the default) means the highest physical
    // level, nz - h - 1. An explicit level inside the halo is an error.
    template<size_t Dim>
    void calculate_horizontal_mean(
        const Field<Dim>& field,
        ScalarView d_mean_result,
        int k_level = -1) const
    {
        static_assert(Dim == 2 || Dim == 3,
                      "calculate_horizontal_mean supports 2D and 3D fields only.");

        auto view = field.get_device_data();

        const int ny_local = grid_.get_local_physical_points_y();
        const int nx_local = grid_.get_local_physical_points_x();
        const int h = grid_.get_halo_cells();
        const int gnx = grid_.get_global_points_x();
        const int gny = grid_.get_global_points_y();
        // Multiplied as Real, not as int: gnx * gny overflows a 32-bit int past
        // roughly 46k x 46k.
        const VVM::Real total_points_horizontal =
            static_cast<VVM::Real>(gnx) * static_cast<VVM::Real>(gny);

        if constexpr (Dim == 3) {
            const int nz = grid_.get_local_total_points_z();
            if (k_level == -1) k_level = nz - h - 1;
            if (k_level < h || k_level >= nz - h) {
                throw std::out_of_range(
                    "calculate_horizontal_mean: k_level " + std::to_string(k_level) +
                    " is outside the physical range [" + std::to_string(h) + ", " +
                    std::to_string(nz - h - 1) + "] of the 3D field '" +
                    field.get_name() + "'.");
            }
        }

        if (total_points_horizontal == VVM::real(0.0)) {
            Kokkos::deep_copy(d_mean_result, VVM::real(0.0));
            return;
        }

        ScalarView d_local_sum("horizontal_mean_local_sum");

#if defined(VVM_DETERMINISTIC_FP)
        // Fixed-order local sum: each row is summed sequentially over i, and the
        // rows are then summed in order. A parallel_reduce combines its partial
        // sums in whatever order the backend's thread layout produces, and
        // CUDA's shuffle tree and OpenMP's per-thread partials do not agree --
        // measured at ~10 ULP over a million values. Row-then-rank ordering is
        // an order both backends can commit to, and it is the same discipline
        // the rank sums below already use. The rows are still summed in
        // parallel, so the O(nx*ny) work stays parallel; only the O(ny) combine
        // is serial. Off by default because it changes the answer in the last
        // ULP, and the v1.0.0 baselines encode the parallel_reduce order.
        if (static_cast<int>(row_sums_.extent(0)) != ny_local) {
            row_sums_ = Kokkos::View<VVM::Real*, Kokkos::DefaultExecutionSpace::memory_space>(
                "horizontal_mean_row_sums", ny_local);
        }
        auto row_sums = row_sums_;
        const int nx_count = nx_local;
        const int ny_count = ny_local;
        const int halo = h;

        if constexpr (Dim == 3) {
            Kokkos::parallel_for("calculate_3d_row_sums",
                Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(0, ny_local),
                KOKKOS_LAMBDA(const int row) {
                    VVM::Real sum = VVM::real(0.0);
                    for (int i = 0; i < nx_count; ++i) sum += view(k_level, halo + row, halo + i);
                    row_sums(row) = sum;
                });
        }
        else {
            Kokkos::parallel_for("calculate_2d_row_sums",
                Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(0, ny_local),
                KOKKOS_LAMBDA(const int row) {
                    VVM::Real sum = VVM::real(0.0);
                    for (int i = 0; i < nx_count; ++i) sum += view(halo + row, halo + i);
                    row_sums(row) = sum;
                });
        }

        Kokkos::parallel_for("calculate_local_sum",
            Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(0, 1),
            KOKKOS_LAMBDA(const int) {
                VVM::Real sum = VVM::real(0.0);
                for (int row = 0; row < ny_count; ++row) sum += row_sums(row);
                d_local_sum() = sum;
            });
#else
        // The backend's own reduction order. Faster to reach, and what the
        // v1.0.0 baselines encode, at the cost of a last-ULP difference between
        // a CUDA and an OpenMP build -- see
        // docs/developer-guides/reproducibility.md.
        if constexpr (Dim == 3) {
            Kokkos::parallel_reduce("calculate_3d_local_sum",
                Kokkos::MDRangePolicy<Kokkos::Rank<2>>({h, h}, {ny_local + h, nx_local + h}),
                KOKKOS_LAMBDA(const int j, const int i, VVM::Real& update_sum) {
                    update_sum += view(k_level, j, i);
                }, d_local_sum);
        }
        else {
            Kokkos::parallel_reduce("calculate_2d_local_sum",
                Kokkos::MDRangePolicy<Kokkos::Rank<2>>({h, h}, {ny_local + h, nx_local + h}),
                KOKKOS_LAMBDA(const int j, const int i, VVM::Real& update_sum) {
                    update_sum += view(j, i);
                }, d_local_sum);
        }
#endif

        Kokkos::fence();

        // Both backends gather the per-rank partial sums and then add them in
        // rank order in the same device kernel below.
        //
        // The point is that floating-point addition is not associative, so a
        // reduction's answer depends on the order it combines the ranks --
        // NCCL's ring and MPI_Allreduce's algorithm agree on most data but not
        // all. Measured on a 2048^2 run over 8 ranks: they differed by one ULP
        // in vtopmn, which shifted v by a constant everywhere and grew to ~1e-12
        // over 120 steps. Rank order is an order both backends can commit to,
        // and it also stops the answer depending on which algorithm the library
        // picks for the current message size, topology or version.
        int comm_size = 1;
#if defined(ENABLE_NCCL)
        const ncclResult_t count_result = ncclCommCount(nccl_comm_, &comm_size);
        if (count_result != ncclSuccess) {
            throw std::runtime_error(
                "calculate_horizontal_mean could not query the NCCL communicator size: " +
                std::string(ncclGetErrorString(count_result)));
        }
#else
        MPI_Comm_size(grid_.get_comm(), &comm_size);
#endif

        if (static_cast<int>(rank_sums_.extent(0)) != comm_size) {
            rank_sums_ = Kokkos::View<VVM::Real*, Kokkos::DefaultExecutionSpace::memory_space>(
                "horizontal_mean_rank_sums", comm_size);
        }

#if defined(ENABLE_NCCL)
        const ncclResult_t result = ncclAllGather(
            d_local_sum.data(),
            rank_sums_.data(),
            1,
            VVM_NCCL_REAL,
            nccl_comm_,
            nccl_stream_
        );

        if (result != ncclSuccess) {
            throw std::runtime_error(
                "calculate_horizontal_mean NCCL all-gather failed for '" +
                field.get_name() + "': " + ncclGetErrorString(result));
        }

        cudaStreamSynchronize(nccl_stream_);
#else
        VVM::Real local_sum = VVM::real(0.0);
        Kokkos::deep_copy(local_sum, d_local_sum);

        std::vector<VVM::Real> host_rank_sums(static_cast<size_t>(comm_size), VVM::real(0.0));
        const int mpi_result = MPI_Allgather(
            &local_sum, 1, VVM_MPI_REAL,
            host_rank_sums.data(), 1, VVM_MPI_REAL, grid_.get_comm());

        if (mpi_result != MPI_SUCCESS) {
            throw std::runtime_error(
                "calculate_horizontal_mean MPI_Allgather failed for '" +
                field.get_name() + "'.");
        }

        Kokkos::deep_copy(
            rank_sums_,
            Kokkos::View<const VVM::Real*, Kokkos::HostSpace>(host_rank_sums.data(), comm_size));
#endif

        // Shared by both backends, deliberately: summing and normalizing in one
        // device kernel means the same additions in the same order and the same
        // division, so the two builds cannot drift apart here. (Normalizing on
        // the host in one backend would also risk differing under -use_fast_math
        // whenever gnx*gny is not a power of two.)
        auto rank_sums = rank_sums_;
        const int num_ranks = comm_size;
        Kokkos::parallel_for("horizontal_mean_finalize",
            Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(0, 1),
            KOKKOS_LAMBDA(const int) {
                VVM::Real global_sum = VVM::real(0.0);
                for (int r = 0; r < num_ranks; ++r) global_sum += rank_sums(r);
                d_mean_result() = global_sum / total_points_horizontal;
            });
    }

    // Provide iterators to loop over all fields
    auto begin() { return fields_.begin(); } // First value
    auto end() { return fields_.end(); } // Last value
    auto begin() const { return fields_.cbegin(); } // First key
    auto end() const { return fields_.cend(); } // Last key

    size_t get_step() const { return step_; }
    void set_step(size_t step) { step_ = step; }
    void increment_step() { step_++; }
    VVM::Real get_time() const { return time_; }
    void set_time(VVM::Real time) { time_ = time; }
    void advance_time(VVM::Real dt) { time_ += dt; }

    bool has_field(const std::string& name) const {
        return fields_.find(name) != fields_.end();
    }

    const std::vector<std::string>& get_tracer_names() const {
        return tracer_names_;
    }

    const std::vector<std::string>& get_tracer_source_targets() const {
        return tracer_source_targets_;
    }

    const std::vector<std::string>& get_tracer_source_names() const {
        return tracer_source_names_;
    }

    bool is_tracer(const std::string& name) const {
        return std::find(tracer_names_.begin(), tracer_names_.end(), name) != tracer_names_.end();
    }

    bool is_tracer_source(const std::string& name) const {
        return std::find(tracer_source_names_.begin(), tracer_source_names_.end(), name) !=
               tracer_source_names_.end();
    }

private:
    const Utils::ConfigurationManager& config_ref_;
    const Grid& grid_;
    const Parameters& parameters_;
    std::map<std::string, AnyField> fields_;
    std::vector<std::string> tracer_names_;
    std::vector<std::string> tracer_source_targets_;
    std::vector<std::string> tracer_source_names_;

    size_t step_ = 0;
    VVM::Real time_ = 0.0;

    // Per-rank partial sums for calculate_horizontal_mean(), kept across calls
    // so the mean does not allocate every time it is asked for.
    mutable Kokkos::View<VVM::Real*, Kokkos::DefaultExecutionSpace::memory_space> rank_sums_;
#if defined(VVM_DETERMINISTIC_FP)
    mutable Kokkos::View<VVM::Real*, Kokkos::DefaultExecutionSpace::memory_space> row_sums_;
#endif

#if defined(ENABLE_NCCL)
    ncclComm_t nccl_comm_;
    cudaStream_t nccl_stream_;
#endif
};

} // namespace Core
} // namespace VVM

#endif // VVM_CORE_VVMSTATE_HPP
