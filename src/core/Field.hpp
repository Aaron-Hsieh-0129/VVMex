// Field class is a Kokkos::View-based data structure
// that represents a 3D field in the simulation grid.
// It provides methods for initialization, data access, and printing field information.

#ifndef VVM_CORE_FIELD_HPP
#define VVM_CORE_FIELD_HPP

#include <Kokkos_Core.hpp>
#include <type_traits>
#include "Grid.hpp"

namespace VVM {
namespace Core {

enum class GridStaggering {
    Unspecified, NotApplicable,

    Centered,       // (i,     j,     k)

    StaggeredX,     // (i+1/2, j,     k)     : u
    StaggeredY,     // (i,     j+1/2, k)     : v
    StaggeredZ,     // (i,     j,     k+1/2) : w

    StaggeredYZ,    // (i,     j+1/2, k+1/2) : xi
    StaggeredXZ,    // (i+1/2, j,     k+1/2) : eta
    StaggeredXY,    // (i+1/2, j+1/2, k)     : zeta

    StaggeredXYZ,   // (i+1/2, j+1/2, k+1/2)
    Surface         // At surface
};

struct FieldMetadata {
    std::string units;
    std::string long_name;
    std::string standard_name;
    std::string comment;
    GridStaggering grid_staggering = GridStaggering::Unspecified;

    FieldMetadata() = default;

    FieldMetadata(
        GridStaggering grid_staggering_in,
        std::string units_in,
        std::string long_name_in,
        std::string standard_name_in = {},
        std::string comment_in = {})
        : grid_staggering(grid_staggering_in),
          units(std::move(units_in)),
          long_name(std::move(long_name_in)),
          standard_name(std::move(standard_name_in)),
          comment(std::move(comment_in)) {}
};

inline const char* grid_staggering_to_string(GridStaggering staggering) noexcept {
    switch (staggering) {
        case GridStaggering::Unspecified:
            return "unspecified";
        case GridStaggering::NotApplicable:
            return "not_applicable";
        case GridStaggering::Centered:
            return "centered";
        case GridStaggering::StaggeredX:
            return "staggered_x";
        case GridStaggering::StaggeredY:
            return "staggered_y";
        case GridStaggering::StaggeredZ:
            return "staggered_z";
        case GridStaggering::StaggeredYZ:
            return "staggered_yz";
        case GridStaggering::StaggeredXZ:
            return "staggered_xz";
        case GridStaggering::StaggeredXY:
            return "staggered_xy";
        case GridStaggering::StaggeredXYZ:
            return "staggered_xyz";
    }
    return "unspecified";
}

// Helper to create Kokkos::View of varying dimensions
template<size_t Dim, typename ScalarType = VVM::Real>
struct ViewTypeHelper;

template<typename ScalarType> struct ViewTypeHelper<0, ScalarType> { using type = Kokkos::View<ScalarType>; };
template<typename ScalarType> struct ViewTypeHelper<1, ScalarType> { using type = Kokkos::View<ScalarType*>; };
template<typename ScalarType> struct ViewTypeHelper<2, ScalarType> { using type = Kokkos::View<ScalarType**>; };
template<typename ScalarType> struct ViewTypeHelper<3, ScalarType> { using type = Kokkos::View<ScalarType***>; };
template<typename ScalarType> struct ViewTypeHelper<4, ScalarType> { using type = Kokkos::View<ScalarType****>; };

// Layout = void means "let Kokkos pick" (LayoutLeft on the CUDA backend), which is
// what every State field uses. Solver-private work arrays may pin a layout instead:
// a kernel threaded over (j,i) with a serial k loop wants x contiguous, the opposite
// of what the default gives, and those arrays are not shared with the rest of the model.
template<size_t Dim, typename Layout = void>
class Field {
public:
    static constexpr size_t DimValue = Dim;

    // Kokkos::View to store the field data
    // The dimensions will be (total_z_points, total_y_points, total_x_points)
    using DefaultViewType = typename ViewTypeHelper<Dim>::type;
    using ViewType = std::conditional_t<
        std::is_void_v<Layout>,
        DefaultViewType,
        Kokkos::View<typename DefaultViewType::data_type, Layout>>;
    using HostMirrorType = typename ViewType::HostMirror;

    // Constructor
    explicit Field(const std::string& field_name, const std::array<int, Dim>& dims, FieldMetadata metadata = {})
        : name_(field_name), metadata_(std::move(metadata)) {

        if constexpr (Dim == 0) data_ = ViewType(name_);
        else if constexpr (Dim == 1) data_ = ViewType(name_, dims[0]);
        else if constexpr (Dim == 2) data_ = ViewType(name_, dims[0], dims[1]);
        else if constexpr (Dim == 3) data_ = ViewType(name_, dims[0], dims[1], dims[2]);
        else if constexpr (Dim == 4) data_ = ViewType(name_, dims[0], dims[1], dims[2], dims[3]);

        Kokkos::deep_copy(data_, real(0.0));
        Kokkos::fence();
    }

    // Destructor (Kokkos::View manages its own memory, so usually empty here)
    ~Field() = default; // Or {} if you need to add custom cleanup

    // Delete copy constructor and assignment operator to prevent shallow copies
    Field(const Field&) = delete;
    Field& operator=(const Field&) = delete;

    // --- Initialization Methods (executed on device) ---

    // Initialize all field values to zero
    void set_to_zero() {
        Kokkos::deep_copy(Kokkos::DefaultExecutionSpace(), data_, real(0.0));
    }

    // Get a const reference to the Kokkos::View (for device computations)
    ViewType& get_mutable_device_data() { return data_; }
    const ViewType& get_device_data() const { return data_; }

    HostMirrorType get_host_data() const {
        HostMirrorType host_data = Kokkos::create_mirror_view(data_);
        Kokkos::deep_copy(host_data, data_);
        Kokkos::fence();
        return host_data;
    }

    const std::string& get_name() const { return name_; }

    const FieldMetadata& get_metadata() const noexcept { return metadata_; }

    // --- Printing/Debugging Methods ---
    void print_field_info() const;
    void print_slice_z_at_k(const Grid& grid, int N_idx, int k_local_idx, int halo=-1) const;
    void print_profile(const Grid& grid, int N_idx, int j_local_idx, int i_local_idx) const;
    void print_xz_cross_at_j(const Grid& grid, int N_idx, int j_local_idx, int halo=-1) const;

private:
    std::string name_; // Name of the field for identification
    ViewType data_;
    FieldMetadata metadata_;
};

template<size_t Dim, typename Layout>
inline void Field<Dim, Layout>::print_slice_z_at_k(const Grid& grid, int N_idx, int k_local_idx, int halo) const {
    int rank;
    MPI_Comm_rank(grid.get_comm(), &rank);

    auto host_data = get_host_data();

    std::cout << "Rank " << rank << ": Field '" << name_ << "' (" << Dim << "D)" << std::endl;

    // Use if constexpr to handle different dimensions at compile time
    if constexpr (Dim == 4) {
        if (N_idx < 0 || N_idx >= host_data.extent(0) ||
            k_local_idx < 0 || k_local_idx >= host_data.extent(1)) {
            std::cerr << "Warning: Slice index (" << N_idx << ", " << k_local_idx 
                      << ") out of bounds for field '" << name_ << "'." << std::endl;
            return;
        }
        std::cout << "  Slice at N=" << N_idx << ", k=" << k_local_idx << std::endl;
        int i_start = halo == -1 ? 0 : halo;
        int j_start = halo == -1 ? 0 : halo;
        int i_end = halo == -1 ? host_data.extent(3) : host_data.extent(3)-halo;
        int j_end = halo == -1 ? host_data.extent(2) : host_data.extent(2)-halo;
        for (int j = j_start; j < j_end; ++j) {
            for (int i = i_start; i < i_end; ++i) {
                std::cout << host_data(N_idx, k_local_idx, j, i) << "\t";
            }
            std::cout << std::endl;
        }
    } 
    else if constexpr (Dim == 3) {
        if (k_local_idx < 0 || k_local_idx >= host_data.extent(0)) {
            std::cerr << "Warning: Z-slice index " << k_local_idx << " out of bounds for field '" << name_ << "'." << std::endl;
            return;
        }
        std::cout << "  Z-slice at k=" << k_local_idx << std::endl;
        int i_start = halo == -1 ? 0 : halo;
        int j_start = halo == -1 ? 0 : halo;
        int i_end = halo == -1 ? host_data.extent(2) : host_data.extent(2)-halo;
        int j_end = halo == -1 ? host_data.extent(1) : host_data.extent(1)-halo;
        for (int j = j_start; j < j_end; ++j) {
            for (int i = i_start; i < i_end; ++i) {
                std::cout << host_data(k_local_idx, j, i) << "\t";
            }
            std::cout << std::endl;
        }
    } 
    else if constexpr (Dim == 2) {
        // For a 2D field, we ignore indices and print the whole field
        std::cout << "  Full 2D data:" << std::endl;
        int i_start = halo == -1 ? 0 : halo;
        int j_start = halo == -1 ? 0 : halo;
        int i_end = halo == -1 ? host_data.extent(1) : host_data.extent(1)-halo;
        int j_end = halo == -1 ? host_data.extent(0) : host_data.extent(0)-halo;
        for (int j = j_start; j < j_end; ++j) {
            for (int i = i_start; i < i_end; ++i) {
                std::cout << host_data(j, i) << "\t";
            }
            std::cout << std::endl;
        }
    } 
    else {
        // For other dimensions, this function is not applicable
        if (rank == 0) {
            std::cout << "  Printing is not implemented for " << Dim << "D fields." << std::endl;
        }
    }
    std::cout << "--------------------------------------" << std::endl;
}


template<size_t Dim, typename Layout>
inline void Field<Dim, Layout>::print_profile(const Grid& grid, int N_idx, int j_local_idx, int i_local_idx) const {
    int rank;
    MPI_Comm_rank(grid.get_comm(), &rank);

    auto host_data = get_host_data();

    std::cout << "Rank " << rank << ": Field '" << name_ << "'" << std::endl;

    // Use if constexpr to handle different dimensions at compile time
    if constexpr (Dim == 4) {
        std::cout << "  Slice at N=" << N_idx << ", j=" << j_local_idx << ", i=" << i_local_idx<< std::endl;
        for (int k = 0; k < host_data.extent(1); ++k) {
            std::cout << host_data(N_idx, k, j_local_idx, i_local_idx) << "\t";
        }
        std::cout << std::endl;
    } 
    else if constexpr (Dim == 3) {
        std::cout << ", j=" << j_local_idx << ", i=" << i_local_idx<< std::endl;
        for (int k = 0; k < host_data.extent(0); ++k) {
            std::cout << std::scientific << std::setprecision(6) << host_data(k, j_local_idx, i_local_idx) << "\t";
        }
        std::cout << std::endl;
    } 
    else if constexpr (Dim == 1) {
        // For a 2D field, we ignore indices and print the whole field
        std::cout << "  1D profile:" << std::endl;
        for (int k = 0; k < host_data.extent(0); ++k) {
            std::cout << host_data(k) << "\t";
        }
        std::cout << std::endl;
    } 
    else {
        // For other dimensions, this function is not applicable
        if (rank == 0) {
            std::cout << "  Printing is not implemented for " << Dim << "D fields." << std::endl;
        }
    }
    std::cout << "--------------------------------------" << std::endl;
}

template<size_t Dim, typename Layout>
inline void Field<Dim, Layout>::print_xz_cross_at_j(const Grid& grid, int N_idx, int j_local_idx, int halo) const {
    int rank;
    MPI_Comm_rank(grid.get_comm(), &rank);

    auto host_data = get_host_data();

    std::cout << "Rank " << rank << ": Field '" << name_ << "' (" << Dim << "D)" << std::endl;

    // Use if constexpr to handle different dimensions at compile time
    if constexpr (Dim == 4) {
        if (N_idx < 0 || N_idx >= host_data.extent(0) ||
            j_local_idx < 0 || j_local_idx >= host_data.extent(1)) {
            std::cerr << "Warning: Slice index (" << N_idx << ", " << j_local_idx 
                      << ") out of bounds for field '" << name_ << "'." << std::endl;
            return;
        }
        std::cout << "  Slice at N=" << N_idx << ", j=" << j_local_idx << std::endl;
        int i_start = halo == -1 ? 0 : halo;
        int k_start = halo == -1 ? 0 : halo;
        int i_end = halo == -1 ? host_data.extent(3) : host_data.extent(3)-halo;
        int k_end = halo == -1 ? host_data.extent(1) : host_data.extent(1)-halo;
        for (int k = k_start; k < k_end; ++k) {
            for (int i = i_start; i < i_end; ++i) {
                std::cout << host_data(N_idx, k, j_local_idx, i) << "\t";
            }
            std::cout << std::endl;
        }
    } 
    else if constexpr (Dim == 3) {
        if (j_local_idx < 0 || j_local_idx >= host_data.extent(1)) {
            std::cerr << "Warning: Y-slice index " << j_local_idx << " out of bounds for field '" << name_ << "'." << std::endl;
            return;
        }
        std::cout << "  Y-slice at j=" << j_local_idx << std::endl;
        int i_start = halo == -1 ? 0 : halo;
        int k_start = halo == -1 ? 0 : halo;
        int i_end = halo == -1 ? host_data.extent(2) : host_data.extent(2)-halo;
        int k_end = halo == -1 ? host_data.extent(0) : host_data.extent(0)-halo;

        for (int k = k_start; k < k_end; ++k) {
            for (int i = i_start; i < i_end; ++i) {
                std::cout << host_data(k, j_local_idx, i) << "\t";
            }
            std::cout << std::endl;
        }
    } 
    else {
        // For other dimensions, this function is not applicable
        if (rank == 0) {
            std::cout << "  Printing is not implemented for " << Dim << "D fields." << std::endl;
        }
    }
    std::cout << "--------------------------------------" << std::endl;
}

} // namespace Core
} // namespace VVM

#endif // VVM_CORE_FIELD_HPP
