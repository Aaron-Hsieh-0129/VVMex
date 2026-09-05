#include <sstream>
#include <stdexcept>
#include <cmath>
#include <string>
#include <algorithm>

#include "Grid.hpp"
#include "core/geometry/HorizontalGeometryFactory.hpp"

namespace VVM {
namespace Core {

Grid::Grid(const VVM::Utils::ConfigurationManager& config, MPI_Comm comm)
    : dims_device_view_("GridDimensions", 3), // Initialize Kokkos::View for 3 dimensions (Z, Y, X)
      dims_host_mirror_("GridDimensions_Host", 3),   // Initialize host mirror
      comm_(comm),
      cart_comm_(MPI_COMM_NULL)        // Initialize MPI_Comm to NULL for safety
{
    // Get MPI rank and size
    MPI_Comm_rank(comm_, &mpi_rank_); 
    MPI_Comm_size(comm_, &mpi_size_);

    // Read grid parameters from ConfigurationManager and populate the host mirror
    try {
        grid_specification_ = GridSpecification::from_config(config);
        const auto& horizontal = grid_specification_.horizontal;
        const auto& vertical = grid_specification_.vertical;

        dims_host_mirror_(0).global_size = vertical.nz;
        dims_host_mirror_(1).global_size = horizontal.ny;
        dims_host_mirror_(2).global_size = horizontal.nx;

        dims_host_mirror_(0).d_coord = vertical.dz;
        dims_host_mirror_(1).d_coord = horizontal.geometry.dq2;
        dims_host_mirror_(2).d_coord = horizontal.geometry.dq1;
        // VVMex currently uses one halo width for all three dimensions.
        // Preserve that behavior until vertical halo configuration is separated.
        for (int dim = 0; dim < 3; ++dim) {
            dims_host_mirror_(dim).num_halo_cells = horizontal.n_halo_cells;
        }
    }
    catch (const std::exception& e) {
        throw std::runtime_error("Grid initialization failed: " + std::string(e.what()));
    }

    // Copy the initialized data from host mirror to the device view
    Kokkos::deep_copy(dims_device_view_, dims_host_mirror_);
    Kokkos::fence(); // Ensure data is on device before proceeding with device-dependent operations

    // Calculate local grid distribution
    radiation_enabled_ = config.get_value<bool>("physics.rrtmgp.enable_rrtmgp", false);

    calculate_local_grid_distribution();

    initialize_horizontal_geometry();

    if (mpi_rank_ == 0) {
        std::cout << "Grid initialized successfully." << std::endl;
    }
}

Grid::~Grid() {
    if (cart_comm_ != MPI_COMM_NULL) {
        MPI_Comm_free(&cart_comm_);
    }
}

void Grid::calculate_local_grid_distribution() {
    // Assume Z dimension is not decomposed, each process has the full Z axis
    // Assume X and Y dimensions are decomposed

    // Z dimension (not decomposed, each process has the full Z axis)
    dims_host_mirror_(0).local_physical_size = dims_host_mirror_(0).global_size;
    dims_host_mirror_(0).local_physical_start_idx = 0;
    dims_host_mirror_(0).local_physical_end_idx = dims_host_mirror_(0).global_size - 1;

    // Decomposing the grid into a 2D topology (Y, X) using MPI Cartesian topology
    // Determine process topology: Px * Py = mpi_size_
    // Try to find a process grid that is close to square
    int process_dimensions[2] = {1, 1};

    const auto& horizontal = grid_specification_.horizontal;
    const auto& topology = horizontal.topology;

    // MPI Cartesian dimensions are ordered (q2, q1), corresponding to (Y, X).
    // Singleton physical dimensions are never connected to themselves.
    int periods[2] = {
        horizontal.ny > 1 && topology.q2 == HorizontalEdgeTopology::Periodic ? 1 : 0,
        horizontal.nx > 1 && topology.q1 == HorizontalEdgeTopology::Periodic ? 1 : 0
    };

    int reorder = 1;         // Allow MPI to reorder processes to optimize topology
    int ndims_cart = 2; // Default to 2D decomposition, even if we only decompose in X and Y

    // Set p_dims based on global grid size
    if (dims_host_mirror_(1).global_size == 1 && dims_host_mirror_(2).global_size > 1) {
        // Only x direction has multiple points, perform 1D X decomposition
        process_dimensions[0] = 1; // Y direction only uses 1 process
        process_dimensions[1] = mpi_size_; // X direction uses all processes
        ndims_cart = 2; // Still create 2D topology, but one dimension has only 1 process

        periods[0] = 0; // No periodic b.c. along y-axis
        // Keep allocated Y halos even for a singleton physical dimension.
        // HaloExchanger packs with the model stencil width in every direction;
        // removing this storage made its Y pack kernels index outside the field.
        if (mpi_rank_ == 0) {
            std::cout << "Detected 1D X-decomposition (ny=1, nx>1)" << std::endl;
        }
    } 
    else if (dims_host_mirror_(2).global_size == 1 && dims_host_mirror_(1).global_size > 1) {
        // Only y direction has multiple points, perform 1D Y decomposition
        process_dimensions[0] = mpi_size_; // Y direction uses all processes
        process_dimensions[1] = 1; // X direction only uses 1 process
        ndims_cart = 2; // Still create 2D topology, but one dimension has only 1 process

        periods[1] = 0; // No periodic b.c. along x-axis
        if (mpi_rank_ == 0) {
            std::cout << "Detected 1D Y-decomposition (nx=1, ny>1)" << std::endl;
        }
    } 
    else if (dims_host_mirror_(2).global_size > 1 && dims_host_mirror_(1).global_size > 1) {
        // Both X and Y directions have multiple points, perform 2D (X,Y) decomposition
        process_dimensions[0] = 0; // Let MPI decide the number of Y processes
        process_dimensions[1] = 0; // Let MPI decide the number of X processes
        const int status = MPI_Dims_create(mpi_size_, 2, process_dimensions); // Let MPI find the best 2D distribution
        if (status != MPI_SUCCESS) {
            throw std::runtime_error("MPI_Dims_create failed to construct the horizontal process topology.");
        }
        ndims_cart = 2;
        if (mpi_rank_ == 0) {
            std::cout << "Detected 2D (X,Y) decomposition (nx>1, ny>1)" << std::endl;
        }
    } 
    else { // dims_[2].global_size == 1 && dims_[1].global_size == 1 (single point grid or full copy)
        // In this case, all processes will copy the entire microgrid
        process_dimensions[0] = 1;
        process_dimensions[1] = 1;
        if (mpi_rank_ == 0) {
            std::cout << "Detected no decomposition (nx=1, ny=1)" << std::endl;
            if (mpi_size_ > 1) {
                std::cout << "WARNING: With nx=1 and ny=1, all MPI ranks will have full copy of horizontal domain." << std::endl;
            }
        }
    }
    // If mpi_rank_ == 0, print decomposition information
    if (mpi_rank_ == 0) {
        std::cout << "MPI 2D Decomposition: Px=" << process_dimensions[1] << ", Py=" << process_dimensions[0] << std::endl;

        if (periods[1] == 1) std::cout << "Boundary Conditions: X-Periodic, ";
        else std::cout << "Boundary Conditions: X-Non Periodic, ";

        if (periods[0] == 1) std::cout << "Y-Periodic, ";
        else std::cout << "Y-Non Periodic, ";

        std::cout << "Z-NonPeriodic" << std::endl; // Added for clarity
    }

    const int nx_global = dims_host_mirror_(2).global_size;
    const int ny_global = dims_host_mirror_(1).global_size;
    if (nx_global % process_dimensions[1] != 0 || ny_global % process_dimensions[0] != 0) {
        std::ostringstream msg;
        msg << "[Grid] Grid does not divide evenly over the ranks: "
            << "nx=" << nx_global << " over Px=" << process_dimensions[1] << " ("
            << nx_global % process_dimensions[1] << " left over), "
            << "ny=" << ny_global << " over Py=" << process_dimensions[0] << " ("
            << ny_global % process_dimensions[0] << " left over), on " << mpi_size_ << " ranks.\n"
            << "  Use a rank count whose 2-D factors divide nx and ny, or change "
               "grid.nx / grid.ny.";
        if (radiation_enabled_) {
            throw std::runtime_error(msg.str());
        }
        if (mpi_rank_ == 0) {
            std::cout << "WARNING: " << msg.str() << "\n"
                      << "  Tolerated because radiation is off; it would abort with "
                         "physics.rrtmgp.enable_rrtmgp=true." << std::endl;
        }
    }

    // 3. Create MPI Cartesian Communicator
    const int cart_status = MPI_Cart_create(comm_, ndims_cart, process_dimensions, periods, reorder, &cart_comm_);
    if (cart_status != MPI_SUCCESS || cart_comm_ == MPI_COMM_NULL) {
        throw std::runtime_error("MPI_Cart_create failed; the process-grid dimensions do not match the communicator.");
    }

    // When reorder is true, a rank in comm_ is not necessarily the same rank
    // in cart_comm_. All Cartesian topology queries must use the latter.
    MPI_Comm_rank(cart_comm_, &cart_rank_);

    // 4. Get current process coordinates in 2D topology
    int coords[2]; // coords[0] for Y-coordinate, coords[1] for X-coordinate
    MPI_Cart_coords(cart_comm_, cart_rank_, 2, coords);

    // 5. Calculate local grid range based on process coordinates
    // --- Y dimension decomposition (dims_[1]) ---
    if (process_dimensions[0] == 1) { // Only one process in Y direction, so no decomposition
        dims_host_mirror_(1).local_physical_size = dims_host_mirror_(1).global_size;
        dims_host_mirror_(1).local_physical_start_idx = 0;
        dims_host_mirror_(1).local_physical_end_idx = dims_host_mirror_(1).global_size - 1;
    } 
    else { // Multiple processes in Y direction, proceed with decomposition
        int base_local_N_y = dims_host_mirror_(1).global_size / process_dimensions[0];
        int remainder_y = dims_host_mirror_(1).global_size % process_dimensions[0];

        dims_host_mirror_(1).local_physical_start_idx = coords[0] * base_local_N_y + std::min(coords[0], remainder_y);
        dims_host_mirror_(1).local_physical_size = base_local_N_y + (coords[0] < remainder_y ? 1 : 0);
        dims_host_mirror_(1).local_physical_end_idx = dims_host_mirror_(1).local_physical_start_idx + dims_host_mirror_(1).local_physical_size - 1;
    }


    // --- X dimension decomposition (dims_[2]) ---
    if (dims_host_mirror_(2).global_size == 1) {
        dims_host_mirror_(2).local_physical_size = dims_host_mirror_(2).global_size;
        dims_host_mirror_(2).local_physical_start_idx = 0;
        dims_host_mirror_(2).local_physical_end_idx = dims_host_mirror_(2).global_size - 1;
    } 
    else { 
        int base_local_N_x = dims_host_mirror_(2).global_size / process_dimensions[1];
        int remainder_x = dims_host_mirror_(2).global_size % process_dimensions[1];

        dims_host_mirror_(2).local_physical_start_idx = coords[1] * base_local_N_x + std::min(coords[1], remainder_x);
        dims_host_mirror_(2).local_physical_size = base_local_N_x + (coords[1] < remainder_x ? 1 : 0);
        dims_host_mirror_(2).local_physical_end_idx = dims_host_mirror_(2).local_physical_start_idx + dims_host_mirror_(2).local_physical_size - 1;
    }

    const int halo_width = dims_host_mirror_(0).num_halo_cells;
    if ((dims_host_mirror_(1).global_size > 1 &&
         dims_host_mirror_(1).local_physical_size < halo_width) ||
        (dims_host_mirror_(2).global_size > 1 &&
         dims_host_mirror_(2).local_physical_size < halo_width)) {
        throw std::runtime_error(
            "Local horizontal domain is narrower than the configured halo width.");
    }

    // Copy the updated data from host mirror back to the device view
    Kokkos::deep_copy(dims_device_view_, dims_host_mirror_);
    Kokkos::fence(); // Ensure data is on device after modification
    
    if (mpi_rank_ == 0) {
        std::cout << "Grid initialized successfully with 2D decomposition." << std::endl;
    }
}


void Grid::print_info() const {
    // Get current process coordinates in the 2D topology
    // Use the stored cart_comm_ member, not create a temporary one, for consistency
    int p_dims_retrieved[2]; // To retrieve actual dimensions from the comm
    int periods_retrieved[2]; // To retrieve actual periods from the comm
    int coords[2];

    // Check if cart_comm_ is valid before using it
    if (cart_comm_ != MPI_COMM_NULL) {
        MPI_Cart_coords(cart_comm_, cart_rank_, 2, coords);
        MPI_Cart_get(cart_comm_, 2, p_dims_retrieved, periods_retrieved, coords);
    } 
    else {
        // Handle case where communicator is NULL (e.g., in a single-process run or error)
        coords[0] = -1; coords[1] = -1; // Indicate invalid coords
        p_dims_retrieved[0] = 0; p_dims_retrieved[1] = 0;
        periods_retrieved[0] = 0; periods_retrieved[1] = 0;
    }

    std::cout << "MPI Rank " << mpi_rank_ << " (Coords: Y=" << coords[0] << ", X=" << coords[1] << ") Local Grid Info:" << std::endl;
    std::cout << "  Periodic (Y,X): (" << (periods_retrieved[0] ? "Yes" : "No") << ", " << (periods_retrieved[1] ? "Yes" : "No") << ")" << std::endl;
    std::cout << "  Z: Global Start=" << dims_host_mirror_(0).local_physical_start_idx
              << ", Global End=" << dims_host_mirror_(0).local_physical_end_idx
              << ", Physical Size=" << dims_host_mirror_(0).local_physical_size
              << ", Total Size (incl. Halo)=" << (dims_host_mirror_(0).local_physical_size + 2 * dims_host_mirror_(0).num_halo_cells) << std::endl;
    std::cout << "  Y: Global Start=" << dims_host_mirror_(1).local_physical_start_idx
              << ", Global End=" << dims_host_mirror_(1).local_physical_end_idx
              << ", Physical Size=" << dims_host_mirror_(1).local_physical_size
              << ", Total Size (incl. Halo)=" << (dims_host_mirror_(1).local_physical_size + 2 * dims_host_mirror_(1).num_halo_cells) << std::endl;
    std::cout << "  X: Global Start=" << dims_host_mirror_(2).local_physical_start_idx
              << ", Global End=" << dims_host_mirror_(2).local_physical_end_idx
              << ", Physical Size=" << dims_host_mirror_(2).local_physical_size
              << ", Total Size (incl. Halo)=" << (dims_host_mirror_(2).local_physical_size + 2 * dims_host_mirror_(2).num_halo_cells) << std::endl;
    std::cout << "------------------------------------" << std::endl;
}

void Grid::initialize_horizontal_geometry() {
    Geometry::HorizontalDomainLayout layout;

    layout.global_nx = get_global_points_x();
    layout.global_ny = get_global_points_y();
    layout.local_physical_nx = get_local_physical_points_x();
    layout.local_physical_ny = get_local_physical_points_y();
    layout.global_start_i = get_local_physical_start_x();
    layout.global_start_j = get_local_physical_start_y();
    layout.halo = get_halo_cells();
    layout.panel_id = -1;

    geometry_ = Geometry::HorizontalGeometryFactory::create(grid_specification_.horizontal.geometry, layout);

    if (!geometry_) {
        throw std::runtime_error("HorizontalGeometryFactory returned a null geometry.");
    }
}

} // namespace Core
} // namespace VVM
