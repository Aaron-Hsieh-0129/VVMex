#include "NumericalConfigurationValidation.hpp"

#include <cmath>
#include <sstream>
#include <stdexcept>

#include <mpi.h>

namespace VVM::Utils {
namespace {

void require_finite_positive(double value, const char* key) {
    if (!std::isfinite(value) || value <= 0.0) {
        throw std::runtime_error(std::string("Configuration error: '") + key + "' must be finite and greater than zero.");
    }
}

void require_local_width(
    const char* axis,
    int global_width,
    int process_width,
    int halo_width) {
    const int smallest_local_width = global_width / process_width;
    if (smallest_local_width < halo_width) {
        std::ostringstream message;
        message << "Configuration error: the " << axis << " domain has "
                << global_width << " cells over " << process_width
                << " process(es), giving a smallest local width of "
                << smallest_local_width << ", but grid.n_halo_cells is "
                << halo_width << ". Reduce the compute-rank count or enlarge "
                << "the domain.";
        throw std::runtime_error(message.str());
    }
}

} // namespace

void validate_numerical_configuration_values(
    const NumericalConfigurationValues& values,
    int compute_ranks) {
    if (compute_ranks <= 0) {
        throw std::runtime_error(
            "Configuration error: the number of compute ranks must be positive.");
    }
    if (values.nx <= 0 || values.ny <= 0 || values.nz <= 0) {
        throw std::runtime_error(
            "Configuration error: grid.nx, grid.ny, and grid.nz must be positive integers.");
    }

    // Takacs is always used for diagnostic dynamics and reaches two cells away
    // from the physical domain, even when a tracer selects another scheme.
    if (values.halo_width < 2) {
        throw std::runtime_error(
            "Configuration error: grid.n_halo_cells must be at least 2 for the Takacs stencil.");
    }

    require_finite_positive(values.dx, "grid.dx");
    require_finite_positive(values.dy, "grid.dy");
    require_finite_positive(values.dz, "grid.dz");
    require_finite_positive(values.dt, "simulation.dt_s");
    require_finite_positive(values.output_interval, "simulation.output_interval_s");
    if (!std::isfinite(values.total_time) || values.total_time < 0.0) {
        throw std::runtime_error(
            "Configuration error: 'simulation.total_time_s' must be finite and nonnegative.");
    }

    int process_dims[2] = {1, 1}; // Y, X -- must match Grid's decomposition.
    if (values.nx == 1 && values.ny == 1) {
        if (compute_ranks != 1) {
            throw std::runtime_error(
                "Configuration error: a 1x1 horizontal domain supports exactly one compute rank.");
        }
    } 
    else if (values.ny == 1) {
        process_dims[1] = compute_ranks;
    } 
    else if (values.nx == 1) {
        process_dims[0] = compute_ranks;
    } 
    else {
        process_dims[0] = 0;
        process_dims[1] = 0;
        const int status = MPI_Dims_create(compute_ranks, 2, process_dims);
        if (status != MPI_SUCCESS) {
            throw std::runtime_error(
                "Configuration error: MPI could not construct the requested 2-D process topology.");
        }
    }

    if (values.nx > 1) {
        require_local_width("X", values.nx, process_dims[1], values.halo_width);
    }
    if (values.ny > 1) {
        require_local_width("Y", values.ny, process_dims[0], values.halo_width);
    }
}

void validate_numerical_configuration(
    const ConfigurationManager& config,
    int compute_ranks) {
    validate_numerical_configuration_values({
        config.get_value<int>("grid.nx"),
        config.get_value<int>("grid.ny"),
        config.get_value<int>("grid.nz"),
        config.get_value<int>("grid.n_halo_cells"),
        config.get_value<double>("grid.dx"),
        config.get_value<double>("grid.dy"),
        config.get_value<double>("grid.dz"),
        config.get_value<double>("simulation.dt_s"),
        config.get_value<double>("simulation.total_time_s"),
        config.get_value<double>("simulation.output_interval_s")},
        compute_ranks);
}

} // namespace VVM::Utils
