#ifndef VVM_UTILS_NUMERICAL_CONFIGURATION_VALIDATION_HPP
#define VVM_UTILS_NUMERICAL_CONFIGURATION_VALIDATION_HPP

#include "ConfigurationManager.hpp"

namespace VVM::Utils {

struct NumericalConfigurationValues {
    int nx;
    int ny;
    int nz;
    int halo_width;
    double dx;
    double dy;
    double dz;
    double dt;
    double total_time;
    double output_interval;
};

void validate_numerical_configuration_values(
    const NumericalConfigurationValues& values,
    int compute_ranks);

void validate_numerical_configuration(
    const ConfigurationManager& config,
    int compute_ranks);

} // namespace VVM::Utils

#endif
