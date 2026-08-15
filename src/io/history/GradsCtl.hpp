#ifndef VVM_IO_HISTORY_GRADS_CTL_HPP
#define VVM_IO_HISTORY_GRADS_CTL_HPP

#include <cstddef>
#include <filesystem>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

#include <mpi.h>

#include "core/Grid.hpp"
#include "core/State.hpp"
#include "core/vvm_types.hpp"

namespace VVM::IO {

struct GradsAxis {
    VVM::Real start = real(0.0);
    VVM::Real increment = real(1.0);
};

struct GradsVariable {
    // The name the file stores ("/Step0/th" for HDF5, "th" for BP5) and the
    // name GrADS exposes. They differ whenever the stored name is not a legal
    // GrADS identifier, which the descriptor spells as "stored=>exposed".
    std::string dataset_name;
    std::string grads_name;
    std::size_t levels = 0;  // 0 marks a surface field
    std::string dimensions;  // "z,y,x", "y,x" or "z"
    std::string description;
};

struct GradsCtl {
    std::string dset;
    std::string dtype;
    bool templated = false;
    std::string title = "VVMex";
    std::string undef = "-9999.0";
    GradsAxis x;
    GradsAxis y;
    std::size_t nx = 0;
    std::size_t ny = 0;
    std::vector<VVM::Real> z_levels;
    std::size_t time_count = 1;
    std::string time_start;
    std::string time_increment;
    std::vector<GradsVariable> variables;
    // Emitted as '*' comment lines ahead of the VARS block.
    std::vector<std::string> notes;
};

// Collective: the taiwanvvm branch reduces lon/lat corner values over comm, so
// every rank must call this even though only rank 0 writes a descriptor.
std::pair<GradsAxis, GradsAxis> grads_horizontal_axes(
    const Core::Grid& grid,
    const Core::State& state,
    bool use_taiwanvvm_coordinates,
    MPI_Comm comm);

std::string grads_start_time(int start_hour);
std::string grads_time_increment(VVM::Real output_interval_s);

// GrADS identifiers are lowercase, alphanumeric plus underscore, start with a
// letter and are at most 15 characters, so names such as "Tg" or
// "precip_liq_surf_mass" cannot be used verbatim. Inserts the result into
// `taken` so a truncated name never collides with another field.
std::string unique_grads_variable_name(
    const std::string& field_name,
    std::unordered_set<std::string>& taken);

void write_grads_ctl(const std::filesystem::path& path, const GradsCtl& ctl);

} // namespace VVM::IO

#endif
