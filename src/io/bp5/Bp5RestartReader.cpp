#include "Bp5RestartReader.hpp"

#include <algorithm>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>

#include <Kokkos_Core.hpp>

#include "core/vvm_types.hpp"
#include "io/RestartVariables.hpp"

namespace VVM::IO::BP5 {

namespace {
constexpr const char* kTag = "Bp5RestartReader";

std::string describe(const adios2::Dims& dims) {
    std::string text = "{";
    for (std::size_t i = 0; i < dims.size(); ++i) {
        if (i > 0) text += ", ";
        text += std::to_string(dims[i]);
    }
    return text + "}";
}

adios2::Dims expected_shape(std::size_t dimensions, const Core::Grid& grid) {
    const auto nz = static_cast<std::size_t>(grid.get_global_points_z());
    const auto ny = static_cast<std::size_t>(grid.get_global_points_y());
    const auto nx = static_cast<std::size_t>(grid.get_global_points_x());
    if (dimensions == 1) return {nz};
    if (dimensions == 2) return {ny, nx};
    return {nz, ny, nx};
}
} // namespace

class Bp5RestartReader::OpenDataset {
public:
    OpenDataset(const std::string& path, MPI_Comm comm, adios2::ADIOS& adios,
                const char* io_name)
        : io_(adios.DeclareIO(io_name)) {
        io_.SetEngine("BP5");
        engine_ = io_.Open(path, adios2::Mode::ReadRandomAccess, comm);
        steps_ = engine_.Steps();
        if (steps_ == 0) {
            throw std::runtime_error(
                std::string("[") + kTag + "] Restart dataset '" + path +
                "' contains no steps.");
        }
    }

    ~OpenDataset() {
        if (engine_) engine_.Close();
    }

    OpenDataset(const OpenDataset&) = delete;
    OpenDataset& operator=(const OpenDataset&) = delete;

    adios2::IO& io() { return io_; }
    adios2::Engine& engine() { return engine_; }
    std::size_t steps() const { return steps_; }
    std::size_t step() const { return step_; }
    void set_step(std::size_t step) { step_ = step; }

private:
    adios2::IO io_;
    adios2::Engine engine_;
    std::size_t steps_ = 0;
    std::size_t step_ = 0;
};

Bp5RestartReader::Bp5RestartReader(const std::string& dataset_path,
                                   const Core::Grid& grid,
                                   const Core::Parameters& params,
                                   const Utils::ConfigurationManager& config,
                                   Core::HaloExchanger& halo_exchanger)
    : dataset_path_(dataset_path),
      grid_(grid),
      params_(params),
      config_(config),
      halo_exchanger_(halo_exchanger),
      comm_(grid.get_cart_comm()) {
    MPI_Comm_rank(comm_, &rank_);
}

std::size_t Bp5RestartReader::resolve_step(std::size_t available_steps) const {
    const std::int64_t requested =
        config_.get_value<std::int64_t>("restart.step_index", -1);
    if (requested == -1) return available_steps - 1;
    if (requested < 0 || static_cast<std::size_t>(requested) >= available_steps) {
        throw std::runtime_error(
            std::string("[") + kTag + "] restart.step_index " +
            std::to_string(requested) + " is outside the " +
            std::to_string(available_steps) + " step(s) in '" + dataset_path_ + "'.");
    }
    return static_cast<std::size_t>(requested);
}

template <typename Stored, std::size_t Dim>
void Bp5RestartReader::read_typed_field(OpenDataset& dataset,
                                        const std::string& var_name,
                                        Core::Field<Dim>& field) const {
    auto variable = dataset.io().InquireVariable<Stored>(var_name);
    if (!variable) {
        throw std::runtime_error(
            std::string("[") + kTag + "] Variable '" + var_name +
            "' is missing from restart dataset.");
    }

    const adios2::Dims shape = variable.Shape();
    const adios2::Dims wanted = expected_shape(Dim, grid_);
    if (shape != wanted) {
        throw std::runtime_error(
            std::string("[") + kTag + "] Variable '" + var_name + "' has shape " +
            describe(shape) + ", which does not match this grid " + describe(wanted) + ".");
    }

    const auto halo = static_cast<std::size_t>(grid_.get_halo_cells());
    const auto nz = static_cast<std::size_t>(grid_.get_global_points_z());
    const auto ny = static_cast<std::size_t>(grid_.get_local_physical_points_y());
    const auto nx = static_cast<std::size_t>(grid_.get_local_physical_points_x());
    const auto y0 = static_cast<std::size_t>(grid_.get_local_physical_start_y());
    const auto x0 = static_cast<std::size_t>(grid_.get_local_physical_start_x());

    adios2::Dims start;
    adios2::Dims count;
    if constexpr (Dim == 1) {
        start = {0};
        count = {nz};
    } 
    else if constexpr (Dim == 2) {
        start = {y0, x0};
        count = {ny, nx};
    } 
    else {
        start = {0, y0, x0};
        count = {nz, ny, nx};
    }

    std::size_t elements = 1;
    for (const auto n : count) elements *= n;
    std::vector<Stored> buffer(elements);

    variable.SetStepSelection({dataset.step(), 1});
    if (elements > 0) variable.SetSelection({start, count});
    dataset.engine().Get(variable, buffer.data(), adios2::Mode::Sync);

    auto device_view = field.get_mutable_device_data();
    auto host_view = Kokkos::create_mirror_view(device_view);
    Kokkos::deep_copy(host_view, device_view);

    if constexpr (Dim == 1) {
        for (std::size_t k = 0; k < nz; ++k) {
            host_view(k + halo) = static_cast<VVM::Real>(buffer[k]);
        }
    } else if constexpr (Dim == 2) {
        for (std::size_t j = 0; j < ny; ++j) {
            for (std::size_t i = 0; i < nx; ++i) {
                host_view(j + halo, i + halo) =
                    static_cast<VVM::Real>(buffer[j * nx + i]);
            }
        }
    } else {
        for (std::size_t k = 0; k < nz; ++k) {
            for (std::size_t j = 0; j < ny; ++j) {
                for (std::size_t i = 0; i < nx; ++i) {
                    host_view(k + halo, j + halo, i + halo) =
                        static_cast<VVM::Real>(buffer[(k * ny + j) * nx + i]);
                }
            }
        }
    }

    Kokkos::deep_copy(device_view, host_view);
}

template <std::size_t Dim>
void Bp5RestartReader::read_field(OpenDataset& dataset,
                                  const std::string& var_name,
                                  Core::Field<Dim>& field) const {
    const std::string type = dataset.io().VariableType(var_name);
    if (type == "float") {
        read_typed_field<float, Dim>(dataset, var_name, field);
    } else if (type == "double") {
        read_typed_field<double, Dim>(dataset, var_name, field);
    } else if (type.empty()) {
        throw std::runtime_error(
            std::string("[") + kTag + "] Variable '" + var_name +
            "' is missing from restart dataset '" + dataset_path_ + "'.");
    } else {
        throw std::runtime_error(
            std::string("[") + kTag + "] Variable '" + var_name + "' has type '" +
            type + "', which is not a restartable field type.");
    }
}

void Bp5RestartReader::read_and_initialize(Core::State& state) {
    const RestartVariables variables =
        select_restart_variables(config_, state, rank_, kTag);
    if (variables.empty()) {
        throw std::runtime_error(
            std::string("[") + kTag + "] No restart variables selected from " + dataset_path_);
    }
    print_restart_variables(variables, dataset_path_, rank_, kTag);

    adios2::ADIOS adios(comm_);
    OpenDataset dataset(dataset_path_, comm_, adios, "VVM_BP5_RESTART_FIELDS");
    dataset.set_step(resolve_step(dataset.steps()));

    if (rank_ == 0) {
        std::cout << "  [" << kTag << "] Reading step " << dataset.step() << " of "
                  << dataset.steps() << " from " << dataset_path_ << std::endl;
    }

    for (const auto& name : variables.vars_1d) read_field(dataset, name, state.get_field<1>(name));
    for (const auto& name : variables.vars_2d) read_field(dataset, name, state.get_field<2>(name));
    for (const auto& name : variables.vars_3d) read_field(dataset, name, state.get_field<3>(name));

    for (const auto& name : variables.vars_1d) {
        if (state.has_field(name)) halo_exchanger_.exchange_halos(state.get_field<1>(name));
    }
    for (const auto& name : variables.vars_2d) {
        if (state.has_field(name)) halo_exchanger_.exchange_halos(state.get_field<2>(name));
    }
    for (const auto& name : variables.vars_3d) {
        if (state.has_field(name)) halo_exchanger_.exchange_halos(state.get_field<3>(name));
    }
}

VVM::Utils::RestartFileMetadata Bp5RestartReader::read_restart_metadata() {
    VVM::Utils::RestartFileMetadata metadata;

    adios2::ADIOS adios(comm_);
    OpenDataset dataset(dataset_path_, comm_, adios, "VVM_BP5_RESTART_CLOCK");
    dataset.set_step(resolve_step(dataset.steps()));

    const auto read_scalar = [&](auto sample, const std::string& name, auto& out) -> bool {
        using T = decltype(sample);
        auto variable = dataset.io().InquireVariable<T>(name);
        if (!variable) return false;
        variable.SetStepSelection({dataset.step(), 1});
        T value{};
        dataset.engine().Get(variable, &value, adios2::Mode::Sync);
        out = static_cast<std::decay_t<decltype(out)>>(value);
        return true;
    };

    double time_s = 0.0;
    if (read_scalar(double{}, "model_time_s", time_s)) {
        metadata.time_s = time_s;
        metadata.has_time = true;
    }

    std::int64_t step = 0;
    if (read_scalar(std::int64_t{}, "model_step", step)) {
        metadata.step = step;
        metadata.has_step = true;
    }

    if (!metadata.has_time) {
        double elapsed = 0.0;
        if (read_scalar(double{}, "time", elapsed) || read_scalar(float{}, "time", elapsed)) {
            metadata.time_s = elapsed;
            metadata.has_time = true;
        }
    }

    if (rank_ == 0) {
        if (!metadata.has_time && !metadata.has_step) {
            std::cout << "  [" << kTag << "] No restart clock stored in " << dataset_path_
                      << " (looked for model_time_s, model_step, time)." << std::endl;
        } else {
            std::cout << "  [" << kTag << "] Restart clock read from " << dataset_path_
                      << " step " << dataset.step() << ":";
            if (metadata.has_time) std::cout << " time=" << metadata.time_s << " s";
            if (metadata.has_step) std::cout << " step=" << metadata.step;
            std::cout << std::endl;
        }
    }
    return metadata;
}

} // namespace VVM::IO::BP5
