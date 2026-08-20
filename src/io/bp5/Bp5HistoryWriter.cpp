#include "Bp5HistoryWriter.hpp"
#include "Bp5CollectiveValidation.hpp"
#include "Bp5PathPolicy.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <type_traits>
#include <unordered_set>
#include <variant>

#include <Kokkos_Core.hpp>

#include "utils/Timer.hpp"

namespace VVM::IO::BP5 {
namespace {

double seconds_since(const std::chrono::steady_clock::time_point& start) {
    return std::chrono::duration<double>(
               std::chrono::steady_clock::now() - start).count();
}

} // namespace

Bp5HistoryWriter::Bp5HistoryWriter(
    const Utils::ConfigurationManager& config,
    const Core::Grid& grid,
    const Core::Parameters& parameters,
    Core::State& state,
    MPI_Comm comm)
    : grid_(grid),
      parameters_(parameters),
      state_(state),
      comm_(comm),
      config_(Bp5OutputConfig::from_config(config)),
      schema_(
          Bp5FieldSchema::parse_bounds(
              config,
              static_cast<std::size_t>(grid.get_global_points_x()),
              static_cast<std::size_t>(grid.get_global_points_y()),
              static_cast<std::size_t>(grid.get_global_points_z())),
          Bp5FieldSchema::from_grid(grid),
          grid.get_mpi_rank()),
      field_names_(config.get_value<std::vector<std::string>>("output.fields_to_output")),
      adios_(comm),
      field_source_(state_, buffers_) {
    MPI_Comm_rank(comm_, &rank_);
    MPI_Comm_size(comm_, &size_);

    element_type_ = config_.element_type();
    effective_buffer_mode_ = config_.effective_buffer_mode();
    narrowed_restart_source_ =
        element_type_ == OutputElementType::Float32 &&
        !output_element_matches_real(element_type_) &&
        config.get_value<bool>("restart.enable", false);

    if (field_names_.empty()) {
        throw std::invalid_argument("output.fields_to_output must not be empty for BP5.");
    }
    std::vector<std::string> sorted_names = field_names_;
    std::sort(sorted_names.begin(), sorted_names.end());
    if (std::adjacent_find(sorted_names.begin(), sorted_names.end()) != sorted_names.end()) {
        throw std::invalid_argument("output.fields_to_output contains a duplicate field name.");
    }

    validate_collective_configuration(config);
    prepare_dataset_path(config);

    try {
        io_ = adios_.DeclareIO("VVM_BP5_HISTORY");
        io_.SetEngine("BP5");
        io_.SetParameters(config_.adios_parameters());
        define_schema();
        prepare_coordinates();
        validate_coverage();
        write_grads_ctl_file(config);
        print_configuration();
        writer_ = io_.Open(dataset_path_.string(), adios2::Mode::Write, comm_);
        writer_.LockWriterDefinitions();
    } catch (const std::exception& e) {
        throw_operation("construct/open", e);
    }
}

void Bp5HistoryWriter::validate_collective_configuration(
    const Utils::ConfigurationManager& config) const {
    std::ostringstream resolved;
    const auto add = [&](const std::string& value) {
        resolved << value.size() << ':' << value;
    };
    add(config.get_value<std::string>("output.output_dir"));
    add(config.get_value<std::string>("output.output_filename_prefix"));
    add(config_.aggregation_type);
    resolved << config_.num_subfiles << ':' << config_.stats_level << ':'
             << config_.async_write << ':' << static_cast<int>(config_.buffer_mode)
             << ':' << static_cast<int>(config_.precision)
             << ':' << static_cast<int>(element_type_)
             << ':' << static_cast<int>(effective_buffer_mode_)
             << ':' << config_.overwrite << ':';
    const auto& bounds = schema_.bounds();
    resolved << bounds.x_start << ':' << bounds.x_end << ':'
             << bounds.y_start << ':' << bounds.y_end << ':'
             << bounds.z_start << ':' << bounds.z_end << ':';
    for (const auto& field : field_names_) add(field);
    require_collective_match(resolved.str(), comm_, "resolved output configuration");
}

Bp5HistoryWriter::~Bp5HistoryWriter() {
    if (!closed_) {
        try {
            close();
        } catch (const std::exception& e) {
            std::cerr << "[BP5 rank " << rank_
                      << "] destructor close failed for '" << dataset_path_.string()
                      << "': " << e.what() << std::endl;
        }
    }
}

void Bp5HistoryWriter::prepare_dataset_path(
    const Utils::ConfigurationManager& config) {
    const std::string output_dir =
        config.get_value<std::string>("output.output_dir");
    const std::string prefix =
        config.get_value<std::string>("output.output_filename_prefix");
    dataset_path_ = std::filesystem::path(output_dir) /
        (std::filesystem::path(prefix).extension() == ".bp" ? prefix : prefix + ".bp");

    int path_status = 0;
    if (rank_ == 0) {
        try {
            if (config.get_value<bool>("restart.enable", false) &&
                config.has_key("restart.source_file")) {
                std::error_code ec;
                const auto source = std::filesystem::weakly_canonical(
                    std::filesystem::path(
                        config.get_value<std::string>("restart.source_file")), ec);
                const auto target =
                    std::filesystem::weakly_canonical(dataset_path_, ec);
                if (!ec && source == target) {
                    throw std::runtime_error(
                        "restart.source_file is the dataset this run would write ('" +
                        dataset_path_.string() +
                        "'); choose a different output.output_dir or "
                        "output.output_filename_prefix so the resumed history survives");
                }
            }

            dataset_path_ = prepare_bp5_dataset_path(
                output_dir, prefix, config_.overwrite);
        } 
        catch (const std::exception& e) {
            std::cerr << "[BP5 rank 0] path preparation failed for '"
                      << dataset_path_.string() << "': " << e.what() << std::endl;
            path_status = 1;
        }
    }
    MPI_Bcast(&path_status, 1, MPI_INT, 0, comm_);
    if (path_status != 0) {
        throw std::runtime_error(
            "BP5 dataset path preparation failed for '" + dataset_path_.string() + "'.");
    }
    MPI_Barrier(comm_);
}

void Bp5HistoryWriter::define_metadata(
    const std::string& field_name,
    const Core::FieldMetadata& metadata) {
    if (!metadata.units.empty()) {
        io_.DefineAttribute<std::string>("units", metadata.units, field_name);
    }
    if (!metadata.long_name.empty()) {
        io_.DefineAttribute<std::string>("long_name", metadata.long_name, field_name);
    }
    if (!metadata.standard_name.empty()) {
        io_.DefineAttribute<std::string>("standard_name", metadata.standard_name, field_name);
    }
    if (!metadata.comment.empty()) {
        io_.DefineAttribute<std::string>("comment", metadata.comment, field_name);
    }
    if (metadata.grid_staggering != Core::GridStaggering::Unspecified) {
        io_.DefineAttribute<std::string>(
            "grid_staggering",
            Core::grid_staggering_to_string(metadata.grid_staggering),
            field_name);
    }
}

void Bp5HistoryWriter::skip_field(const std::string& field_name, const char* reason) const {
    if (rank_ == 0) {
        std::cout << "  [BP5] Skipping output field '" << field_name << "': " << reason
                  << std::endl;
    }
}

void Bp5HistoryWriter::define_field(const std::string& field_name) {
    auto it = state_.begin();
    while (it != state_.end() && it->first != field_name) ++it;
    if (it == state_.end()) {
        skip_field(field_name, "not registered in this run's state");
        return;
    }

    std::visit(
        [&](const auto& field) {
            using FieldType = std::decay_t<decltype(field)>;
            if constexpr (std::is_same_v<FieldType, std::monostate>) {
                skip_field(field_name, "registered but holds no field");
            } else if constexpr (FieldType::DimValue == 0) {
                skip_field(field_name, "0-D fields are not supported by BP5 output");
            } else {
                const auto& view = field.get_device_data();
                const std::size_t components =
                    FieldType::DimValue == 4 ? view.extent(0) : 1;
                FieldSelection selection =
                    schema_.selection(FieldType::DimValue, components);
                for (std::size_t d = 0; d < selection.memory_count.size(); ++d) {
                    if (view.extent(d) != selection.memory_count[d]) {
                        throw std::invalid_argument(
                            "Configured BP5 field '" + field_name +
                            "' dimensions do not match its grid role.");
                    }
                }
                if (effective_buffer_mode_ == CpuBufferMode::Direct) {
                    using Layout = typename std::decay_t<decltype(view)>::array_layout;
                    if constexpr (!std::is_same_v<Layout, Kokkos::LayoutRight>) {
                        throw std::invalid_argument(
                            "BP5 direct mode requires LayoutRight for field '" + field_name +
                            "'; use output.bp5.buffer_mode='pack'.");
                    }
                }

                const bool direct = (effective_buffer_mode_ == CpuBufferMode::Direct);
                FieldVariable variable;
                if (element_type_ == OutputElementType::Float32) {
                    auto typed = io_.DefineVariable<float>(
                        field_name, selection.shape, selection.start, selection.count,
                        adios2::ConstantDims);
                    if (direct && !selection.empty()) {
                        typed.SetMemorySelection(
                            {selection.memory_start, selection.memory_count});
                    }
                    variable = typed;
                } else {
                    auto typed = io_.DefineVariable<double>(
                        field_name, selection.shape, selection.start, selection.count,
                        adios2::ConstantDims);
                    if (direct && !selection.empty()) {
                        typed.SetMemorySelection(
                            {selection.memory_start, selection.memory_count});
                    }
                    variable = typed;
                }
                const auto& metadata = field.get_metadata();
                define_metadata(field_name, metadata);
                fields_.push_back({field_name, describe_field(field_name, metadata),
                                   std::move(selection), std::move(variable)});
            }
        },
        it->second);
}

void Bp5HistoryWriter::define_schema() {
    time_variable_ = io_.DefineVariable<VVM::Real>("time");
    model_time_variable_ = io_.DefineVariable<double>("model_time_s");
    model_step_variable_ = io_.DefineVariable<std::int64_t>("model_step");

    const auto& region = schema_.grid();
    x_variable_ = io_.DefineVariable<VVM::Real>(
        "coordinates/x", {region.global_nx}, {0},
        {rank_ == 0 ? region.global_nx : 0}, adios2::ConstantDims);
    y_variable_ = io_.DefineVariable<VVM::Real>(
        "coordinates/y", {region.global_ny}, {0},
        {rank_ == 0 ? region.global_ny : 0}, adios2::ConstantDims);
    z_variable_ = io_.DefineVariable<VVM::Real>(
        "coordinates/z_mid", {region.global_nz}, {0},
        {rank_ == 0 ? region.global_nz : 0}, adios2::ConstantDims);

    io_.DefineAttribute<std::string>("units", "s", "time");
    io_.DefineAttribute<std::string>("long_name", "elapsed simulation time", "time");
    io_.DefineAttribute<std::string>("units", "s", "model_time_s");
    io_.DefineAttribute<std::string>("long_name", "elapsed simulation time", "model_time_s");
    io_.DefineAttribute<std::string>("units", "1", "model_step");
    io_.DefineAttribute<std::string>("long_name", "integration step count", "model_step");
    io_.DefineAttribute<std::string>("units", "meter", "coordinates/x");
    io_.DefineAttribute<std::string>("units", "meter", "coordinates/y");
    io_.DefineAttribute<std::string>("units", "meter", "coordinates/z_mid");

    io_.DefineAttribute<std::string>("vvm_schema_version", "1");
    io_.DefineAttribute<std::string>("vvm_output_role", "history");
    io_.DefineAttribute<std::string>(
        "vvm_real_precision", sizeof(VVM::Real) == 8 ? "float64" : "float32");
    io_.DefineAttribute<std::string>(
        "vvm_field_precision", output_element_type_name(element_type_));
    io_.DefineAttribute<std::string>("vvm_coordinate_order", "z,y,x");
    const std::array<std::uint64_t, 3> grid_shape = {
        region.global_nz, region.global_ny, region.global_nx};
    io_.DefineAttribute<std::uint64_t>(
        "vvm_global_grid_shape", grid_shape.data(), grid_shape.size());
    const auto& bounds = schema_.bounds();
    const std::array<std::uint64_t, 6> output_bounds = {
        bounds.z_start, bounds.z_end,
        bounds.y_start, bounds.y_end,
        bounds.x_start, bounds.x_end};
    io_.DefineAttribute<std::uint64_t>(
        "vvm_output_bounds_zyx", output_bounds.data(), output_bounds.size());

    fields_.reserve(field_names_.size());
    for (const auto& field_name : field_names_) define_field(field_name);
}

std::string Bp5HistoryWriter::describe_field(
    const std::string& field_name,
    const Core::FieldMetadata& metadata) {
    std::string description =
        metadata.long_name.empty() ? field_name : metadata.long_name;
    if (!metadata.units.empty()) description += " (" + metadata.units + ")";
    return description;
}

void Bp5HistoryWriter::write_grads_ctl_file(
    const Utils::ConfigurationManager& config) {
    const bool use_taiwanvvm_coordinates =
        config.get_value<std::string>("grid.vertical_coordinate_type", "default") ==
        "taiwanvvm";
    // Collective on every rank; only rank 0 writes the descriptor.
    const auto axes =
        grads_horizontal_axes(grid_, state_, use_taiwanvvm_coordinates, comm_);
    if (rank_ != 0) return;

    const auto& region = schema_.grid();

    GradsCtl ctl;
    ctl.dset = "^" + dataset_path_.filename().string();
    ctl.dtype = "bp5";
    ctl.x = axes.first;
    ctl.y = axes.second;
    ctl.nx = region.global_nx;
    ctl.ny = region.global_ny;
    ctl.z_levels = z_coordinates_;

    const double total_time = config.get_value<double>("simulation.total_time_s", 0.0);
    const double output_interval =
        std::max(1e-6, config.get_value<double>("simulation.output_interval_s", 1.0));
    const auto intervals_before = [&](double time) {
        return std::floor(time / output_interval + 1e-6);
    };
    const double start_time = static_cast<double>(state_.get_time());
    ctl.time_count = static_cast<std::size_t>(
        std::max(0.0, intervals_before(total_time) - intervals_before(start_time)));
    if (config.get_value<bool>("output.output_initial_step", true)) ++ctl.time_count;
    const int start_hour = (config.get_value<int>("physics.rrtmgp.time.hour", 16) + 8) % 24;
    ctl.time_start = grads_start_time(start_hour);
    ctl.time_increment = grads_time_increment(output_interval);

    std::unordered_set<std::string> taken;
    std::vector<std::string> profile_fields;
    for (const auto& field : fields_) {
        GradsVariable variable;
        variable.dataset_name = field.name;
        variable.description = field.description;
        if (field.selection.dimensions == 2) {
            variable.levels = 0;
            variable.dimensions = "y,x";
        } 
        else if (field.selection.dimensions == 3) {
            variable.levels = region.global_nz;
            variable.dimensions = "z,y,x";
        } 
        else if (field.selection.dimensions == 4) {
            variable.levels = region.global_nz;
            variable.dimensions = "0,z,y,x";
        } 
        else {
            profile_fields.push_back(field.name);
            continue;
        }
        variable.grads_name = unique_grads_variable_name(field.name, taken);
        ctl.variables.push_back(std::move(variable));
    }
    if (!profile_fields.empty()) {
        std::string note =
            "Profile (z-only) variables in the dataset, not usable from GrADS:";
        for (const auto& name : profile_fields) note += " " + name;
        ctl.notes.push_back(note);
    }

    const std::filesystem::path ctl_path =
        std::filesystem::path(config.get_value<std::string>("output.output_dir")) /
        "vvm.ctl";
    write_grads_ctl(ctl_path, ctl);
    std::cout << "  [BP5] GrADS descriptor: " << ctl_path.string() << std::endl;
}

void Bp5HistoryWriter::prepare_coordinates() {
    if (rank_ != 0) return;
    const auto& region = schema_.grid();
    x_coordinates_.resize(region.global_nx);
    y_coordinates_.resize(region.global_ny);
    z_coordinates_.resize(region.global_nz);
    for (std::size_t i = 0; i < region.global_nx; ++i) {
        x_coordinates_[i] = static_cast<VVM::Real>(i) * grid_.get_dx();
    }
    for (std::size_t j = 0; j < region.global_ny; ++j) {
        y_coordinates_[j] = static_cast<VVM::Real>(j) * grid_.get_dy();
    }
    const auto z_host = parameters_.z_mid.get_host_data();
    for (std::size_t k = 0; k < region.global_nz; ++k) {
        z_coordinates_[k] = z_host(k + region.halo);
    }
}

void Bp5HistoryWriter::validate_coverage() {
    const auto check = [&](std::size_t dimensions, std::size_t expected) {
        const FieldSelection selection = schema_.selection(dimensions, 1);
        const unsigned long long local =
            static_cast<unsigned long long>(selection.elements());
        unsigned long long global = 0;
        MPI_Allreduce(&local, &global, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, comm_);
        if (global != expected) {
            throw std::runtime_error(
                "BP5 rank selections do not cover the requested " +
                std::to_string(dimensions) + "-D output domain exactly once (covered " +
                std::to_string(global) + ", expected " + std::to_string(expected) + ").");
        }
    };
    const auto& bounds = schema_.bounds();
    check(1, bounds.nz());
    check(2, bounds.ny() * bounds.nx());
    check(3, bounds.nz() * bounds.ny() * bounds.nx());

    std::size_t local_bytes = 0;
    for (const auto& field : fields_) {
        local_bytes +=
            field.selection.elements() * output_element_size(element_type_);
    }
    unsigned long long local = static_cast<unsigned long long>(local_bytes);
    unsigned long long global = 0;
    MPI_Allreduce(&local, &global, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, comm_);
    global_bytes_per_step_ =
        static_cast<std::size_t>(global) +
        (schema_.grid().global_nx + schema_.grid().global_ny + schema_.grid().global_nz) *
            sizeof(VVM::Real) +
        sizeof(VVM::Real) + sizeof(double) + sizeof(std::int64_t);
}

void Bp5HistoryWriter::print_configuration() const {
    if (rank_ != 0) return;
    std::cout << "  [BP5] Direct compute-rank history output\n"
              << "  [BP5] Dataset: " << dataset_path_.string() << "\n"
              << "  [BP5] AggregationType: " << config_.aggregation_type << "\n"
              << "  [BP5] NumSubFiles: " << config_.num_subfiles << "\n"
              << "  [BP5] StatsLevel: " << config_.stats_level << "\n"
              << "  [BP5] AsyncWrite: " << (config_.async_write ? "true" : "false") << "\n"
              << "  [BP5] Field precision: " << output_element_type_name(element_type_)
              << " (requested '" << output_precision_name(config_.precision)
              << "', model VVM::Real is "
              << (sizeof(VVM::Real) == 8 ? "float64" : "float32") << ")\n"
              << "  [BP5] Field buffer mode: "
              << cpu_buffer_mode_name(effective_buffer_mode_);
    if (effective_buffer_mode_ != config_.buffer_mode) {
        std::cout << "  (requested '" << cpu_buffer_mode_name(config_.buffer_mode)
                  << "'; ";
#if defined(KOKKOS_ENABLE_CUDA)
        std::cout << "CUDA fields require host staging)";
#else
        std::cout << "converting precision requires a staging buffer)";
#endif
    }
    std::cout << "\n"
              << "  [BP5] Estimated logical bytes/step: " << global_bytes_per_step_
              << std::endl;
    if (narrowed_restart_source_) {
        std::cout << "  [BP5] WARNING: narrowed output is a restart source; "
                     "restarts from this dataset lose precision." << std::endl;
    }
}

void Bp5HistoryWriter::write(std::size_t step, VVM::Real time) {
    if (closed_) throw std::logic_error("Cannot write to a closed BP5 history dataset.");

    std::vector<FieldInput> inputs;
    inputs.reserve(fields_.size());
    double prepare_s = 0.0;
    double put_s = 0.0;
    double end_step_s = 0.0;

    try {
        {
            Utils::Timer timer("bp5_field_prepare");
            const auto start = std::chrono::steady_clock::now();
            Kokkos::fence("bp5_history_source_ready");
            for (const auto& field : fields_) {
                inputs.push_back(field_source_.prepare(
                    field.name, field.selection, effective_buffer_mode_,
                    element_type_));
            }
            prepare_s = seconds_since(start);
        }

        {
            Utils::Timer timer("bp5_begin_step");
            writer_.BeginStep();
        }

        {
            Utils::Timer timer("bp5_put_submit");
            const auto start = std::chrono::steady_clock::now();
            const adios2::Mode put_mode =
                config_.async_write ? adios2::Mode::Sync : adios2::Mode::Deferred;
            const double model_time_s = static_cast<double>(time);
            const std::int64_t model_step = static_cast<std::int64_t>(step);
            if (rank_ == 0) {
                writer_.Put(time_variable_, &time, adios2::Mode::Sync);
                writer_.Put(model_time_variable_, &model_time_s, adios2::Mode::Sync);
                writer_.Put(model_step_variable_, &model_step, adios2::Mode::Sync);
                writer_.Put(x_variable_, x_coordinates_.data(), put_mode);
                writer_.Put(y_variable_, y_coordinates_.data(), put_mode);
                writer_.Put(z_variable_, z_coordinates_.data(), put_mode);
            }
            for (std::size_t i = 0; i < fields_.size(); ++i) {
                if (fields_[i].selection.empty()) continue;
                if (inputs[i].data == nullptr) {
                    throw std::logic_error(
                        "BP5 field source returned a null pointer for '" +
                        fields_[i].name + "'.");
                }
                std::visit(
                    [&](auto& variable) {
                        using VariableType = std::decay_t<decltype(variable)>;
                        if constexpr (std::is_same_v<VariableType,
                                                     adios2::Variable<float>>) {
                            writer_.Put(variable,
                                        static_cast<const float*>(inputs[i].data),
                                        put_mode);
                        } else {
                            writer_.Put(variable,
                                        static_cast<const double*>(inputs[i].data),
                                        put_mode);
                        }
                    },
                    fields_[i].variable);
            }
            put_s = seconds_since(start);
        }

        {
            Utils::Timer timer("bp5_end_step");
            const auto start = std::chrono::steady_clock::now();
            writer_.EndStep();
            end_step_s = seconds_since(start);
        }
        ++steps_written_;
    } catch (const std::exception& e) {
        throw_operation("write step", e);
    }

    const std::array<double, 3> local = {prepare_s, put_s, end_step_s};
    std::array<double, 3> maximum = {0.0, 0.0, 0.0};
    MPI_Reduce(local.data(), maximum.data(), static_cast<int>(local.size()),
               MPI_DOUBLE, MPI_MAX, 0, comm_);
    if (rank_ == 0) {
        std::cout << "  [BP5] Step " << steps_written_ - 1
                  << " model_step=" << step
                  << " prepare_max=" << maximum[0] << " s"
                  << " put_max=" << maximum[1] << " s"
                  << " end_step_max=" << maximum[2] << " s"
                  << std::endl;
    }
}

void Bp5HistoryWriter::close() {
    if (closed_) return;
    const auto start = std::chrono::steady_clock::now();
    try {
        Utils::Timer timer("bp5_close");
        if (writer_) writer_.Close();
        closed_ = true;
    } catch (const std::exception& e) {
        throw_operation("close", e);
    }
    const double local = seconds_since(start);
    double maximum = 0.0;
    MPI_Reduce(&local, &maximum, 1, MPI_DOUBLE, MPI_MAX, 0, comm_);
    if (rank_ == 0) {
        std::cout << "  [BP5] Closed " << dataset_path_.string()
                  << " after " << steps_written_ << " step(s); close_max="
                  << maximum << " s" << std::endl;
    }
}

[[noreturn]] void Bp5HistoryWriter::throw_operation(
    const char* operation,
    const std::exception& error) const {
    throw std::runtime_error(
        "BP5 rank " + std::to_string(rank_) + " failed to " + operation +
        " dataset '" + dataset_path_.string() + "': " + error.what());
}

} // namespace VVM::IO::BP5
