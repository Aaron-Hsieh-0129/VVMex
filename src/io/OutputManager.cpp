#include "OutputManager.hpp"
#include "Hdf5DimensionScales.hpp"
#include "io/history/GradsCtl.hpp"
#include <sys/stat.h>
#include <cerrno>
#include <algorithm>
#include <cstdlib>
#include <array>
#include <cctype>
#include <cstdint>
#include <cmath>
#include <filesystem>
#include <system_error>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <unordered_set>
#include <hdf5.h>

namespace VVM {
namespace IO {

namespace {
std::string uppercase_transport_name(std::string value) {
    std::transform(
        value.begin(),
        value.end(),
        value.begin(),
        [](unsigned char c) { return static_cast<char>(std::toupper(c)); }
    );
    return value;
}
} // namespace

std::string OutputManager::format_to_six_digits(int number) {
    std::stringstream ss;
    ss << std::setfill('0') << std::setw(6) << number;
    return ss.str();
}

OutputManager::OutputManager(const Utils::ConfigurationManager& config, const VVM::Core::Grid& grid, const VVM::Core::Parameters& params, VVM::Core::State& state, MPI_Comm comm)
    : grid_(grid), params_(params), state_(state), comm_(comm), adios_(comm) {
    
    MPI_Comm_rank(comm_, &rank_);
    MPI_Comm_size(comm_, &mpi_size_);

    output_dir_ = config.get_value<std::string>("output.output_dir");
    filename_prefix_ = config.get_value<std::string>("output.output_filename_prefix");
    fields_to_output_ = config.get_value<std::vector<std::string>>("output.fields_to_output");
    if (config.has_key("dynamics.tracers")) {
        const auto tracer_config = config.get_value<nlohmann::json>("dynamics.tracers");
        static const std::unordered_set<std::string> optional_builtin_output_fields = {
            "qc", "qr", "qi", "qm", "nc", "nr", "ni", "bm",
            "sw_heating", "lw_heating", "swdn", "lwdn", "lwup",
            "swup_toa", "swdn_toa", "lwup_toa", "lwdn_toa",
            "swup_sfc", "swdn_sfc", "lwup_sfc", "lwdn_sfc",
            "precip_liq_surf_mass", "precip_ice_surf_mass",
            "sfc_flux_th", "sfc_flux_qv", "sfc_flux_u", "sfc_flux_v",
            "le", "hfx", "st1", "st2", "st3", "st4", "gfx",
            "slc1", "slc2", "slc3", "slc4", "sfemis", "zorl",
            "chx", "cmx", "albedo"
        };
        for (const auto& field_name : fields_to_output_) {
            if (tracer_config.contains(field_name) && !state_.is_tracer(field_name)) {
                throw std::runtime_error("Output field '" + field_name +
                                         "' names a disabled or unregistered tracer.");
            }
            if (!state_.has_field(field_name) &&
                optional_builtin_output_fields.count(field_name) == 0) {
                throw std::runtime_error("Output field '" + field_name +
                                         "' is not registered. If this is a tracer, check its name and enable setting.");
            }
        }
    }
    output_interval_s_ = config.get_value<VVM::Real>("simulation.output_interval_s");
    total_time_ = config.get_value<VVM::Real>("simulation.total_time_s");
    use_taiwanvvm_coordinates_ = (config.get_value<std::string>("grid.vertical_coordinate_type", "default") == "taiwanvvm");
    const int config_start_hour = config.get_value<int>("physics.rrtmgp.time.hour", 16);
    grads_start_hour_ = (config_start_hour + 8) % 24;

    // Default to HDF5 if not specified
    if (config.has_key("output.engine")) {
        engine_type_ = config.get_value<std::string>("output.engine");
    } 
    else {
        engine_type_ = "HDF5"; 
    }

    element_type_ = resolve_output_element_type(configured_output_precision(config));
    converts_precision_ = !output_element_matches_real(element_type_);
    if (rank_ == 0 && converts_precision_) {
        std::cout << "  [OutputManager] Field precision: "
                  << output_element_type_name(element_type_) << " (model VVM::Real is "
                  << (sizeof(VVM::Real) == 8 ? "float64" : "float32")
                  << "); clocks and coordinates stay VVM::Real" << std::endl;
        if (element_type_ == OutputElementType::Float32 &&
            config.get_value<bool>("restart.enable", false)) {
            // HDF5 output is also the restart source, so narrowing it narrows
            // what a later restart can recover. Worth saying out loud rather
            // than discovering from a restarted run's trajectory.
            std::cout << "  [OutputManager] WARNING: narrowed output is the restart "
                         "source; restarts from these files lose precision."
                      << std::endl;
        }
    }

    output_x_start_  = config.get_value<size_t>("output.output_grid.x_start");
    output_y_start_  = config.get_value<size_t>("output.output_grid.y_start");
    output_z_start_  = config.get_value<size_t>("output.output_grid.z_start");

    output_x_end_    = config.get_value<size_t>("output.output_grid.x_end");
    output_y_end_    = config.get_value<size_t>("output.output_grid.y_end");
    output_z_end_    = config.get_value<size_t>("output.output_grid.z_end");

    // Create the output directory with std::filesystem rather than
    // system("mkdir -p " + output_dir_): a space in output.output_dir split the
    // shell command into two paths, and a metacharacter ran as a command.
    // Rank 0 creates it and tells everyone the result, so it cannot fail alone
    // while the other ranks wait in the barrier below.
    if (!output_dir_.empty()) {
        int mkdir_failed = 0;
        if (rank_ == 0) {
            std::error_code ec;
            // Returns false when the directory already exists, which is not an
            // error -- only ec is conclusive here.
            std::filesystem::create_directories(std::filesystem::path(output_dir_), ec);
            if (ec) {
                std::cerr << "[OutputManager] ERROR: cannot create output.output_dir '"
                          << output_dir_ << "': " << ec.message() << std::endl;
                mkdir_failed = 1;
            }
        }
        MPI_Bcast(&mkdir_failed, 1, MPI_INT, 0, comm_);
        if (mkdir_failed != 0) {
            MPI_Abort(MPI_COMM_WORLD, 4);
        }
    }

    MPI_Barrier(comm_);
    grads_ctl_file();

    io_ = adios_.DeclareIO("VVM_IO");
    io_.SetEngine(engine_type_);

    if (engine_type_ == "HDF5") {
        const bool use_collective_mpio = config.get_value<bool>("output.hdf5_collective_mpio", false);
        const std::string use_collective = use_collective_mpio ? "true" : "false";
        if (rank_ == 0) std::cout << "  [OutputManager] Engine: HDF5. Collective: " << use_collective << std::endl;
        
        io_.SetParameter("IdleH5Writer", "true");
        io_.SetParameter("H5CollectiveMPIO", use_collective);
    } 
    else if (engine_type_ == "SST") {
        const int queue_limit = config.get_value<int>("output.queue_limit", 1);
        const std::string data_transport =
            uppercase_transport_name(config.get_value<std::string>("output.data_transport", "WAN"));
        const std::string control_transport =
            config.get_value<std::string>("output.control_transport", "sockets");

        if (rank_ == 0) {
            std::cout << "  [OutputManager] Engine: SST" << std::endl;
            std::cout << "  [OutputManager] SST DataTransport: "
                      << ((data_transport.empty() || data_transport == "AUTO")
                              ? "AUTO"
                              : data_transport)
                      << std::endl;
            std::cout << "  [OutputManager] SST ControlTransport: "
                      << control_transport << std::endl;
            std::cout << "  [OutputManager] SST QueueLimit: " << queue_limit << std::endl;
        }

        io_.SetParameter("MarshalMethod", "BP5");

        if (!data_transport.empty() && data_transport != "AUTO") {
            io_.SetParameter("DataTransport", data_transport);
        }
        if (data_transport == "WAN") {
            io_.SetParameter("WANDataTransport", "sockets");
        }
        if (!control_transport.empty()) {
            io_.SetParameter("ControlTransport", control_transport);
        }

        io_.SetParameter("RendezvousReaderCount", "1");
        io_.SetParameter("QueueLimit", std::to_string(queue_limit));
        io_.SetParameter("QueueFullPolicy", "Block");
        io_.SetParameter("ReserveQueueLimit", "0");

        define_variables();
        variables_defined_ = true;

        const std::string stream_name = output_dir_ + "/" + filename_prefix_;
        if (rank_ == 0) {
            std::cout << "  [OutputManager] SST Streaming: "
                    << stream_name << std::endl;
        }

        writer_ = io_.Open(stream_name, adios2::Mode::Write, comm_);
    }
}

OutputManager::~OutputManager() {
    if (writer_) writer_.Close();
}

void OutputManager::define_adios_field_metadata(
    const std::string& field_name,
    const VVM::Core::FieldMetadata& metadata)
{
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
    if (metadata.grid_staggering != VVM::Core::GridStaggering::Unspecified) {
        io_.DefineAttribute<std::string>(
            "grid_staggering",
            VVM::Core::grid_staggering_to_string(metadata.grid_staggering),
            field_name);
    }
}

void OutputManager::define_variables() {
    const size_t gnx = grid_.get_global_points_x();
    const size_t gny = grid_.get_global_points_y();
    const size_t gnz = grid_.get_global_points_z();

    io_.DefineVariable<VVM::Real>("time");

    // Restart clock. "time" has always held elapsed seconds and analysis tools
    // read it, so it keeps its name and meaning; these two are the unambiguous,
    // self-describing pair a restart is recovered from -- a double so long runs
    // keep full precision in single-precision builds, and an exact integer step
    // so the resumed run does not have to re-derive it from dt.
    io_.DefineVariable<double>("model_time_s");
    io_.DefineVariable<int64_t>("model_step");

    io_.DefineVariable<VVM::Real>("coordinates/x", {gnx}, {0}, {rank_ == 0 ? gnx : 0});
    io_.DefineVariable<VVM::Real>("coordinates/y", {gny}, {0}, {rank_ == 0 ? gny : 0});
    io_.DefineVariable<VVM::Real>("coordinates/z_mid", {gnz}, {0}, {rank_ == 0 ? gnz : 0});

    // The value written here is elapsed simulation seconds, not a calendar date.
    // It used to be labelled "hours since 2025-10-07 00:00:00", which was wrong
    // in both unit and origin.
    io_.DefineAttribute<std::string>("units", "s", "time");
    io_.DefineAttribute<std::string>("long_name", "elapsed simulation time", "time");
    io_.DefineAttribute<std::string>("units", "s", "model_time_s");
    io_.DefineAttribute<std::string>("long_name", "elapsed simulation time", "model_time_s");
    io_.DefineAttribute<std::string>("units", "1", "model_step");
    io_.DefineAttribute<std::string>("long_name", "integration step count", "model_step");
    io_.DefineAttribute<std::string>("units", "meter", "coordinates/z_mid");
    io_.DefineAttribute<std::string>("units", "meter", "coordinates/y");
    io_.DefineAttribute<std::string>("units", "meter", "coordinates/x");

    // A reader needs both to tell whether the file is a lossless copy of the
    // model state: the model's working precision, and what the field variables
    // actually are.
    io_.DefineAttribute<std::string>(
        "vvm_real_precision", sizeof(VVM::Real) == 8 ? "float64" : "float32");
    io_.DefineAttribute<std::string>(
        "vvm_field_precision", output_element_type_name(element_type_));

    const size_t rank_lnx = grid_.get_local_physical_points_x();
    const size_t rank_lny = grid_.get_local_physical_points_y();
    const size_t rank_lnz = grid_.get_local_physical_points_z();

    const size_t rank_offset_x = grid_.get_local_physical_start_x();
    const size_t rank_offset_y = grid_.get_local_physical_start_y();
    const size_t rank_offset_z = grid_.get_local_physical_start_z();

    for (const auto& field_name : fields_to_output_) {
        auto it = state_.begin();
        while (it != state_.end() && it->first != field_name) ++it;

        if (it != state_.end()) {
            std::visit([&](const auto& field) {
                using T = std::decay_t<decltype(field)>;
                if constexpr (!std::is_same_v<T, std::monostate>) {
                    size_t actual_out_x_start = std::max(rank_offset_x, output_x_start_);
                    size_t actual_out_x_end = std::min(rank_offset_x + rank_lnx - 1, output_x_end_);
                    size_t local_nx = (actual_out_x_end >= actual_out_x_start) ? (actual_out_x_end - actual_out_x_start + 1) : 0;

                    size_t actual_out_y_start = std::max(rank_offset_y, output_y_start_);
                    size_t actual_out_y_end = std::min(rank_offset_y + rank_lny - 1, output_y_end_);
                    size_t local_ny = (actual_out_y_end >= actual_out_y_start) ? (actual_out_y_end - actual_out_y_start + 1) : 0;

                    size_t actual_out_z_start = std::max(rank_offset_z, output_z_start_);
                    size_t actual_out_z_end = std::min(rank_offset_z + rank_lnz - 1, output_z_end_);
                    size_t local_nz = (actual_out_z_end >= actual_out_z_start) ? (actual_out_z_end - actual_out_z_start + 1) : 0;
                    
                    if constexpr (T::DimValue == 1) {
                        size_t count = (rank_ == 0) ? local_nz : 0;
                        define_field_variable(field_name, {gnz}, {actual_out_z_start}, {count});
                    }
                    else if constexpr (T::DimValue == 2) {
                        define_field_variable(field_name, {gny, gnx}, {actual_out_y_start, actual_out_x_start}, {local_ny, local_nx});
                    }
                    else if constexpr (T::DimValue == 3) {
                        define_field_variable(field_name, {gnz, gny, gnx}, {actual_out_z_start, actual_out_y_start, actual_out_x_start}, {local_nz, local_ny, local_nx});
                    }
                    else if constexpr (T::DimValue == 4) {
                        const size_t dim4 = field.get_device_data().extent(0);
                        define_field_variable(field_name, {dim4, gnz, gny, gnx}, {0, actual_out_z_start, actual_out_y_start, actual_out_x_start}, {dim4, local_nz, local_ny, local_nx});
                    }

                    const auto& metadata = field.get_metadata();
                    if (field_counts_.count(field_name)) {
                        define_adios_field_metadata(field_name, metadata);
                    }
                }
            }, it->second);
        }
    }
}

void OutputManager::define_field_variable(
    const std::string& field_name,
    const adios2::Dims& shape,
    const adios2::Dims& start,
    const adios2::Dims& count)
{
    if (converts_precision_) {
        converted_variables_[field_name] =
            io_.DefineVariable<ConvertedReal>(field_name, shape, start, count);
    } else {
        field_variables_[field_name] =
            io_.DefineVariable<VVM::Real>(field_name, shape, start, count);
    }
    field_counts_[field_name] = count;
}

void OutputManager::put_field(
    const std::string& field_name,
    const VVM::Real* data,
    size_t elements)
{
    if (!converts_precision_) {
        writer_.Put(field_variables_.at(field_name), data, adios2::Mode::Sync);
        return;
    }

    // The staged host copy is VVM::Real, so narrowing or widening happens here,
    // once per field per output, into a buffer that is allocated once. Never
    // empty, so a rank with no selected cells still hands ADIOS2 a valid
    // pointer, as the VVM::Real path does.
    auto& buffer = converted_buffers_[field_name];
    if (buffer.size() < std::max<size_t>(elements, 1)) {
        buffer.resize(std::max<size_t>(elements, 1));
    }
    for (size_t i = 0; i < elements; ++i) {
        buffer[i] = static_cast<ConvertedReal>(data[i]);
    }
    writer_.Put(converted_variables_.at(field_name), buffer.data(), adios2::Mode::Sync);
}

void OutputManager::write(size_t step, VVM::Real time) {
    if (!variables_defined_) {
        define_variables();
        variables_defined_ = true;
    }

    std::string filename;
    if (engine_type_ == "HDF5") {
        filename = output_dir_ + "/" + filename_prefix_ + "_" + format_to_six_digits((int) (time/output_interval_s_)) + ".h5";
        if (rank_ == 0) std::cout << "  [OutputManager] HDF5 Writing: " << filename << std::endl;
        writer_ = io_.Open(filename, adios2::Mode::Write, comm_);
    } 
    else if (engine_type_ == "SST") {
        if (!writer_) {
            filename = output_dir_ + "/" + filename_prefix_;
            if (rank_ == 0) std::cout << "  [OutputManager] SST Streaming: " << filename << std::endl;
            writer_ = io_.Open(filename, adios2::Mode::Write, comm_);
        }
    }

    writer_.BeginStep();

    auto var_time = io_.InquireVariable<VVM::Real>("time");
    writer_.Put<VVM::Real>(var_time, &time, adios2::Mode::Sync);

    const double model_time_s = static_cast<double>(time);
    const int64_t model_step = static_cast<int64_t>(step);
    auto var_model_time_s = io_.InquireVariable<double>("model_time_s");
    writer_.Put<double>(var_model_time_s, &model_time_s, adios2::Mode::Sync);
    auto var_model_step = io_.InquireVariable<int64_t>("model_step");
    writer_.Put<int64_t>(var_model_step, &model_step, adios2::Mode::Sync);

    write_static_data();

    const size_t h = grid_.get_halo_cells();
    const size_t rank_off_x = grid_.get_local_physical_start_x();
    const size_t rank_off_y = grid_.get_local_physical_start_y();
    const size_t rank_off_z = grid_.get_local_physical_start_z();

    size_t out_x_start = std::max(rank_off_x, output_x_start_);
    size_t out_y_start = std::max(rank_off_y, output_y_start_);
    size_t out_z_start = std::max(rank_off_z, output_z_start_);

    for (const auto& field_name : fields_to_output_) {
        if (field_counts_.count(field_name)) {
            const auto& adios_count = field_counts_.at(field_name);
            auto it = state_.begin();
            while (it != state_.end() && it->first != field_name) ++it;
             
            if (it != state_.end()) {
                std::visit([&](const auto& field) {
                    using T = std::decay_t<decltype(field)>;
                    if constexpr (!std::is_same_v<T, std::monostate>) {
                        auto full_data_view = field.get_device_data();
                        using DevMemSpace = typename decltype(full_data_view)::memory_space;

                        size_t k_start = (out_z_start - rank_off_z) + h;
                        size_t j_start = (out_y_start - rank_off_y) + h;
                        size_t i_start = (out_x_start - rank_off_x) + h;

                        if constexpr (T::DimValue == 1) {
                            size_t count = adios_count[0];
                            auto subview = Kokkos::subview(full_data_view, std::make_pair(k_start, k_start + count));

                            if (host_buffers_1d_.find(field_name) == host_buffers_1d_.end()) {
                                host_buffers_1d_[field_name] = Kokkos::View<VVM::Real*, Kokkos::HostSpace>(field_name + "_host", count);
                            }
                            auto& host_view = host_buffers_1d_[field_name];

                            Kokkos::deep_copy(host_view, subview);
                            put_field(field_name, host_view.data(), count);
                        }
                        else if constexpr (T::DimValue == 2) {
                            size_t ny = adios_count[0];
                            size_t nx = adios_count[1];
                            
                            // 2-Step Copy: Strided Device -> Contiguous Device -> Contiguous Host
                            if (dev_buffers_2d_.find(field_name) == dev_buffers_2d_.end()) {
                                dev_buffers_2d_[field_name] = Kokkos::View<VVM::Real**, Kokkos::LayoutRight>(field_name + "_dev", ny, nx);
                            }
                            auto& dev_contig = dev_buffers_2d_[field_name];

                            auto subview = Kokkos::subview(full_data_view, 
                                std::make_pair(j_start, j_start + ny), 
                                std::make_pair(i_start, i_start + nx));
                            Kokkos::deep_copy(dev_contig, subview);

                            if (host_buffers_2d_.find(field_name) == host_buffers_2d_.end()) {
                                host_buffers_2d_[field_name] = Kokkos::View<VVM::Real**, Kokkos::LayoutRight, Kokkos::HostSpace>(field_name + "_host", ny, nx);
                            }
                            auto& host_view = host_buffers_2d_[field_name];
                            Kokkos::deep_copy(host_view, dev_contig);

                            // UNCONDITIONAL PUT: Even if ny*nx is 0, we pass the pointer (which is valid/empty)
                            put_field(field_name, host_view.data(), ny * nx);
                        }
                        else if constexpr (T::DimValue == 3) {
                            size_t nz = adios_count[0];
                            size_t ny = adios_count[1];
                            size_t nx = adios_count[2];

                            if (dev_buffers_3d_.find(field_name) == dev_buffers_3d_.end()) {
                                dev_buffers_3d_[field_name] = Kokkos::View<VVM::Real***, Kokkos::LayoutRight>(field_name + "_dev", nz, ny, nx);
                            }
                            auto& dev_contig = dev_buffers_3d_[field_name];

                            auto subview = Kokkos::subview(full_data_view,
                                std::make_pair(k_start, k_start + nz),
                                std::make_pair(j_start, j_start + ny),
                                std::make_pair(i_start, i_start + nx));
                            Kokkos::deep_copy(dev_contig, subview);

                            if (host_buffers_3d_.find(field_name) == host_buffers_3d_.end()) {
                                host_buffers_3d_[field_name] = Kokkos::View<VVM::Real***, Kokkos::LayoutRight, Kokkos::HostSpace>(
                                    field_name + "_host", nz, ny, nx);
                            }
                            auto& host_view = host_buffers_3d_[field_name];
                            Kokkos::deep_copy(host_view, dev_contig);

                            put_field(field_name, host_view.data(), nz * ny * nx);
                        }
                        else if constexpr (T::DimValue == 4) {
                            size_t d4 = adios_count[0];
                            size_t nz = adios_count[1];
                            size_t ny = adios_count[2];
                            size_t nx = adios_count[3];

                            if (dev_buffers_4d_.find(field_name) == dev_buffers_4d_.end()) {
                                dev_buffers_4d_[field_name] = Kokkos::View<VVM::Real****, Kokkos::LayoutRight>(field_name + "_dev", d4, nz, ny, nx);
                            }
                            auto& dev_contig = dev_buffers_4d_[field_name];

                            auto subview = Kokkos::subview(full_data_view, Kokkos::ALL(),
                                std::make_pair(k_start, k_start + nz),
                                std::make_pair(j_start, j_start + ny),
                                std::make_pair(i_start, i_start + nx));
                            Kokkos::deep_copy(dev_contig, subview);

                            if (host_buffers_4d_.find(field_name) == host_buffers_4d_.end()) {
                                host_buffers_4d_[field_name] = Kokkos::View<VVM::Real****, Kokkos::LayoutRight, Kokkos::HostSpace>(
                                    field_name + "_host", d4, nz, ny, nx);
                            }
                            auto& host_view = host_buffers_4d_[field_name];
                            Kokkos::deep_copy(host_view, dev_contig);

                            put_field(field_name, host_view.data(), d4 * nz * ny * nx);
                        }
                    }
                }, it->second);
             }
        }
    }

    writer_.PerformPuts();
    writer_.EndStep();
    
    if (engine_type_ == "HDF5") {
        writer_.Close();

        MPI_Barrier(comm_);
        attach_hdf5_field_metadata(filename);
        MPI_Barrier(comm_);
    }
}

void OutputManager::write_static_data() {
    const size_t gnx = grid_.get_global_points_x();
    const size_t gny = grid_.get_global_points_y();
    const size_t gnz = grid_.get_global_points_z();
    const size_t h = grid_.get_halo_cells();

    auto var_x = io_.InquireVariable<VVM::Real>("coordinates/x");
    std::vector<VVM::Real> x_coords;
    if (rank_ == 0) {
        x_coords.resize(gnx);
        for(size_t i = 0; i < gnx; ++i) x_coords[i] = i * grid_.get_dx();
    } 
    writer_.Put<VVM::Real>(var_x, x_coords.data(), adios2::Mode::Sync);

    auto var_y = io_.InquireVariable<VVM::Real>("coordinates/y");
    std::vector<VVM::Real> y_coords;
    if (rank_ == 0) {
        y_coords.resize(gny);
        for(size_t i = 0; i < gny; ++i) y_coords[i] = i * grid_.get_dy();
    }
    writer_.Put<VVM::Real>(var_y, y_coords.data(), adios2::Mode::Sync);

    auto var_z_mid = io_.InquireVariable<VVM::Real>("coordinates/z_mid");
    std::vector<VVM::Real> z_mid_physical;
    if (rank_ == 0) {
        z_mid_physical.resize(gnz);
        auto z_mid_host = params_.z_mid.get_host_data();
        for (size_t i = 0; i < gnz; ++i) z_mid_physical[i] = z_mid_host(i + h);
    }
    writer_.Put<VVM::Real>(var_z_mid, z_mid_physical.data(), adios2::Mode::Sync);
}

void OutputManager::attach_hdf5_field_metadata(
    const std::string& filename)
{
    if (rank_ != 0) return;

    hid_t file = H5Fopen(filename.c_str(), H5F_ACC_RDWR, H5P_DEFAULT);

    if (file < 0) {
        std::cerr << "[OutputManager] Failed to reopen HDF5 file '"
                  << filename << "' for metadata output.\n";
        return;
    }

    const auto write_attribute = [](hid_t dataset, const std::string& name, const std::string& value)
    {
        if (value.empty()) return;

        hid_t space = H5Screate(H5S_SCALAR);
        hid_t type = H5Tcopy(H5T_C_S1);

        H5Tset_size(type, value.size() + 1);
        H5Tset_strpad(type, H5T_STR_NULLTERM);

        hid_t attribute = H5Acreate2(dataset, name.c_str(), type, space, H5P_DEFAULT, H5P_DEFAULT);

        if (attribute >= 0) {
            H5Awrite(attribute, type, value.c_str());
            H5Aclose(attribute);
        }

        H5Tclose(type);
        H5Sclose(space);
    };

    for (const auto& field_name : fields_to_output_) {
        const std::string dataset_path = "/Step0/" + field_name;

        hid_t dataset = H5Dopen2(file, dataset_path.c_str(), H5P_DEFAULT);

        if (dataset < 0) continue;

        auto it = state_.begin();
        while (it != state_.end() && it->first != field_name) ++it;

        if (it != state_.end()) {
            std::visit([&](const auto& field)
            {
                using T = std::decay_t<decltype(field)>;

                if constexpr (
                    !std::is_same_v<T, std::monostate>) {

                    const auto& metadata = field.get_metadata();

                    write_attribute(dataset, "units", metadata.units);
                    write_attribute(dataset, "long_name", metadata.long_name);
                    write_attribute(dataset, "standard_name", metadata.standard_name);
                    write_attribute(dataset, "comment", metadata.comment);
                    if (metadata.grid_staggering != VVM::Core::GridStaggering::Unspecified) {
                        write_attribute(
                            dataset,
                            "grid_staggering",
                            VVM::Core::grid_staggering_to_string(metadata.grid_staggering));
                    }
                }
            },
            it->second);
        }

        H5Dclose(dataset);
    }

    // The clocks and coordinates are not State fields, so the loop above never
    // reaches them. Label them in the file itself: a reader that opens
    // vvm_output_XXXXXX.h5 should be able to see that these are elapsed seconds,
    // an exact step count, and metres, without consulting the source. Done
    // before the dimension scales are copied, so the x/y/z scales inherit the
    // coordinate units.
    const std::array<std::array<const char*, 3>, 6> labelled_datasets = {{
        {{"time",              "s",     "elapsed simulation time"}},
        {{"model_time_s",      "s",     "elapsed simulation time"}},
        {{"model_step",        "1",     "integration step count"}},
        {{"coordinates/x",     "meter", ""}},
        {{"coordinates/y",     "meter", ""}},
        {{"coordinates/z_mid", "meter", ""}},
    }};

    for (const auto& entry : labelled_datasets) {
        hid_t dataset = H5Dopen2(file, (std::string("/Step0/") + entry[0]).c_str(), H5P_DEFAULT);
        if (dataset < 0) continue;
        write_attribute(dataset, "units", entry[1]);
        write_attribute(dataset, "long_name", entry[2]);
        H5Dclose(dataset);
    }

    attach_hdf5_dimension_scales(file, fields_to_output_);

    H5Fclose(file);
}


void OutputManager::grads_ctl_file() {
    const auto axes = grads_horizontal_axes(grid_, state_, use_taiwanvvm_coordinates_, comm_);

    if (rank_ != 0) return;

    const auto z_mid_host = params_.z_mid.get_host_data();
    const int h = grid_.get_halo_cells();
    const int nz_phy = grid_.get_global_points_z();

    GradsCtl ctl;
    ctl.dset = "^" + filename_prefix_ + "_%tm6.h5";
    ctl.dtype = "hdf5_grid";
    ctl.templated = true;
    ctl.x = axes.first;
    ctl.y = axes.second;
    ctl.nx = grid_.get_global_points_x();
    ctl.ny = grid_.get_global_points_y();
    ctl.z_levels.reserve(nz_phy);
    for (int k = h; k < h + nz_phy; ++k) ctl.z_levels.push_back(z_mid_host(k));
    ctl.time_count = static_cast<std::size_t>(total_time_ / output_interval_s_ + 1);
    ctl.time_start = grads_start_time(grads_start_hour_);
    ctl.time_increment = grads_time_increment(output_interval_s_);
    ctl.variables = grads_variables("/Step0/", nz_phy);

    write_grads_ctl(output_dir_ + "/vvm.ctl", ctl);
}

std::vector<GradsVariable> OutputManager::grads_variables(
    const std::string& dataset_prefix,
    std::size_t levels) const {
    std::vector<GradsVariable> variables;
    std::unordered_set<std::string> taken;

    for (const auto& field_name : fields_to_output_) {
        auto it = state_.begin();
        while (it != state_.end() && it->first != field_name) ++it;
        if (it == state_.end()) continue;

        std::visit([&](const auto& field) {
            using T = std::decay_t<decltype(field)>;
            if constexpr (!std::is_same_v<T, std::monostate>) {
                GradsVariable variable;
                variable.dataset_name = dataset_prefix + field_name;
                variable.grads_name = unique_grads_variable_name(field_name, taken);
                variable.description = field_name;
                if constexpr (T::DimValue == 3 || T::DimValue == 4) {
                    variable.levels = levels;
                    variable.dimensions = "z,y,x";
                } else if constexpr (T::DimValue == 2) {
                    variable.levels = 0;
                    variable.dimensions = "y,x";
                } else if constexpr (T::DimValue == 1) {
                    variable.levels = levels;
                    variable.dimensions = "z";
                } else {
                    return;
                }
                variables.push_back(std::move(variable));
            }
        }, it->second);
    }

    return variables;
}

void OutputManager::write_static_topo_file() {
    if (rank_ == 0) std::cout << "Writing static topography file..." << std::endl;

    adios2::IO topo_io = adios_.DeclareIO("TOPO_IO");
    topo_io.SetEngine("HDF5");
    topo_io.SetParameter("IdleH5Writer", "true");
    topo_io.SetParameter("H5CollectiveMPIO", "no");

    std::string filename = output_dir_ + "/topo.h5";
    adios2::Engine topo_writer = topo_io.Open(filename, adios2::Mode::Write, comm_);

    const size_t gnx = grid_.get_global_points_x();
    const size_t gny = grid_.get_global_points_y();
    const size_t gnz = grid_.get_global_points_z();
    const size_t h = grid_.get_halo_cells();

    const size_t rank_lnx = grid_.get_local_physical_points_x();
    const size_t rank_lny = grid_.get_local_physical_points_y();
    const size_t rank_offset_x = grid_.get_local_physical_start_x();
    const size_t rank_offset_y = grid_.get_local_physical_start_y();

    auto var_topo = topo_io.DefineVariable<VVM::Real>("topo", {gny, gnx}, {rank_offset_y, rank_offset_x}, {rank_lny, rank_lnx});
    topo_io.DefineAttribute<std::string>("units", "meter", var_topo.Name());

    topo_writer.BeginStep();

    try {
        const auto& topo_field = state_.get_field<2>("topo");
        auto topo_data_view = topo_field.get_device_data();

        Kokkos::View<VVM::Real**, Kokkos::LayoutRight> topo_phys_subview("topo_phys_subview", rank_lny, rank_lnx);
        auto subview_from_full = Kokkos::subview(topo_data_view, 
                                                std::make_pair(h, h + rank_lny), 
                                                std::make_pair(h, h + rank_lnx));
        Kokkos::deep_copy(topo_phys_subview, subview_from_full);
        auto topo_phys_host = Kokkos::create_mirror_view(topo_phys_subview);
        Kokkos::deep_copy(topo_phys_host, topo_phys_subview);
        
        topo_writer.Put<VVM::Real>(var_topo, topo_phys_host.data());
    } 
    catch (const std::exception& e) {
        if (rank_ == 0) std::cerr << "Warning: Could not write 'topo': " << e.what() << std::endl;
    }

    topo_writer.EndStep();
    topo_writer.Close();
}

} // namespace IO
} // namespace VVM
