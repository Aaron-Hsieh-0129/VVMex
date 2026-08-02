#include "IOServer.hpp"
#include "Hdf5DimensionScales.hpp"
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <sstream>
#include <iomanip>
#include <map>
#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <adios2.h>
#include <hdf5.h>
#include <sys/stat.h>

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

using FieldMetadataCache =
    std::map<std::string, std::map<std::string, std::string>>;

void cache_adios_field_metadata(
    adios2::IO& input_io,
    const std::string& field_name,
    FieldMetadataCache& metadata_cache)
{
    auto& metadata = metadata_cache[field_name];
    const auto copy_attribute = [&](const std::string& attribute_name)
    {
        if (metadata.count(attribute_name)) {
            return;
        }

        const auto input_attribute =
            input_io.InquireAttribute<std::string>(attribute_name, field_name);
        if (!input_attribute) {
            return;
        }

        const auto values = input_attribute.Data();
        if (values.empty() || values.front().empty()) {
            return;
        }

        metadata[attribute_name] = values.front();
    };

    copy_attribute("units");
    copy_attribute("long_name");
    copy_attribute("standard_name");
    copy_attribute("comment");
    copy_attribute("grid_staggering");
}

void write_hdf5_string_attribute(
    hid_t dataset,
    const std::string& name,
    const std::string& value)
{
    if (value.empty()) return;

    hid_t space = H5Screate(H5S_SCALAR);
    hid_t type = H5Tcopy(H5T_C_S1);

    H5Tset_size(type, value.size() + 1);
    H5Tset_strpad(type, H5T_STR_NULLTERM);

    hid_t attribute =
        H5Acreate2(dataset, name.c_str(), type, space, H5P_DEFAULT, H5P_DEFAULT);

    if (attribute >= 0) {
        H5Awrite(attribute, type, value.c_str());
        H5Aclose(attribute);
    }

    H5Tclose(type);
    H5Sclose(space);
}

void attach_sst_hdf5_field_metadata(
    const std::string& filename,
    const FieldMetadataCache& metadata_cache)
{
    hid_t file = H5Fopen(filename.c_str(), H5F_ACC_RDWR, H5P_DEFAULT);
    if (file < 0) {
        std::cerr << "[IO-Server] Failed to reopen HDF5 file '"
                  << filename << "' for metadata output.\n";
        return;
    }

    for (const auto& field_metadata : metadata_cache) {
        const std::string dataset_path = "/Step0/" + field_metadata.first;
        hid_t dataset = H5Dopen2(file, dataset_path.c_str(), H5P_DEFAULT);
        if (dataset < 0) continue;

        for (const auto& attribute : field_metadata.second) {
            write_hdf5_string_attribute(
                dataset,
                attribute.first,
                attribute.second);
        }

        H5Dclose(dataset);
    }

    std::vector<std::string> field_names;
    field_names.reserve(metadata_cache.size());
    for (const auto& field_metadata : metadata_cache) {
        field_names.push_back(field_metadata.first);
    }
    attach_hdf5_dimension_scales(file, field_names);

    H5Fclose(file);
}
} // namespace

std::string format_six_digits(int number) {
    std::stringstream ss;
    ss << std::setfill('0') << std::setw(6) << number;
    return ss.str();
}

void get_local_range(size_t global_size, int rank, int size, size_t &start, size_t &count) {
    size_t base = global_size / size;
    size_t rem = global_size % size;
    if (rank < rem) { start = rank * (base + 1); count = base + 1; } 
    else { start = rank * base + rem; count = base; }
}

void run_io_server(MPI_Comm io_comm, const VVM::Utils::ConfigurationManager& config) {
    int rank, size;
    MPI_Comm_rank(io_comm, &rank);
    MPI_Comm_size(io_comm, &size);

    adios2::ADIOS adios(io_comm);

    const std::string output_dir =
        config.get_value<std::string>("output.output_dir");
    const std::string filename_prefix =
        config.get_value<std::string>("output.output_filename_prefix");
    const std::vector<std::string> fields_to_output =
        config.get_value<std::vector<std::string>>("output.fields_to_output");
    const VVM::Real output_interval_s =
        config.get_value<VVM::Real>("simulation.output_interval_s");

    const std::string input_stream_name = output_dir + "/" + filename_prefix;

    if (rank == 0) {
        std::cout << "  [IO-Server] Listening on stream: "
                  << input_stream_name << std::endl;
        mkdir(output_dir.c_str(), 0777);
    }

    MPI_Barrier(io_comm);

    // -------------------------
    // SST reader: safe settings
    // -------------------------
    adios2::IO inIO = adios.DeclareIO("InputSST");
    inIO.SetEngine("SST");
    const std::string data_transport =
        uppercase_transport_name(config.get_value<std::string>("output.data_transport", "WAN"));
    const std::string control_transport =
        config.get_value<std::string>("output.control_transport", "sockets");

    if (rank == 0) {
        std::cout << "  [IO-Server] SST DataTransport: "
                  << ((data_transport.empty() || data_transport == "AUTO")
                          ? "AUTO"
                          : data_transport)
                  << std::endl;
        std::cout << "  [IO-Server] SST ControlTransport: "
                  << control_transport << std::endl;
    }

    if (!data_transport.empty() && data_transport != "AUTO") {
        inIO.SetParameter("DataTransport", data_transport);
    }
    if (data_transport == "WAN") {
        inIO.SetParameter("WANDataTransport", "sockets");
    }
    if (!control_transport.empty()) {
        inIO.SetParameter("ControlTransport", control_transport);
    }

    // Restart startup can be long before the writer opens.
    inIO.SetParameter("OpenTimeoutSecs", "14400");
    inIO.SetParameter("SpeculativePreloadMode", "OFF");
    inIO.SetParameter("AlwaysProvideLatestTimestep", "false");
    
    // -------------------------
    // HDF5 writer: safe settings
    // -------------------------
    adios2::IO outIO = adios.DeclareIO("OutputHDF5");
    outIO.SetEngine("HDF5");
    outIO.SetParameter("IdleH5Writer", "true");

    // For the safe SST run, use 1 IO rank first.
    // Serial HDF5 is safer than collective HDF5 while debugging SST stability.
    outIO.SetParameter("H5CollectiveMPIO", "false");

    adios2::Engine reader;
    try {
        reader = inIO.Open(input_stream_name, adios2::Mode::Read);
    } catch (const std::exception& e) {
        if (rank == 0) {
            std::cerr << "[IO-Server] FATAL: failed to open SST stream: "
                      << e.what() << std::endl;
        }
        MPI_Abort(MPI_COMM_WORLD, 10);
    }

    FieldMetadataCache field_metadata_cache;
    while (true) {
        std::map<std::string, std::vector<VVM::Real>> data_buffers;
        // Integer scalars travel separately from the float relay below: the
        // restart step count must reach the HDF5 file as an exact integer, and
        // rounding it through VVM::Real is exactly what the restart clock is
        // not allowed to depend on.
        std::map<std::string, int64_t> int_scalars;
        std::vector<std::string> current_step_vars;

        adios2::StepStatus status;
        try {
            status = reader.BeginStep();
        } catch (const std::exception& e) {
            if (rank == 0) {
                std::cerr << "[IO-Server] FATAL: BeginStep failed: "
                          << e.what() << std::endl;
            }
            MPI_Abort(MPI_COMM_WORLD, 11);
        }

        if (status != adios2::StepStatus::OK) {
            if (rank == 0) {
                std::cout << "[IO-Server] SST stream ended." << std::endl;
            }
            break;
        }

        const int step = reader.CurrentStep();
        VVM::Real step_time = static_cast<VVM::Real>(step) * output_interval_s;
        bool has_time_variable = false;

        const auto& varTypeMap = inIO.AvailableVariables();

        try {
            for (const auto& varPair : varTypeMap) {
                const std::string& name = varPair.first;

                auto typeIt = varPair.second.find("Type");
                if (typeIt == varPair.second.end()) continue;

                const std::string& type = typeIt->second;

                if (type == "int64_t") {
                    auto intVarIn = inIO.InquireVariable<int64_t>(name);
                    if (!intVarIn || !intVarIn.Shape().empty()) continue;

                    if (!outIO.InquireVariable<int64_t>(name)) {
                        outIO.DefineVariable<int64_t>(name);
                    }
                    current_step_vars.push_back(name);
                    int_scalars[name] = 0;
                    reader.Get(intVarIn, &int_scalars[name], adios2::Mode::Deferred);
                    continue;
                }

                if (type != "double" && type != "float") continue;

                auto varIn = inIO.InquireVariable<VVM::Real>(name);
                if (!varIn) continue;

                const adios2::Dims shape = varIn.Shape();
                current_step_vars.push_back(name);

                if (!outIO.InquireVariable<VVM::Real>(name)) {
                    if (shape.empty()) {
                        outIO.DefineVariable<VVM::Real>(name);
                    } else {
                        adios2::Dims start(shape.size(), 0);
                        adios2::Dims count = shape;
                        outIO.DefineVariable<VVM::Real>(name, shape, start, count);
                    }
                }

                if (std::find(fields_to_output.begin(), fields_to_output.end(), name) !=
                    fields_to_output.end()) {
                    cache_adios_field_metadata(inIO, name, field_metadata_cache);
                }

                // Scalar
                if (shape.empty()) {
                    data_buffers[name].resize(1);

                    reader.Get(varIn, data_buffers[name].data(), adios2::Mode::Deferred);

                    if (name == "time") {
                        has_time_variable = true;
                    }
                    continue;
                }

                // Array: split first dimension across IO ranks.
                size_t my_start = 0;
                size_t my_count = 0;
                get_local_range(shape[0], rank, size, my_start, my_count);

                if (my_count == 0) {
                    continue;
                }

                adios2::Dims start(shape.size(), 0);
                adios2::Dims count = shape;
                start[0] = my_start;
                count[0] = my_count;

                varIn.SetSelection({start, count});

                size_t elements = 1;
                for (const auto c : count) {
                    elements *= c;
                }

                data_buffers[name].resize(elements);

                reader.Get(varIn, data_buffers[name].data(), adios2::Mode::Deferred);
            }

            reader.PerformGets();

            if (has_time_variable) {
                auto timeIt = data_buffers.find("time");
                if (timeIt != data_buffers.end() && !timeIt->second.empty()) {
                    step_time = timeIt->second[0];
                }
            }

            reader.EndStep();

        } catch (const std::exception& e) {
            if (rank == 0) {
                std::cerr << "[IO-Server] FATAL: SST read/Get/EndStep failed: "
                          << e.what() << std::endl;
            }
            MPI_Abort(MPI_COMM_WORLD, 12);
        }

        const int output_index =
            static_cast<int>(std::llround(step_time / output_interval_s));

        if (rank == 0) {
            std::cout << "  [IO-Server] Writing Step "
                      << output_index << "..." << std::endl;
        }

        const std::string h5_name =
            output_dir + "/" + filename_prefix + "_" +
            format_six_digits(output_index) + ".h5";

        std::sort(current_step_vars.begin(), current_step_vars.end());

        try {
            adios2::Engine writer = outIO.Open(h5_name, adios2::Mode::Write, io_comm);
            writer.BeginStep();

            for (const auto& name : current_step_vars) {
                const auto intIt = int_scalars.find(name);
                if (intIt != int_scalars.end()) {
                    auto intVarOut = outIO.InquireVariable<int64_t>(name);
                    if (intVarOut) {
                        writer.Put(intVarOut, &intIt->second, adios2::Mode::Deferred);
                    }
                    continue;
                }

                auto varOut = outIO.InquireVariable<VVM::Real>(name);
                if (!varOut) continue;

                auto bufIt = data_buffers.find(name);
                if (bufIt == data_buffers.end()) continue;

                auto& buffer = bufIt->second;

                // Scalar
                if (varOut.Shape().empty()) {
                    writer.Put(varOut, buffer.data(), adios2::Mode::Deferred);
                    continue;
                }

                const adios2::Dims shape = varOut.Shape();
                if (shape.empty()) continue;

                size_t s_start = 0;
                size_t s_count = 0;
                get_local_range(shape[0], rank, size, s_start, s_count);

                if (s_count == 0) {
                    continue;
                }

                adios2::Dims start(shape.size(), 0);
                adios2::Dims count = shape;
                start[0] = s_start;
                count[0] = s_count;

                varOut.SetSelection({start, count});

                writer.Put(varOut, buffer.data(), adios2::Mode::Deferred);
            }

            writer.PerformPuts();
            writer.EndStep();
            writer.Close();

        } catch (const std::exception& e) {
            if (rank == 0) {
                std::cerr << "[IO-Server] FATAL: HDF5 write failed for "
                          << h5_name << ": " << e.what() << std::endl;
            }
            MPI_Abort(MPI_COMM_WORLD, 13);
        }

        MPI_Barrier(io_comm);
        if (rank == 0) {
            attach_sst_hdf5_field_metadata(h5_name, field_metadata_cache);
        }
        MPI_Barrier(io_comm);
    }

    try {
        reader.Close();
    } catch (const std::exception& e) {
        if (rank == 0) {
            std::cerr << "[IO-Server] Warning: reader.Close() failed: "
                      << e.what() << std::endl;
        }
    }
}

} // namespace IO
} // namespace VVM
