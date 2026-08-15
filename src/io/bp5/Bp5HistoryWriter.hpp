#ifndef VVM_IO_BP5_HISTORY_WRITER_HPP
#define VVM_IO_BP5_HISTORY_WRITER_HPP

#include <cstddef>
#include <filesystem>
#include <map>
#include <string>
#include <variant>
#include <vector>

#include <adios2.h>
#include <mpi.h>

#include "Bp5BufferSet.hpp"
#include "Bp5FieldSchema.hpp"
#include "Bp5OutputConfig.hpp"
#include "CpuFieldSource.hpp"
#include "core/Grid.hpp"
#include "core/Parameters.hpp"
#include "core/State.hpp"
#include "io/history/GradsCtl.hpp"
#include "io/history/HistoryWriter.hpp"
#include "utils/ConfigurationManager.hpp"

namespace VVM::IO::BP5 {

class Bp5HistoryWriter final : public HistoryWriter {
public:
    Bp5HistoryWriter(
        const Utils::ConfigurationManager& config,
        const Core::Grid& grid,
        const Core::Parameters& parameters,
        Core::State& state,
        MPI_Comm comm);
    ~Bp5HistoryWriter() override;

    Bp5HistoryWriter(const Bp5HistoryWriter&) = delete;
    Bp5HistoryWriter& operator=(const Bp5HistoryWriter&) = delete;

    void write(std::size_t step, VVM::Real time) override;
    void close() override;

    const std::filesystem::path& dataset_path() const noexcept {
        return dataset_path_;
    }

private:
    // Field data carries the configured output precision, which need not be
    // VVM::Real. Coordinates and clocks deliberately do not: they are a few
    // kilobytes per step against tens of gigabytes of field data, and
    // narrowing a coordinate or a timestamp is an analysis hazard for no gain.
    using FieldVariable =
        std::variant<adios2::Variable<float>, adios2::Variable<double>>;

    struct FieldRecord {
        std::string name;
        std::string description;  // GrADS descriptor text
        FieldSelection selection;
        FieldVariable variable;
    };

    const Core::Grid& grid_;
    const Core::Parameters& parameters_;
    Core::State& state_;
    MPI_Comm comm_;
    int rank_ = 0;
    int size_ = 1;

    Bp5OutputConfig config_;
    // Resolved once in the constructor: element_type_ turns a 'native' request
    // into a concrete on-disk type, and effective_buffer_mode_ downgrades a
    // 'direct' request to packing when the precision needs converting.
    OutputElementType element_type_ = OutputElementType::Float64;
    CpuBufferMode effective_buffer_mode_ = CpuBufferMode::Direct;
    Bp5FieldSchema schema_;
    std::filesystem::path dataset_path_;
    std::vector<std::string> field_names_;

    adios2::ADIOS adios_;
    adios2::IO io_;
    adios2::Engine writer_;
    adios2::Variable<VVM::Real> time_variable_;
    adios2::Variable<double> model_time_variable_;
    adios2::Variable<std::int64_t> model_step_variable_;
    adios2::Variable<VVM::Real> x_variable_;
    adios2::Variable<VVM::Real> y_variable_;
    adios2::Variable<VVM::Real> z_variable_;
    std::vector<FieldRecord> fields_;

    std::vector<VVM::Real> x_coordinates_;
    std::vector<VVM::Real> y_coordinates_;
    std::vector<VVM::Real> z_coordinates_;
    Bp5BufferSet buffers_;
    CpuFieldSource field_source_;

    bool closed_ = false;
    std::size_t steps_written_ = 0;
    std::size_t global_bytes_per_step_ = 0;

    void prepare_dataset_path(const Utils::ConfigurationManager& config);
    void validate_collective_configuration(
        const Utils::ConfigurationManager& config) const;
    void define_schema();
    void define_field(const std::string& field_name);
    void define_metadata(const std::string& field_name,
                         const Core::FieldMetadata& metadata);
    void skip_field(const std::string& field_name, const char* reason) const;
    static std::string describe_field(const std::string& field_name,
                                      const Core::FieldMetadata& metadata);
    void write_grads_ctl_file(const Utils::ConfigurationManager& config);
    void prepare_coordinates();
    void validate_coverage();
    void print_configuration() const;
    [[noreturn]] void throw_operation(const char* operation,
                                      const std::exception& error) const;
};

} // namespace VVM::IO::BP5

#endif
