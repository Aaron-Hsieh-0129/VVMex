#ifndef VVM_IO_OutputManager_HPP
#define VVM_IO_OutputManager_HPP

#include <string>
#include <vector>
#include <map>
#include <utility>
#include <adios2.h>
#include <Kokkos_Core.hpp>
#include <adios2/cxx/KokkosView.h>

#include "core/Grid.hpp"
#include "core/State.hpp"
#include "core/Parameters.hpp"
#include "core/vvm_types.hpp"
#include "io/OutputPrecision.hpp"
#include "io/history/GradsCtl.hpp"
#include "utils/ConfigurationManager.hpp"
#include "utils/Timer.hpp"
#include "utils/TimingManager.hpp"

namespace VVM {
namespace IO {

class OutputManager {
public:
    OutputManager(const Utils::ConfigurationManager& config, const VVM::Core::Grid& grid, const VVM::Core::Parameters& params, VVM::Core::State& state, MPI_Comm comm);
    ~OutputManager();

    OutputManager(const OutputManager&) = delete;
    OutputManager& operator=(const OutputManager&) = delete;
    OutputManager(OutputManager&&) = delete;
    OutputManager& operator=(OutputManager&&) = delete;

    void write(size_t step, VVM::Real time);
    void write_static_data();
    void write_static_topo_file();

private:
    const VVM::Core::Grid& grid_;
    const VVM::Core::Parameters& params_;
    VVM::Core::State& state_;

    std::string output_dir_;
    std::string filename_prefix_;
    std::vector<std::string> fields_to_output_;
    bool use_taiwanvvm_coordinates_ = false;
    int grads_start_hour_ = 0;

    std::vector<std::pair<std::string, VVM::Core::FieldMetadata>> output_field_metadata_;

    VVM::Real output_interval_s_;
    VVM::Real total_time_;
    std::string engine_type_;

    int rank_;
    int mpi_size_;
    MPI_Comm comm_;
    adios2::ADIOS adios_;
    adios2::IO io_;
    adios2::Engine writer_;

    // Field data carries the configured output precision, which need not be
    // VVM::Real; clocks and coordinates deliberately stay VVM::Real. Exactly
    // one of these two maps is populated, decided once by converts_precision_.
    OutputElementType element_type_ = OutputElementType::Float64;
    bool converts_precision_ = false;
    std::map<std::string, adios2::Variable<VVM::Real>> field_variables_;
    std::map<std::string, adios2::Variable<ConvertedReal>> converted_variables_;
    std::map<std::string, adios2::Dims> field_counts_;
    std::map<std::string, std::vector<ConvertedReal>> converted_buffers_;

    size_t output_x_start_, output_y_start_, output_z_start_;
    size_t output_x_end_, output_y_end_, output_z_end_;
    // size_t output_x_stride_, output_y_stride_, output_z_stride_;

    bool variables_defined_ = false;
    adios2::Variable<VVM::Real> var_time_;

    void define_variables();
    void define_field_variable(
        const std::string& field_name,
        const adios2::Dims& shape,
        const adios2::Dims& start,
        const adios2::Dims& count);
    void put_field(const std::string& field_name, const VVM::Real* data, size_t elements);
    void define_adios_field_metadata(
        const std::string& field_name,
        const VVM::Core::FieldMetadata& metadata);
    void attach_hdf5_field_metadata(const std::string& filename);

    void grads_ctl_file();
    std::vector<GradsVariable> grads_variables(
        const std::string& dataset_prefix,
        std::size_t levels) const;

    std::string format_to_six_digits(int number);


    std::map<std::string, Kokkos::View<VVM::Real*, Kokkos::HostSpace>> host_buffers_1d_;
    std::map<std::string, Kokkos::View<VVM::Real**, Kokkos::LayoutRight, Kokkos::HostSpace>> host_buffers_2d_;
    std::map<std::string, Kokkos::View<VVM::Real***, Kokkos::LayoutRight, Kokkos::HostSpace>> host_buffers_3d_;
    std::map<std::string, Kokkos::View<VVM::Real****, Kokkos::LayoutRight, Kokkos::HostSpace>> host_buffers_4d_;

    std::map<std::string, Kokkos::View<VVM::Real**, Kokkos::LayoutRight>> dev_buffers_2d_;
    std::map<std::string, Kokkos::View<VVM::Real***, Kokkos::LayoutRight>> dev_buffers_3d_;
    std::map<std::string, Kokkos::View<VVM::Real****, Kokkos::LayoutRight>> dev_buffers_4d_;
};

} // namespace IO
} // namespace VVM

#endif // VVM_CORE_OutputManager_HPP
