#ifndef VVM_IO_BP5_RESTART_READER_HPP
#define VVM_IO_BP5_RESTART_READER_HPP

#include <cstddef>
#include <string>
#include <vector>

#include <adios2.h>
#include <mpi.h>

#include "core/Grid.hpp"
#include "core/HaloExchanger.hpp"
#include "core/Parameters.hpp"
#include "core/State.hpp"
#include "io/Reader.hpp"
#include "utils/ConfigurationManager.hpp"

namespace VVM::IO::BP5 {

class Bp5RestartReader : public VVM::IO::Reader {
public:
    Bp5RestartReader(const std::string& dataset_path,
                     const Core::Grid& grid,
                     const Core::Parameters& params,
                     const Utils::ConfigurationManager& config,
                     Core::HaloExchanger& halo_exchanger);

    void read_and_initialize(Core::State& state) override;

    VVM::Utils::RestartFileMetadata read_restart_metadata() override;

private:
    class OpenDataset;

    std::size_t resolve_step(std::size_t available_steps) const;

    template <std::size_t Dim>
    void read_field(OpenDataset& dataset,
                    const std::string& var_name,
                    Core::Field<Dim>& field) const;

    template <typename Stored, std::size_t Dim>
    void read_typed_field(OpenDataset& dataset,
                          const std::string& var_name,
                          Core::Field<Dim>& field) const;

    std::string dataset_path_;
    const Core::Grid& grid_;
    const Core::Parameters& params_;
    const Utils::ConfigurationManager& config_;
    Core::HaloExchanger& halo_exchanger_;
    MPI_Comm comm_;
    int rank_ = 0;
};

} // namespace VVM::IO::BP5

#endif
