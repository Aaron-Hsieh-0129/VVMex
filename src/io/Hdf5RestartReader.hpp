#ifndef VVM_IO_HDF5RESTARTREADER_HPP
#define VVM_IO_HDF5RESTARTREADER_HPP

#include "io/Reader.hpp"
#include "core/Grid.hpp"
#include "core/haloexchange/HaloExchanger.hpp"
#include "core/Parameters.hpp"
#include "core/State.hpp"
#include "utils/ConfigurationManager.hpp"

#include <hdf5.h>
#include <mpi.h>
#include <string>
#include <vector>

namespace VVM {
namespace IO {

class Hdf5RestartReader : public Reader {
public:
    Hdf5RestartReader(const std::string& filepath,
                      const Core::Grid& grid,
                      const Core::Parameters& params,
                      const Utils::ConfigurationManager& config,
                      Core::HaloExchanger& halo_exchanger);

    void read_and_initialize(Core::State& state) override;

    // Restart clock stored in the file: /Step0/model_time_s + /Step0/model_step,
    // falling back to the long-standing elapsed-seconds /Step0/time.
    VVM::Utils::RestartFileMetadata read_restart_metadata() override;

private:
    template<size_t Dim>
    void read_field(hid_t file_id,
                    const std::string& var_name,
                    Core::Field<Dim>& field) const;

    std::string source_file_;
    const Core::Grid& grid_;
    const Core::Parameters& params_;
    const Utils::ConfigurationManager& config_;
    Core::HaloExchanger& halo_exchanger_;
    MPI_Comm comm_;
    int rank_ = 0;
};

} // namespace IO
} // namespace VVM

#endif // VVM_IO_HDF5RESTARTREADER_HPP
