#ifndef VVM_CLEO_PROCESS_INTERFACE_HPP
#define VVM_CLEO_PROCESS_INTERFACE_HPP

#include "core/grid/Grid.hpp"
#include "core/state/State.hpp"
#include "core/Parameters.hpp"
#include "core/haloexchange/HaloExchanger.hpp"

namespace VVM {
namespace Physics {

class VVM_CLEO_Interface {
public:
    // setup required state field
    VVM_CLEO_Interface(const VVM::Utils::ConfigurationManager& config,
        const VVM::Core::Grid& grid,
        const VVM::Core::Parameters& params,
        Core::HaloExchanger& halo_exchanger,
        Core::State& state);
    ~VVM_CLEO_Interface();

    void initialize(VVM::Core::State& state);
    void finalize();

    void run(VVM::Core::State& state, const VVM::Real dt);

private:
    // CLEO headers require C++20 while the rest of VVM builds as C++17;
    // Pimpl confines all CLEO types and includes to cleo_interface.cpp.
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}
}

#endif
