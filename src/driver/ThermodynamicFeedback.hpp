#pragma once

#include "core/Field.hpp"
#include <vector>

namespace VVM::Core { class State; }
namespace VVM::Driver {

// Applied at the transport/physics split. The receiver is visible only here;
// the independent producer never receives a reference to it.
class ReceiverTransport {
public:
    virtual ~ReceiverTransport() = default;
    virtual void apply(Core::State&, VVM::Real dt, VVM::Real strength) const = 0;
};

// Borrowed, read-only forcing. The producer owns the fields; no receiver state
// or model reference crosses this interface.
struct ThermodynamicFeedback {
    const Core::Field<3>* theta = nullptr;
    const Core::Field<3>* qv = nullptr;
    VVM::Real strength = VVM::real(1.);
    bool limit_vapor = false;
    const ReceiverTransport* transport = nullptr;
    struct DynamicTendency {
        std::string variable;
        const Core::Field<3>* field;
    };
    std::vector<DynamicTendency> dynamics;
};

} // namespace VVM::Driver
