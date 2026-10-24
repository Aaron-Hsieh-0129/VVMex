#ifndef VVM_READERS_READER_HPP
#define VVM_READERS_READER_HPP

#include "core/state/State.hpp"
#include "core/grid/Grid.hpp"
#include "utils/RestartTime.hpp"

namespace VVM {
namespace IO {

class Reader {
public:
    virtual ~Reader() = default;
    virtual void read_and_initialize(VVM::Core::State& state) = 0;

    bool
    restored_canonical_horizontal_state() const {
        return restored_canonical_horizontal_state_;
    }

    virtual VVM::Utils::RestartFileMetadata
    read_restart_metadata() {
        return VVM::Utils::RestartFileMetadata{};
    }

protected:
    bool restored_canonical_horizontal_state_ = false;
};

} // namespace IO
} // namespace VVM

#endif // VVM_READERS_READER_HPP
