#ifndef VVM_READERS_READER_HPP
#define VVM_READERS_READER_HPP

#include "core/State.hpp"
#include "core/Grid.hpp"
#include "utils/RestartTime.hpp"

namespace VVM {
namespace IO {

class Reader {
public:
    virtual ~Reader() = default;
    virtual void read_and_initialize(VVM::Core::State& state) = 0;

    virtual VVM::Utils::RestartFileMetadata read_restart_metadata() {
        return VVM::Utils::RestartFileMetadata{};
    }
};

} // namespace IO
} // namespace VVM

#endif // VVM_READERS_READER_HPP
