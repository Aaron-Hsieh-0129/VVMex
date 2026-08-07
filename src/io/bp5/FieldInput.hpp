#ifndef VVM_IO_BP5_FIELD_INPUT_HPP
#define VVM_IO_BP5_FIELD_INPUT_HPP

#include <cstddef>

#include "core/vvm_types.hpp"

namespace VVM::IO::BP5 {

enum class FieldMemoryLocation { Host, Device };

struct FieldInput {
    const VVM::Real* data = nullptr;
    std::size_t elements = 0;
    FieldMemoryLocation location = FieldMemoryLocation::Host;
    bool packed = false;
};

} // namespace VVM::IO::BP5

#endif
