#ifndef VVM_IO_BP5_FIELD_INPUT_HPP
#define VVM_IO_BP5_FIELD_INPUT_HPP

#include <cstddef>

#include "core/vvm_types.hpp"

namespace VVM::IO::BP5 {

enum class FieldMemoryLocation { Host, Device };

// The pointer is untyped because the on-disk element type is a configuration
// choice, not VVM::Real. Precision is fixed for the whole dataset, so the
// writer knows which adios2::Variable<T> this belongs to and casts back once.
struct FieldInput {
    const void* data = nullptr;
    std::size_t elements = 0;
    FieldMemoryLocation location = FieldMemoryLocation::Host;
    bool packed = false;
};

} // namespace VVM::IO::BP5

#endif
