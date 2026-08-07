#ifndef VVM_IO_HISTORY_HISTORY_WRITER_HPP
#define VVM_IO_HISTORY_HISTORY_WRITER_HPP

#include <cstddef>

#include "core/vvm_types.hpp"

namespace VVM::IO {

class HistoryWriter {
public:
    virtual ~HistoryWriter() = default;

    virtual void write(std::size_t step, VVM::Real time) = 0;
    virtual void close() = 0;
};

} // namespace VVM::IO

#endif
