#ifndef VVM_IO_BP5_COLLECTIVE_VALIDATION_HPP
#define VVM_IO_BP5_COLLECTIVE_VALIDATION_HPP

#include <string>

#include <mpi.h>

namespace VVM::IO::BP5 {

void require_collective_match(
    const std::string& local_value,
    MPI_Comm comm,
    const std::string& description);

} // namespace VVM::IO::BP5

#endif
