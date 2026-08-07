#ifndef VVM_IO_BP5_COLLECTIVE_VALIDATION_HPP
#define VVM_IO_BP5_COLLECTIVE_VALIDATION_HPP

#include <string>

#include <mpi.h>

namespace VVM::IO::BP5 {

// Throw on every rank if any rank resolved a different configuration string.
// This prevents one rank entering an ADIOS collective with a different schema.
void require_collective_match(
    const std::string& local_value,
    MPI_Comm comm,
    const std::string& description);

} // namespace VVM::IO::BP5

#endif
