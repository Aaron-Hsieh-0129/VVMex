#include "Bp5CollectiveValidation.hpp"

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <vector>

namespace VVM::IO::BP5 {

void require_collective_match(
    const std::string& local_value,
    MPI_Comm comm,
    const std::string& description) {
    int rank = 0;
    MPI_Comm_rank(comm, &rank);
    if (local_value.size() > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
        throw std::length_error("BP5 " + description + " is too large to validate.");
    }
    int root_size = rank == 0 ? static_cast<int>(local_value.size()) : 0;
    MPI_Bcast(&root_size, 1, MPI_INT, 0, comm);
    std::vector<char> root_value(static_cast<std::size_t>(root_size));
    if (rank == 0) {
        std::copy(local_value.begin(), local_value.end(), root_value.begin());
    }
    if (root_size > 0) MPI_Bcast(root_value.data(), root_size, MPI_CHAR, 0, comm);
    const bool local_matches =
        local_value.size() == root_value.size() &&
        std::equal(local_value.begin(), local_value.end(), root_value.begin());
    int local_mismatch = local_matches ? 0 : 1;
    int mismatch_count = 0;
    MPI_Allreduce(&local_mismatch, &mismatch_count, 1, MPI_INT, MPI_SUM, comm);
    if (mismatch_count != 0) {
        throw std::runtime_error(
            "BP5 " + description + " differs across " +
            std::to_string(mismatch_count) + " non-root MPI rank(s).");
    }
}

} // namespace VVM::IO::BP5
