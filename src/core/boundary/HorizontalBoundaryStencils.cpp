#include <stdexcept>

#include "core/boundary/HorizontalBoundaryStencils.hpp"

namespace VVM {
namespace Core {
namespace Boundary {

HorizontalBoundaryStencils::HorizontalBoundaryStencils(const Grid& grid)
    : grid_(grid) {

    const auto& horizontal = grid_.horizontal_specification();

    if (horizontal.ny > 1 && horizontal.topology.q2 != HorizontalEdgeTopology::Bounded) {
        throw std::invalid_argument("HorizontalBoundaryStencils requires bounded q2 topology.");
    }
}

} // namespace Boundary
} // namespace Core
} // namespace VVM
