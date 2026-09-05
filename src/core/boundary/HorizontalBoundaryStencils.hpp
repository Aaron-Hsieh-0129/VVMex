#ifndef VVM_CORE_BOUNDARY_HORIZONTAL_BOUNDARY_STENCILS_HPP
#define VVM_CORE_BOUNDARY_HORIZONTAL_BOUNDARY_STENCILS_HPP

#include "core/Field.hpp"
#include "core/Grid.hpp"

namespace VVM {
namespace Core {
namespace Boundary {

class HorizontalBoundaryStencils {
public:
    explicit HorizontalBoundaryStencils(const Grid& grid);

    // Constantly extrapolate the nearest physical q2 row into every exterior
    // q2 halo row. This reproduces the MODE=2 normal-boundary stencil used by
    // CVVM's BOUND_NORMAL routine.
    //
    // This is a low-level stencil, not a complete free-slip boundary policy.
    void fill_constant_q2_halos(Field<2>& field) const;
    void fill_constant_q2_halos(Field<3>& field) const;

private:
    const Grid& grid_;
};

} // namespace Boundary
} // namespace Core
} // namespace VVM

#endif // VVM_CORE_BOUNDARY_HORIZONTAL_BOUNDARY_STENCILS_HPP
