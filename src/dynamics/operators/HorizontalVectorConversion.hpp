#ifndef VVM_DYNAMICS_OPERATORS_HORIZONTAL_VECTOR_CONVERSION_HPP
#define VVM_DYNAMICS_OPERATORS_HORIZONTAL_VECTOR_CONVERSION_HPP

#include "core/vvm_types.hpp"

namespace VVM {
namespace Dynamics {
namespace Operators {

// Stateless horizontal vector-component conversion helpers.
//
// These helpers intentionally do not own/capture geometry views. The caller
// supplies the local metric factor at the required staggered location.
//
// This is important for CUDA graph compatibility: keeping geometry views in a
// large conversion object can make Kokkos lambdas large enough to use the
// constant-memory kernel-launch path, which is not stream-capture safe.
//
// For the currently supported orthogonal horizontal coordinates:
//
//     physical = h_i * contravariant
//     contravariant = physical / h_i
//
// and
//
//     covariant = h_i * physical
//     physical = covariant / h_i
//
// The caller provides either h_i or 1 / h_i explicitly so that the existing
// floating-point operation form can be preserved.
struct HorizontalVectorConversion {
    KOKKOS_INLINE_FUNCTION
    static VVM::Real
    physical_to_contravariant(const VVM::Real physical,
        const VVM::Real inverse_scale_factor) noexcept {

        return physical * inverse_scale_factor;
    }

    KOKKOS_INLINE_FUNCTION
    static VVM::Real
    contravariant_to_physical(const VVM::Real contravariant,
        const VVM::Real scale_factor) noexcept {

        return contravariant * scale_factor;
    }

    KOKKOS_INLINE_FUNCTION
    static VVM::Real
    physical_to_covariant(const VVM::Real physical, const VVM::Real scale_factor) noexcept {

        // Keep scale-factor-first multiplication because existing RLL
        // circulation code evaluates h_i * physical.
        return scale_factor * physical;
    }

    KOKKOS_INLINE_FUNCTION
    static VVM::Real
    covariant_to_physical(const VVM::Real covariant,
        const VVM::Real inverse_scale_factor) noexcept {

        return covariant * inverse_scale_factor;
    }
};

} // namespace Operators
} // namespace Dynamics
} // namespace VVM

#endif // VVM_DYNAMICS_OPERATORS_HORIZONTAL_VECTOR_CONVERSION_HPP
