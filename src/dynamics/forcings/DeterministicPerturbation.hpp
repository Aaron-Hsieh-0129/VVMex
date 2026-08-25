#pragma once

#include "core/vvm_types.hpp"

#include <cstdint>

namespace VVM {
namespace Dynamics {
namespace RandomForcingDetail {

KOKKOS_INLINE_FUNCTION constexpr std::uint64_t mix64(std::uint64_t value) {
    value += UINT64_C(0x9e3779b97f4a7c15);
    value = (value ^ (value >> 30)) * UINT64_C(0xbf58476d1ce4e5b9);
    value = (value ^ (value >> 27)) * UINT64_C(0x94d049bb133111eb);
    return value ^ (value >> 31);
}

KOKKOS_INLINE_FUNCTION constexpr std::uint64_t random_bits(
    int seed, std::uint64_t step, std::uint64_t global_k,
    std::uint64_t global_j, std::uint64_t global_i) {
    std::uint64_t key = mix64(static_cast<std::uint32_t>(seed));
    key = mix64(key ^ step);
    key = mix64(key ^ global_k);
    key = mix64(key ^ global_j);
    return mix64(key ^ global_i);
}

KOKKOS_INLINE_FUNCTION VVM::Real signed_unit_random(
    int seed, std::uint64_t step, std::uint64_t global_k,
    std::uint64_t global_j, std::uint64_t global_i) {
    const std::uint64_t bits = random_bits(seed, step, global_k, global_j, global_i);
#ifdef VVM_USE_DOUBLE_PRECISION
    const VVM::Real unit = static_cast<VVM::Real>(bits >> 11) * VVM::real(0x1.0p-53);
#else
    const VVM::Real unit = static_cast<VVM::Real>(bits >> 40) * VVM::real(0x1.0p-24);
#endif
    return VVM::real(2.0) * unit - VVM::real(1.0);
}

} // namespace RandomForcingDetail
} // namespace Dynamics
} // namespace VVM
