#ifndef VVM_IO_OUTPUT_PRECISION_HPP
#define VVM_IO_OUTPUT_PRECISION_HPP

#include <cstddef>
#include <string>
#include <type_traits>

#include "core/vvm_types.hpp"
#include "utils/ConfigurationManager.hpp"

namespace VVM::IO {

// What the configuration asked for. Native follows VVM::Real, so a
// configuration written before this option existed keeps producing exactly the
// bytes it produced before.
enum class OutputPrecision { Native, Float32, Float64 };

// What actually lands on disk, once Native has been resolved against this
// build. History output is deliberately independent of VVM::Real: a
// double-precision model may write float32 history without changing how it
// computes.
enum class OutputElementType { Float32, Float64 };

// The float width VVM::Real is not, which is what a converting run writes.
using ConvertedReal =
    std::conditional_t<sizeof(VVM::Real) == sizeof(float), double, float>;

const char* output_precision_name(OutputPrecision precision) noexcept;
const char* output_element_type_name(OutputElementType type) noexcept;
std::size_t output_element_size(OutputElementType type) noexcept;

// True when the on-disk type is VVM::Real, so writing a field costs no
// conversion and the model's own memory can be handed straight to ADIOS2.
bool output_element_matches_real(OutputElementType type) noexcept;

OutputElementType resolve_output_element_type(OutputPrecision precision) noexcept;

// Accepts native, float32/float/single, and float64/double, case-insensitively.
// `key` names the configuration key the value came from, for the error message.
OutputPrecision parse_output_precision(const std::string& value, const char* key);

// The engine-neutral `output.precision`; Native when the key is absent.
OutputPrecision configured_output_precision(const Utils::ConfigurationManager& config);

} // namespace VVM::IO

#endif
