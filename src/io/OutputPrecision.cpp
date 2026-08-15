#include "io/OutputPrecision.hpp"

#include <algorithm>
#include <cctype>
#include <stdexcept>

namespace VVM::IO {

const char* output_precision_name(OutputPrecision precision) noexcept {
    switch (precision) {
        case OutputPrecision::Float32: return "float32";
        case OutputPrecision::Float64: return "float64";
        case OutputPrecision::Native:  break;
    }
    return "native";
}

const char* output_element_type_name(OutputElementType type) noexcept {
    return type == OutputElementType::Float32 ? "float32" : "float64";
}

std::size_t output_element_size(OutputElementType type) noexcept {
    return type == OutputElementType::Float32 ? sizeof(float) : sizeof(double);
}

bool output_element_matches_real(OutputElementType type) noexcept {
    // VVM::Real is float or double, so comparing widths is a valid identity
    // test and stays correct if the model's precision switch ever moves.
    return output_element_size(type) == sizeof(VVM::Real);
}

OutputElementType resolve_output_element_type(OutputPrecision precision) noexcept {
    switch (precision) {
        case OutputPrecision::Float32: return OutputElementType::Float32;
        case OutputPrecision::Float64: return OutputElementType::Float64;
        case OutputPrecision::Native:  break;
    }
    return sizeof(VVM::Real) == sizeof(float) ? OutputElementType::Float32
                                              : OutputElementType::Float64;
}

OutputPrecision parse_output_precision(const std::string& value, const char* key) {
    std::string name = value;
    std::transform(name.begin(), name.end(), name.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });

    // Aliases are accepted because this is the knob a user flips between runs,
    // and 'float'/'double' is how most people say it.
    if (name == "native") return OutputPrecision::Native;
    if (name == "float32" || name == "float" || name == "single") {
        return OutputPrecision::Float32;
    }
    if (name == "float64" || name == "double") return OutputPrecision::Float64;

    throw std::invalid_argument(
        std::string(key) + " must be 'native', 'float32', or 'float64'.");
}

OutputPrecision configured_output_precision(const Utils::ConfigurationManager& config) {
    if (!config.has_key("output.precision")) return OutputPrecision::Native;
    return parse_output_precision(
        config.get_value<std::string>("output.precision"), "output.precision");
}

} // namespace VVM::IO
