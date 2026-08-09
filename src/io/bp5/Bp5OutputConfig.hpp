#ifndef VVM_IO_BP5_OUTPUT_CONFIG_HPP
#define VVM_IO_BP5_OUTPUT_CONFIG_HPP

#include <cstddef>
#include <map>
#include <string>

#include "core/vvm_types.hpp"
#include "utils/ConfigurationManager.hpp"

namespace VVM::IO::BP5 {

enum class CpuBufferMode { Direct, Pack };

// What the configuration asked for. Native follows VVM::Real, so a
// configuration written before this option existed keeps producing exactly the
// bytes it produced before.
enum class OutputPrecision { Native, Float32, Float64 };

// What actually lands on disk, once Native has been resolved against this
// build. History output is deliberately independent of VVM::Real: a
// double-precision model may write float32 history without changing how it
// computes.
enum class OutputElementType { Float32, Float64 };

const char* cpu_buffer_mode_name(CpuBufferMode mode) noexcept;
const char* output_precision_name(OutputPrecision precision) noexcept;
const char* output_element_type_name(OutputElementType type) noexcept;
std::size_t output_element_size(OutputElementType type) noexcept;

// True when the on-disk type is VVM::Real, so writing a field costs no
// conversion and the model's own memory can be handed straight to ADIOS2.
bool output_element_matches_real(OutputElementType type) noexcept;

struct Bp5OutputConfig {
    std::string aggregation_type = "TwoLevelShm";
    unsigned int num_subfiles = 10;
    unsigned int stats_level = 0;
    bool async_write = false;
    CpuBufferMode buffer_mode = CpuBufferMode::Direct;
    OutputPrecision precision = OutputPrecision::Native;
    bool overwrite = false;

    static Bp5OutputConfig from_json(const nlohmann::json& value);
    static Bp5OutputConfig from_config(
        const Utils::ConfigurationManager& config);

    OutputElementType element_type() const noexcept;

    // Converting to a different width has to go through a staging buffer, so a
    // direct memory selection into the model's own field is only available when
    // no conversion is needed. Asking for 'direct' with a converting precision
    // resolves to packing rather than failing.
    CpuBufferMode effective_buffer_mode() const noexcept;

    std::map<std::string, std::string> adios_parameters() const;
};

} // namespace VVM::IO::BP5

#endif
