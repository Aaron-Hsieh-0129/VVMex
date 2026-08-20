#ifndef VVM_IO_BP5_OUTPUT_CONFIG_HPP
#define VVM_IO_BP5_OUTPUT_CONFIG_HPP

#include <cstddef>
#include <map>
#include <string>

#include "core/vvm_types.hpp"
#include "io/OutputPrecision.hpp"
#include "utils/ConfigurationManager.hpp"

namespace VVM::IO::BP5 {

enum class CpuBufferMode { Direct, Pack };

using VVM::IO::OutputElementType;
using VVM::IO::OutputPrecision;
using VVM::IO::output_element_matches_real;
using VVM::IO::output_element_size;
using VVM::IO::output_element_type_name;
using VVM::IO::output_precision_name;

const char* cpu_buffer_mode_name(CpuBufferMode mode) noexcept;

struct Bp5OutputConfig {
    std::string aggregation_type = "TwoLevelShm";
    unsigned int num_subfiles = 10;
    unsigned int stats_level = 0;
    bool async_write = false;
    CpuBufferMode buffer_mode = CpuBufferMode::Direct;
    OutputPrecision precision = OutputPrecision::Native;
    bool overwrite = false;

    static Bp5OutputConfig from_json(
        const nlohmann::json& value,
        OutputPrecision default_precision = OutputPrecision::Native);
    static Bp5OutputConfig from_config(
        const Utils::ConfigurationManager& config);

    OutputElementType element_type() const noexcept;

    CpuBufferMode effective_buffer_mode() const noexcept;

    std::map<std::string, std::string> adios_parameters() const;
};

} // namespace VVM::IO::BP5

#endif
