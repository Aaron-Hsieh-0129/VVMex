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

// Output precision is engine-neutral and lives in VVM::IO. Re-exported here so
// existing BP5 code and tests keep naming it VVM::IO::BP5::OutputPrecision.
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

    // `default_precision` is what `output.precision` asked for; the BP5 block's
    // own `precision` key overrides it when present.
    static Bp5OutputConfig from_json(
        const nlohmann::json& value,
        OutputPrecision default_precision = OutputPrecision::Native);
    static Bp5OutputConfig from_config(
        const Utils::ConfigurationManager& config);

    OutputElementType element_type() const noexcept;

    // Converting to a different width has to go through a staging buffer, so a
    // direct memory selection into the model's own field is only available when
    // no conversion is needed. CUDA fields also always stage through host
    // memory because the external ADIOS2 build is not GPU-aware. Asking for
    // 'direct' in either case resolves to packing rather than failing.
    CpuBufferMode effective_buffer_mode() const noexcept;

    std::map<std::string, std::string> adios_parameters() const;
};

} // namespace VVM::IO::BP5

#endif
