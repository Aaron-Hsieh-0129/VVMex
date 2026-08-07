#ifndef VVM_IO_BP5_OUTPUT_CONFIG_HPP
#define VVM_IO_BP5_OUTPUT_CONFIG_HPP

#include <map>
#include <string>

#include "utils/ConfigurationManager.hpp"

namespace VVM::IO::BP5 {

enum class CpuBufferMode { Direct, Pack };

const char* cpu_buffer_mode_name(CpuBufferMode mode) noexcept;

struct Bp5OutputConfig {
    std::string aggregation_type = "TwoLevelShm";
    unsigned int num_subfiles = 10;
    unsigned int stats_level = 0;
    bool async_write = false;
    CpuBufferMode buffer_mode = CpuBufferMode::Direct;
    bool overwrite = false;

    static Bp5OutputConfig from_json(const nlohmann::json& value);
    static Bp5OutputConfig from_config(
        const Utils::ConfigurationManager& config);

    std::map<std::string, std::string> adios_parameters() const;
};

} // namespace VVM::IO::BP5

#endif
