#include "Bp5OutputConfig.hpp"

#include <algorithm>
#include <cctype>
#include <set>
#include <stdexcept>

namespace VVM::IO::BP5 {
namespace {

std::string lower(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return value;
}

template <typename T>
T read_optional(const nlohmann::json& value, const char* key, const T& fallback) {
    if (!value.contains(key)) return fallback;
    try {
        return value.at(key).get<T>();
    } catch (const nlohmann::json::exception& e) {
        throw std::invalid_argument(
            std::string("output.bp5.") + key + " has the wrong type: " + e.what());
    }
}

} // namespace

const char* cpu_buffer_mode_name(CpuBufferMode mode) noexcept {
    return mode == CpuBufferMode::Direct ? "direct" : "pack";
}

Bp5OutputConfig Bp5OutputConfig::from_json(const nlohmann::json& value) {
    if (!value.is_object()) {
        throw std::invalid_argument("output.bp5 must be a JSON object.");
    }

    static const std::set<std::string> allowed = {
        "aggregation_type", "num_subfiles", "stats_level",
        "async_write", "buffer_mode", "overwrite"
    };
    for (const auto& item : value.items()) {
        if (!item.key().empty() && item.key().front() == '_') continue;
        if (allowed.count(item.key()) == 0) {
            throw std::invalid_argument(
                "Unknown BP5 configuration key 'output.bp5." + item.key() + "'.");
        }
    }

    Bp5OutputConfig result;
    result.aggregation_type =
        read_optional<std::string>(value, "aggregation_type", result.aggregation_type);
    result.num_subfiles =
        read_optional<unsigned int>(value, "num_subfiles", result.num_subfiles);
    result.stats_level =
        read_optional<unsigned int>(value, "stats_level", result.stats_level);
    result.async_write =
        read_optional<bool>(value, "async_write", result.async_write);
    result.overwrite = read_optional<bool>(value, "overwrite", result.overwrite);

    const std::string mode = lower(
        read_optional<std::string>(value, "buffer_mode", "direct"));
    if (mode == "direct") {
        result.buffer_mode = CpuBufferMode::Direct;
    } else if (mode == "pack" || mode == "packed") {
        result.buffer_mode = CpuBufferMode::Pack;
    } else {
        throw std::invalid_argument(
            "output.bp5.buffer_mode must be 'direct' or 'pack'.");
    }

    if (result.aggregation_type != "TwoLevelShm") {
        throw std::invalid_argument(
            "output.bp5.aggregation_type currently supports only 'TwoLevelShm'.");
    }
    if (result.num_subfiles == 0) {
        throw std::invalid_argument("output.bp5.num_subfiles must be positive.");
    }
    if (result.stats_level > 1) {
        throw std::invalid_argument("output.bp5.stats_level must be 0 or 1.");
    }

    return result;
}

Bp5OutputConfig Bp5OutputConfig::from_config(
    const Utils::ConfigurationManager& config) {
    if (!config.has_key("output.bp5")) return Bp5OutputConfig{};
    return from_json(config.get_value<nlohmann::json>("output.bp5"));
}

std::map<std::string, std::string> Bp5OutputConfig::adios_parameters() const {
    return {
        {"AggregationType", aggregation_type},
        {"NumSubFiles", std::to_string(num_subfiles)},
        {"StatsLevel", std::to_string(stats_level)},
        {"AsyncWrite", async_write ? "true" : "false"}
    };
}

} // namespace VVM::IO::BP5
