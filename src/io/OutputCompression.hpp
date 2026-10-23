#pragma once

#include "utils/ConfigurationManager.hpp"
#include <cstdint>
#include <stdexcept>

namespace VVM::IO {

// Engine-neutral defaults. Explicit legacy engine settings override these
// field by field, so existing configurations keep their meaning.
inline nlohmann::json merge_output_compression(
    const nlohmann::json& common, const nlohmann::json& engine) {
    if (!common.is_object() || !engine.is_object())
        throw std::invalid_argument("Output compression settings must be JSON objects");
    for (const auto& item : common.items()) {
        const auto& key = item.key();
        if (!key.empty() && key.front() == '_') continue;
        const auto& value = item.value();
        if (key == "enabled") {
            if (!value.is_boolean())
                throw std::invalid_argument("output.compression.enabled must be boolean");
        } else if (key == "level" || key == "minimum_bytes") {
            if (!value.is_number_integer())
                throw std::invalid_argument("output.compression." + key + " must be an integer");
            const int64_t maximum = key == "level" ? 9 : INT64_MAX;
            if ((value.is_number_unsigned() && value.get<uint64_t>() > uint64_t(maximum)) ||
                value.get<int64_t>() < (key == "level" ? 0 : 1) ||
                value.get<int64_t>() > maximum)
                throw std::invalid_argument("output.compression." + key + " is out of range");
        } else {
            throw std::invalid_argument("Unknown output.compression key: " + key);
        }
    }
    auto resolved = common;
    for (const auto& item : engine.items()) resolved[item.key()] = item.value();
    return resolved;
}

inline nlohmann::json configured_output_compression(
    const Utils::ConfigurationManager& config, const char* engine_key) {
    const auto common = config.has_key("output.compression")
        ? config.get_value<nlohmann::json>("output.compression") : nlohmann::json::object();
    const auto engine = config.has_key(engine_key)
        ? config.get_value<nlohmann::json>(engine_key) : nlohmann::json::object();
    return merge_output_compression(common, engine);
}
} // namespace VVM::IO
