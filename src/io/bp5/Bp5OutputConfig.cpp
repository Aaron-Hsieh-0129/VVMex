#include "Bp5OutputConfig.hpp"
#include "io/OutputCompression.hpp"

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <limits>
#include <set>
#include <stdexcept>

namespace VVM::IO::BP5 {
namespace {

std::string
lower(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    return value;
}

template <typename T>
T
read_optional(const nlohmann::json& value, const char* key, const T& fallback) {
    if (!value.contains(key)) {
        return fallback;
    }
    try {
        return value.at(key).get<T>();
    }
    catch (const nlohmann::json::exception& e) {
        throw std::invalid_argument(
            std::string("output.bp5.") + key + " has the wrong type: " + e.what());
    }
}

} // namespace

const char*
cpu_buffer_mode_name(CpuBufferMode mode) noexcept {
    return mode == CpuBufferMode::Direct ? "direct" : "pack";
}

const char*
existing_dataset_policy_name(ExistingDatasetPolicy policy) noexcept {
    switch (policy) {
    case ExistingDatasetPolicy::Error:
        return "error";
    case ExistingDatasetPolicy::Replace:
        return "replace";
    case ExistingDatasetPolicy::Append:
        return "append";
    }
    return "unknown";
}

OutputElementType
Bp5OutputConfig::element_type() const noexcept {
    return resolve_output_element_type(precision);
}

CpuBufferMode
Bp5OutputConfig::effective_buffer_mode() const noexcept {
    // Operators consume contiguous host blocks. Do not expose halo-strided
    // model memory to compression, even when output precision is native.
    if (compression_enabled) return CpuBufferMode::Pack;
#if defined(KOKKOS_ENABLE_CUDA)
    // ADIOS2 is deliberately built without Kokkos/CUDA support, so every CUDA
    // field must be copied to host memory before it is handed to the writer.
    return CpuBufferMode::Pack;
#else
    return output_element_matches_real(element_type()) ? buffer_mode : CpuBufferMode::Pack;
#endif
}

Bp5OutputConfig
Bp5OutputConfig::from_json(const nlohmann::json& value, OutputPrecision default_precision) {
    if (!value.is_object()) {
        throw std::invalid_argument("output.bp5 must be a JSON object.");
    }

    static const std::set<std::string> allowed = {"aggregation_type",
        "num_subfiles",
        "stats_level",
        "async_write",
        "buffer_mode",
        "existing_dataset",
        "overwrite",
        "precision",
        "compression"};
    for (const auto& item : value.items()) {
        if (!item.key().empty() && item.key().front() == '_') {
            continue;
        }
        if (allowed.count(item.key()) == 0) {
            throw std::invalid_argument(
                "Unknown BP5 configuration key 'output.bp5." + item.key() + "'.");
        }
    }

    Bp5OutputConfig result;
    result.precision = default_precision;
    result.aggregation_type =
        read_optional<std::string>(value, "aggregation_type", result.aggregation_type);
    result.num_subfiles = read_optional<unsigned int>(value, "num_subfiles", result.num_subfiles);
    result.stats_level = read_optional<unsigned int>(value, "stats_level", result.stats_level);
    result.async_write = read_optional<bool>(value, "async_write", result.async_write);
    if (value.contains("compression")) {
        const auto& c = value.at("compression");
        if (!c.is_object())
            throw std::invalid_argument("output.bp5.compression must be a JSON object");
        const std::set<std::string> keys = {"enabled", "codec", "level", "threads", "minimum_bytes"};
        for (const auto& item : c.items()) {
            if (!item.key().empty() && item.key().front() == '_') continue;
            if (!keys.count(item.key()))
                throw std::invalid_argument("Unknown output.bp5.compression key: " + item.key());
        }
        try {
            if (c.contains("enabled")) result.compression_enabled = c.at("enabled").get<bool>();
            if (c.contains("codec")) result.compression_codec = lower(c.at("codec").get<std::string>());
            const auto integer = [&](const char* key, int64_t fallback, int64_t maximum) {
                if (!c.contains(key)) return fallback;
                const auto& v = c.at(key);
                if (!v.is_number_integer() || (v.is_number_unsigned() && v.get<uint64_t>() > uint64_t(maximum)))
                    throw std::invalid_argument(std::string("output.bp5.compression.") + key + " must be an integer in range");
                const auto n = v.get<int64_t>();
                if (n < 0 || n > maximum)
                    throw std::invalid_argument(std::string("output.bp5.compression.") + key + " is out of range");
                return n;
            };
            result.compression_level = static_cast<int>(integer("level", 1, 9));
            result.compression_threads = static_cast<int>(integer("threads", 1, 32767));
            const auto minimum = integer("minimum_bytes", 1048576, INT64_MAX);
            if (result.compression_threads == 0 || minimum == 0)
                throw std::invalid_argument("output.bp5.compression threads and minimum_bytes must be positive");
            if (uint64_t(minimum) > std::numeric_limits<std::size_t>::max())
                throw std::invalid_argument("output.bp5.compression.minimum_bytes exceeds size_t");
            result.compression_minimum_bytes = static_cast<std::size_t>(minimum);
        } catch (const nlohmann::json::exception& e) {
            throw std::invalid_argument(std::string("Invalid output.bp5.compression: ") + e.what());
        }
        if (result.compression_codec != "zstd" && result.compression_codec != "lz4" &&
            result.compression_codec != "zlib")
            throw std::invalid_argument("output.bp5.compression.codec must be zstd, lz4, or zlib");
    }
    if (value.contains("existing_dataset") && value.contains("overwrite")) {
        throw std::invalid_argument(
            "output.bp5.existing_dataset and the legacy output.bp5.overwrite "
            "cannot both be set.");
    }
    if (value.contains("existing_dataset")) {
        const std::string policy =
            lower(read_optional<std::string>(value, "existing_dataset", "error"));
        if (policy == "error") {
            result.existing_dataset = ExistingDatasetPolicy::Error;
        }
        else if (policy == "replace") {
            result.existing_dataset = ExistingDatasetPolicy::Replace;
        }
        else if (policy == "append") {
            result.existing_dataset = ExistingDatasetPolicy::Append;
        }
        else {
            throw std::invalid_argument(
                "output.bp5.existing_dataset must be 'error', 'replace', or 'append'.");
        }
    }
    else if (value.contains("overwrite")) {
        result.existing_dataset = read_optional<bool>(value, "overwrite", false)
                                      ? ExistingDatasetPolicy::Replace
                                      : ExistingDatasetPolicy::Error;
        result.existing_dataset_from_legacy_overwrite = true;
    }

    const std::string mode = lower(read_optional<std::string>(value, "buffer_mode", "direct"));
    if (mode == "direct") {
        result.buffer_mode = CpuBufferMode::Direct;
    }
    else if (mode == "pack" || mode == "packed") {
        result.buffer_mode = CpuBufferMode::Pack;
    }
    else {
        throw std::invalid_argument("output.bp5.buffer_mode must be 'direct' or 'pack'.");
    }

    // Deprecated: precision is engine-neutral and belongs at output.precision.
    // Still honoured, with the original override semantics, so configurations
    // written before it moved keep producing the same dataset.
    if (value.contains("precision")) {
        result.precision =
            parse_output_precision(read_optional<std::string>(value, "precision", "native"),
                "output.bp5.precision");
        result.precision_from_bp5_block = true;
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

Bp5OutputConfig
Bp5OutputConfig::from_config(const Utils::ConfigurationManager& config) {
    const OutputPrecision default_precision = configured_output_precision(config);
    auto value = config.has_key("output.bp5")
        ? config.get_value<nlohmann::json>("output.bp5") : nlohmann::json::object();
    if (!value.is_object()) throw std::invalid_argument("output.bp5 must be a JSON object");
    value["compression"] = configured_output_compression(config, "output.bp5.compression");
    return from_json(value, default_precision);
}

std::map<std::string, std::string>
Bp5OutputConfig::adios_parameters() const {
    return {{"AggregationType", aggregation_type},
        {"NumSubFiles", std::to_string(num_subfiles)},
        {"StatsLevel", std::to_string(stats_level)},
        {"AsyncWrite", async_write ? "true" : "false"}};
}

std::map<std::string, std::string>
Bp5OutputConfig::compression_parameters() const {
    return {{"compressor", compression_codec},
        {"clevel", std::to_string(compression_level)},
        {"nthreads", std::to_string(compression_threads)},
        {"threshold", std::to_string(compression_minimum_bytes)},
        {"doshuffle", "BLOSC_SHUFFLE"}};
}

} // namespace VVM::IO::BP5
