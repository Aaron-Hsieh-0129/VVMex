#include "io/bp5/Bp5OutputConfig.hpp"

#include <cstdio>
#include <functional>
#include <stdexcept>
#include <string>

using VVM::IO::BP5::Bp5OutputConfig;
using VVM::IO::BP5::CpuBufferMode;

namespace {
int failures = 0;

void check(bool condition, const char* message) {
    if (!condition) {
        std::fprintf(stderr, "FAIL: %s\n", message);
        ++failures;
    }
}

void expect_invalid(const nlohmann::json& value, const char* message) {
    try {
        (void)Bp5OutputConfig::from_json(value);
        check(false, message);
    } catch (const std::invalid_argument&) {
    }
}
} // namespace

int main() {
    const auto defaults = Bp5OutputConfig::from_json(nlohmann::json::object());
    check(defaults.aggregation_type == "TwoLevelShm", "default aggregation");
    check(defaults.num_subfiles == 10, "default subfiles");
    check(defaults.stats_level == 0, "default stats");
    check(!defaults.async_write, "async defaults off");
    check(defaults.buffer_mode == CpuBufferMode::Direct, "direct defaults on");
    check(!defaults.overwrite, "overwrite defaults off");

    const nlohmann::json configured = {
        {"aggregation_type", "TwoLevelShm"},
        {"num_subfiles", 4},
        {"stats_level", 1},
        {"async_write", true},
        {"buffer_mode", "pack"},
        {"overwrite", true},
        {"__comment", "ignored documentation key"}};
    const auto parsed = Bp5OutputConfig::from_json(configured);
    check(parsed.num_subfiles == 4, "configured subfiles");
    check(parsed.stats_level == 1, "configured stats");
    check(parsed.async_write, "configured async");
    check(parsed.buffer_mode == CpuBufferMode::Pack, "configured pack");
    check(parsed.overwrite, "configured overwrite");
    check(parsed.adios_parameters().at("AsyncWrite") == "true", "ADIOS async parameter");

    expect_invalid(nlohmann::json::array(), "array config accepted");
    expect_invalid({{"unknown", 1}}, "unknown key accepted");
    expect_invalid({{"num_subfiles", 0}}, "zero subfiles accepted");
    expect_invalid({{"stats_level", 2}}, "invalid stats accepted");
    expect_invalid({{"buffer_mode", "mystery"}}, "invalid buffer mode accepted");
    expect_invalid({{"aggregation_type", "EveryoneWrites"}}, "unsupported aggregation accepted");
    expect_invalid({{"async_write", "yes"}}, "wrong async type accepted");

    if (failures == 0) std::puts("test_bp5_output_config: PASS");
    return failures == 0 ? 0 : 1;
}
