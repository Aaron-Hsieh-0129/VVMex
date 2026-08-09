#include "io/bp5/Bp5OutputConfig.hpp"

#include <cstdio>
#include <functional>
#include <stdexcept>
#include <string>

using VVM::IO::BP5::Bp5OutputConfig;
using VVM::IO::BP5::CpuBufferMode;
using VVM::IO::BP5::OutputElementType;
using VVM::IO::BP5::output_element_type_name;
using VVM::IO::BP5::OutputPrecision;

// The on-disk type a 'native' request resolves to in this build. Everything
// about precision is relative to it, so the assertions below stay valid whether
// VVM_USE_DOUBLE_PRECISION is on or off.
constexpr OutputElementType kNative = sizeof(VVM::Real) == sizeof(float)
                                          ? OutputElementType::Float32
                                          : OutputElementType::Float64;
constexpr OutputElementType kConverting = sizeof(VVM::Real) == sizeof(float)
                                              ? OutputElementType::Float64
                                              : OutputElementType::Float32;

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
    // The default must stay lossless and byte-identical to the behaviour that
    // existed before precision was configurable.
    check(defaults.precision == OutputPrecision::Native, "precision defaults to native");
    check(defaults.element_type() == kNative, "native resolves to VVM::Real width");
    check(defaults.effective_buffer_mode() == CpuBufferMode::Direct,
          "native precision leaves direct mode intact");

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

    // Explicit precision, independent of VVM::Real in both directions.
    const auto as_float32 =
        Bp5OutputConfig::from_json({{"precision", "float32"}});
    check(as_float32.precision == OutputPrecision::Float32, "float32 parsed");
    check(as_float32.element_type() == OutputElementType::Float32,
          "float32 resolves to float32");

    const auto as_float64 =
        Bp5OutputConfig::from_json({{"precision", "float64"}});
    check(as_float64.precision == OutputPrecision::Float64, "float64 parsed");
    check(as_float64.element_type() == OutputElementType::Float64,
          "float64 resolves to float64");

    // "float"/"double"/"single" are the words people actually reach for.
    check(Bp5OutputConfig::from_json({{"precision", "float"}}).element_type() ==
              OutputElementType::Float32,
          "'float' alias");
    check(Bp5OutputConfig::from_json({{"precision", "single"}}).element_type() ==
              OutputElementType::Float32,
          "'single' alias");
    check(Bp5OutputConfig::from_json({{"precision", "double"}}).element_type() ==
              OutputElementType::Float64,
          "'double' alias");
    check(Bp5OutputConfig::from_json({{"precision", "FLOAT32"}}).element_type() ==
              OutputElementType::Float32,
          "precision is case-insensitive");

    // Requesting the width VVM::Real already has is not a conversion, so it
    // must not silently cost a staging buffer.
    const nlohmann::json same_width = {
        {"precision", output_element_type_name(kNative)}, {"buffer_mode", "direct"}};
    check(Bp5OutputConfig::from_json(same_width).effective_buffer_mode() ==
              CpuBufferMode::Direct,
          "matching precision keeps direct mode");

    // Converting cannot hand ADIOS2 the model's own memory, so 'direct'
    // resolves to packing rather than failing or silently writing raw bytes.
    const nlohmann::json converting = {
        {"precision", output_element_type_name(kConverting)}, {"buffer_mode", "direct"}};
    const auto converted = Bp5OutputConfig::from_json(converting);
    check(converted.buffer_mode == CpuBufferMode::Direct,
          "the requested buffer mode is preserved as asked");
    check(converted.effective_buffer_mode() == CpuBufferMode::Pack,
          "converting precision forces packing");

    expect_invalid(nlohmann::json::array(), "array config accepted");
    expect_invalid({{"unknown", 1}}, "unknown key accepted");
    expect_invalid({{"num_subfiles", 0}}, "zero subfiles accepted");
    expect_invalid({{"stats_level", 2}}, "invalid stats accepted");
    expect_invalid({{"buffer_mode", "mystery"}}, "invalid buffer mode accepted");
    expect_invalid({{"aggregation_type", "EveryoneWrites"}}, "unsupported aggregation accepted");
    expect_invalid({{"async_write", "yes"}}, "wrong async type accepted");
    expect_invalid({{"precision", "float16"}}, "unsupported precision accepted");
    expect_invalid({{"precision", "int32"}}, "non-floating precision accepted");
    expect_invalid({{"precision", 32}}, "wrong precision type accepted");

    if (failures == 0) std::puts("test_bp5_output_config: PASS");
    return failures == 0 ? 0 : 1;
}
