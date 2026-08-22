#include "io/bp5/Bp5OutputConfig.hpp"

#include <cstdio>
#include <functional>
#include <stdexcept>
#include <string>

using VVM::IO::BP5::Bp5OutputConfig;
using VVM::IO::BP5::CpuBufferMode;
using VVM::IO::BP5::OutputElementType;
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

// The same two, as the request that produces them. Precision now arrives as
// the resolved output.precision rather than a key inside the BP5 block.
constexpr OutputPrecision kNativePrecision = OutputPrecision::Native;
constexpr OutputPrecision kConvertingPrecision = sizeof(VVM::Real) == sizeof(float)
                                                     ? OutputPrecision::Float64
                                                     : OutputPrecision::Float32;

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
#if defined(KOKKOS_ENABLE_CUDA)
    check(defaults.effective_buffer_mode() == CpuBufferMode::Pack,
          "CUDA fields always resolve to host packing");
#else
    check(defaults.effective_buffer_mode() == CpuBufferMode::Direct,
          "native precision leaves direct mode intact");
#endif

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

    // Precision reaches BP5 as the resolved output.precision, not as a key of
    // its own. Spelling and aliases are test_output_precision's business.
    const auto as_float32 =
        Bp5OutputConfig::from_json(nlohmann::json::object(), OutputPrecision::Float32);
    check(as_float32.precision == OutputPrecision::Float32, "float32 carried in");
    check(as_float32.element_type() == OutputElementType::Float32,
          "float32 resolves to float32");

    const auto as_float64 =
        Bp5OutputConfig::from_json(nlohmann::json::object(), OutputPrecision::Float64);
    check(as_float64.precision == OutputPrecision::Float64, "float64 carried in");
    check(as_float64.element_type() == OutputElementType::Float64,
          "float64 resolves to float64");
    check(!as_float64.precision_from_bp5_block,
          "precision from output.precision is not flagged as deprecated");

    // Deprecated but still honoured: configurations written before precision
    // moved up a level must keep producing the same dataset, which means the
    // BP5-scoped key still wins over the engine-neutral one.
    const auto legacy = Bp5OutputConfig::from_json(
        {{"precision", "float64"}}, OutputPrecision::Float32);
    check(legacy.precision == OutputPrecision::Float64,
          "output.bp5.precision still overrides output.precision");
    check(legacy.element_type() == OutputElementType::Float64,
          "the deprecated key resolves the on-disk type");
    check(legacy.precision_from_bp5_block,
          "the deprecated key is flagged so the writer can say so");

    // Requesting the width VVM::Real already has is not a conversion, so it
    // must not silently cost a staging buffer.
    const nlohmann::json same_width = {{"buffer_mode", "direct"}};
#if defined(KOKKOS_ENABLE_CUDA)
    check(Bp5OutputConfig::from_json(same_width, kNativePrecision).effective_buffer_mode() ==
              CpuBufferMode::Pack,
          "matching precision still stages CUDA fields");
#else
    check(Bp5OutputConfig::from_json(same_width, kNativePrecision).effective_buffer_mode() ==
              CpuBufferMode::Direct,
          "matching precision keeps direct mode");
#endif

    // Converting cannot hand ADIOS2 the model's own memory, so 'direct'
    // resolves to packing rather than failing or silently writing raw bytes.
    const nlohmann::json converting = {{"buffer_mode", "direct"}};
    const auto converted =
        Bp5OutputConfig::from_json(converting, kConvertingPrecision);
    check(converted.element_type() == kConverting,
          "a converting request resolves to the other width");
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
    // Deprecated, but not a licence to accept nonsense.
    expect_invalid({{"precision", "float16"}}, "unsupported precision accepted");
    expect_invalid({{"precision", 32}}, "wrong precision type accepted");

    if (failures == 0) std::puts("test_bp5_output_config: PASS");
    return failures == 0 ? 0 : 1;
}
