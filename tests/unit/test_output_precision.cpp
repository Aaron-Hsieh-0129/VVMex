#include "io/OutputPrecision.hpp"
#include "io/bp5/Bp5OutputConfig.hpp"

#include <cstdio>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <unistd.h>

#include <mpi.h>

using VVM::IO::OutputElementType;
using VVM::IO::OutputPrecision;
using VVM::IO::configured_output_precision;
using VVM::IO::output_element_matches_real;
using VVM::IO::parse_output_precision;
using VVM::IO::resolve_output_element_type;
using VVM::IO::BP5::Bp5OutputConfig;

// Everything about precision is relative to what 'native' resolves to in this
// build, so the assertions hold whether or not VVM::Real is double.
constexpr OutputElementType kNative = sizeof(VVM::Real) == sizeof(float)
                                          ? OutputElementType::Float32
                                          : OutputElementType::Float64;
constexpr OutputElementType kConverting = sizeof(VVM::Real) == sizeof(float)
                                              ? OutputElementType::Float64
                                              : OutputElementType::Float32;

namespace {
int failures = 0;

void check(bool condition, const char* message) {
    if (condition) return;
    ++failures;
    std::fprintf(stderr, "FAIL: %s\n", message);
}

void expect_invalid(const char* value, const char* message) {
    try {
        (void)parse_output_precision(value, "output.precision");
        check(false, message);
    } catch (const std::invalid_argument&) {
    }
}

void test_parsing() {
    check(parse_output_precision("native", "k") == OutputPrecision::Native,
          "native parses");
    check(parse_output_precision("float32", "k") == OutputPrecision::Float32,
          "float32 parses");
    check(parse_output_precision("float", "k") == OutputPrecision::Float32,
          "float is an alias for float32");
    check(parse_output_precision("single", "k") == OutputPrecision::Float32,
          "single is an alias for float32");
    check(parse_output_precision("Float64", "k") == OutputPrecision::Float64,
          "parsing is case-insensitive");
    check(parse_output_precision("double", "k") == OutputPrecision::Float64,
          "double is an alias for float64");
    expect_invalid("half", "an unknown precision is rejected");
    expect_invalid("", "an empty precision is rejected");

    check(resolve_output_element_type(OutputPrecision::Native) == kNative,
          "native follows VVM::Real");
    check(resolve_output_element_type(OutputPrecision::Float32) ==
              OutputElementType::Float32,
          "float32 resolves to float32");
    check(output_element_matches_real(kNative), "native needs no conversion");
    check(!output_element_matches_real(kConverting), "the other width converts");
}

std::string write_config(const std::filesystem::path& path, const std::string& output_body) {
    std::ofstream file(path);
    file << "{\n  \"output\": {\n" << output_body << "\n  }\n}\n";
    file.close();
    return path.string();
}

void test_configuration(const std::filesystem::path& root) {
    using VVM::Utils::ConfigurationManager;

    const ConfigurationManager none(
        write_config(root / "none.json", "    \"output_dir\": \".\""));
    check(configured_output_precision(none) == OutputPrecision::Native,
          "output.precision defaults to native");
    check(Bp5OutputConfig::from_config(none).precision == OutputPrecision::Native,
          "BP5 defaults to native");

    // The engine-neutral key drives every engine, BP5 included.
    const ConfigurationManager shared(
        write_config(root / "shared.json", "    \"precision\": \"float32\""));
    check(configured_output_precision(shared) == OutputPrecision::Float32,
          "output.precision is read");
    check(Bp5OutputConfig::from_config(shared).precision == OutputPrecision::Float32,
          "BP5 inherits output.precision when its own block is absent");

    const ConfigurationManager shared_with_block(
        write_config(root / "shared_block.json",
                     "    \"precision\": \"float32\",\n"
                     "    \"bp5\": { \"num_subfiles\": 4 }"));
    check(Bp5OutputConfig::from_config(shared_with_block).precision ==
              OutputPrecision::Float32,
          "a BP5 block without precision still inherits output.precision");

    // output.bp5.precision is deprecated -- precision is engine-neutral and a
    // run has only one engine -- but configurations predating the move must
    // keep working, so it is still read and still wins.
    const ConfigurationManager overridden(
        write_config(root / "override.json",
                     "    \"precision\": \"float32\",\n"
                     "    \"bp5\": { \"precision\": \"native\" }"));
    const auto legacy = Bp5OutputConfig::from_config(overridden);
    check(legacy.precision == OutputPrecision::Native,
          "the deprecated output.bp5.precision still overrides output.precision");
    check(legacy.precision_from_bp5_block,
          "using the deprecated key is recorded");

    const ConfigurationManager bp5_only(
        write_config(root / "bp5_only.json", "    \"bp5\": { \"precision\": \"double\" }"));
    check(Bp5OutputConfig::from_config(bp5_only).precision == OutputPrecision::Float64,
          "the deprecated key still works on its own");

    const ConfigurationManager invalid(
        write_config(root / "invalid.json", "    \"precision\": \"half\""));
    try {
        (void)configured_output_precision(invalid);
        check(false, "an invalid output.precision is rejected");
    } catch (const std::invalid_argument&) {
    }
}
} // namespace

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    namespace fs = std::filesystem;
    const fs::path root = fs::temp_directory_path() /
        ("vvm_output_precision_" + std::to_string(static_cast<long long>(getpid())));
    std::error_code ec;
    fs::remove_all(root, ec);
    fs::create_directories(root, ec);

    test_parsing();
    test_configuration(root);

    fs::remove_all(root, ec);
    if (failures == 0) std::fprintf(stdout, "test_output_precision: all checks passed\n");

    MPI_Finalize();
    return failures == 0 ? 0 : 1;
}
