#include "io/history/GradsCtl.hpp"

#include <cstdio>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <unordered_set>
#include <vector>
#include <unistd.h>

namespace {
int failures = 0;

void check(bool condition, const char* message) {
    if (condition) return;
    ++failures;
    std::fprintf(stderr, "FAIL: %s\n", message);
}

std::vector<std::string> read_lines(const std::filesystem::path& path) {
    std::vector<std::string> lines;
    std::ifstream file(path);
    std::string line;
    while (std::getline(file, line)) lines.push_back(line);
    return lines;
}

bool contains(const std::vector<std::string>& lines, const std::string& line) {
    for (const auto& candidate : lines) {
        if (candidate == line) return true;
    }
    return false;
}

VVM::IO::GradsVariable make_variable(
    const std::string& dataset_name,
    const std::string& grads_name,
    std::size_t levels,
    const std::string& dimensions) {
    VVM::IO::GradsVariable variable;
    variable.dataset_name = dataset_name;
    variable.grads_name = grads_name;
    variable.levels = levels;
    variable.dimensions = dimensions;
    variable.description = dataset_name + " description";
    return variable;
}

void test_variable_names() {
    using VVM::IO::unique_grads_variable_name;
    std::unordered_set<std::string> taken;

    // GrADS lowercases the name and keeps at most 15 characters, and rejects a
    // descriptor whose first character is not a lowercase letter.
    check(unique_grads_variable_name("w", taken) == "w", "short name is unchanged");
    check(unique_grads_variable_name("Tg", taken) == "tg", "name is lowercased");
    check(unique_grads_variable_name("precip_liq_surf_mass", taken) == "precip_liq_surf",
          "long name is truncated to 15 characters");
    check(unique_grads_variable_name("precip_liq_surf_flux", taken) == "precip_liq_sur2",
          "a truncation collision gets a suffix");
    check(unique_grads_variable_name("2m_temperature", taken) == "v2m_temperature",
          "a leading digit is prefixed");
    check(unique_grads_variable_name("q c", taken) == "q_c",
          "an illegal character becomes an underscore");
}

void test_descriptor(const std::filesystem::path& root) {
    VVM::IO::GradsCtl ctl;
    ctl.dset = "^vvm_output.bp";
    ctl.dtype = "bp5";
    ctl.nx = 4;
    ctl.ny = 3;
    ctl.x.start = VVM::real(-1.5);
    ctl.x.increment = VVM::real(0.5);
    ctl.y = ctl.x;
    ctl.z_levels = {VVM::real(50.0), VVM::real(150.5), VVM::real(300.0)};
    ctl.time_count = 7;
    ctl.time_start = VVM::IO::grads_start_time(0);
    ctl.time_increment = VVM::IO::grads_time_increment(VVM::real(300.0));
    ctl.notes.push_back("thbar is a profile");
    ctl.variables.push_back(make_variable("th", "th", 3, "z,y,x"));
    ctl.variables.push_back(make_variable("Tg", "tg", 0, "y,x"));
    ctl.variables.push_back(make_variable("qbands", "qbands", 3, "0,z,y,x"));

    const std::filesystem::path path = root / "bp5.ctl";
    VVM::IO::write_grads_ctl(path, ctl);
    const auto lines = read_lines(path);

    check(contains(lines, "DSET ^vvm_output.bp"), "writes the dataset line");
    check(contains(lines, "DTYPE bp5"), "writes the engine type");
    check(!contains(lines, "OPTIONS template"),
          "a single multi-step dataset is not templated");
    check(contains(lines, "XDEF 4 LINEAR -1.5000000 .5000000"), "writes the x axis");
    check(contains(lines, "YDEF 3 LINEAR -1.5000000 .5000000"), "writes the y axis");
    // Levels are whole metres; GrADS only uses them as labels.
    check(contains(lines, "ZDEF 3 LEVELS 50 150 300"), "writes the z levels");
    check(contains(lines, "TDEF 7 LINEAR 00z01JAN1998 5mn"), "writes the time axis");
    check(contains(lines, "* thbar is a profile"), "notes become comment lines");
    check(contains(lines, "VARS 3"), "counts only real variable records");
    check(contains(lines, "th 3 z,y,x th description"),
          "a name GrADS accepts needs no alias");
    check(contains(lines, "Tg=>tg 0 y,x Tg description"),
          "a name GrADS cannot use is aliased");
    check(contains(lines, "qbands 3 0,z,y,x qbands description"),
          "a component axis is pinned to a fixed index");
    check(contains(lines, "ENDVARS"), "closes the variable block");

    // The legacy HDF5 descriptor differs only in its header and dataset names.
    ctl.dset = "^vvm_output_%tm6.h5";
    ctl.dtype = "hdf5_grid";
    ctl.templated = true;
    ctl.notes.clear();
    ctl.variables = {make_variable("/Step0/th", "th", 3, "z,y,x")};
    const std::filesystem::path h5_path = root / "hdf5.ctl";
    VVM::IO::write_grads_ctl(h5_path, ctl);
    const auto h5_lines = read_lines(h5_path);
    check(contains(h5_lines, "DSET ^vvm_output_%tm6.h5"), "writes the template dataset");
    check(contains(h5_lines, "DTYPE hdf5_grid"), "writes the hdf5 engine type");
    check(contains(h5_lines, "OPTIONS template"), "one file per step is templated");
    check(contains(h5_lines, "/Step0/th=>th 3 z,y,x /Step0/th description"),
          "the hdf5 dataset path is aliased to the GrADS name");
}
} // namespace

int main() {
    namespace fs = std::filesystem;
    const fs::path root = fs::temp_directory_path() /
        ("vvm_grads_ctl_" + std::to_string(static_cast<long long>(getpid())));
    std::error_code ec;
    fs::remove_all(root, ec);
    fs::create_directories(root, ec);

    test_variable_names();
    test_descriptor(root);

    fs::remove_all(root, ec);
    if (failures == 0) std::fprintf(stdout, "test_grads_ctl: all checks passed\n");
    return failures == 0 ? 0 : 1;
}
