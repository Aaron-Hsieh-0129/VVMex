#include "io/bp5/Bp5PathPolicy.hpp"

#include <cstdio>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <unistd.h>

namespace {
int failures = 0;

void check(bool condition, const char* message) {
    if (condition) return;
    ++failures;
    std::fprintf(stderr, "FAIL: %s\n", message);
}

template <typename Function>
void check_throws(Function&& function, const char* message) {
    try {
        function();
        check(false, message);
    } catch (const std::exception&) {
    }
}
} // namespace

int main() {
    namespace fs = std::filesystem;
    using VVM::IO::BP5::ExistingDatasetPolicy;
    using VVM::IO::BP5::prepare_bp5_dataset_path;
    const fs::path root = fs::temp_directory_path() /
        ("vvm_bp5_path_policy_" + std::to_string(static_cast<long long>(getpid())));
    std::error_code ec;
    fs::remove_all(root, ec);

    const fs::path dataset = prepare_bp5_dataset_path(
        root.string(), "history", ExistingDatasetPolicy::Error);
    check(dataset == root / "history.bp", "adds the .bp extension");
    check(fs::is_directory(root), "creates the output directory");

    fs::create_directories(dataset);
    std::ofstream(dataset / "sentinel") << "keep";
    check_throws([&] { prepare_bp5_dataset_path(root.string(), "history", ExistingDatasetPolicy::Error); },
                 "existing dataset is refused with the error policy");
    check(fs::exists(dataset / "sentinel"), "refusal preserves existing dataset");

    const fs::path appended = prepare_bp5_dataset_path(
        root.string(), "history", ExistingDatasetPolicy::Append);
    check(appended == dataset, "append preserves the dataset path");
    check(fs::exists(dataset / "sentinel"), "append preserves existing data");
    const fs::path replaced = prepare_bp5_dataset_path(
        root.string(), "history.bp", ExistingDatasetPolicy::Replace);
    check(replaced == dataset, "an existing .bp suffix is preserved");
    check(!fs::exists(dataset), "replace removes only the exact dataset");

    const fs::path outside = root / "outside";
    fs::create_directories(outside);
    std::ofstream(outside / "sentinel") << "keep";
    fs::create_directory_symlink(outside, dataset);
    check_throws([&] { prepare_bp5_dataset_path(root.string(), "history", ExistingDatasetPolicy::Replace); },
                 "symlink dataset is refused");
    check(fs::exists(outside / "sentinel"), "symlink target is preserved");

    check_throws([&] { prepare_bp5_dataset_path(root.string(), "history", ExistingDatasetPolicy::Append); },
                 "symlink append target is refused");
    fs::remove(dataset, ec);
    check_throws([&] { prepare_bp5_dataset_path(root.string(), "history", ExistingDatasetPolicy::Append); },
                 "missing append target is refused");
    check_throws([&] { prepare_bp5_dataset_path(root.string(), "../escape", ExistingDatasetPolicy::Replace); },
                 "parent traversal prefix is refused");
    check_throws([&] { prepare_bp5_dataset_path(root.string(), "", ExistingDatasetPolicy::Replace); },
                 "empty prefix is refused");
    check_throws([&] { prepare_bp5_dataset_path("", "history", ExistingDatasetPolicy::Replace); },
                 "empty output directory is refused");

    fs::remove_all(root, ec);
    if (failures == 0) std::puts("test_bp5_path_policy: PASS");
    return failures == 0 ? 0 : 1;
}
