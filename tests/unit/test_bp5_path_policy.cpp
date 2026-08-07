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
    using VVM::IO::BP5::prepare_bp5_dataset_path;
    const fs::path root = fs::temp_directory_path() /
        ("vvm_bp5_path_policy_" + std::to_string(static_cast<long long>(getpid())));
    std::error_code ec;
    fs::remove_all(root, ec);

    const fs::path dataset = prepare_bp5_dataset_path(root.string(), "history", false);
    check(dataset == root / "history.bp", "adds the .bp extension");
    check(fs::is_directory(root), "creates the output directory");

    fs::create_directories(dataset);
    std::ofstream(dataset / "sentinel") << "keep";
    check_throws([&] { prepare_bp5_dataset_path(root.string(), "history", false); },
                 "existing dataset is refused without overwrite");
    check(fs::exists(dataset / "sentinel"), "refusal preserves existing dataset");

    const fs::path replaced = prepare_bp5_dataset_path(root.string(), "history.bp", true);
    check(replaced == dataset, "an existing .bp suffix is preserved");
    check(!fs::exists(dataset), "overwrite removes only the exact dataset");

    const fs::path outside = root / "outside";
    fs::create_directories(outside);
    std::ofstream(outside / "sentinel") << "keep";
    fs::create_directory_symlink(outside, dataset);
    check_throws([&] { prepare_bp5_dataset_path(root.string(), "history", true); },
                 "symlink dataset is refused");
    check(fs::exists(outside / "sentinel"), "symlink target is preserved");

    check_throws([&] { prepare_bp5_dataset_path(root.string(), "../escape", true); },
                 "parent traversal prefix is refused");
    check_throws([&] { prepare_bp5_dataset_path(root.string(), "", true); },
                 "empty prefix is refused");
    check_throws([&] { prepare_bp5_dataset_path("", "history", true); },
                 "empty output directory is refused");

    fs::remove_all(root, ec);
    if (failures == 0) std::puts("test_bp5_path_policy: PASS");
    return failures == 0 ? 0 : 1;
}
