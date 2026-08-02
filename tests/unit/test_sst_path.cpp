// Unit tests for the stale-SST cleanup path logic (src/utils/SstPath.hpp).
//
// The cleanup used to be system("rm -rf " + output_dir + "/" + prefix + ".sst"),
// i.e. two configuration strings pasted into a shell command line. These tests
// pin the two properties that replaced it:
//
//   1. Every character in the configuration is an ordinary filename character.
//      Spaces do not split the path, and ';', '$', '&' and backticks do not
//      mean anything -- there is no shell to interpret them.
//   2. The only thing the model will ever delete is a plain <prefix>.sst file
//      directly inside the configured output directory. An empty output_dir or
//      prefix, a '..' in the prefix, "/", "." and the output directory itself
//      are all refused rather than normalised into something plausible.
//
// The filesystem cases (a nested .sst directory, a missing path, a directory
// with the write bit cleared) run in a private directory under the system temp
// directory and clean up after themselves.
//
// Header-only and free of Kokkos/MPI, so this runs on any machine -- no GPU,
// no ranks, no simulation.
//
// Reporting goes through <cstdio> rather than <iostream> on purpose. The NVHPC
// build compiles against the system GCC 13 headers while LD_LIBRARY_PATH puts
// GCC 11's libstdc++ first at run time, and that mismatch faults inside the
// ostream sentry. C stdio does not touch it.

#include "utils/SstPath.hpp"

#include <cstdio>
#include <filesystem>
#include <fstream>
#include <string>
#include <system_error>
#include <unistd.h>
#include <vector>

namespace fs = std::filesystem;

using VVM::Utils::remove_stale_sst_path;
using VVM::Utils::resolve_sst_path;
using VVM::Utils::SstCleanupOutcome;
using VVM::Utils::SstPathError;
using VVM::Utils::SstPathResult;

namespace {

int failures = 0;

void check(bool condition, const std::string& what) {
    if (!condition) {
        std::fprintf(stderr, "FAIL: %s\n", what.c_str());
        ++failures;
    }
}

void check_target(const std::string& output_dir, const std::string& prefix,
                  const fs::path& expected, const std::string& what) {
    const SstPathResult r = resolve_sst_path(output_dir, prefix);
    if (!r.ok) {
        std::fprintf(stderr, "FAIL: %s (rejected: %s)\n", what.c_str(), r.message.c_str());
        ++failures;
        return;
    }
    check(r.path == expected,
          what + ": expected '" + expected.string() + "', got '" + r.path.string() + "'");
}

void check_rejected(const std::string& output_dir, const std::string& prefix,
                    SstPathError expected, const std::string& what) {
    const SstPathResult r = resolve_sst_path(output_dir, prefix);
    if (r.ok) {
        std::fprintf(stderr, "FAIL: %s (accepted '%s')\n", what.c_str(), r.path.string().c_str());
        ++failures;
        return;
    }
    check(r.error == expected,
          what + ": rejected for the wrong reason (" + r.message + ")");
    check(!r.message.empty(), what + ": a rejection must explain itself");
}

// ---------------------------------------------------------------------------
// Path construction
// ---------------------------------------------------------------------------

// The ordinary case, and the same directory written with a trailing separator.
// Shipped configs use both forms ("./output/grass" and
// "./output/testing_output_2dbubble/"), and they must name the same file.
void test_normal_path() {
    check_target("/data/run", "vvm_output", "/data/run/vvm_output.sst",
                 "a normal absolute output_dir");
    check_target("/data/run/", "vvm_output", "/data/run/vvm_output.sst",
                 "a trailing separator on output_dir");
    check_target("/data/run//", "vvm_output", "/data/run/vvm_output.sst",
                 "a doubled trailing separator on output_dir");
    check_target("/data/run/./sub", "vvm_output", "/data/run/sub/vvm_output.sst",
                 "a '.' component inside output_dir");

    // The naming convention itself: <output_dir>/<prefix>.sst, matching the
    // stream name OutputManager opens.
    const SstPathResult r = resolve_sst_path("/data/run", "vvm_output");
    check(r.ok, "the normal case is accepted");
    check(r.path.filename() == "vvm_output.sst", "the file is named <prefix>.sst");
    check(r.path.parent_path() == r.directory, "the file sits directly in output_dir");
}

// A relative output_dir is resolved against the working directory -- the same
// directory OutputManager writes into.
void test_relative_output_dir() {
    const SstPathResult r = resolve_sst_path("./output/grass", "vvm_output");
    check(r.ok, "a relative output_dir is accepted");
    if (!r.ok) return;
    check(r.path.is_absolute(), "the target is made absolute");
    check(r.path.filename() == "vvm_output.sst", "the target is <prefix>.sst");
    check(r.path.parent_path().filename() == "grass",
          "the target sits in the configured directory, got '" + r.path.string() + "'");
    // Resolved against the working directory -- the same one OutputManager
    // writes into -- with no "." left in the path.
    check(r.path.string().find("/./") == std::string::npos,
          "the '.' component is normalised away: '" + r.path.string() + "'");
    check(r.path.string().rfind(fs::current_path().string(), 0) == 0 ||
              !fs::exists(fs::current_path() / "output"),
          "a relative output_dir resolves under the working directory");
}

// The whole point: no shell, so no quoting. A space is a character in a
// filename, not an argument separator, and the target keeps both halves.
void test_spaces_are_ordinary_characters() {
    check_target("/data/my run", "vvm_output", "/data/my run/vvm_output.sst",
                 "a space in output_dir");
    check_target("/data/run", "vvm output", "/data/run/vvm output.sst",
                 "a space in the prefix");
    check_target("/data/my run dir", "my output file",
                 "/data/my run dir/my output file.sst",
                 "spaces in both output_dir and the prefix");

    // The old code would have run: rm -rf /data/my run/vvm_output.sst
    // i.e. deleted "/data/my" and "run/vvm_output.sst". Nothing here splits.
    const SstPathResult r = resolve_sst_path("/data/my run", "vvm_output");
    check(r.ok && r.path.parent_path() == fs::path("/data/my run"),
          "the directory with a space stays one path component");
}

// Shell metacharacters are just bytes in a filename now. Under the old code
// each of these was an injection: `; rm -rf /` and the rest ran as commands.
void test_shell_metacharacters_are_inert() {
    struct Case { std::string dir; std::string prefix; std::string expected; };
    const std::vector<Case> cases = {
        {"/data/run", "vvm;rm -rf ~", "/data/run/vvm;rm -rf ~.sst"},
        {"/data/run", "vvm$HOME", "/data/run/vvm$HOME.sst"},
        {"/data/run", "vvm&sleep", "/data/run/vvm&sleep.sst"},
        {"/data/run", "vvm`id`", "/data/run/vvm`id`.sst"},
        {"/data/run", "vvm$(id)", "/data/run/vvm$(id).sst"},
        {"/data/run", "vvm|tee", "/data/run/vvm|tee.sst"},
        {"/data/run", "vvm\"'x", "/data/run/vvm\"'x.sst"},
        {"/data/run", "vvm\nnewline", "/data/run/vvm\nnewline.sst"},
        {"/data/run", "vvm*?[]", "/data/run/vvm*?[].sst"},
        {"/data;rm -rf /", "vvm_output", "/data;rm -rf /vvm_output.sst"},
        {"/data/$USER run", "vvm_output", "/data/$USER run/vvm_output.sst"},
        {"/data/`id`", "vvm_output", "/data/`id`/vvm_output.sst"},
    };
    for (const auto& c : cases) {
        check_target(c.dir, c.prefix, c.expected,
                     "metacharacters are literal: dir '" + c.dir + "', prefix '" + c.prefix + "'");
    }

    // Whatever the characters are, the target stays a child of the output
    // directory and still ends in .sst.
    for (const auto& c : cases) {
        const SstPathResult r = resolve_sst_path(c.dir, c.prefix);
        check(r.ok, "accepted: '" + c.dir + "' / '" + c.prefix + "'");
        if (!r.ok) continue;
        check(r.path.parent_path() == r.directory,
              "still a direct child: '" + r.path.string() + "'");
        check(r.path.extension() == ".sst", "still a .sst path: '" + r.path.string() + "'");
    }
}

// An empty output_dir used to build "/<prefix>.sst" -- a delete at the root of
// the filesystem. An empty prefix used to build "<output_dir>/.sst", and an
// empty pair built "/.sst".
void test_empty_configuration_is_refused() {
    check_rejected("", "vvm_output", SstPathError::EmptyOutputDir,
                   "an empty output_dir");
    check_rejected("/data/run", "", SstPathError::EmptyPrefix,
                   "an empty prefix");
    check_rejected("", "", SstPathError::EmptyOutputDir,
                   "an empty output_dir and prefix");
}

// '..' in the prefix is the escape the old concatenation allowed:
// "/data/run" + "/" + "../../etc/passwd" + ".sst". The prefix must be one
// filename component, so every one of these is refused outright -- it is never
// normalised into a path outside the output directory.
void test_traversal_in_the_prefix_is_refused() {
    const std::vector<std::string> bad_prefixes = {
        "..", ".", "../vvm_output", "../../etc/passwd", "sub/vvm_output",
        "/etc/passwd", "/", "vvm_output/", "./vvm_output", "a/../b",
        // A shell payload that also happens to contain a separator: refused for
        // being a path, not for the ';' -- and no shell ever sees it either way.
        "vvm;rm -rf /",
    };
    for (const auto& prefix : bad_prefixes) {
        check_rejected("/data/run", prefix, SstPathError::PrefixNotAFileName,
                       "a prefix that is not a plain file name: '" + prefix + "'");
    }

    // A '..' that only appears inside output_dir is a legitimate relative path
    // and is normalised, but the target still lands directly under it.
    const SstPathResult r = resolve_sst_path("/data/run/../other", "vvm_output");
    check(r.ok, "'..' inside output_dir is normalised, not refused");
    if (r.ok) {
        check(r.path == fs::path("/data/other/vvm_output.sst"),
              "output_dir '..' normalises: got '" + r.path.string() + "'");
        check(r.path.parent_path() == r.directory, "and the target stays inside it");
    }
}

// "/", "." and ".." as the output directory are accepted only as directories --
// the target is still a <prefix>.sst file inside them, never the directory
// itself. What must never happen is the target *being* one of those.
void test_dangerous_targets_are_refused() {
    // The root as output_dir: the target is /vvm_output.sst, not "/".
    const SstPathResult root = resolve_sst_path("/", "vvm_output");
    check(root.ok, "'/' as output_dir names a file inside it");
    if (root.ok) {
        check(root.path == fs::path("/vvm_output.sst"), "target under the root");
        check(root.path != root.path.root_path(), "the target is never the root itself");
    }

    // "." and ".." as output_dir resolve against the cwd, same rule.
    for (const std::string dir : {".", ".."}) {
        const SstPathResult r = resolve_sst_path(dir, "vvm_output");
        check(r.ok, "'" + dir + "' as output_dir names a file inside it");
        if (!r.ok) continue;
        check(r.path.filename() == "vvm_output.sst",
              "'" + dir + "' as output_dir still targets <prefix>.sst");
        check(r.path != r.directory, "the target is never the output directory itself");
        check(r.path.is_absolute(), "the target is absolute");
    }

    // No configuration can make the target the output directory, the root, or a
    // dot entry, because the prefix cannot be empty and cannot contain a
    // separator. Assert the invariant over everything accepted above.
    const std::vector<std::pair<std::string, std::string>> accepted = {
        {"/", "vvm_output"}, {".", "vvm_output"}, {"..", "vvm_output"},
        {"/data/run", "vvm_output"}, {"/data/my run", "a;b$c&d`e"},
    };
    for (const auto& c : accepted) {
        const SstPathResult r = resolve_sst_path(c.first, c.second);
        if (!r.ok) continue;
        const std::string name = r.path.filename().string();
        check(r.path != r.directory, "target != output_dir for '" + c.first + "'");
        check(r.path != r.path.root_path(), "target != '/' for '" + c.first + "'");
        check(name != "." && name != ".." && !name.empty(),
              "target is not a dot entry for '" + c.first + "'");
        check(name.size() > 4 && name.compare(name.size() - 4, 4, ".sst") == 0,
              "target ends in .sst for '" + c.first + "'");
    }
}

// ---------------------------------------------------------------------------
// Removal, against a real filesystem
// ---------------------------------------------------------------------------

fs::path make_scratch_dir() {
    const fs::path dir = fs::temp_directory_path() /
                         ("vvm_sst_path_test_" + std::to_string(getpid()));
    std::error_code ec;
    fs::remove_all(dir, ec);
    fs::create_directories(dir);
    return dir;
}

void write_file(const fs::path& path, const std::string& contents) {
    std::ofstream out(path);
    out << contents;
}

// A path that is not there is not an error, and it must not be reported as a
// removal -- a fresh run would otherwise claim to have cleaned something.
void test_missing_path_is_not_an_error() {
    const fs::path scratch = make_scratch_dir();

    const auto result = remove_stale_sst_path(scratch.string(), "vvm_output");
    check(result.outcome == SstCleanupOutcome::NotPresent,
          "a missing .sst path is 'not present'");
    check(result.ok(), "a missing .sst path is not a failure");
    check(result.removed_entries == 0, "nothing is reported as removed");
    check(result.path == scratch / "vvm_output.sst", "the target is still reported");

    // An output directory that does not exist yet -- the first run of a case.
    const auto absent_dir =
        remove_stale_sst_path((scratch / "not_created_yet").string(), "vvm_output");
    check(absent_dir.outcome == SstCleanupOutcome::NotPresent,
          "a missing output directory is 'not present'");

    std::error_code ec;
    fs::remove_all(scratch, ec);
}

// The real case: ADIOS leaves a .sst directory with files in it.
void test_existing_sst_directory_with_nested_files() {
    const fs::path scratch = make_scratch_dir();
    const fs::path sst = scratch / "vvm_output.sst";
    fs::create_directories(sst / "sub");
    write_file(sst / "md.0", "metadata");
    write_file(sst / "sub" / "data.0", "data");

    // A sibling that must survive: cleanup deletes the .sst path only.
    const fs::path keep = scratch / "vvm_output_000001.h5";
    write_file(keep, "output");

    const auto result = remove_stale_sst_path(scratch.string(), "vvm_output");
    check(result.outcome == SstCleanupOutcome::Removed, "the stale .sst tree is removed");
    check(!fs::exists(sst), "the .sst path is gone");
    check(result.removed_entries == 4,
          "all 4 entries are counted, got " + std::to_string(result.removed_entries));
    check(fs::exists(keep), "non-.sst output is left alone");
    check(fs::exists(scratch), "the output directory itself is left alone");

    // Removing it again is a no-op, not a failure.
    const auto again = remove_stale_sst_path(scratch.string(), "vvm_output");
    check(again.outcome == SstCleanupOutcome::NotPresent, "a second cleanup is a no-op");

    std::error_code ec;
    fs::remove_all(scratch, ec);
}

// A plain file at the .sst path (some transports) is removed the same way.
void test_existing_sst_file() {
    const fs::path scratch = make_scratch_dir();
    write_file(scratch / "vvm_output.sst", "stale contact info");

    const auto result = remove_stale_sst_path(scratch.string(), "vvm_output");
    check(result.outcome == SstCleanupOutcome::Removed, "a stale .sst file is removed");
    check(result.removed_entries == 1, "one entry removed");
    check(!fs::exists(scratch / "vvm_output.sst"), "the file is gone");

    std::error_code ec;
    fs::remove_all(scratch, ec);
}

// Names that a shell would have mangled, exercised end to end on disk.
void test_removal_with_hostile_names() {
    const fs::path scratch = make_scratch_dir();
    const fs::path dir = scratch / "run dir; rm -rf $HOME";
    fs::create_directories(dir);

    const std::string prefix = "vvm out;put `id` $x & y";
    const fs::path sst = dir / (prefix + ".sst");
    fs::create_directories(sst);
    write_file(sst / "md.0", "metadata");

    // A decoy the old shell command would have deleted, since it sits at the
    // word boundary the space introduced.
    const fs::path decoy = scratch / "run";
    fs::create_directories(decoy);

    const auto result = remove_stale_sst_path(dir.string(), prefix);
    check(result.outcome == SstCleanupOutcome::Removed,
          "a .sst path full of metacharacters is removed");
    check(!fs::exists(sst), "the hostile-named .sst path is gone");
    check(fs::exists(decoy), "the decoy at the space boundary survives");
    check(fs::exists(dir), "the output directory survives");
    check(fs::exists(scratch), "and so does everything above it");

    std::error_code ec;
    fs::remove_all(scratch, ec);
}

// A rejected configuration must not touch the filesystem at all.
void test_rejected_configuration_removes_nothing() {
    const fs::path scratch = make_scratch_dir();
    write_file(scratch / ".sst", "would be hit by an empty prefix");
    const fs::path victim = scratch / "victim.sst";
    fs::create_directories(victim);
    write_file(victim / "keep", "keep");

    const auto empty_prefix = remove_stale_sst_path(scratch.string(), "");
    check(empty_prefix.outcome == SstCleanupOutcome::Rejected, "an empty prefix is rejected");
    check(!empty_prefix.ok(), "a rejected cleanup is not ok()");
    check(fs::exists(scratch / ".sst"), "an empty prefix deletes nothing");

    const auto empty_dir = remove_stale_sst_path("", "vvm_output");
    check(empty_dir.outcome == SstCleanupOutcome::Rejected, "an empty output_dir is rejected");

    // '..' traversal aimed at the victim directory one level up.
    const auto traversal =
        remove_stale_sst_path((scratch / "sub").string(), "../victim");
    check(traversal.outcome == SstCleanupOutcome::Rejected, "'..' in the prefix is rejected");
    check(fs::exists(victim / "keep"), "the traversal target is untouched");

    check(fs::exists(scratch), "the scratch directory survives every rejection");

    std::error_code ec;
    fs::remove_all(scratch, ec);
}

// A valid path that cannot be removed is a failure, not a silent success: the
// run would otherwise continue and fail later inside ADIOS. Skipped for root,
// which ignores the permission bits.
void test_removal_failure_is_reported() {
    if (geteuid() == 0) {
        std::printf("  (skipping the permission-denied case: running as root)\n");
        return;
    }
    const fs::path scratch = make_scratch_dir();
    const fs::path locked = scratch / "locked";
    const fs::path sst = locked / "vvm_output.sst";
    fs::create_directories(sst);
    write_file(sst / "md.0", "metadata");

    std::error_code ec;
    fs::permissions(locked, fs::perms::owner_read | fs::perms::owner_exec,
                    fs::perm_options::replace, ec);
    if (ec) {
        std::fprintf(stderr, "FAIL: could not clear the write bit: %s\n", ec.message().c_str());
        ++failures;
        fs::remove_all(scratch, ec);
        return;
    }

    const auto result = remove_stale_sst_path(locked.string(), "vvm_output");
    check(result.outcome == SstCleanupOutcome::Failed,
          "a removal blocked by permissions is reported as a failure");
    check(!result.ok(), "a failed cleanup is not ok()");
    check(!result.message.empty(), "the failure explains itself");

    fs::permissions(locked, fs::perms::owner_all, fs::perm_options::replace, ec);
    fs::remove_all(scratch, ec);
}

} // namespace

int main() {
    test_normal_path();
    test_relative_output_dir();
    test_spaces_are_ordinary_characters();
    test_shell_metacharacters_are_inert();
    test_empty_configuration_is_refused();
    test_traversal_in_the_prefix_is_refused();
    test_dangerous_targets_are_refused();

    test_missing_path_is_not_an_error();
    test_existing_sst_directory_with_nested_files();
    test_existing_sst_file();
    test_removal_with_hostile_names();
    test_rejected_configuration_removes_nothing();
    test_removal_failure_is_reported();

    if (failures != 0) {
        std::fprintf(stderr, "%d check(s) failed\n", failures);
        return 1;
    }
    std::printf("sst path: all checks passed\n");
    return 0;
}
