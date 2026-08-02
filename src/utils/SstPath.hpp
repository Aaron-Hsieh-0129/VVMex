#ifndef VVM_UTILS_SST_PATH_HPP
#define VVM_UTILS_SST_PATH_HPP

#include <cstdint>
#include <filesystem>
#include <string>
#include <system_error>

namespace VVM {
namespace Utils {

// Where the stale-SST cleanup gets its path, and what it is allowed to delete.

enum class SstPathError {
    None,
    EmptyOutputDir,        // output.output_dir is "" -- would target "/<prefix>.sst"
    EmptyPrefix,           // output.output_filename_prefix is "" -- would target the directory
    PrefixNotAFileName,    // prefix contains a separator, or is "." / ".."
    NotAnSstName,          // constructed target does not end in .sst
    EscapesOutputDir,      // target is not a direct child of the output directory
    IsRootOrOutputDir,     // target resolved to "/" or to the output directory itself
    UnresolvableOutputDir, // output_dir could not be made absolute
};

struct SstPathResult {
    bool ok = false;
    SstPathError error = SstPathError::None;
    std::string message;                // human-readable reason when !ok
    std::filesystem::path directory;    // the resolved output directory
    std::filesystem::path path;         // the .sst target; only meaningful when ok
};

inline std::filesystem::path strip_trailing_separator(std::filesystem::path p) {
    if (p.has_relative_path() && p.filename().empty()) {
        p = p.parent_path();
    }
    return p;
}

// Build the stale-SST target from the two configuration values, or explain why
// the configuration must not be acted on. Never deletes anything.
inline SstPathResult resolve_sst_path(const std::string& output_dir,
                                      const std::string& prefix) {
    namespace fs = std::filesystem;

    SstPathResult result;

    auto reject = [&result](SstPathError error, std::string message) {
        result.ok = false;
        result.error = error;
        result.message = std::move(message);
        return result;
    };

    if (output_dir.empty()) {
        return reject(SstPathError::EmptyOutputDir,
                      "output.output_dir is empty");
    }
    if (prefix.empty()) {
        return reject(SstPathError::EmptyPrefix,
                      "output.output_filename_prefix is empty");
    }

    // The prefix must be a single filename component. This is what keeps the
    // target inside the output directory: "../../etc" or "/etc/passwd" as a
    // prefix is rejected here rather than normalised into something plausible
    // later. Spaces, ';', '$', '&' and backticks are fine -- they are ordinary
    // filename characters once no shell is involved.
    const fs::path prefix_path(prefix);
    if (prefix == "." || prefix == ".." ||
        prefix_path.has_root_path() || prefix_path.has_parent_path() ||
        prefix_path.filename() != prefix_path) {
        return reject(SstPathError::PrefixNotAFileName,
                      "output.output_filename_prefix must be a plain file name "
                      "(no '/', '.' or '..'), got '" + prefix + "'");
    }

    // Make the directory absolute and normalise it lexically. weakly_canonical()
    // additionally resolves symlinks over the part that already exists and
    // leaves the rest lexical, which is what we need here: the output directory
    // may not have been created yet on a first run. It is allowed to fail (a
    // permission error part-way down the path), in which case the lexically
    // normalised absolute path is still a safe basis for the checks below.
    std::error_code ec;
    fs::path directory = fs::absolute(fs::path(output_dir), ec);
    if (ec) {
        return reject(SstPathError::UnresolvableOutputDir,
                      "cannot resolve output.output_dir '" + output_dir +
                          "': " + ec.message());
    }
    directory = strip_trailing_separator(directory.lexically_normal());

    std::error_code canonical_ec;
    const fs::path canonical_directory = fs::weakly_canonical(directory, canonical_ec);
    if (!canonical_ec && !canonical_directory.empty()) {
        directory = strip_trailing_separator(canonical_directory);
    }
    result.directory = directory;

    if (directory.empty()) {
        return reject(SstPathError::UnresolvableOutputDir,
                      "output.output_dir '" + output_dir + "' resolved to an empty path");
    }

    const fs::path target = directory / fs::path(prefix + ".sst");

    const std::string filename = target.filename().string();
    if (target.extension() != ".sst" || filename.size() <= 4 ||
        filename.compare(filename.size() - 4, 4, ".sst") != 0) {
        return reject(SstPathError::NotAnSstName,
                      "refusing to remove '" + target.string() + "': not a .sst path");
    }
    if (filename == "." || filename == "..") {
        return reject(SstPathError::NotAnSstName,
                      "refusing to remove '" + target.string() + "'");
    }
    if (target == directory || target == target.root_path()) {
        return reject(SstPathError::IsRootOrOutputDir,
                      "refusing to remove '" + target.string() +
                          "': it is the output directory or the filesystem root");
    }
    if (strip_trailing_separator(target.lexically_normal()) != target ||
        target.parent_path() != directory) {
        return reject(SstPathError::EscapesOutputDir,
                      "refusing to remove '" + target.string() +
                          "': not a direct child of output.output_dir '" +
                          directory.string() + "'");
    }

    result.ok = true;
    result.error = SstPathError::None;
    result.path = target;
    return result;
}

enum class SstCleanupOutcome {
    NotPresent, // nothing to do -- no stale stream from a previous run
    Removed,    // a stale path existed and is gone
    Rejected,   // the configuration does not name a path we are willing to delete
    Failed,     // the path is valid but could not be removed
};

struct SstCleanupResult {
    SstCleanupOutcome outcome = SstCleanupOutcome::NotPresent;
    std::filesystem::path path;
    std::uintmax_t removed_entries = 0; // files + directories actually removed
    std::string message;                // set for Rejected and Failed

    bool ok() const {
        return outcome == SstCleanupOutcome::NotPresent ||
               outcome == SstCleanupOutcome::Removed;
    }
};

inline SstCleanupResult remove_stale_sst_path(const std::string& output_dir,
                                              const std::string& prefix) {
    namespace fs = std::filesystem;

    SstCleanupResult result;

    const SstPathResult resolved = resolve_sst_path(output_dir, prefix);
    if (!resolved.ok) {
        result.outcome = SstCleanupOutcome::Rejected;
        result.message = resolved.message;
        return result;
    }
    result.path = resolved.path;

    // symlink_status(), not status(): a stale ".sst" that is a symlink should be
    // reported on its own terms, and remove_all() removes the link rather than
    // following it.
    std::error_code status_ec;
    const fs::file_status status = fs::symlink_status(resolved.path, status_ec);
    if (status.type() == fs::file_type::not_found) {
        result.outcome = SstCleanupOutcome::NotPresent;
        return result;
    }
    if (status_ec) {
        result.outcome = SstCleanupOutcome::Failed;
        result.message = "cannot inspect '" + resolved.path.string() + "': " +
                         status_ec.message();
        return result;
    }

    std::error_code remove_ec;
    const std::uintmax_t removed = fs::remove_all(resolved.path, remove_ec);
    if (remove_ec) {
        result.outcome = SstCleanupOutcome::Failed;
        result.message = "failed to remove '" + resolved.path.string() + "': " +
                         remove_ec.message();
        return result;
    }

    // remove_all() reports how many entries it deleted. Zero means the path went
    // away between the status() above and here; report it as "nothing was there"
    // so the caller never logs a removal that did not happen.
    result.outcome = (removed == 0) ? SstCleanupOutcome::NotPresent
                                    : SstCleanupOutcome::Removed;
    result.removed_entries = removed;
    return result;
}

} // namespace Utils
} // namespace VVM

#endif // VVM_UTILS_SST_PATH_HPP
