#ifndef VVM_UTILS_RESTART_TIME_HPP
#define VVM_UTILS_RESTART_TIME_HPP

#include <cctype>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace VVM {
namespace Utils {

// Where a restart run picks up from.
struct RestartFileMetadata {
    bool has_time = false;
    double time_s = 0.0;   // elapsed simulation seconds
    bool has_step = false;
    long long step = 0;    // integration-step count
};

// The configured policy: dt, plus the explicit opt-ins that cover files with no
// metadata at all. Nothing here is consulted while the file itself has an
// answer.
struct RestartTimePolicy {
    double dt_s = 0.0;                    // simulation.dt_s
    bool has_legacy_time = false;         // restart.legacy_time_s was given
    double legacy_time_s = 0.0;
    bool allow_filename_fallback = false; // restart.allow_filename_time_fallback
    double filename_interval_s = 3600.0;  // restart.file_interval_s
    bool ignore_stored_step = false;      // restart.ignore_stored_step
};

enum class RestartTimeSource {
    FileTimeAndStep,      // both read from the file
    FileTimeDerivedStep,  // time from the file, step = round(time / dt)
    FileStepDerivedTime,  // step from the file, time = step * dt
    LegacyConfig,         // restart.legacy_time_s
    Filename,             // trailing digits * restart.file_interval_s
};

inline const char* to_string(RestartTimeSource source) {
    switch (source) {
        case RestartTimeSource::FileTimeAndStep:     return "file model_time_s + model_step";
        case RestartTimeSource::FileTimeDerivedStep: return "file model_time_s, step derived from dt";
        case RestartTimeSource::FileStepDerivedTime: return "file model_step, time derived from dt";
        case RestartTimeSource::LegacyConfig:        return "restart.legacy_time_s";
        case RestartTimeSource::Filename:            return "file name digits (legacy fallback)";
    }
    return "unknown";
}

struct RestartTimeResolution {
    bool ok = false;
    std::string error;                 // only when !ok
    double time_s = 0.0;
    long long step = 0;
    RestartTimeSource source = RestartTimeSource::FileTimeAndStep;
    std::vector<std::string> warnings; // printed on rank 0
};

inline double restart_time_tolerance_s(double a, double b) {
    const double scale = std::max(std::fabs(a), std::fabs(b));
    return std::max(1e-6, 1e-6 * scale);
}

inline std::string format_restart_seconds(double value) {
    std::ostringstream oss;
    oss << std::setprecision(17) << value;
    return oss.str();
}

inline bool parse_trailing_file_index(const std::string& source_file,
                                      long long& index) {
    const std::size_t dot_pos = source_file.find_last_of('.');
    const std::size_t search_end =
        (dot_pos == std::string::npos) ? source_file.size() : dot_pos;

    std::size_t digit_end = search_end;
    while (digit_end > 0 &&
           !std::isdigit(static_cast<unsigned char>(source_file[digit_end - 1]))) {
        --digit_end;
    }

    std::size_t digit_start = digit_end;
    while (digit_start > 0 &&
           std::isdigit(static_cast<unsigned char>(source_file[digit_start - 1]))) {
        --digit_start;
    }

    if (digit_start == digit_end) return false;

    try {
        index = std::stoll(source_file.substr(digit_start, digit_end - digit_start));
    } catch (const std::exception&) {
        return false;
    }
    return true;
}

// Turn (file metadata, configured policy) into the time and step the restarted
// run starts from, in the documented priority order:
//
//   1. stored time + stored step        (checked against each other)
//   2. stored time, step from dt        (or stored step, time from dt)
//   3. restart.legacy_time_s            (explicit, warns)
//   4. file name digits                 (explicit opt-in only, warns loudly)
//
// Returns ok == false with a fully populated message rather than throwing, so
// the caller decides how to report it and the logic stays trivially testable.
inline RestartTimeResolution resolve_restart_time(const RestartFileMetadata& meta,
                                                  const RestartTimePolicy& policy,
                                                  const std::string& source_file) {
    RestartTimeResolution result;

    auto fail = [&result](const std::string& message) {
        result.ok = false;
        result.error = message;
        return result;
    };

    if (!std::isfinite(policy.dt_s) || policy.dt_s <= 0.0) {
        return fail("simulation.dt_s must be a positive, finite number to recover a "
                    "restart step; got " + format_restart_seconds(policy.dt_s) + ".");
    }

    // Reject a stored time that cannot describe a simulation clock before it can
    // reach state_.set_time().
    auto validate_time = [&](double time_s, const std::string& origin) {
        if (!std::isfinite(time_s)) {
            return "restart time from " + origin + " is not finite (" +
                   format_restart_seconds(time_s) + "); source file '" + source_file + "'.";
        }
        if (time_s < 0.0) {
            return "restart time from " + origin + " is negative (" +
                   format_restart_seconds(time_s) + " s); source file '" + source_file + "'.";
        }
        return std::string();
    };

    // Derive the step from an authoritative time. time / dt is integral for any
    // file this model wrote, so a non-integral ratio means dt changed since the
    // file was written -- worth saying out loud, but the time still stands.
    auto derive_step = [&](double time_s) {
        const long long step = static_cast<long long>(std::llround(time_s / policy.dt_s));
        const double expected = static_cast<double>(step) * policy.dt_s;
        const double tolerance = restart_time_tolerance_s(time_s, expected);
        if (std::fabs(expected - time_s) > tolerance) {
            std::ostringstream oss;
            oss << "[WARNING] Restart time " << format_restart_seconds(time_s)
                << " s is not an integer multiple of simulation.dt_s = "
                << format_restart_seconds(policy.dt_s)
                << " s. Using the nearest step " << step << " (= "
                << format_restart_seconds(expected)
                << " s); the recovered step count is approximate.";
            result.warnings.push_back(oss.str());
        }
        return step;
    };

    if (meta.has_time) {
        const std::string time_error = validate_time(meta.time_s, "the restart file");
        if (!time_error.empty()) return fail(time_error);

        if (meta.has_step && policy.ignore_stored_step) {
            std::ostringstream oss;
            oss << "[WARNING] restart.ignore_stored_step is set: the step " << meta.step
                << " stored in '" << source_file
                << "' is discarded and the step is derived from simulation.dt_s.";
            result.warnings.push_back(oss.str());
        }

        if (meta.has_step && !policy.ignore_stored_step) {
            if (meta.step < 0) {
                std::ostringstream oss;
                oss << "restart step stored in '" << source_file << "' is negative ("
                    << meta.step << ").";
                return fail(oss.str());
            }

            const double expected = static_cast<double>(meta.step) * policy.dt_s;
            const double tolerance = restart_time_tolerance_s(meta.time_s, expected);
            if (std::fabs(expected - meta.time_s) > tolerance) {
                std::ostringstream oss;
                oss << "Inconsistent restart metadata in '" << source_file << "':\n"
                    << "    stored time          = " << format_restart_seconds(meta.time_s) << " s\n"
                    << "    stored step          = " << meta.step << "\n"
                    << "    configured dt        = " << format_restart_seconds(policy.dt_s) << " s\n"
                    << "    expected time (step * dt) = " << format_restart_seconds(expected) << " s\n"
                    << "    difference           = "
                    << format_restart_seconds(expected - meta.time_s) << " s (tolerance "
                    << format_restart_seconds(tolerance) << " s)\n"
                    << "  Neither value is silently preferred. If simulation.dt_s was changed on "
                       "purpose, set \"restart\": {\"ignore_stored_step\": true} to keep the stored "
                       "time and re-derive the step from the new dt.";
                return fail(oss.str());
            }

            result.ok = true;
            result.time_s = meta.time_s;
            result.step = meta.step;
            result.source = RestartTimeSource::FileTimeAndStep;
            return result;
        }

        result.ok = true;
        result.time_s = meta.time_s;
        result.step = derive_step(meta.time_s);
        result.source = RestartTimeSource::FileTimeDerivedStep;
        return result;
    }

    if (meta.has_step) {
        if (meta.step < 0) {
            std::ostringstream oss;
            oss << "restart step stored in '" << source_file << "' is negative ("
                << meta.step << ").";
            return fail(oss.str());
        }
        std::ostringstream oss;
        oss << "[WARNING] Restart file '" << source_file
            << "' stores model_step but no model time. Using time = step * dt = "
            << format_restart_seconds(static_cast<double>(meta.step) * policy.dt_s) << " s.";
        result.warnings.push_back(oss.str());
        result.ok = true;
        result.step = meta.step;
        result.time_s = static_cast<double>(meta.step) * policy.dt_s;
        result.source = RestartTimeSource::FileStepDerivedTime;
        return result;
    }

    // No metadata in the file. Everything below is an explicit, visible opt-in.
    if (policy.has_legacy_time) {
        const std::string time_error =
            validate_time(policy.legacy_time_s, "restart.legacy_time_s");
        if (!time_error.empty()) return fail(time_error);

        std::ostringstream oss;
        oss << "[WARNING] Restart file contains no model-time metadata.\n"
            << "Using restart.legacy_time_s=" << format_restart_seconds(policy.legacy_time_s) << ".";
        result.warnings.push_back(oss.str());

        result.ok = true;
        result.time_s = policy.legacy_time_s;
        result.step = derive_step(policy.legacy_time_s);
        result.source = RestartTimeSource::LegacyConfig;
        return result;
    }

    if (policy.allow_filename_fallback) {
        long long index = 0;
        if (!parse_trailing_file_index(source_file, index)) {
            return fail("restart.allow_filename_time_fallback is enabled but '" + source_file +
                        "' has no trailing digits to derive a restart time from.");
        }
        if (index < 0) {
            std::ostringstream oss;
            oss << "restart file name index parsed from '" << source_file
                << "' is negative (" << index << ").";
            return fail(oss.str());
        }
        if (!std::isfinite(policy.filename_interval_s) || policy.filename_interval_s <= 0.0) {
            return fail("restart.allow_filename_time_fallback is enabled but "
                        "restart.file_interval_s is not a positive, finite number (" +
                        format_restart_seconds(policy.filename_interval_s) + ").");
        }

        const double time_s = static_cast<double>(index) * policy.filename_interval_s;
        const std::string time_error = validate_time(time_s, "the file name");
        if (!time_error.empty()) return fail(time_error);

        std::ostringstream oss;
        oss << "[WARNING] Restart file contains no model-time metadata and "
               "restart.allow_filename_time_fallback is enabled.\n"
            << "[WARNING] The restart time is being taken from the DIGITS IN THE FILE NAME: "
            << index << " * " << format_restart_seconds(policy.filename_interval_s) << " s = "
            << format_restart_seconds(time_s) << " s.\n"
            << "[WARNING] Renaming '" << source_file
            << "' will change the simulation clock. Prefer restart.legacy_time_s.";
        result.warnings.push_back(oss.str());

        result.ok = true;
        result.time_s = time_s;
        result.step = derive_step(time_s);
        result.source = RestartTimeSource::Filename;
        return result;
    }

    return fail(
        "Restart file '" + source_file +
        "' contains no model-time metadata (no 'model_time_s', 'model_step' or elapsed-seconds "
        "'time').\n"
        "  Files written by this model carry it; older files do not. Choose one:\n"
        "    \"restart\": { \"legacy_time_s\": <elapsed seconds> }   (recommended)\n"
        "    \"restart\": { \"allow_filename_time_fallback\": true } (legacy: index * "
        "restart.file_interval_s, changes with the file name)");
}

} // namespace Utils
} // namespace VVM

#endif // VVM_UTILS_RESTART_TIME_HPP
