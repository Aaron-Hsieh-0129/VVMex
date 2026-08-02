// Unit tests for restart-clock recovery (src/utils/RestartTime.hpp).
//
// The restart time used to be reconstructed from the digits in the restart file
// name: index * restart.file_interval_s. Renaming a file therefore moved
// state.time, state.step, and with them the radiation, surface-process and
// output schedules -- silently. These tests pin the replacement:
//
//   1. The metadata stored inside the file decides. The same metadata resolves
//      to the same time and step no matter what the file is called, and beats a
//      file name whose digits say something else.
//   2. A stored time and a stored step must agree to within a serialization
//      tolerance. When they do not, the run stops and says so; neither value is
//      silently preferred.
//   3. A file with no metadata is only usable through an explicit, visible
//      opt-in (restart.legacy_time_s, or the old file-name behaviour behind
//      restart.allow_filename_time_fallback), and both of them warn.
//
// Header-only and free of Kokkos/MPI/HDF5, so this runs on any machine -- no
// GPU, no ranks, no simulation. tests/unit/test_restart_metadata_io.cpp covers
// getting the numbers out of real HDF5 and NetCDF files.
//
// Reporting goes through <cstdio> rather than <iostream> on purpose: the NVHPC
// build compiles against the system GCC 13 headers while LD_LIBRARY_PATH puts
// GCC 11's libstdc++ first at run time, and that mismatch faults inside the
// ostream sentry.

#include "utils/RestartTime.hpp"

#include <cmath>
#include <cstdio>
#include <limits>
#include <string>

using VVM::Utils::parse_trailing_file_index;
using VVM::Utils::resolve_restart_time;
using VVM::Utils::RestartFileMetadata;
using VVM::Utils::RestartTimePolicy;
using VVM::Utils::RestartTimeResolution;
using VVM::Utils::RestartTimeSource;

namespace {

int failures = 0;

void check(bool condition, const std::string& what) {
    if (!condition) {
        std::fprintf(stderr, "FAIL: %s\n", what.c_str());
        ++failures;
    }
}

RestartFileMetadata time_and_step(double time_s, long long step) {
    RestartFileMetadata m;
    m.has_time = true;
    m.time_s = time_s;
    m.has_step = true;
    m.step = step;
    return m;
}

RestartFileMetadata time_only(double time_s) {
    RestartFileMetadata m;
    m.has_time = true;
    m.time_s = time_s;
    return m;
}

RestartFileMetadata step_only(long long step) {
    RestartFileMetadata m;
    m.has_step = true;
    m.step = step;
    return m;
}

RestartTimePolicy policy_with_dt(double dt_s) {
    RestartTimePolicy p;
    p.dt_s = dt_s;
    return p;
}

bool warns(const RestartTimeResolution& r) { return !r.warnings.empty(); }

std::string all_warnings(const RestartTimeResolution& r) {
    std::string joined;
    for (const auto& w : r.warnings) joined += w + "\n";
    return joined;
}

bool contains(const std::string& haystack, const std::string& needle) {
    return haystack.find(needle) != std::string::npos;
}

void check_resolved(const RestartTimeResolution& r, double time_s, long long step,
                    RestartTimeSource source, const std::string& what) {
    if (!r.ok) {
        std::fprintf(stderr, "FAIL: %s (rejected: %s)\n", what.c_str(), r.error.c_str());
        ++failures;
        return;
    }
    check(r.time_s == time_s, what + ": time " + std::to_string(r.time_s) +
                                  " != " + std::to_string(time_s));
    check(r.step == step, what + ": step " + std::to_string(r.step) +
                              " != " + std::to_string(step));
    check(r.source == source, what + ": unexpected source '" +
                                  VVM::Utils::to_string(r.source) + "'");
}

void check_rejected(const RestartTimeResolution& r, const std::string& what) {
    if (r.ok) {
        std::fprintf(stderr, "FAIL: %s (accepted: time=%.17g step=%lld)\n", what.c_str(),
                     r.time_s, r.step);
        ++failures;
    }
}

// --- 1. a file that stores both time and step ------------------------------
void test_time_and_step() {
    const auto r = resolve_restart_time(time_and_step(7200.0, 3600),
                                        policy_with_dt(2.0), "vvm_output_000002.h5");
    check_resolved(r, 7200.0, 3600, RestartTimeSource::FileTimeAndStep,
                   "time + step are taken from the file");
    check(!warns(r), "a consistent time/step pair warns about nothing");
}

// --- 2. a file that stores time but no step --------------------------------
void test_time_without_step() {
    const auto r = resolve_restart_time(time_only(7200.0), policy_with_dt(2.0),
                                        "vvm_output_000002.h5");
    check_resolved(r, 7200.0, 3600, RestartTimeSource::FileTimeDerivedStep,
                   "step is derived from time / dt when the file has no step");
    check(!warns(r), "an exactly divisible time / dt warns about nothing");
}

// A file with only a step is not something this model writes, but the step is
// still authoritative over anything outside the file.
void test_step_without_time() {
    const auto r = resolve_restart_time(step_only(1800), policy_with_dt(4.0),
                                        "vvm_output_000002.h5");
    check_resolved(r, 7200.0, 1800, RestartTimeSource::FileStepDerivedTime,
                   "time is derived from step * dt when the file has no time");
    check(warns(r), "deriving the time from the step is worth a warning");
}

// --- 3. renaming the file changes nothing ----------------------------------
void test_rename_does_not_change_the_clock() {
    const RestartFileMetadata meta = time_and_step(7200.0, 3600);
    const RestartTimePolicy policy = policy_with_dt(2.0);

    const char* names[] = {
        "vvm_output_000002.h5",
        "vvm_output_000999.h5",
        "renamed.h5",
        "/some/other/dir/copy_of_the_restart.h5",
        "no_digits_at_all.h5",
    };

    for (const char* name : names) {
        const auto r = resolve_restart_time(meta, policy, name);
        check_resolved(r, 7200.0, 3600, RestartTimeSource::FileTimeAndStep,
                       std::string("renaming to '") + name + "' does not move the clock");
    }

    // Same, for a file carrying only the time.
    for (const char* name : names) {
        const auto r = resolve_restart_time(time_only(7200.0), policy, name);
        check_resolved(r, 7200.0, 3600, RestartTimeSource::FileTimeDerivedStep,
                       std::string("renaming to '") + name + "' does not move a time-only clock");
    }
}

// --- 4 & 5. file metadata beats the file name ------------------------------
void test_file_metadata_beats_the_filename() {
    RestartTimePolicy policy = policy_with_dt(2.0);
    // The old behaviour would have read 42 * 600 = 25200 s out of this name.
    policy.allow_filename_fallback = true;
    policy.filename_interval_s = 600.0;

    const auto r = resolve_restart_time(time_and_step(7200.0, 3600), policy,
                                        "vvm_output_000042.h5");
    check_resolved(r, 7200.0, 3600, RestartTimeSource::FileTimeAndStep,
                   "file metadata wins over a conflicting file-name index");
    check(!warns(r), "no file-name warning when the file answered the question");
}

// --- 6. odd and even step counts -------------------------------------------
// The step's parity picks the AB2 history slot, so an off-by-one here is not
// cosmetic.
void test_odd_and_even_steps() {
    check_resolved(resolve_restart_time(time_and_step(7200.0, 3600), policy_with_dt(2.0), "r.h5"),
                   7200.0, 3600, RestartTimeSource::FileTimeAndStep, "even stored step");
    check_resolved(resolve_restart_time(time_and_step(7202.0, 3601), policy_with_dt(2.0), "r.h5"),
                   7202.0, 3601, RestartTimeSource::FileTimeAndStep, "odd stored step");

    const auto even_derived =
        resolve_restart_time(time_only(7200.0), policy_with_dt(2.0), "r.h5");
    check_resolved(even_derived, 7200.0, 3600, RestartTimeSource::FileTimeDerivedStep,
                   "even derived step");
    const auto odd_derived =
        resolve_restart_time(time_only(7202.0), policy_with_dt(2.0), "r.h5");
    check_resolved(odd_derived, 7202.0, 3601, RestartTimeSource::FileTimeDerivedStep,
                   "odd derived step");
}

// --- 7. time / dt is not an integer, and no step is stored -----------------
// The time is authoritative and is kept exactly; the derived step is the nearest
// one, and the mismatch is reported rather than hidden.
void test_non_integer_time_over_dt() {
    const auto r = resolve_restart_time(time_only(7201.0), policy_with_dt(3.0), "r.h5");
    check_resolved(r, 7201.0, 2400, RestartTimeSource::FileTimeDerivedStep,
                   "a non-integer time / dt keeps the stored time");
    check(warns(r), "a non-integer time / dt warns");
    check(contains(all_warnings(r), "not an integer multiple"),
          "the non-integer warning says what is wrong");
}

// --- 8. stored time and stored step disagree -------------------------------
void test_inconsistent_time_and_step() {
    // 3600 steps at dt = 2 s is 7200 s, not 3600 s.
    const auto r = resolve_restart_time(time_and_step(3600.0, 3600), policy_with_dt(2.0),
                                        "vvm_output_000002.h5");
    check_rejected(r, "an inconsistent time/step pair is refused");

    check(contains(r.error, "3600"), "the message quotes the stored step");
    check(contains(r.error, "stored time"), "the message labels the stored time");
    check(contains(r.error, "stored step"), "the message labels the stored step");
    check(contains(r.error, "configured dt"), "the message quotes the configured dt");
    check(contains(r.error, "expected time"), "the message gives the expected time");
    check(contains(r.error, "vvm_output_000002.h5"), "the message names the source file");

    // One step of drift is caught, not rounded away.
    check_rejected(resolve_restart_time(time_and_step(7202.0, 3600), policy_with_dt(2.0), "r.h5"),
                   "a one-step disagreement is refused");

    // Serialization noise is not a disagreement: a time round-tripped through
    // 32-bit float still matches its step.
    const auto float_noise = resolve_restart_time(
        time_and_step(static_cast<double>(static_cast<float>(7200.0)), 3600),
        policy_with_dt(2.0), "r.h5");
    check(float_noise.ok, "float round-tripping does not count as a disagreement");

    // With restart.ignore_stored_step the stored time wins and the step is
    // re-derived -- the documented escape when dt changed on purpose.
    RestartTimePolicy policy = policy_with_dt(4.0);
    policy.ignore_stored_step = true;
    const auto ignored = resolve_restart_time(time_and_step(7200.0, 3600), policy, "r.h5");
    check_resolved(ignored, 7200.0, 1800, RestartTimeSource::FileTimeDerivedStep,
                   "ignore_stored_step re-derives the step from the new dt");
    check(warns(ignored), "discarding the stored step warns");
}

// --- 9 & 10. times that cannot describe a clock ----------------------------
void test_invalid_times() {
    check_rejected(resolve_restart_time(time_only(-1.0), policy_with_dt(2.0), "r.h5"),
                   "a negative stored time is refused");
    check_rejected(resolve_restart_time(time_and_step(-7200.0, -3600), policy_with_dt(2.0), "r.h5"),
                   "a negative stored time/step pair is refused");
    check_rejected(resolve_restart_time(time_and_step(7200.0, -3600), policy_with_dt(2.0), "r.h5"),
                   "a negative stored step is refused");
    check_rejected(resolve_restart_time(step_only(-1), policy_with_dt(2.0), "r.h5"),
                   "a negative step-only file is refused");

    const double nan_value = std::numeric_limits<double>::quiet_NaN();
    const double inf_value = std::numeric_limits<double>::infinity();
    check_rejected(resolve_restart_time(time_only(nan_value), policy_with_dt(2.0), "r.h5"),
                   "a NaN stored time is refused");
    check_rejected(resolve_restart_time(time_only(inf_value), policy_with_dt(2.0), "r.h5"),
                   "an infinite stored time is refused");

    // dt itself has to be usable before a step can mean anything.
    check_rejected(resolve_restart_time(time_only(7200.0), policy_with_dt(0.0), "r.h5"),
                   "dt = 0 is refused");
    check_rejected(resolve_restart_time(time_only(7200.0), policy_with_dt(-2.0), "r.h5"),
                   "a negative dt is refused");
    check_rejected(resolve_restart_time(time_only(7200.0), policy_with_dt(nan_value), "r.h5"),
                   "a NaN dt is refused");

    // Zero is a legitimate restart time: a run resumed from its own step 0 dump.
    check_resolved(resolve_restart_time(time_and_step(0.0, 0), policy_with_dt(2.0), "r.h5"),
                   0.0, 0, RestartTimeSource::FileTimeAndStep, "time 0 / step 0 is valid");
}

// --- 11. no metadata, but restart.legacy_time_s is given -------------------
void test_legacy_time_s() {
    RestartTimePolicy policy = policy_with_dt(2.0);
    policy.has_legacy_time = true;
    policy.legacy_time_s = 7200.0;

    const auto r = resolve_restart_time(RestartFileMetadata{}, policy, "old_restart.h5");
    check_resolved(r, 7200.0, 3600, RestartTimeSource::LegacyConfig,
                   "restart.legacy_time_s carries a file with no metadata");
    check(contains(all_warnings(r), "no model-time metadata"),
          "the legacy path says the file had no metadata");
    check(contains(all_warnings(r), "legacy_time_s"),
          "the legacy warning names the configuration key it used");

    // It is a fallback, not an override: a file that knows its own time is not
    // second-guessed by the configuration.
    const auto from_file = resolve_restart_time(time_and_step(600.0, 300), policy, "r.h5");
    check_resolved(from_file, 600.0, 300, RestartTimeSource::FileTimeAndStep,
                   "legacy_time_s does not override metadata stored in the file");

    // The legacy value is validated like any other time.
    policy.legacy_time_s = -1.0;
    check_rejected(resolve_restart_time(RestartFileMetadata{}, policy, "old_restart.h5"),
                   "a negative restart.legacy_time_s is refused");
}

// --- 12. no metadata and no explicit fallback ------------------------------
void test_missing_metadata_without_fallback() {
    const auto r = resolve_restart_time(RestartFileMetadata{}, policy_with_dt(2.0),
                                        "vvm_output_000002.h5");
    check_rejected(r, "a file with no metadata and no policy is refused");
    check(contains(r.error, "vvm_output_000002.h5"), "the message names the file");
    check(contains(r.error, "legacy_time_s"), "the message offers restart.legacy_time_s");
    check(contains(r.error, "allow_filename_time_fallback"),
          "the message mentions the file-name opt-in");
    check(!contains(r.error, "2 * 3600"), "the message does not quietly compute a file-name time");
}

// --- 13. the file-name fallback, switched on explicitly --------------------
void test_filename_fallback() {
    RestartTimePolicy policy = policy_with_dt(2.0);
    policy.allow_filename_fallback = true;
    policy.filename_interval_s = 3600.0;

    const auto r = resolve_restart_time(RestartFileMetadata{}, policy, "vvm_output_000002.h5");
    check_resolved(r, 7200.0, 3600, RestartTimeSource::Filename,
                   "the enabled file-name fallback reproduces the old index * interval");
    check(warns(r), "the file-name fallback warns");
    check(contains(all_warnings(r), "FILE NAME"), "the warning is prominent about the file name");
    check(contains(all_warnings(r), "Renaming"), "the warning says renaming will change the clock");

    // Which is exactly the hazard: with the fallback on, the name *is* the clock.
    const auto renamed = resolve_restart_time(RestartFileMetadata{}, policy, "vvm_output_000003.h5");
    check_resolved(renamed, 10800.0, 5400, RestartTimeSource::Filename,
                   "with the fallback on, the file name still decides");

    // legacy_time_s takes precedence over the file name when both are offered.
    policy.has_legacy_time = true;
    policy.legacy_time_s = 60.0;
    check_resolved(resolve_restart_time(RestartFileMetadata{}, policy, "vvm_output_000002.h5"),
                   60.0, 30, RestartTimeSource::LegacyConfig,
                   "restart.legacy_time_s outranks the file-name fallback");

    // A name with no digits cannot be guessed at.
    policy.has_legacy_time = false;
    check_rejected(resolve_restart_time(RestartFileMetadata{}, policy, "restart.h5"),
                   "the file-name fallback refuses a name with no digits");

    // Nor can a nonsensical interval.
    policy.filename_interval_s = 0.0;
    check_rejected(resolve_restart_time(RestartFileMetadata{}, policy, "vvm_output_000002.h5"),
                   "the file-name fallback refuses a non-positive file_interval_s");
}

void test_parse_trailing_file_index() {
    long long index = -1;

    check(parse_trailing_file_index("vvm_output_000002.h5", index) && index == 2,
          "trailing index of vvm_output_000002.h5");
    check(parse_trailing_file_index("/a/b/vvm_output_000144.h5", index) && index == 144,
          "trailing index survives a directory path");
    check(parse_trailing_file_index("vvm_output_000002", index) && index == 2,
          "trailing index without an extension");
    check(parse_trailing_file_index("run12_output_7.nc", index) && index == 7,
          "only the last run of digits counts");
    check(!parse_trailing_file_index("restart.h5", index), "no digits, no index");
    check(!parse_trailing_file_index("", index), "an empty name has no index");
    // Kept bug-compatible with the original parser: it scans backwards past the
    // separator, so a digit in a *directory* name becomes the index. One more
    // reason the fallback is opt-in.
    check(parse_trailing_file_index("run2/restart.h5", index) && index == 2,
          "digits in a directory name are still picked up (legacy behaviour)");
}

} // namespace

int main() {
    test_time_and_step();
    test_time_without_step();
    test_step_without_time();
    test_rename_does_not_change_the_clock();
    test_file_metadata_beats_the_filename();
    test_odd_and_even_steps();
    test_non_integer_time_over_dt();
    test_inconsistent_time_and_step();
    test_invalid_times();
    test_legacy_time_s();
    test_missing_metadata_without_fallback();
    test_filename_fallback();
    test_parse_trailing_file_index();

    if (failures != 0) {
        std::fprintf(stderr, "test_restart_time: %d check(s) failed\n", failures);
        return 1;
    }
    std::printf("test_restart_time: all checks passed\n");
    return 0;
}
