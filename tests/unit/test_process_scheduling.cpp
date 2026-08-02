// Unit tests for VVM::Utils process scheduling (src/utils/ProcessScheduling.hpp).
//
// Two conventions live in that header on purpose, and this file pins both:
//
//   is_process_step()                 0, N, 2N, ...  -- correct; radiation uses it
//   is_legacy_surface_compute_step()  1, N+1, ...    -- v1.0.0; the surface uses it
//
// The surface process is deliberately left on the legacy phase so results stay
// reproducible for the paper under review. See the FIXME(surface-scheduling)
// notes in ProcessScheduling.hpp and Model.cpp. The legacy tests below are not
// endorsing that schedule -- they describe exactly what is frozen, so that
// flipping the call site later is a visible, deliberate change rather than a
// silent drift in the numbers.
//
// Header-only and free of Kokkos/MPI, so this runs on any machine -- no GPU,
// no ranks, no simulation.
//
// Reporting goes through <cstdio> rather than <iostream> on purpose. The NVHPC
// build compiles against the system GCC 13 headers while LD_LIBRARY_PATH puts
// GCC 11's libstdc++ first at run time, and that mismatch faults inside the
// ostream sentry. C stdio does not touch it.

#include "utils/ProcessScheduling.hpp"

#include <cstddef>
#include <cstdio>
#include <limits>
#include <string>
#include <vector>

namespace {

int failures = 0;

void check(bool condition, const std::string& what) {
    if (!condition) {
        std::fprintf(stderr, "FAIL: %s\n", what.c_str());
        ++failures;
    }
}

void check_throws(void (*fn)(), const std::string& what) {
    try {
        fn();
    } catch (const std::runtime_error&) {
        return;
    }
    std::fprintf(stderr, "FAIL: %s (expected a runtime_error)\n", what.c_str());
    ++failures;
}

using VVM::Utils::interval_steps_from_frequency;
using VVM::Utils::is_legacy_surface_compute_step;
using VVM::Utils::is_process_step;

// The intervals named in the bug report, plus 1 and 2. 3, 5 and 15 matter
// because SIZE_MAX % N == 0 for them, which is why the legacy phase looks
// interval-dependent instead of uniformly wrong. 24 is rcemip (dt 5 s), 120 is
// every other physics case (dt 1 s).
const std::vector<int> kIntervals = {1, 2, 3, 5, 15, 24, 120};

// ---------------------------------------------------------------------------
// The correct convention: 0, N, 2N, ...
// ---------------------------------------------------------------------------

// Step 0 is a compute step for every interval. This is the property the surface
// process needs and does not yet have: calculate_tendencies() reads
// sfc_flux_th / sfc_flux_qv / sfc_flux_u / sfc_flux_v, and compute_coefficients()
// is the only thing that ever writes them, so a skipped step 0 means the whole
// first step runs on the zero-initialised fields from SurfaceProcess::initialize().
void test_step_zero_always_computes() {
    for (int n : kIntervals) {
        check(is_process_step(0, n),
              "step 0 must be a compute step for interval " + std::to_string(n));
    }
    // The reported configuration: dt_s 1.0, frequency_s 120.
    const int n = interval_steps_from_frequency(120.0, 1.0, "surface process");
    check(n == 120, "120 s at dt 1 s is 120 steps");
    check(is_process_step(0, n), "step 0 computes at a 120-step interval");
}

// Interval 1 means every step, including step 1.
void test_interval_one_runs_every_step() {
    for (std::size_t step = 0; step < 10; ++step) {
        check(is_process_step(step, 1),
              "interval 1 computes at step " + std::to_string(step));
    }
    check(is_process_step(1, 1), "step 1 computes at a 1-step interval");
}

// The convention itself: compute at 0, N, 2N, ... and nowhere else.
void test_multiples_only() {
    for (int n : kIntervals) {
        for (std::size_t step = 0; step < 500; ++step) {
            const bool expected = (step % static_cast<std::size_t>(n) == 0);
            check(is_process_step(step, n) == expected,
                  "interval " + std::to_string(n) + " at step " + std::to_string(step) +
                      ": expected " + (expected ? "compute" : "skip"));
        }
        // ... and far from the origin, where an int/size_t mix-up would show.
        for (std::size_t k = 1; k <= 4; ++k) {
            const std::size_t big = 1000000 * k;
            check(is_process_step(big * static_cast<std::size_t>(n), n),
                  "interval " + std::to_string(n) + " computes at a large multiple");
        }
    }

    // Nothing wraps at the top of the step counter either.
    check(is_process_step(std::numeric_limits<std::size_t>::max(), 1),
          "interval 1 computes at the maximum step");
}

// ---------------------------------------------------------------------------
// The frozen v1.0.0 surface convention. These tests describe the bug; they must
// be deleted, not "fixed", when the surface call site moves to is_process_step()
// and the references are regenerated.
// ---------------------------------------------------------------------------

void test_legacy_phase_is_reproduced() {
    // Step 0 is skipped at the intervals every shipped physics case uses.
    check(!is_legacy_surface_compute_step(0, 120),
          "v1.0.0 skips step 0 at a 120-step interval (dt 1 s, frequency 120 s)");
    check(!is_legacy_surface_compute_step(0, 24),
          "v1.0.0 skips step 0 at a 24-step interval (rcemip, dt 5 s)");

    // ... but computes at step 0 for the intervals where SIZE_MAX % N == 0.
    for (int n : {3, 5, 15}) {
        check(is_legacy_surface_compute_step(0, n),
              "v1.0.0 computes at step 0 for interval " + std::to_string(n) +
                  " (SIZE_MAX % N == 0)");
    }

    // The phase after step 0: 1, N+1, 2N+1, ...
    for (int n : {3, 5, 15, 24, 120}) {
        const std::size_t un = static_cast<std::size_t>(n);
        for (std::size_t k = 0; k < 4; ++k) {
            check(is_legacy_surface_compute_step(1 + k * un, n),
                  "v1.0.0 computes at step " + std::to_string(1 + k * un) +
                      " for interval " + std::to_string(n));
        }
        check(!is_legacy_surface_compute_step(un, n),
              "v1.0.0 skips step N for interval " + std::to_string(n));
    }
}

// The one case where the two conventions agree. Any config with
// frequency_s == dt_s is unaffected by the pending fix.
void test_conventions_agree_at_interval_one() {
    for (std::size_t step = 0; step < 100; ++step) {
        check(is_legacy_surface_compute_step(step, 1) == is_process_step(step, 1),
              "the conventions agree at interval 1, step " + std::to_string(step));
    }
}

// What the pending fix will change, stated as a test so the diff in results is
// predictable before anyone regenerates a reference.
void test_pending_fix_changes_only_the_phase() {
    for (int n : {3, 5, 15, 24, 120}) {
        int disagreements = 0;
        for (std::size_t step = 0; step < 1000; ++step) {
            if (is_process_step(step, n) != is_legacy_surface_compute_step(step, n)) {
                ++disagreements;
            }
        }
        check(disagreements > 0,
              "the conventions differ for interval " + std::to_string(n));
    }
    // Over a horizon that is a whole number of intervals the fix is a pure phase
    // shift -- same number of surface updates -- EXCEPT for the intervals where
    // the wrapped step 0 fires accidentally, where v1.0.0 does one extra update.
    // 12000 is divisible by every interval below.
    const std::size_t horizon = 12000;
    for (int n : {3, 5, 15, 24, 120}) {
        int fixed = 0, legacy = 0;
        for (std::size_t step = 0; step < horizon; ++step) {
            if (is_process_step(step, n)) ++fixed;
            if (is_legacy_surface_compute_step(step, n)) ++legacy;
        }
        const int accidental = is_legacy_surface_compute_step(0, n) ? 1 : 0;
        check(legacy == fixed + accidental,
              "over " + std::to_string(horizon) + " steps at interval " +
                  std::to_string(n) + ", v1.0.0 does " + std::to_string(accidental) +
                  " extra update(s): expected " + std::to_string(fixed + accidental) +
                  ", got " + std::to_string(legacy));
        check(fixed == static_cast<int>(horizon) / n,
              "the fixed convention fires horizon/N times at interval " + std::to_string(n));
    }
}

// ---------------------------------------------------------------------------
// Interval validation -- shared by both conventions.
// ---------------------------------------------------------------------------

// A zero or negative interval must never reach the modulo, on either path.
// Configuration validation rejects both, and the helpers fall back to
// "every step" rather than dividing by zero if ever handed one anyway.
void test_no_modulo_by_zero() {
    check(is_process_step(0, 0), "interval 0 falls back to every step");
    check(is_process_step(7, 0), "interval 0 falls back to every step (nonzero step)");
    check(is_process_step(7, -1), "negative interval falls back to every step");
    check(is_process_step(7, -120), "negative interval falls back to every step");
    check(is_process_step(std::numeric_limits<std::size_t>::max(), 0),
          "interval 0 is safe at the maximum step");

    check(is_legacy_surface_compute_step(0, 0), "legacy: interval 0 is safe");
    check(is_legacy_surface_compute_step(7, 0), "legacy: interval 0 is safe (nonzero step)");
    check(is_legacy_surface_compute_step(7, -120), "legacy: negative interval is safe");
    check(is_legacy_surface_compute_step(std::numeric_limits<std::size_t>::max(), 0),
          "legacy: interval 0 is safe at the maximum step");
}

// interval_steps_from_frequency() is the only path that produces an interval,
// and it never returns anything below 1.
void test_frequency_conversion() {
    struct Case { double freq_s; double dt_s; int steps; };
    const std::vector<Case> ok = {
        {1.0, 1.0, 1}, {120.0, 1.0, 120}, {3.0, 1.0, 3}, {15.0, 1.0, 15},
        {120.0, 5.0, 24}, {120.0, 8.0, 15}, {600.0, 5.0, 120}, {0.5, 0.5, 1},
    };
    for (const auto& c : ok) {
        const int steps = interval_steps_from_frequency(c.freq_s, c.dt_s, "test process");
        check(steps == c.steps,
              "frequency " + std::to_string(c.freq_s) + " s at dt " +
                  std::to_string(c.dt_s) + " s is " + std::to_string(c.steps) + " steps, got " +
                  std::to_string(steps));
        check(steps >= 1, "a converted interval is never below 1");
    }

    // frequency 0 used to pass the divisibility check (fmod(0, dt) == 0) and
    // round to 0 steps -- a modulo by zero on the first step.
    check_throws([] { interval_steps_from_frequency(0.0, 1.0, "test process"); },
                 "frequency 0 s is rejected");
    check_throws([] { interval_steps_from_frequency(-120.0, 1.0, "test process"); },
                 "a negative frequency is rejected");
    // Positive but under half a step: divides dt to within epsilon, rounds to 0.
    check_throws([] { interval_steps_from_frequency(1e-9, 1.0, "test process"); },
                 "a frequency shorter than one step is rejected");
    // Unchanged from v1.0.0: not a whole multiple of dt.
    check_throws([] { interval_steps_from_frequency(1.5, 1.0, "test process"); },
                 "a frequency that does not divide dt is rejected");
}

} // namespace

int main() {
    test_step_zero_always_computes();
    test_interval_one_runs_every_step();
    test_multiples_only();
    test_legacy_phase_is_reproduced();
    test_conventions_agree_at_interval_one();
    test_pending_fix_changes_only_the_phase();
    test_no_modulo_by_zero();
    test_frequency_conversion();

    if (failures != 0) {
        std::fprintf(stderr, "%d check(s) failed\n", failures);
        return 1;
    }
    std::printf("process scheduling: all checks passed\n");
    return 0;
}
