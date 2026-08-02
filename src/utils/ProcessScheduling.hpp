#ifndef VVM_UTILS_PROCESS_SCHEDULING_HPP
#define VVM_UTILS_PROCESS_SCHEDULING_HPP

#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <string>

namespace VVM {
namespace Utils {

// Scheduling for processes that are called less often than every time step
// (radiation, surface). The intended convention is:
//
//     a process with an interval of N steps runs at absolute steps 0, N, 2N, ...
//
// Step 0 is always a compute step. That is not cosmetic: these processes own
// fields (surface fluxes, radiative heating) that the tendency code reads on
// every step, including the first. Phasing them to start at step 1 leaves those
// fields at their zero-initialised values for the whole first step.
//
// The phase is on the absolute step counter, so a restart at step S resumes the
// same schedule the run would have had without the restart.
//
// Two conventions are implemented here, and the two call sites disagree today:
//
//     radiation  -> is_process_step()                 0, N, 2N, ...  (correct)
//     surface    -> is_legacy_surface_compute_step()  1, N+1, ...    (v1.0.0)
//
// That is a deliberate freeze, not an oversight. See the FIXME below.

// FIXME(surface-scheduling): the surface process does NOT use is_process_step()
// yet -- it still calls is_legacy_surface_compute_step() below, which reproduces
// the off-by-one phase shipped in v1.0.0. That is deliberate and temporary: the
// v1.0.0 numbers are the ones in the paper under review, and the stored
// bit-for-bit references in tests/references/ were generated with it. Switching
// the call site in Model.cpp to is_process_step() is the real fix; it changes
// results for every case with frequency_s > dt_s (all seven surface cases, by
// ~1e-6 relative in the surface fields), so it needs the references regenerated
// in the same commit. Do it once the paper is out.

// True when `step` is a compute step for a process with this interval.
// This is the correct convention. Radiation already uses it.
inline bool is_process_step(std::size_t step, int interval_in_steps) {
    // An interval that is not strictly positive must never reach the modulo:
    // 0 is a division by zero, and a negative int converts to a huge value in
    // the unsigned arithmetic below. interval_steps_from_frequency() already
    // rejects both at construction, so this guard only matters if a process is
    // ever scheduled without going through it. Falling back to "every step" is
    // the safe direction -- it can cost time, but it cannot skip physics.
    //
    // Interval 1 takes the same branch because every step is a compute step.
    if (interval_in_steps <= 1) return true;
    return step % static_cast<std::size_t>(interval_in_steps) == 0;
}

// FIXME: (surface-scheduling): frozen bug-compatible schedule -- do not "fix" this
// function, replace its callers with is_process_step(). See the note above.
//
// The v1.0.0 surface schedule, written inline in Model.cpp as
//
//     (state_.get_step() - 1) % surface_process_steps_ == 0
//
// get_step() returns size_t, so at step 0 the subtraction wraps to SIZE_MAX and
// the first compute step depends on the interval: SIZE_MAX % N is 0 for N of
// 3, 5 and 15 (step 0 computes by accident) but 15 for N of 24 and 120 (step 0
// is skipped, and the whole first step runs on the zero-initialised sfc_flux_*
// fields). After step 0 the schedule is 1, N+1, 2N+1, ... instead of 0, N, 2N.
//
// Kept as a named function so the behaviour is reproduced on purpose rather
// than re-derived from an expression that looks like a typo.
inline bool is_legacy_surface_compute_step(std::size_t step, int interval_in_steps) {
    // Same non-positive guard as is_process_step(): unreachable for a validated
    // interval, and interval 1 computes every step under both conventions, so
    // this does not change any v1.0.0 result.
    if (interval_in_steps <= 1) return true;
    return (step - 1) % static_cast<std::size_t>(interval_in_steps) == 0;
}

// Convert a calling frequency in seconds into a whole number of time steps.
// Rejects frequencies that are not a positive whole multiple of dt, so the
// interval handed to is_process_step() is always >= 1.
inline int interval_steps_from_frequency(double frequency_s, double dt_s,
                                         const std::string& process_name) {
    const double epsilon = 1e-6;

    if (!(frequency_s > 0.0)) {
        throw std::runtime_error("Error: " + process_name +
                                 " calling frequency must be positive, got " +
                                 std::to_string(frequency_s) + " s.");
    }

    const double remainder = std::fmod(frequency_s, dt_s);
    if (remainder > epsilon && (dt_s - remainder) > epsilon) {
        throw std::runtime_error("Error: " + process_name +
                                 " calling frequency can't be evenly divided by dt.");
    }

    const int steps = static_cast<int>(std::round(frequency_s / dt_s));
    if (steps < 1) {
        // frequency_s > 0 but under half a time step: it divides dt to within
        // epsilon yet rounds to zero.
        throw std::runtime_error("Error: " + process_name +
                                 " calling frequency (" + std::to_string(frequency_s) +
                                 " s) is shorter than one time step (" +
                                 std::to_string(dt_s) + " s).");
    }
    return steps;
}

} // namespace Utils
} // namespace VVM

#endif // VVM_UTILS_PROCESS_SCHEDULING_HPP
