// The VVMex <-> CLEO coupling, checked at the level of VVM_CLEO_Interface:
// what goes into CLEO's gridboxes, what comes back into State, and when.
//
//   test_cleo_coupling <config.json> <scenario>
//
// The configuration and CLEO's input binaries are made by
// tests/scripts/make_cleo_test_case.py from tests/configs/cleo_coupling.json;
// the scenario names one of its test_scenarios entries:
//
//   transport     condensation and collision off, haze everywhere. The
//                 thermodynamics must round-trip through CLEO unchanged, the
//                 bulk diagnostics must read "no cloud", halos must be fresh.
//   condensation  2% supersaturation, coupling every 2 model steps. Pins the
//                 cadence, water conservation and the latent heating, which
//                 between them pin the unit conversion in both directions.
//   average_y     average_over_y: every y column gets the column mean.
//   rain          large drops in the lowest level only, condensation off.
//                 Pins the cloud/rain split, the effective radii, the surface
//                 flux against the water that leaves the domain, and the
//                 accumulate/average/reset cycle of precip_liq_surf_mass.
//   rain_collision  the same with collisions on: collision-coalescence and
//                 breakup move water between superdroplets, never in or out.
//   advection     uniform diagonal wind, any number of ranks. Checks the water budget here; where every
//                 superdroplet went is checked from CLEO's zarr by
//                 tests/scripts/check_cleo_superdroplets.py, which also
//                 compares runs at different rank counts droplet by droplet.
//   average_y_split  average_over_y with y split across ranks must be refused:
//                 the y mean would cover only this rank's columns.
//
// The base state is written here rather than by Core::Initializer, so every
// value CLEO receives is known: th carries a horizontal pattern, which is what
// turns a gridbox-index mapping error into a failure rather than a no-op.
#include "core/Grid.hpp"
#include "core/haloexchange/HaloExchanger.hpp"
#include "core/Parameters.hpp"
#include "core/State.hpp"
#include "physics/cleo/VVM_cleo_process_interface.hpp"
#include "utils/ConfigurationManager.hpp"

#include <Kokkos_Core.hpp>
#include <mpi.h>

#include <cmath>
#include <cstdio>
#include <functional>
#include <map>
#include <sstream>
#include <string>
#include <vector>

#if defined(ENABLE_NCCL)
#include <cuda_runtime.h>
#include <nccl.h>
#endif

namespace {

// CLEO's own thermodynamic constants (CLEO libs/cleoconstants.hpp). The
// coupling hands CLEO pressure and temperature; what CLEO does with them, and
// so what comes back, follows these rather than VVMex's.
constexpr double LATENT_V = 2500930.0;
constexpr double CP_DRY = 1004.64;
constexpr double CP_V = 1865.01;
constexpr double C_L = 4192.664;
constexpr double RGAS_UNIV = 8.314462618;
constexpr double MR_WATER = 0.01801528;
constexpr double MR_DRY = 0.028966216;
constexpr double RGAS_DRY = RGAS_UNIV / MR_DRY;
constexpr double RGAS_V = RGAS_UNIV / MR_WATER;
constexpr double MR_RATIO = MR_WATER / MR_DRY;

// P3's values for an empty cell (p3_main_impl.hpp), which the coupling copies
// so that RRTMGP is handed the same thing by either scheme.
constexpr double EMPTY_CLOUD_RADIUS_UM = 10.0;
constexpr double EMPTY_RAIN_RADIUS_UM = 500.0;
constexpr double ICE_RADIUS_UM = 25.0;

int failures = 0;
int rank = 0;
int nranks = 1;

// Report only the first few failures of each kind: a mapping error fails every
// cell, and the first one says everything the next thousand would.
std::map<std::string, int> reported;

void
fail(const std::string& what, const std::string& detail = "") {
    ++failures;
    if (reported[what]++ < 3) {
        const std::string where = nranks > 1 ? "[rank " + std::to_string(rank) + "] " : "";
        std::fprintf(stderr, "%sFAIL: %s%s%s\n", where.c_str(), what.c_str(),
            detail.empty() ? "" : ": ", detail.c_str());
    }
}

void
check(bool condition, const std::string& what, const std::string& detail = "") {
    if (!condition) {
        fail(what, detail);
    }
}

std::string
at(int k, int j, int i, double got, double want) {
    std::ostringstream s;
    s.precision(17);
    s << "(k,j,i)=(" << k << "," << j << "," << i << ") got " << got << ", expected " << want;
    return s.str();
}

// For one-sided checks, where there is a bound rather than an expected value.
std::string
at_bound(int k, int j, int i, double got, const char* relation, double bound) {
    std::ostringstream s;
    s.precision(17);
    s << "(k,j,i)=(" << k << "," << j << "," << i << ") got " << got << ", needs " << relation
      << " " << bound;
    return s.str();
}

bool
close(double got, double want, double rtol, double atol = 0.0) {
    return std::abs(got - want) <= atol + rtol * std::abs(want);
}

// CLEO's saturation pressure over liquid [Pa] (thermodynamic_equations.cpp).
double
saturation_pressure(double temp_k) {
    return 611.655 * std::exp(17.4146 * (temp_k - 273.16) / (temp_k - 33.639));
}

// The vapour mixing ratio CLEO reads as saturation ratio `s` at (temp, press).
double
qv_at_saturation_ratio(double s, double temp_k, double press_pa) {
    const double pv = s * saturation_pressure(temp_k);
    return MR_RATIO * pv / (press_pa - pv);
}

// CLEO's dry-air density, the denominator of every mixing ratio it returns.
double
dry_air_density(double press_pa, double temp_k, double qv) {
    return press_pa / ((RGAS_DRY + RGAS_V * qv) * temp_k);
}

using Host3 = decltype(std::declval<const VVM::Core::Field<3>&>().get_host_data());
using Host2 = decltype(std::declval<const VVM::Core::Field<2>&>().get_host_data());
using Host1 = decltype(std::declval<const VVM::Core::Field<1>&>().get_host_data());

const char* const fields3[] = {"th", "qv", "qp", "qc", "qr", "nc", "nr", "qi",
    "diag_eff_radius_qc", "diag_eff_radius_qr", "diag_eff_radius_qi"};
const char* const fields2[] = {"precip_liq_surf_mass", "precip_liq_surf_flux",
    "precip_ice_surf_mass", "precip_ice_surf_flux"};

struct Snapshot {
    std::map<std::string, Host3> f3;
    std::map<std::string, Host2> f2;
};

Snapshot
take_snapshot(const VVM::Core::State& state) {
    Kokkos::fence();
    Snapshot s;
    for (const char* name : fields3) {
        s.f3.emplace(name, state.get_field<3>(name).get_host_data());
    }
    for (const char* name : fields2) {
        s.f2.emplace(name, state.get_field<2>(name).get_host_data());
    }
    return s;
}

struct Case {
    const VVM::Utils::ConfigurationManager& config;
    const VVM::Core::Grid& grid;
    VVM::Core::State& state;
    VVM::Physics::VVM_CLEO_Interface& cleo;
    int h, lnz, lny, lnx, nzt, nyt, nxt;
    double dx, dy, dz, dt;
    Host1 pbar, pibar;

    // Loops over the physical cells only: k, j, i in array (halo-inclusive)
    // indices.
    void
    for_physical(const std::function<void(int, int, int)>& f) const {
        for (int k = h; k < h + lnz; ++k) {
            for (int j = h; j < h + lny; ++j) {
                for (int i = h; i < h + lnx; ++i) {
                    f(k, j, i);
                }
            }
        }
    }

    void
    for_surface(const std::function<void(int, int)>& f) const {
        for (int j = h; j < h + lny; ++j) {
            for (int i = h; i < h + lnx; ++i) {
                f(j, i);
            }
        }
    }

    void
    step() {
        cleo.run(state, VVM::real(dt));
        Kokkos::fence();
        state.advance_time(VVM::real(dt));
    }

    double
    temperature(const Snapshot& s, int k, int j, int i) const {
        return s.f3.at("th")(k, j, i) * pibar(k);
    }

    static double
    global_sum(double local) {
        double total = 0.0;
        MPI_Allreduce(&local, &total, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        return total;
    }

    // Total condensate on this rank [kg]: qp times CLEO's dry-air mass.
    double
    condensate_mass(const Snapshot& s) const {
        double total = 0.0;
        for_physical([&](int k, int j, int i) {
            const double rho = dry_air_density(pbar(k),
                temperature(s, k, j, i),
                s.f3.at("qv")(k, j, i));
            total += s.f3.at("qp")(k, j, i) * rho * dx * dy * dz;
        });
        return total;
    }
};

// pbar/pibar for an isothermal-ish column, th with a horizontal pattern, and qv
// at saturation ratio `s` everywhere -- halos included, with the pattern
// periodic, so they already hold what an exchange would put there.
void
write_base_state(Case& c, double s) {
    auto& state = c.state;
    const int nx_global = c.grid.get_global_points_x();
    const int ny_global = c.grid.get_global_points_y();
    const int x0 = c.grid.get_local_physical_start_x();
    const int y0 = c.grid.get_local_physical_start_y();

    auto pbar = state.get_field<1>("pbar").get_host_data();
    auto pibar = state.get_field<1>("pibar").get_host_data();
    std::vector<double> z(c.nzt);
    for (int k = 0; k < c.nzt; ++k) {
        z[k] = (k - c.h + 0.5) * c.dz;
        pbar(k) = 1.0e5 * std::exp(-z[k] / 8000.0);
        pibar(k) = std::pow(pbar(k) / 1.0e5, 287.04 / 1004.5);
    }
    Kokkos::deep_copy(state.get_field<1>("pbar").get_mutable_device_data(), pbar);
    Kokkos::deep_copy(state.get_field<1>("pibar").get_mutable_device_data(), pibar);

    auto th = state.get_field<3>("th").get_host_data();
    auto qv = state.get_field<3>("qv").get_host_data();
    const auto wrap = [](int n, int period) { return ((n % period) + period) % period; };
    for (int k = 0; k < c.nzt; ++k) {
        for (int j = 0; j < c.nyt; ++j) {
            for (int i = 0; i < c.nxt; ++i) {
                const int gi = wrap(x0 + i - c.h, nx_global);
                const int gj = wrap(y0 + j - c.h, ny_global);
                th(k, j, i) = 300.0 + 0.003 * z[k] + 0.05 * gi + 0.02 * gj;
                qv(k, j, i) = qv_at_saturation_ratio(s, th(k, j, i) * pibar(k), pbar(k));
            }
        }
    }
    Kokkos::deep_copy(state.get_field<3>("th").get_mutable_device_data(), th);
    Kokkos::deep_copy(state.get_field<3>("qv").get_mutable_device_data(), qv);

    for (const char* name : {"u", "v", "w"}) {
        state.get_field<3>(name).set_to_zero();
    }
    state.get_field<2>("topo").set_to_zero();
    state.set_time(VVM::real(0.0));
    Kokkos::fence();

    c.pbar = pbar;
    c.pibar = pibar;
}

// Halos of a horizontally periodic field equal their periodic image. Only the
// edges are checked: corners depend on the exchange's ordering, not on CLEO.
void
check_periodic_halos(const Case& c, const Snapshot& s, const char* name) {
    const auto& f = s.f3.at(name);
    for (int k = c.h; k < c.h + c.lnz; ++k) {
        for (int j = c.h; j < c.h + c.lny; ++j) {
            for (int i = 0; i < c.h; ++i) {
                check(f(k, j, i) == f(k, j, i + c.lnx),
                    std::string(name) + " west halo is stale after the coupling",
                    at(k, j, i, f(k, j, i), f(k, j, i + c.lnx)));
                const int e = c.h + c.lnx + i;
                check(f(k, j, e) == f(k, j, e - c.lnx),
                    std::string(name) + " east halo is stale after the coupling",
                    at(k, j, e, f(k, j, e), f(k, j, e - c.lnx)));
            }
        }
        for (int i = c.h; i < c.h + c.lnx; ++i) {
            for (int j = 0; j < c.h; ++j) {
                check(f(k, j, i) == f(k, j + c.lny, i),
                    std::string(name) + " south halo is stale after the coupling",
                    at(k, j, i, f(k, j, i), f(k, j + c.lny, i)));
                const int n = c.h + c.lny + j;
                check(f(k, n, i) == f(k, n - c.lny, i),
                    std::string(name) + " north halo is stale after the coupling",
                    at(k, n, i, f(k, n, i), f(k, n - c.lny, i)));
            }
        }
    }
}

void
check_no_ice(const Case& c, const Snapshot& s) {
    c.for_physical([&](int k, int j, int i) {
        check(s.f3.at("qi")(k, j, i) == 0.0, "qi is not zero (CLEO is warm-phase)");
        check(s.f3.at("diag_eff_radius_qi")(k, j, i) == ICE_RADIUS_UM,
            "diag_eff_radius_qi moved off 25 um");
    });
    c.for_surface([&](int j, int i) {
        check(s.f2.at("precip_ice_surf_mass")(j, i) == 0.0, "precip_ice_surf_mass is not zero");
        check(s.f2.at("precip_ice_surf_flux")(j, i) == 0.0, "precip_ice_surf_flux is not zero");
    });
}

// ---------------------------------------------------------------------------

void
run_transport(Case& c) {
    write_base_state(c, 0.8);
    c.cleo.initialize(c.state);
    const Snapshot initial = take_snapshot(c.state);

    // Every superdroplet is haze (radius < cloud_min_radius_um = 1 um), which
    // the bulk diagnostics leave out: no cloud, no rain, P3's empty values.
    // initialize() must already have diagnosed them -- radiation runs before
    // CLEO in a step and would otherwise see whatever add_field left.
    c.for_physical([&](int k, int j, int i) {
        check(initial.f3.at("qc")(k, j, i) == 0.0, "haze counted as cloud water after initialize");
        check(initial.f3.at("qr")(k, j, i) == 0.0, "haze counted as rain after initialize");
        check(initial.f3.at("nc")(k, j, i) == 0.0, "haze counted in nc after initialize");
        check(initial.f3.at("nr")(k, j, i) == 0.0, "haze counted in nr after initialize");
        check(initial.f3.at("diag_eff_radius_qc")(k, j, i) == EMPTY_CLOUD_RADIUS_UM,
            "empty-cell cloud effective radius is not P3's 10 um after initialize",
            at(k, j, i, initial.f3.at("diag_eff_radius_qc")(k, j, i), EMPTY_CLOUD_RADIUS_UM));
        check(initial.f3.at("diag_eff_radius_qr")(k, j, i) == EMPTY_RAIN_RADIUS_UM,
            "empty-cell rain effective radius is not P3's 500 um after initialize",
            at(k, j, i, initial.f3.at("diag_eff_radius_qr")(k, j, i), EMPTY_RAIN_RADIUS_UM));
    });

    for (int n = 0; n < 2; ++n) {
        c.step();
        const Snapshot s = take_snapshot(c.state);

        // With no microphysics, CLEO must hand back exactly the th and qv it
        // was given: th -> T/TEMP0 -> th is a round trip through pibar, so only
        // a few ulp may be lost. A wrong gridbox index or pibar level moves
        // the horizontal pattern and fails here.
        c.for_physical([&](int k, int j, int i) {
            const double th0 = initial.f3.at("th")(k, j, i);
            const double th1 = s.f3.at("th")(k, j, i);
            check(close(th1, th0, 1e-12),
                "th changed with condensation off (send/receive do not round-trip)",
                at(k, j, i, th1, th0));
            const double qv0 = initial.f3.at("qv")(k, j, i);
            const double qv1 = s.f3.at("qv")(k, j, i);
            check(close(qv1, qv0, 1e-12),
                "qv changed with condensation off (send/receive do not round-trip)",
                at(k, j, i, qv1, qv0));

            // Every gridbox was seeded, so every cell holds some condensate,
            // and qp (all superdroplets, haze included) bounds qc + qr.
            const double qp = s.f3.at("qp")(k, j, i);
            check(qp > 0.0, "qp is not positive in a seeded gridbox", at_bound(k, j, i, qp, ">", 0.0));
            check(qp >= s.f3.at("qc")(k, j, i) + s.f3.at("qr")(k, j, i),
                "qp is smaller than qc + qr");
        });

        // The coupling writes physical cells only; the dynamics differentiates
        // across halos, so they must be refreshed before it runs.
        for (const char* name : {"th", "qv", "qp"}) {
            check_periodic_halos(c, s, name);
        }
        check_no_ice(c, s);
    }
}

void
run_condensation(Case& c) {
    const double coupling =
        c.config.get_value<double>("physics.microphysics.cleo.timesteps_s.coupling");
    if (!close(coupling, 2.0 * c.dt, 1e-12)) {
        fail("condensation scenario expects coupling = 2 * dt; config does not match");
        return;
    }
    write_base_state(c, 1.02);
    c.cleo.initialize(c.state);
    const Snapshot initial = take_snapshot(c.state);

    c.step(); // t = 0: couples
    const Snapshot s0 = take_snapshot(c.state);
    c.step(); // t = 1: between couplings
    const Snapshot s1 = take_snapshot(c.state);
    c.step(); // t = 2: couples
    const Snapshot s2 = take_snapshot(c.state);

    // Cadence. The first step couples -- radiation and the dynamics must not
    // wait a whole coupling interval for CLEO's first feedback -- and the step
    // in between leaves State exactly as it was.
    c.for_physical([&](int k, int j, int i) {
        check(s0.f3.at("th")(k, j, i) > initial.f3.at("th")(k, j, i),
            "th did not warm on the first step in supersaturated air (no coupling at t=0?)",
            at_bound(k, j, i, s0.f3.at("th")(k, j, i), ">", initial.f3.at("th")(k, j, i)));
        check(s0.f3.at("qv")(k, j, i) < initial.f3.at("qv")(k, j, i),
            "qv did not fall on the first step in supersaturated air (no coupling at t=0?)",
            at_bound(k, j, i, s0.f3.at("qv")(k, j, i), "<", initial.f3.at("qv")(k, j, i)));
        for (const char* name : fields3) {
            check(s1.f3.at(name)(k, j, i) == s0.f3.at(name)(k, j, i),
                std::string(name) + " changed on a step between couplings",
                at(k, j, i, s1.f3.at(name)(k, j, i), s0.f3.at(name)(k, j, i)));
        }
    });

    // Across the second coupling (t = 1 -> 2), when qp already carries CLEO's
    // condensate. Before the first one qp is whatever add_field left.
    c.for_physical([&](int k, int j, int i) {
        const double dqv = s2.f3.at("qv")(k, j, i) - s1.f3.at("qv")(k, j, i);
        const double dqp = s2.f3.at("qp")(k, j, i) - s1.f3.at("qp")(k, j, i);
        const double dth = s2.f3.at("th")(k, j, i) - s1.f3.at("th")(k, j, i);
        check(dqv < 0.0, "no condensation at the second coupling in supersaturated air",
            at_bound(k, j, i, dqv, "<", 0.0));

        // Condensation moves water from vapour to droplets and nowhere else.
        // The 2% allows for CLEO recomputing its condensate with the post-step dry-air
        // density and for the few micrometres a haze droplet falls in 1 s.
        check(close(dqp, -dqv, 2e-2, 1e-14),
            "condensed water is not conserved: dqp != -dqv",
            at(k, j, i, dqp, -dqv));

        // dT = Lv / cp_moist * dq (CLEO's EffectOnThermodynamicState), and the
        // coupling turns dT into dth through pibar. A missing or misplaced
        // pibar, or a TEMP0 applied once too often, fails this.
        const double cp_moist =
            CP_DRY + CP_V * s2.f3.at("qv")(k, j, i) + C_L * s2.f3.at("qp")(k, j, i);
        const double want = LATENT_V / cp_moist * (-dqv) / c.pibar(k);
        check(close(dth, want, 2e-3),
            "latent heating is not Lv/cp * dqv / pibar",
            at(k, j, i, dth, want));
    });

    check_no_ice(c, s2);
}

void
run_average_y(Case& c) {
    if (!c.config.get_value<bool>("physics.microphysics.cleo.average_over_y", false)) {
        fail("average_y scenario needs physics.microphysics.cleo.average_over_y = true");
        return;
    }
    if (c.lny < 2 || c.grid.get_global_points_y() != c.lny) {
        fail("average_y scenario needs ny >= 2 with all of y on this rank");
        return;
    }
    write_base_state(c, 1.02);
    c.cleo.initialize(c.state);

    const auto check_uniform_in_y = [&](const Snapshot& s, const char* when) {
        for (const char* name : fields3) {
            const auto& f = s.f3.at(name);
            c.for_physical([&](int k, int j, int i) {
                check(f(k, j, i) == f(k, c.h, i),
                    std::string(name) + " differs between y columns " + when +
                        " with average_over_y",
                    at(k, j, i, f(k, j, i), f(k, c.h, i)));
            });
        }
        for (const char* name : fields2) {
            const auto& f = s.f2.at(name);
            c.for_surface([&](int j, int i) {
                check(f(j, i) == f(c.h, i),
                    std::string(name) + " differs between y columns " + when +
                        " with average_over_y");
            });
        }
    };

    // th starts with a y pattern, so only the averaged feedback can make the
    // columns agree -- and the diagnostics must agree as well, or radiation
    // reintroduces the y asymmetry averaging is there to remove.
    const Snapshot after_init = take_snapshot(c.state);
    for (const char* name : {"qc", "qr", "nc", "nr", "diag_eff_radius_qc", "diag_eff_radius_qr"}) {
        const auto& f = after_init.f3.at(name);
        c.for_physical([&](int k, int j, int i) {
            check(f(k, j, i) == f(k, c.h, i),
                std::string(name) + " differs between y columns after initialize");
        });
    }

    c.step();
    check_uniform_in_y(take_snapshot(c.state), "after a coupling");
    c.step();
    check_uniform_in_y(take_snapshot(c.state), "after two couplings");
}

void
run_rain(Case& c) {
    const double coupling =
        c.config.get_value<double>("physics.microphysics.cleo.timesteps_s.coupling");
    const double output_interval = c.config.get_value<double>("simulation.output_interval_s");
    if (!close(coupling, c.dt, 1e-12) || !close(output_interval, 3.0 * c.dt, 1e-12)) {
        fail("rain scenario expects coupling = dt and output_interval_s = 3 * dt");
        return;
    }
    write_base_state(c, 0.8);
    c.cleo.initialize(c.state);
    const Snapshot initial = take_snapshot(c.state);

    // Only the lowest level is seeded, with drops of 0.3-1 mm: all rain
    // (radius >= rain_min_radius_um = 40 um), none of it cloud.
    const int k_seeded = c.h;
    c.for_physical([&](int k, int j, int i) {
        const double qr = initial.f3.at("qr")(k, j, i);
        const double qc = initial.f3.at("qc")(k, j, i);
        const double nc = initial.f3.at("nc")(k, j, i);
        const double re_c = initial.f3.at("diag_eff_radius_qc")(k, j, i);
        const double re_r = initial.f3.at("diag_eff_radius_qr")(k, j, i);
        check(qc == 0.0, "rain drops counted as cloud water", at(k, j, i, qc, 0.0));
        check(nc == 0.0, "rain drops counted in nc (cloud number)", at(k, j, i, nc, 0.0));
        check(re_c == EMPTY_CLOUD_RADIUS_UM, "cloud effective radius is not 10 um with no cloud",
            at(k, j, i, re_c, EMPTY_CLOUD_RADIUS_UM));
        const double nr = initial.f3.at("nr")(k, j, i);
        if (k == k_seeded) {
            check(qr > 0.0, "seeded level has no rain water", at_bound(k, j, i, qr, ">", 0.0));
            check(nr > 0.0, "seeded level has no rain drops in nr", at_bound(k, j, i, nr, ">", 0.0));
            // sum(xi r^3) / sum(xi r^2) of drops in [300, 1000] um, in um.
            check(re_r >= 300.0 && re_r <= 1000.0,
                "rain effective radius is not in the seeded [300, 1000] um (units?)",
                at_bound(k, j, i, re_r, "to be in", 300.0) + " .. 1000");
        }
        else {
            check(qr == 0.0, "rain water above the seeded level", at(k, j, i, qr, 0.0));
            check(nr == 0.0, "rain drops above the seeded level", at(k, j, i, nr, 0.0));
            check(re_r == EMPTY_RAIN_RADIUS_UM, "rain effective radius is not 500 um with no rain",
                at(k, j, i, re_r, EMPTY_RAIN_RADIUS_UM));
        }
    });

    std::vector<Snapshot> after;
    for (int n = 0; n < 5; ++n) {
        c.step();
        after.push_back(take_snapshot(c.state));
    }

    // Surface flux of each coupling step, per column.
    const auto flux = [&](int n, int j, int i) {
        return after[n].f2.at("precip_liq_surf_flux")(j, i);
    };
    const auto acc = [&](int n, int j, int i) {
        return after[n].f2.at("precip_liq_surf_mass")(j, i);
    };

    double total_flux = 0.0;
    for (int n = 0; n < 5; ++n) {
        c.for_surface([&](int j, int i) {
            check(flux(n, j, i) >= 0.0, "negative surface precipitation flux");
            total_flux += flux(n, j, i);
        });
    }
    check(total_flux > 0.0, "no rain reached the surface from drops seeded in the lowest level");

    // What leaves through the bottom is exactly what the surface flux reports.
    // From step 1 on: qp is not written before the first coupling.
    const double area = c.dx * c.dy;
    for (int n = 1; n < 5; ++n) {
        const double lost = c.condensate_mass(after[n - 1]) - c.condensate_mass(after[n]);
        double precipitated = 0.0;
        c.for_surface([&](int j, int i) { precipitated += flux(n, j, i) * area * coupling; });
        std::ostringstream detail;
        detail.precision(17);
        detail << "step " << n << ": domain lost " << lost << " kg, surface flux says "
               << precipitated << " kg";
        check(close(precipitated, lost, 1e-6, 1e-12 * c.condensate_mass(after[n - 1])),
            "surface precipitation does not match the condensate that left the domain",
            detail.str());
    }

    // precip_liq_surf_mass follows P3: accumulated over the output interval,
    // turned into a mean rate on the step that ends it, and restarted on the
    // step after. Output is due when time + dt reaches 3 s, i.e. on step 2.
    c.for_surface([&](int j, int i) {
        const double f0 = flux(0, j, i), f1 = flux(1, j, i), f2 = flux(2, j, i);
        const double f3 = flux(3, j, i), f4 = flux(4, j, i);
        const std::pair<double, double> expected[] = {
            {acc(0, j, i), f0 * coupling},
            {acc(1, j, i), (f0 + f1) * coupling},
            {acc(2, j, i), (f0 + f1 + f2) / 3.0},
            {acc(3, j, i), f3 * coupling},
            {acc(4, j, i), (f3 + f4) * coupling},
        };
        for (int n = 0; n < 5; ++n) {
            std::ostringstream detail;
            detail.precision(17);
            detail << "after step " << n << " at (j,i)=(" << j << "," << i << "): got "
                   << expected[n].first << ", expected " << expected[n].second;
            check(close(expected[n].first, expected[n].second, 1e-9, 1e-30),
                "precip_liq_surf_mass does not follow accumulate / average / reset",
                detail.str());
        }
    });

    check_no_ice(c, after.back());
}

// Where each rank's gridboxes start, for check_cleo_superdroplets.py: CLEO's
// sdgbxindex is the index within the rank holding the droplet, so turning it
// into a position needs the decomposition, which the zarr does not record.
void
write_decomposition(const Case& c) {
    const int mine[4] = {c.grid.get_local_physical_start_x(),
        c.grid.get_local_physical_start_y(),
        c.lnx,
        c.lny};
    std::vector<int> all(4 * nranks);
    MPI_Gather(mine, 4, MPI_INT, all.data(), 4, MPI_INT, 0, MPI_COMM_WORLD);
    if (rank != 0) {
        return;
    }
    const std::string path =
        c.config.get_value<std::string>("output.output_dir") + "decomposition.json";
    std::FILE* f = std::fopen(path.c_str(), "w");
    if (f == nullptr) {
        fail("cannot write " + path);
        return;
    }
    std::fprintf(f, "{\"nz\": %d, \"ranks\": [", c.lnz);
    for (int r = 0; r < nranks; ++r) {
        std::fprintf(f, "%s{\"x0\": %d, \"y0\": %d, \"nx\": %d, \"ny\": %d}", r ? ", " : "",
            all[4 * r], all[4 * r + 1], all[4 * r + 2], all[4 * r + 3]);
    }
    std::fprintf(f, "]}\n");
    std::fclose(f);
}

void
run_advection(Case& c) {
    const double u = c.config.get_value<double>("test_cleo.u_m_s");
    const double v = c.config.get_value<double>("test_cleo.v_m_s");
    const double coupling =
        c.config.get_value<double>("physics.microphysics.cleo.timesteps_s.coupling");
    if (!close(coupling, c.dt, 1e-12)) {
        fail("advection scenario expects coupling = dt");
        return;
    }
    write_base_state(c, 0.8);

    // Uniform wind, halos included, so every face CLEO reads carries it.
    auto& state = c.state;
    Kokkos::deep_copy(state.get_field<3>("u").get_mutable_device_data(), VVM::real(u));
    Kokkos::deep_copy(state.get_field<3>("v").get_mutable_device_data(), VVM::real(v));

    Kokkos::fence();

    c.cleo.initialize(state);
    write_decomposition(c);
    const Snapshot initial = take_snapshot(state);

    const int steps =
        static_cast<int>(std::lround(c.config.get_value<double>("simulation.total_time_s") / c.dt));
    const double area = c.dx * c.dy;
    double mass_before = 0.0;
    for (int n = 0; n < steps; ++n) {
        c.step();
        const Snapshot s = take_snapshot(state);

        // Microphysics off: the thermodynamics comes back as sent, on every
        // rank, however many superdroplets crossed into it.
        c.for_physical([&](int k, int j, int i) {
            check(close(s.f3.at("th")(k, j, i), initial.f3.at("th")(k, j, i), 1e-12),
                "th changed with condensation off",
                at(k, j, i, s.f3.at("th")(k, j, i), initial.f3.at("th")(k, j, i)));
            check(close(s.f3.at("qv")(k, j, i), initial.f3.at("qv")(k, j, i), 1e-12),
                "qv changed with condensation off");
        });

        // Water is only carried: what the domain loses is what reached the
        // surface. A superdroplet lost between ranks breaks this.
        const double mass = Case::global_sum(c.condensate_mass(s));
        double flux = 0.0;
        c.for_surface([&](int j, int i) {
            flux += s.f2.at("precip_liq_surf_flux")(j, i) * area * coupling;
        });
        const double precipitated = Case::global_sum(flux);
        if (n == 0) {
            check(mass > 0.0, "no condensate in the domain after the first coupling");
        }
        else {
            std::ostringstream detail;
            detail.precision(17);
            detail << "step " << n << ": domain lost " << (mass_before - mass)
                   << " kg, surface flux says " << precipitated << " kg";
            check(close(precipitated, mass_before - mass, 1e-6, 1e-12 * mass_before),
                "condensate not conserved under advection: it went somewhere other than the "
                "surface",
                detail.str());
        }
        mass_before = mass;
    }
    check_no_ice(c, take_snapshot(state));
}

Case
make_case(const VVM::Utils::ConfigurationManager& config,
    const VVM::Core::Grid& grid,
    VVM::Core::State& state,
    VVM::Physics::VVM_CLEO_Interface& cleo) {
    return Case{config,
        grid,
        state,
        cleo,
        grid.get_halo_cells(),
        grid.get_local_physical_points_z(),
        grid.get_local_physical_points_y(),
        grid.get_local_physical_points_x(),
        grid.get_local_total_points_z(),
        grid.get_local_total_points_y(),
        grid.get_local_total_points_x(),
        static_cast<double>(grid.get_dx()),
        static_cast<double>(grid.get_dy()),
        static_cast<double>(grid.vertical_specification().dz),
        config.get_value<double>("simulation.dt_s"),
        {},
        {}};
}

// average_over_y averages over the y cells this rank holds. Split y, and the
// "mean" differs from rank to rank, which is the y asymmetry it exists to
// remove -- with no error. Refusing is acceptable at construction or at
// initialize().
void
run_average_y_split(const VVM::Utils::ConfigurationManager& config,
    const VVM::Core::Grid& grid,
    const VVM::Core::Parameters& parameters,
    VVM::Core::HaloExchanger& halo,
    VVM::Core::State& state) {
    if (!config.get_value<bool>("physics.microphysics.cleo.average_over_y", false) ||
        grid.get_global_points_y() == grid.get_local_physical_points_y()) {
        fail("average_y_split needs average_over_y = true and y split across ranks "
             "(e.g. 2 ranks, ny = 4)");
        return;
    }
    int refused = 0;
    try {
        VVM::Physics::VVM_CLEO_Interface cleo(config, grid, parameters, halo, state);
        Case c = make_case(config, grid, state, cleo);
        write_base_state(c, 0.8);
        cleo.initialize(state);
        cleo.finalize();
    }
    catch (const std::exception& e) {
        refused = 1;
        if (rank == 0) {
            std::printf("refused as expected: %s\n", e.what());
        }
    }
    int any = 0;
    MPI_Allreduce(&refused, &any, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
    check(any != 0,
        "average_over_y accepted with y split across ranks: the mean would cover only each "
        "rank's own y columns");
}

void
run_scenario(const VVM::Utils::ConfigurationManager& config,
    const VVM::Core::Grid& grid,
    const VVM::Core::Parameters& parameters,
    VVM::Core::HaloExchanger& halo,
    VVM::Core::State& state,
    const std::string& scenario) {
    if (scenario == "average_y_split") {
        run_average_y_split(config, grid, parameters, halo, state);
        return;
    }
    // The other single-rank scenarios check halos against their periodic image
    // on the same rank, and budgets without a reduction.
    if (scenario != "advection" && nranks != 1) {
        fail("scenario '" + scenario + "' runs on 1 rank only");
        return;
    }

    VVM::Physics::VVM_CLEO_Interface cleo(config, grid, parameters, halo, state);
    Case c = make_case(config, grid, state, cleo);

    if (scenario == "transport") {
        run_transport(c);
    }
    else if (scenario == "condensation") {
        run_condensation(c);
    }
    else if (scenario == "average_y") {
        run_average_y(c);
    }
    else if (scenario == "rain" || scenario == "rain_collision") {
        run_rain(c);
    }
    else if (scenario == "advection") {
        run_advection(c);
    }
    else {
        fail("unknown scenario '" + scenario + "'");
        return;
    }
    cleo.finalize();
}

} // namespace

int
main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nranks);

    if (argc != 3) {
        if (rank == 0) {
            std::fprintf(stderr,
                "usage: %s <config.json> <transport|condensation|average_y|rain|"
                "rain_collision|advection|average_y_split>\n",
                argv[0]);
        }
        MPI_Finalize();
        return 2;
    }
    const std::string scenario = argv[2];

    Kokkos::initialize(Kokkos::InitializationSettings().set_device_id(0));
    try {
        VVM::Utils::ConfigurationManager config(argv[1]);
        VVM::Core::Grid grid(config, MPI_COMM_WORLD);
        VVM::Core::Parameters parameters(config, grid);
#if defined(ENABLE_NCCL)
        ncclUniqueId id;
        if (rank == 0) {
            ncclGetUniqueId(&id);
        }
        MPI_Bcast(&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD);
        ncclComm_t nccl_comm;
        ncclCommInitRank(&nccl_comm, nranks, id, rank);
        cudaStream_t stream = Kokkos::Cuda().cuda_stream();
        {
            VVM::Core::State state(config, parameters, grid, nccl_comm, stream);
            VVM::Core::HaloExchanger halo(config, grid, nccl_comm, stream);
            run_scenario(config, grid, parameters, halo, state, scenario);
        }
        ncclCommDestroy(nccl_comm);
#else
        VVM::Core::State state(config, parameters, grid);
        VVM::Core::HaloExchanger halo(grid);
        run_scenario(config, grid, parameters, halo, state, scenario);
#endif
    }
    catch (const std::exception& e) {
        fail("unexpected exception", e.what());
    }
    Kokkos::finalize();

    // Every rank reports its own failures; the verdict is the same on all.
    int total_failures = 0;
    MPI_Allreduce(&failures, &total_failures, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    failures = total_failures;
    if (rank == 0) {
        if (failures == 0) {
            std::printf("test_cleo_coupling %s: all checks passed\n", scenario.c_str());
        }
        else {
            std::fprintf(stderr, "test_cleo_coupling %s: %d check(s) failed\n", scenario.c_str(),
                failures);
        }
    }
    MPI_Finalize();
    return failures == 0 ? 0 : 1;
}
