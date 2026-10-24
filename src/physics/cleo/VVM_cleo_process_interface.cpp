#include "VVM_cleo_process_interface.hpp"

#include <mpi.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "cartesiandomain/cartesian_decomposition.hpp"
#include "cartesiandomain/collect_data_for_collective_dataset.hpp"
#include "cartesiandomain/createcartesianmaps.hpp"
#include "cartesiandomain/movement/cartesian_motion.hpp"
#include "cartesiandomain/movement/cartesian_movement.hpp"
#include "cleoconstants.hpp"
#include "configuration/communicator.hpp"
#include "gridboxes/movesupersindomain.hpp"
#include "initialise/init_all_supers_from_binary.hpp"
#include "initialise/initgbxsnull.hpp"
#include "initialise/readbinary.hpp"
#include "initialise/timesteps.hpp"
#include "observers/gbxindex_observer.hpp"
#include "observers/massmoments_observer.hpp"
#include "observers/nsupers_observer.hpp"
#include "observers/observers.hpp"
#include "observers/sdmmonitor/monitor_condensation_observer.hpp"
#include "observers/sdmmonitor/monitor_massmoments_change_observer.hpp"
#include "observers/sdmmonitor/monitor_precipitation_observer.hpp"
#include "observers/streamout_observer.hpp"
#include "observers/superdrops_observer.hpp"
#include "observers/thermo_observer.hpp"
#include "observers/time_observer.hpp"
#include "observers/totnsupers_observer.hpp"
#include "observers/windvel_observer.hpp"
#include "runcleo/creategbxs.hpp"
#include "runcleo/createsupers.hpp"
#include "runcleo/sdmmethods.hpp"
#include "superdrops/collisions/breakup_nfrags.hpp"
#include "superdrops/collisions/coalbure.hpp"
#include "superdrops/collisions/coalbure_flag.hpp"
#include "superdrops/collisions/collisions.hpp"
#include "superdrops/collisions/longhydroprob.hpp"
#include "superdrops/condensation.hpp"
#include "superdrops/sdmmonitor.hpp"
#include "superdrops/thermodynamic_equations.hpp"
#include "zarr/collective_dataset.hpp"
#include "zarr/fsstore.hpp"

#include "core/state/State.hpp"
#include "core/vvm_types.hpp"

namespace VVM::Physics {

namespace {

using Dataset = CollectiveDataset<FSStore, CartesianDecomposition>;

std::string
key(const std::string& k) {
    return "physics.microphysics.cleo." + k;
}

size_t
global_ngbxs(const Core::Grid& grid) {
    return static_cast<size_t>(grid.get_global_points_z()) * grid.get_global_points_x() *
           grid.get_global_points_y();
}

// zarr chunk size; CLEO's (time, gridbox) arrays need it >= the gridbox count.
size_t
zarr_maxchunk(const Core::Grid& grid) {
    return std::max<size_t>(2500000, global_ngbxs(grid));
}

SuppliedDecomposition
supplied_from_grid(const Core::Grid& grid) {
    const MPI_Comm cart = grid.get_cart_comm();
    int nprocs = 0;
    MPI_Comm_size(cart, &nprocs);

    // CLEO orders dimensions (z, x, y)
    const std::array<size_t, 3> my_origin{0,
        static_cast<size_t>(grid.get_local_physical_start_x()),
        static_cast<size_t>(grid.get_local_physical_start_y())};
    const std::array<size_t, 3> my_size{static_cast<size_t>(grid.get_global_points_z()),
        static_cast<size_t>(grid.get_local_physical_points_x()),
        static_cast<size_t>(grid.get_local_physical_points_y())};

    SuppliedDecomposition supplied;
    supplied.origins.resize(nprocs);
    supplied.sizes.resize(nprocs);
    MPI_Allgather(my_origin.data(), 3, MPI_UNSIGNED_LONG, supplied.origins.data(), 3,
        MPI_UNSIGNED_LONG, cart);
    MPI_Allgather(my_size.data(), 3, MPI_UNSIGNED_LONG, supplied.sizes.data(), 3,
        MPI_UNSIGNED_LONG, cart);

    for (int d = 0; d < 3; ++d) {
        std::vector<size_t> distinct;
        for (const auto& o : supplied.origins) {
            distinct.push_back(o[d]);
        }
        std::sort(distinct.begin(), distinct.end());
        distinct.erase(std::unique(distinct.begin(), distinct.end()), distinct.end());
        supplied.decomposition[d] = distinct.size();
    }
    return supplied;
}

size_t
nsupers_in_binary(const std::filesystem::path& path) {
    std::ifstream file(open_binary(path));
    return metadata_from_binary(file).at(0).nvar;
}

Timesteps
timesteps_from(const Utils::ConfigurationManager& config) {
    RequiredConfigParams::TimestepsParams p;
    p.CONDTSTEP = config.get_value<double>(key("timesteps_s.condensation"));
    p.COLLTSTEP = config.get_value<double>(key("timesteps_s.collision"));
    p.MOTIONTSTEP = config.get_value<double>(key("timesteps_s.motion"));
    p.COUPLTSTEP = config.get_value<double>(key("timesteps_s.coupling"));
    p.OBSTSTEP = config.get_value<double>(key("timesteps_s.observation"));
    p.T_END = config.get_value<double>("simulation.total_time_s");
    return Timesteps(p);
}

inline MicrophysicalProcess auto
create_microphysics(const Timesteps& ts, const Utils::ConfigurationManager& config) {
    OptionalConfigParams::CondensationParams c;
    c.do_alter_thermo = config.get_value(key("process.condensation.do_alter_thermo"), true);
    c.maxniters = config.get_value(key("process.condensation.maxniters"), 100);
    c.MINSUBTSTEP = config.get_value(key("process.condensation.MINSUBTSTEP"), 1e-5);
    c.rtol = config.get_value(key("process.condensation.rtol"), 0.0);
    c.atol = config.get_value(key("process.condensation.atol"), 0.01);

    // An interval of LIMITVALUES::uintmax never fires, which switches a process off.
    // Condensation is built by hand because Condensation() derives the solver step
    // from that interval.
    const auto condint = config.get_value<bool>(key("process.enable_condensation"), true)
                             ? ts.get_condstep()
                             : LIMITVALUES::uintmax;
    const auto cond = ConstTstepMicrophysics(condint,
        DoCondensation(c.do_alter_thermo,
            step2dimlesstime(ts.get_condstep()),
            c.maxniters,
            c.rtol,
            c.atol,
            realtime2dimless(c.MINSUBTSTEP)));

    const PairProbability auto collprob = LongHydroProb();
    const NFragments auto nfrags =
        ConstNFrags(config.get_value<double>(key("process.breakup_nfrags"), 5.0));
    const CoalBuReFlag auto coalbure_flag = TSCoalBuReFlag(RogersGKTerminalVelocity{});
    const auto collint = config.get_value<bool>(key("process.enable_collision"), true)
                             ? ts.get_collstep()
                             : LIMITVALUES::uintmax;

    // 0 seeds from std::random_device
    const auto collision_seed =
        config.get_value<std::uint64_t>(key("process.collision_seed"), 0);
    const MicrophysicalProcess auto colls =
        collision_seed
            ? CoalBuRe(collint, &step2realtime, collprob, nfrags, coalbure_flag, collision_seed)
            : CoalBuRe(collint, &step2realtime, collprob, nfrags, coalbure_flag);
    return colls >> cond;
}

template <GridboxMaps GbxMaps>
inline auto
create_movement(const Timesteps& ts, const GbxMaps& gbxmaps) {
    const Motion<CartesianMaps> auto motion =
        CartesianMotion(ts.get_motionstep(), &step2dimlesstime, RogersGKTerminalVelocity{});
    const BoundaryConditions<CartesianMaps> auto boundary_conditions = NullBoundaryConditions();
    return cartesian_movement(gbxmaps, motion, boundary_conditions);
}

inline Observer auto
create_superdrops_observer(
    const unsigned int interval, Dataset& dataset, FSStore& store, const size_t maxchunk) {
    CollectDataForDataset<Dataset> auto sdid = CollectSdId(dataset, maxchunk);
    CollectDataForDataset<Dataset> auto sdgbxindex = CollectSdgbxindex(dataset, maxchunk);
    CollectDataForDataset<Dataset> auto xi = CollectXi(dataset, maxchunk);
    CollectDataForDataset<Dataset> auto radius = CollectRadius(dataset, maxchunk);
    CollectDataForDataset<Dataset> auto msol = CollectMsol(dataset, maxchunk);
    CollectDataForDataset<Dataset> auto coord3 = CollectCoord3(dataset, maxchunk);
    CollectDataForDataset<Dataset> auto coord1 = CollectCoord1(dataset, maxchunk);
    CollectDataForDataset<Dataset> auto coord2 = CollectCoord2(dataset, maxchunk);

    // sdId must be written first: CollectiveDataset orders the other ragged arrays by it.
    const auto collect_sddata =
        sdid >> sdgbxindex >> xi >> radius >> msol >> coord3 >> coord2 >> coord1;
    return SuperdropsObserver(interval, dataset, store, maxchunk, collect_sddata);
}

inline Observer auto
create_gridboxes_observer(
    const unsigned int interval, Dataset& dataset, const size_t maxchunk, const size_t ngbxs) {
    const CollectDataForDataset<Dataset> auto thermo = CollectThermo(dataset, maxchunk, ngbxs);
    const CollectDataForDataset<Dataset> auto windvel = CollectWindVel(dataset, maxchunk, ngbxs);
    const CollectDataForDataset<Dataset> auto nsupers = CollectNsupers(dataset, maxchunk, ngbxs);
    return WriteToDatasetObserver(interval, dataset, nsupers >> windvel >> thermo);
}

inline Observer auto
create_sdmmonitor_observer(const unsigned int interval,
    Dataset& dataset,
    FSStore& store,
    const size_t maxchunk,
    const size_t ngbxs) {
    const Observer auto obs_cond =
        MonitorCondensationObserver(interval, dataset, store, maxchunk, ngbxs);
    const Observer auto obs_massmoms =
        MonitorMassMomentsChangeObserver(interval, dataset, store, maxchunk, ngbxs);
    const Observer auto obs_rainmassmoms =
        MonitorRainMassMomentsObserver(interval, dataset, store, maxchunk, ngbxs);
    const Observer auto obs_precip =
        MonitorPrecipitationObserver(interval, dataset, store, maxchunk, ngbxs);
    return obs_cond >> obs_massmoms >> obs_rainmassmoms >> obs_precip;
}

// ngbxs is the local gridbox count: CollectiveDataset sums it over ranks.
inline Observer auto
create_observer(const Timesteps& ts,
    Dataset& dataset,
    FSStore& store,
    const size_t maxchunk,
    const size_t ngbxs) {
    const auto obsstep = ts.get_obsstep();

    const Observer auto obs0 = StreamOutObserver(obsstep, &step2realtime);
    const Observer auto obs1 = TimeObserver(obsstep, dataset, store, maxchunk, &step2dimlesstime);
    const Observer auto obs2 = GbxindexObserver(dataset, store, maxchunk, ngbxs);
    const Observer auto obs3 = TotNsupersObserver(obsstep, dataset, store, maxchunk);
    const Observer auto obs4 = MassMomentsObserver(obsstep, dataset, store, maxchunk, ngbxs);
    const Observer auto obs5 =
        MassMomentsRaindropsObserver(obsstep, dataset, store, maxchunk, ngbxs);
    const Observer auto obsgbx = create_gridboxes_observer(obsstep, dataset, maxchunk, ngbxs);
    const Observer auto obssd = create_superdrops_observer(obsstep, dataset, store, maxchunk);
    const Observer auto obsm = create_sdmmonitor_observer(obsstep, dataset, store, maxchunk, ngbxs);
    return obsm >> obssd >> obsgbx >> obs5 >> obs4 >> obs3 >> obs2 >> obs1 >> obs0;
}

/* Condensate mass sum(m * xi) that falls through the bottom face of each local
gridbox. CLEO's own precipitation monitor only reaches the zarr. */
struct BottomFluxMonitor {
    Kokkos::View<double*> d_mass;

    void
    reset_monitor() const {}

    KOKKOS_FUNCTION
    void
    before_timestepping(const TeamMember&, const viewd_constsupers) const {}

    KOKKOS_FUNCTION
    void
    monitor_condensation(const TeamMember&, const double) const {}

    KOKKOS_FUNCTION
    void
    monitor_microphysics(const TeamMember&, const viewd_constsupers) const {}

    void
    monitor_motion(const auto, const auto) const {}

    // Called before the drop's gridbox index is updated; league_rank() is the box it leaves.
    KOKKOS_FUNCTION
    void
    monitor_precipitation(const TeamMember& team_member,
        const unsigned int gbxindex,
        const auto& gbxmaps,
        Superdrop& drop) const {
        if (drop.get_coord3() < gbxmaps.coord3bounds(gbxindex).first) {
            Kokkos::atomic_add(&d_mass(team_member.league_rank()),
                drop.condensate_mass() * static_cast<double>(drop.get_xi()));
        }
    }
};

// Never observes; only puts BottomFluxMonitor into the monitor chain.
struct BottomFluxObserver {
    BottomFluxMonitor mo;

    void
    before_timestepping(const viewd_constgbx, const subviewd_constsupers) const {}
    void
    after_timestepping() const {}
    unsigned int
    next_obs(const unsigned int) const {
        return LIMITVALUES::uintmax;
    }
    bool
    on_step(const unsigned int) const {
        return false;
    }
    void
    at_start_step(const unsigned int, const viewd_constgbx, const subviewd_constsupers) const {}
    SDMMonitor auto
    get_sdmmonitor() const {
        return mo;
    }
};

// Type-erases SDMMethods, whose type depends on the whole process/observer chain.
struct SdmRunner {
    virtual ~SdmRunner() = default;

    virtual unsigned int get_couplstep() const = 0;
    virtual unsigned int next_couplstep(unsigned int t_mdl) const = 0;
    virtual unsigned int next_obs(unsigned int t_mdl) const = 0;
    virtual void prepare_to_timestep(dualview_gbx gbxs, const SupersInDomain& allsupers) const = 0;
    virtual void at_start_step(
        unsigned int t_mdl, dualview_gbx gbxs, const SupersInDomain& allsupers) const = 0;
    virtual void run_step(unsigned int t_mdl,
        unsigned int t_mdl_next,
        viewd_gbx d_gbxs,
        SupersInDomain& allsupers) const = 0;
    virtual void after_timestepping() const = 0;
};

template <typename SDM>
struct SdmRunnerOf final : SdmRunner {
    SDM sdm;

    explicit SdmRunnerOf(SDM sdm) : sdm(std::move(sdm)) {}

    unsigned int
    get_couplstep() const override {
        return sdm.get_couplstep();
    }
    unsigned int
    next_couplstep(unsigned int t_mdl) const override {
        return sdm.next_couplstep(t_mdl);
    }
    unsigned int
    next_obs(unsigned int t_mdl) const override {
        return sdm.obs.next_obs(t_mdl);
    }
    void
    prepare_to_timestep(dualview_gbx gbxs, const SupersInDomain& allsupers) const override {
        sdm.prepare_to_timestep(gbxs, allsupers);
    }
    void
    at_start_step(
        unsigned int t_mdl, dualview_gbx gbxs, const SupersInDomain& allsupers) const override {
        sdm.at_start_step(t_mdl, gbxs, allsupers);
    }
    void
    run_step(unsigned int t_mdl,
        unsigned int t_mdl_next,
        viewd_gbx d_gbxs,
        SupersInDomain& allsupers) const override {
        sdm.run_step(t_mdl, t_mdl_next, d_gbxs, allsupers);
    }
    void
    after_timestepping() const override {
        sdm.obs.after_timestepping();
    }
};

template <typename SDM>
std::unique_ptr<SdmRunner>
erase_sdm(SDM sdm) {
    return std::make_unique<SdmRunnerOf<SDM>>(std::move(sdm));
}

struct CleoState {
    const Timesteps ts;
    FSStore store;
    Dataset dataset;
    const CartesianMaps gbxmaps;
    const size_t total_supers;
    SupersInDomain allsupers;
    dualview_gbx gbxs;
    // Before sdm: the observer chain keeps a copy of the view.
    Kokkos::View<double*> bottom_mass;
    std::unique_ptr<SdmRunner> sdm;

    const size_t lnz, lnx, lny, halo;

    // Quasi-2-D runs: the y columns are replicates, so feed back their mean.
    const bool average_over_y;

    // Haze < cloud_min_radius <= cloud < rain_min_radius <= rain
    const double cloud_min_radius_um, rain_min_radius_um;
    const double cell_area_m2;
    // Per local gridbox: qc, qr, nc, nr, then sum(xi r^3), sum(xi r^2) for cloud and rain
    Kokkos::View<double**> moments;

    CleoState(const Utils::ConfigurationManager& config, const Core::Grid& grid)
        : ts(timesteps_from(config)),
          store(std::filesystem::path(config.get_value<std::string>("output.output_dir")) /
                config.get_value<std::string>(key("output.zarrbasedir"), "vvm_sol.zarr")),
          dataset(store),
          gbxmaps(create_cartesian_maps(global_ngbxs(grid),
              3,
              config.get_value<std::string>(key("init_gbx_path")),
              supplied_from_grid(grid))),
          total_supers(nsupers_in_binary(config.get_value<std::string>(key("init_supers_path")))),
          // Every rank reads the whole binary and keeps its own superdroplets; the
          // capacity factor leaves room for the ones drifting in from neighbours.
          allsupers(create_supers(
              InitAllSupersFromBinary(
                  total_supers, config.get_value<std::string>(key("init_supers_path")), 3),
              gbxmaps,
              config.get_value<double>(key("superdroplets.rank_capacity_factor"), 1.5))),
          gbxs(create_gbxs(
              gbxmaps, InitGbxsNull(gbxmaps.get_local_ngridboxes_hostcopy()), allsupers)),
          bottom_mass("cleo_bottom_mass", gbxmaps.get_local_ngridboxes_hostcopy()),
          sdm(erase_sdm(SDMMethods(ts.get_couplstep(),
              gbxmaps,
              create_microphysics(ts, config),
              create_movement(ts, gbxmaps),
              create_observer(ts,
                  dataset,
                  store,
                  zarr_maxchunk(grid),
                  gbxmaps.get_local_ngridboxes_hostcopy()) >>
                  BottomFluxObserver{BottomFluxMonitor{bottom_mass}}))),
          lnz(grid.get_local_physical_points_z()), lnx(grid.get_local_physical_points_x()),
          lny(grid.get_local_physical_points_y()), halo(grid.get_halo_cells()),
          average_over_y(config.get_value(key("average_over_y"), false)),
          cloud_min_radius_um(config.get_value(key("diagnostics.cloud_min_radius_um"), 1.0)),
          rain_min_radius_um(config.get_value(key("diagnostics.rain_min_radius_um"), 40.0)),
          cell_area_m2(static_cast<double>(grid.get_dx()) * static_cast<double>(grid.get_dy())),
          moments("cleo_hydrometeor_moments", lnz * lnx * lny, 8) {
        dataset.set_decomposition(gbxmaps.get_domain_decomposition());
        dataset.set_max_superdroplets(total_supers);

        sdm->prepare_to_timestep(gbxs, allsupers);

        // The coupling decodes the gridbox index as k + nz * (i + nx * j).
        if (lnz * lnx * lny != gbxs.extent(0)) {
            throw std::runtime_error("CLEO built " + std::to_string(gbxs.extent(0)) +
                                     " local gridboxes but the grid has " + std::to_string(lnz) +
                                     "*" + std::to_string(lnx) + "*" + std::to_string(lny));
        }
        if (average_over_y && lny != static_cast<size_t>(grid.get_global_points_y())) {
            throw std::runtime_error(key("average_over_y") + " needs y undivided across ranks");
        }
    }

    void
    send_dynamics(const Core::State& state) {
        const auto u = state.get_field<3>("u").get_device_data();
        const auto v = state.get_field<3>("v").get_device_data();
        const auto w = state.get_field<3>("w").get_device_data();
        const auto th = state.get_field<3>("th").get_device_data();
        const auto qv = state.get_field<3>("qv").get_device_data();
        const auto pibar = state.get_field<1>("pibar").get_device_data();
        const auto pbar = state.get_field<1>("pbar").get_device_data();

        const auto nz = lnz, nx = lnx, h = halo;
        gbxs.sync_device();
        const auto d_gbxs = gbxs.view_device();
        Kokkos::parallel_for("cleo_send_dynamics",
            Kokkos::RangePolicy<ExecSpace>(0, d_gbxs.extent(0)),
            KOKKOS_LAMBDA(const size_t idx) {
                const auto k = idx % nz;
                const auto i = (idx / nz) % nx;
                const auto j = idx / (nz * nx);
                const auto ka = h + k, ja = h + j, ia = h + i;

                // u, v, w live on the east, north and top faces
                auto& gb_state = d_gbxs(idx).state;
                gb_state.uvel = {u(ka, ja, ia - 1) / dlc::W0, u(ka, ja, ia) / dlc::W0};
                gb_state.vvel = {v(ka, ja - 1, ia) / dlc::W0, v(ka, ja, ia) / dlc::W0};
                gb_state.wvel = {w(ka - 1, ja, ia) / dlc::W0, w(ka, ja, ia) / dlc::W0};
                gb_state.temp = th(ka, ja, ia) * pibar(ka) / dlc::TEMP0;
                gb_state.qvap = qv(ka, ja, ia);
                gb_state.press = pbar(ka) / dlc::P0;
            });
        gbxs.modify_device();
    }

    // qcond from the superdroplets, as CLEO sets it after every motion step
    void
    diagnose_qcond() {
        const auto d_gbxs = gbxs.view_device();
        Kokkos::parallel_for("cleo_diagnose_qcond",
            TeamPolicy(d_gbxs.extent(0), KCS::team_size),
            EffectOnHydrometeorStatesFunctor{d_gbxs, allsupers.domain_supers_readonly()});
    }

    // qp from qcond, and th and qv too if thermo
    void
    receive_dynamics(Core::State& state, Core::HaloExchanger& halo_exchanger, const bool thermo) {
        auto th = state.get_field<3>("th").get_mutable_device_data();
        auto qv = state.get_field<3>("qv").get_mutable_device_data();
        auto qp = state.get_field<3>("qp").get_mutable_device_data();
        const auto pibar = state.get_field<1>("pibar").get_device_data();

        const auto nz = lnz, nx = lnx, ny = lny, h = halo;
        const bool average_y = average_over_y;
        const auto d_gbxs = gbxs.view_device();
        Kokkos::parallel_for("cleo_receive_dynamics",
            Kokkos::RangePolicy<ExecSpace>(0, nz * nx),
            KOKKOS_LAMBDA(const size_t ki) {
                const auto k = ki % nz;
                const auto i = ki / nz;
                const auto ka = h + k, ia = h + i;
                double t_sum = 0.0, qv_sum = 0.0, qc_sum = 0.0;
                if (average_y) {
                    for (size_t j = 0; j < ny; ++j) {
                        const auto& st = d_gbxs(k + nz * (i + nx * j)).state;
                        t_sum += st.temp;
                        qv_sum += st.qvap;
                        qc_sum += st.qcond;
                    }
                }
                const double inv = average_y ? 1.0 / static_cast<double>(ny) : 1.0;
                for (size_t j = 0; j < ny; ++j) {
                    const auto ja = h + j;
                    const auto& st = d_gbxs(k + nz * (i + nx * j)).state;
                    if (thermo) {
                        const double temp = average_y ? t_sum * inv : st.temp;
                        th(ka, ja, ia) = temp * dlc::TEMP0 / pibar(ka);
                        qv(ka, ja, ia) = average_y ? qv_sum * inv : st.qvap;
                    }
                    qp(ka, ja, ia) = average_y ? qc_sum * inv : st.qcond;
                }
            });

        if (thermo) {
            halo_exchanger.exchange_halos(state.get_field<3>("th"));
            halo_exchanger.exchange_halos(state.get_field<3>("qv"));
        }
        halo_exchanger.exchange_halos(state.get_field<3>("qp"));
    }

    // The bulk fields P3 would provide, for radiation; diagnostic only.
    void
    diagnose_hydrometeors(Core::State& state) {
        const auto nz = lnz, nx = lnx, ny = lny, h = halo;
        const bool average_y = average_over_y;
        const double r_cloud = cloud_min_radius_um * 1e-6 / dlc::R0;
        const double r_rain = rain_min_radius_um * 1e-6 / dlc::R0;
        const double um_per_r = dlc::R0 * 1e6;

        const auto d_gbxs = gbxs.view_device();
        const auto domainsupers = allsupers.domain_supers_readonly();
        const auto mom = moments;
        Kokkos::parallel_for("cleo_hydrometeor_moments",
            Kokkos::RangePolicy<ExecSpace>(0, d_gbxs.extent(0)),
            KOKKOS_LAMBDA(const size_t idx) {
                const auto& st = d_gbxs(idx).state;
                const auto supers = d_gbxs(idx).supersingbx.readonly(domainsupers);
                double mc = 0.0, mr = 0.0, nc = 0.0, nr = 0.0;
                double c3 = 0.0, c2 = 0.0, r3 = 0.0, r2 = 0.0;
                for (size_t kk = 0; kk < supers.extent(0); ++kk) {
                    const auto& drop = supers(kk);
                    const double r = drop.get_radius();
                    const double xi = static_cast<double>(drop.get_xi());
                    const double xr2 = xi * r * r;
                    if (r >= r_rain) {
                        mr += drop.condensate_mass() * xi;
                        nr += xi;
                        r3 += xr2 * r;
                        r2 += xr2;
                    }
                    else if (r >= r_cloud) {
                        mc += drop.condensate_mass() * xi;
                        nc += xi;
                        c3 += xr2 * r;
                        c2 += xr2;
                    }
                }
                // dry-air mass of the gridbox [kg], as CLEO uses for qcond
                const double air = st.get_volume() * dlc::VOL0 *
                                   dry_air_density(st.press, st.temp, st.qvap) * dlc::RHO0;
                mom(idx, 0) = mc * dlc::MASS0 / air;
                mom(idx, 1) = mr * dlc::MASS0 / air;
                mom(idx, 2) = nc / air;
                mom(idx, 3) = nr / air;
                mom(idx, 4) = c3;
                mom(idx, 5) = c2;
                mom(idx, 6) = r3;
                mom(idx, 7) = r2;
            });

        auto qc = state.get_field<3>("qc").get_mutable_device_data();
        auto qr = state.get_field<3>("qr").get_mutable_device_data();
        auto ncf = state.get_field<3>("nc").get_mutable_device_data();
        auto nrf = state.get_field<3>("nr").get_mutable_device_data();
        auto re_c = state.get_field<3>("diag_eff_radius_qc").get_mutable_device_data();
        auto re_r = state.get_field<3>("diag_eff_radius_qr").get_mutable_device_data();
        Kokkos::parallel_for("cleo_diagnose_hydrometeors",
            Kokkos::RangePolicy<ExecSpace>(0, nz * nx),
            KOKKOS_LAMBDA(const size_t ki) {
                const auto k = ki % nz;
                const auto i = ki / nz;
                const auto ka = h + k, ia = h + i;
                constexpr int nmom = 8;
                double sum[nmom] = {};
                if (average_y) {
                    for (size_t j = 0; j < ny; ++j) {
                        for (int m = 0; m < nmom; ++m) {
                            sum[m] += mom(k + nz * (i + nx * j), m);
                        }
                    }
                }
                const double inv = average_y ? 1.0 / static_cast<double>(ny) : 1.0;
                for (size_t j = 0; j < ny; ++j) {
                    const auto ja = h + j;
                    double v[nmom];
                    for (int m = 0; m < nmom; ++m) {
                        v[m] = average_y ? sum[m] * inv : mom(k + nz * (i + nx * j), m);
                    }
                    qc(ka, ja, ia) = v[0];
                    qr(ka, ja, ia) = v[1];
                    ncf(ka, ja, ia) = v[2];
                    nrf(ka, ja, ia) = v[3];
                    // P3's values for an empty cell
                    re_c(ka, ja, ia) = v[5] > 0.0 ? v[4] / v[5] * um_per_r : 10.0;
                    re_r(ka, ja, ia) = v[7] > 0.0 ? v[6] / v[7] * um_per_r : 500.0;
                }
            });
    }

    // Mean surface liquid flux over the coupling step just run, taken at the
    // bottom gridboxes; it is held until the next coupling.
    void
    collect_surface_precip(Core::State& state, const double interval_s) {
        const auto nz = lnz, nx = lnx, ny = lny, h = halo;
        const bool average_y = average_over_y;
        const double to_flux = dlc::MASS0 / (cell_area_m2 * interval_s);

        const auto mass = bottom_mass;
        auto flux = state.get_field<2>("precip_liq_surf_flux").get_mutable_device_data();
        auto acc = state.get_field<2>("precip_liq_surf_mass").get_mutable_device_data();
        Kokkos::parallel_for("cleo_surface_precip",
            Kokkos::RangePolicy<ExecSpace>(0, nx),
            KOKKOS_LAMBDA(const size_t i) {
                const auto ia = h + i;
                const auto surface_flux = [&](const size_t j) {
                    return mass(nz * (i + nx * j)) * to_flux;
                };
                double sum = 0.0;
                if (average_y) {
                    for (size_t j = 0; j < ny; ++j) {
                        sum += surface_flux(j);
                    }
                }
                const double inv = average_y ? 1.0 / static_cast<double>(ny) : 1.0;
                for (size_t j = 0; j < ny; ++j) {
                    const double f = average_y ? sum * inv : surface_flux(j);
                    flux(h + j, ia) = f;
                    acc(h + j, ia) += f * interval_s;
                }
            });
        Kokkos::deep_copy(bottom_mass, 0.0);
    }

    // precip_liq_surf_mass [kg m-2] -> mean rate over interval_s
    void
    average_surface_precip(Core::State& state, const double interval_s) const {
        const double inv = 1.0 / interval_s;
        auto acc = state.get_field<2>("precip_liq_surf_mass").get_mutable_device_data();
        Kokkos::parallel_for("cleo_average_precip",
            Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<2>>({0, 0},
                {static_cast<long>(acc.extent(0)), static_cast<long>(acc.extent(1))}),
            KOKKOS_LAMBDA(const long j, const long i) { acc(j, i) *= inv; });
    }
};

}

// PIMPL
struct VVM_CLEO_Interface::Impl {
    const Utils::ConfigurationManager& config;
    const Core::Grid& grid;
    Core::HaloExchanger& halo_exchanger;
    std::unique_ptr<CleoState> cleo;

    double next_couple_s = 0.0;
    unsigned int t_cleo = 0;

    // precip_liq_surf_mass follows P3: accumulated, divided on the step before
    // output, zeroed on the step after.
    const double output_interval_s;
    double next_output_s = -1.0;
    double precip_window_s = 0.0;
    bool need_reset_precip = false;

    Impl(const Utils::ConfigurationManager& config,
        const Core::Grid& grid,
        Core::HaloExchanger& halo_exchanger)
        : config(config), grid(grid), halo_exchanger(halo_exchanger),
          output_interval_s(config.get_value<double>("simulation.output_interval_s")) {}

    void
    couple(Core::State& state) {
        auto& sdm = *cleo->sdm;
        const auto t_mdl = t_cleo;
        const auto t_next = t_mdl + sdm.get_couplstep();
        if (t_next != sdm.next_couplstep(t_mdl)) {
            throw std::runtime_error("CLEO is out of sync with the coupling at t_mdl " +
                                     std::to_string(t_mdl));
        }
        t_cleo = t_next;

        cleo->send_dynamics(state);
        // Observations inside the interval are taken on the way, as CLEO's own driver does.
        for (auto t = t_mdl; t < t_next;) {
            const auto t_sub = std::min(t_next, sdm.next_obs(t));
            sdm.at_start_step(t, cleo->gbxs, cleo->allsupers);
            sdm.run_step(t, t_sub, cleo->gbxs.view_device(), cleo->allsupers);
            t = t_sub;
        }
        cleo->receive_dynamics(state, halo_exchanger, true);
        cleo->diagnose_hydrometeors(state);

        const double couplstep_s = step2realtime(sdm.get_couplstep());
        cleo->collect_surface_precip(state, couplstep_s);
        precip_window_s += couplstep_s;
    }

    // Same arithmetic as main.cpp's output decision. Divides by the coupling time
    // actually accumulated, which can differ from output_interval_s by a step.
    void
    average_precip_if_output_step(Core::State& state, const Real dt) {
        if (next_output_s < 0.0) {
            next_output_s =
                (std::floor(state.get_time() / output_interval_s) + 1.0) * output_interval_s;
        }
        if (state.get_time() + dt < next_output_s) {
            return;
        }
        next_output_s += output_interval_s;
        if (precip_window_s > 0.0) {
            cleo->average_surface_precip(state, precip_window_s);
        }
        precip_window_s = 0.0;
        need_reset_precip = true;
    }

    // Superdroplets would sink into terrain: CLEO's lower boundary is the bottom of the domain.
    void
    require_flat_terrain(const Core::State& state) const {
        const auto topo = state.get_field<2>("topo").get_host_data();
        const int h = grid.get_halo_cells();
        int raised = 0;
        for (int j = h; j < h + grid.get_local_physical_points_y(); ++j) {
            for (int i = h; i < h + grid.get_local_physical_points_x(); ++i) {
                raised = raised || topo(j, i) >= h;
            }
        }
        MPI_Allreduce(MPI_IN_PLACE, &raised, 1, MPI_INT, MPI_MAX, grid.get_cart_comm());
        if (raised) {
            throw std::runtime_error("CLEO coupling needs flat terrain (topo = 0)");
        }
    }

    void
    initialize(Core::State& state) {
        require_flat_terrain(state);
        // Before CleoState: the dataset takes the communicator in its constructor.
        init_communicator::set_communicator(grid.get_cart_comm());
        cleo = std::make_unique<CleoState>(config, grid);

        std::cout << "[CLEO] rank " << grid.get_mpi_rank() << ": "
                  << cleo->gbxmaps.get_local_ngridboxes_hostcopy() << " of "
                  << global_ngbxs(grid) << " gridboxes, couplstep "
                  << cleo->sdm->get_couplstep() << " (TIME0=" << dlc::TIME0 << " s)"
                  << std::endl;

        // qp and qcond from the initial superdroplets, before the first output and observation
        cleo->send_dynamics(state);
        cleo->diagnose_qcond();
        cleo->receive_dynamics(state, halo_exchanger, false);
        // radiation runs before CLEO within a step
        cleo->diagnose_hydrometeors(state);
    }
    void
    run(Core::State& state, const Real dt) {
        if (need_reset_precip) {
            Kokkos::deep_copy(
                state.get_field<2>("precip_liq_surf_mass").get_mutable_device_data(), 0.0);
            need_reset_precip = false;
        }
        if (t_cleo < cleo->ts.get_t_end() && state.get_time() + 0.5 * dt >= next_couple_s) {
            next_couple_s += step2realtime(cleo->sdm->get_couplstep());
            couple(state);
        }
        average_precip_if_output_step(state, dt);
    }
    void
    finalize() {
        if (cleo) {
            // CLEO observes at the start of a step; this is the one at the end of the run.
            cleo->sdm->at_start_step(t_cleo, cleo->gbxs, cleo->allsupers);
            cleo->sdm->after_timestepping();
        }
    }
};

VVM_CLEO_Interface::VVM_CLEO_Interface(const VVM::Utils::ConfigurationManager& config,
    const VVM::Core::Grid& grid,
    const VVM::Core::Parameters& params,
    Core::HaloExchanger& halo_exchanger,
    Core::State& state)
    : impl_(std::make_unique<Impl>(config, grid, halo_exchanger)) {

    if (grid.geometry().kind() != Core::Geometry::GeometryKind::Cartesian) {
        throw std::runtime_error("CLEO coupling supports Cartesian geometry only");
    }
    const auto& topology = grid.horizontal_specification().topology;
    if (topology.q1 != Core::HorizontalEdgeTopology::Periodic ||
        topology.q2 != Core::HorizontalEdgeTopology::Periodic) {
        throw std::runtime_error(
            "CLEO coupling needs periodic grid.boundary_condition.x and .y: CLEO wraps "
            "superdroplets periodically");
    }
    // run() couples at most once per model step
    const double coupling_s = config.get_value<double>(key("timesteps_s.coupling"));
    if (coupling_s < params.get_value_host(params.dt)) {
        throw std::runtime_error(key("timesteps_s.coupling") + " is shorter than simulation.dt_s");
    }
    // Each coupling runs CLEO one interval ahead of the model; the last must end at total_time_s.
    if (realtime2step(config.get_value<double>("simulation.total_time_s")) %
            realtime2step(coupling_s) !=
        0) {
        throw std::runtime_error(
            "simulation.total_time_s is not a whole number of " + key("timesteps_s.coupling"));
    }

    const std::array<int, 3> shape{grid.get_local_total_points_z(),
        grid.get_local_total_points_y(),
        grid.get_local_total_points_x()};

    const auto add_3d_field = [&](const char* name, const char* units, const char* long_name) {
        if (!state.has_field(name)) {
            state.add_field<3>(name,
                shape,
                Core::FieldMetadata{Core::GridStaggering::Centered, units, long_name});
        }
    };
    add_3d_field("qc", "kg kg-1", "cloud liquid water mass mixing ratio (radius <= 40um)");
    add_3d_field("qr", "kg kg-1", "rain water mass mixing ratio (radius > 40um)");
    add_3d_field("qp", "kg kg-1", "total hydrometeor mass mixing ratio");

    add_3d_field("nc", "kg-1", "cloud droplet number mixing ratio");
    add_3d_field("nr", "kg-1", "rain droplet number mixing ratio");
    // No ice in CLEO, always zero
    add_3d_field("qi", "kg kg-1", "total ice mass mixing ratio");
    add_3d_field("diag_eff_radius_qc", "um", "cloud droplet effective radius");
    add_3d_field("diag_eff_radius_qr", "um", "rain drop effective radius");
    add_3d_field("diag_eff_radius_qi", "um", "ice particle effective radius");

    // no ice grid set to 25um, consistent with P3
    Kokkos::deep_copy(state.get_field<3>("diag_eff_radius_qi").get_mutable_device_data(),
        real(25.0));

    const std::array<int, 2> sfc_shape{grid.get_local_total_points_y(),
        grid.get_local_total_points_x()};
    const auto add_2d_field = [&](const char* name, const char* long_name) {
        if (!state.has_field(name)) {
            state.add_field<2>(name,
                sfc_shape,
                Core::FieldMetadata{Core::GridStaggering::Surface, "kg m-2 s-1", long_name});
        }
    };
    add_2d_field("precip_liq_surf_mass", "time-averaged surface liquid precipitation flux");
    add_2d_field("precip_liq_surf_flux", "surface liquid precipitation flux");
    // always zero
    add_2d_field("precip_ice_surf_mass", "time-averaged surface ice precipitation flux");
    add_2d_field("precip_ice_surf_flux", "surface ice precipitation flux");
}

VVM_CLEO_Interface::~VVM_CLEO_Interface() = default;

void
VVM_CLEO_Interface::initialize(Core::State& state) {
    impl_->initialize(state);
}

void
VVM_CLEO_Interface::run(Core::State& state, Real dt) {
    impl_->run(state, dt);
}

void
VVM_CLEO_Interface::finalize() {
    impl_->finalize();
}

}
