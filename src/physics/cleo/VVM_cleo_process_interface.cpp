#include "VVM_cleo_process_interface.hpp"

#include <mpi.h>

#include <algorithm>
#include <array>
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
#include "zarr/collective_dataset.hpp"
#include "zarr/fsstore.hpp"

#include "core/State.hpp"
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

// Type-erases SDMMethods, whose type depends on the whole process/observer chain.
struct SdmRunner {
    virtual ~SdmRunner() = default;

    virtual unsigned int get_couplstep() const = 0;
    virtual unsigned int next_couplstep(unsigned int t_mdl) const = 0;
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
    std::unique_ptr<SdmRunner> sdm;

    const size_t lnz, lnx, lny, halo;

    // Quasi-2-D runs: the y columns are replicates, so feed back their mean.
    const bool average_over_y;

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
          sdm(erase_sdm(SDMMethods(ts.get_couplstep(),
              gbxmaps,
              create_microphysics(ts, config),
              create_movement(ts, gbxmaps),
              create_observer(ts,
                  dataset,
                  store,
                  zarr_maxchunk(grid),
                  gbxmaps.get_local_ngridboxes_hostcopy())))),
          lnz(grid.get_local_physical_points_z()), lnx(grid.get_local_physical_points_x()),
          lny(grid.get_local_physical_points_y()), halo(grid.get_halo_cells()),
          average_over_y(config.get_value(key("average_over_y"), false)) {
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

    void
    receive_dynamics(Core::State& state, Core::HaloExchanger& halo_exchanger) {
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
                    const double temp = average_y ? t_sum * inv : st.temp;
                    th(ka, ja, ia) = temp * dlc::TEMP0 / pibar(ka);
                    qv(ka, ja, ia) = average_y ? qv_sum * inv : st.qvap;
                    qp(ka, ja, ia) = average_y ? qc_sum * inv : st.qcond;
                }
            });

        halo_exchanger.exchange_halos(state.get_field<3>("th"));
        halo_exchanger.exchange_halos(state.get_field<3>("qv"));
        halo_exchanger.exchange_halos(state.get_field<3>("qp"));
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

    Impl(const Utils::ConfigurationManager& config,
        const Core::Grid& grid,
        Core::HaloExchanger& halo_exchanger)
        : config(config), grid(grid), halo_exchanger(halo_exchanger) {}

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
        sdm.at_start_step(t_mdl, cleo->gbxs, cleo->allsupers);
        sdm.run_step(t_mdl, t_next, cleo->gbxs.view_device(), cleo->allsupers);
        cleo->receive_dynamics(state, halo_exchanger);
    }

    void
    initialize(Core::State& state) {
        // Before CleoState: the dataset takes the communicator in its constructor.
        init_communicator::set_communicator(grid.get_cart_comm());
        cleo = std::make_unique<CleoState>(config, grid);

        std::cout << "[CLEO] rank " << grid.get_mpi_rank() << ": "
                  << cleo->gbxmaps.get_local_ngridboxes_hostcopy() << " of "
                  << global_ngbxs(grid) << " gridboxes, couplstep "
                  << cleo->sdm->get_couplstep() << " (TIME0=" << dlc::TIME0 << " s)"
                  << std::endl;

        cleo->send_dynamics(state);
    }
    void
    run(Core::State& state, const Real dt) {
        if (state.get_time() + 0.5 * dt >= next_couple_s) {
            next_couple_s += step2realtime(cleo->sdm->get_couplstep());
            couple(state);
        }
    }
    void
    finalize() {
        if (cleo) {
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
    // run() couples at most once per model step
    if (config.get_value<double>(key("timesteps_s.coupling")) < params.get_value_host(params.dt)) {
        throw std::runtime_error(key("timesteps_s.coupling") + " is shorter than simulation.dt_s");
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
