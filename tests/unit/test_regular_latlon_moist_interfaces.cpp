#include "core/BoundaryConditionManager.hpp"
#include "core/ModelConfigurationValidation.hpp"
#include "core/State.hpp"
#include "core/Initializer.hpp"
#include "core/haloexchange/HaloExchanger.hpp"
#include "dynamics/numerical_methods/NumericalMethodFactory.hpp"
#include "dynamics/operators/RegularLatLonScalarTransport.hpp"
#include <filesystem>
#include <cmath>
#include <fstream>
#include <iostream>
#include <vector>
#include <unistd.h>

using namespace VVM;
using Json = nlohmann::json;

void require(bool value, const char* message) {
    if (!value) throw std::runtime_error(message);
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    Kokkos::initialize(Kokkos::InitializationSettings().set_device_id(0));
    int result = 0;
    const auto directory = std::filesystem::temp_directory_path() /
        ("vvm_rll_moist_interfaces_" + std::to_string(getpid()));
    std::filesystem::create_directory(directory);
#if defined(ENABLE_NCCL)
    ncclComm_t comm = nullptr;
#endif
    try {
        Json base = Json::parse(R"({
          "grid":{"horizontal":{"nx":16,"ny":8,"n_halo_cells":2,
            "geometry":{"kind":"regular_latlon","earth_radius_m":6371220,
              "longitude_bounds_deg":[-1,1],"latitude_bounds_deg":[-1,1]},
            "topology":{"q1":"periodic","q2":"bounded"}},
            "vertical":{"nz":8,"type":"default","dz":250,"dz1":250}},
          "simulation":{"idealized_test":"jung2019_barotropic","dt_s":1,"total_time_s":2,"output_interval_s":1},
          "initial_conditions":{"jung2019":{"case":1}},
          "dynamics":{"solver":{"w_solver_method":"tridiagonal","iteration":10,
            "initial_iterations":10,"vertical_iterations":10,"WRXMU":100}},
          "constants":{"gravity":9.806,"Rd":287.04,"Cp":1004.5,"P0":100000},
          "physics":{},"output":{"engine":"HDF5"}})");
        for (const auto* name : {"xi","eta","zeta"})
            for (const auto* term : {"advection","stretching","twisting"})
                base["dynamics"]["prognostic_variables"][name]["tendency_terms"][term] =
                    {{"enable",true},{"spatial_scheme","Takacs"},{"temporal_scheme","AdamsBashforth2"}};
        const auto save = [&](const char* name, const Json& data) {
            const auto path = directory/name;
            std::ofstream(path) << data.dump(2);
            return path.string();
        };
        Utils::ConfigurationManager config(save("base.json",base));
        Core::Grid grid(config,MPI_COMM_WORLD);
        Core::Parameters params(config,grid);
#if defined(ENABLE_NCCL)
        ncclUniqueId id;
        require(ncclGetUniqueId(&id)==ncclSuccess,"NCCL id");
        require(ncclCommInitRank(&comm,1,id,0)==ncclSuccess,"NCCL init");
        const auto stream=Kokkos::Cuda().cuda_stream();
        Core::State state(config,params,grid,comm,stream);
        Core::HaloExchanger halo(config,grid,comm,stream);
#else
        Core::State state(config,params,grid);
        Core::HaloExchanger halo(grid);
#endif
        Core::BoundaryConditionManager boundary(grid,true);
        const int h=grid.get_halo_cells(), nz=grid.get_local_total_points_z();
        const int ny=grid.get_local_total_points_y(), nx=grid.get_local_total_points_x();
        const std::array<int,3> dims{nz,ny,nx};
        Core::Initializer initializer(config,grid,params,state,halo);
        Kokkos::deep_copy(state.get_field<2>("lon").get_mutable_device_data(),real(-999.));
        Kokkos::deep_copy(state.get_field<2>("lat").get_mutable_device_data(),real(-999.));
        initializer.initialize_geographic_coordinates();
        const auto longitude=state.get_field<2>("lon").get_host_data();
        const auto latitude=state.get_field<2>("lat").get_host_data();
        for (int j=0;j<ny;++j) for (int i=0;i<nx;++i) {
            require(std::abs(longitude(j,i)-(-1.+(i-h+.5)*2./16.))<1e-12,
                "RLL longitude in degrees, including halos");
            require(std::abs(latitude(j,i)-(-1.+(j-h+.5)*2./8.))<1e-12,
                "RLL latitude in degrees, including halos");
        }
        const auto fill = [&](const char* name, Real value) {
            Kokkos::deep_copy(state.get_field<3>(name).get_mutable_device_data(),value);
        };
        for (const char* name : {"qp","qc","qr","qi","qm","nc","nr","ni","bm"})
            if (!state.has_field(name)) state.add_field<3>(name,dims);
        for (const char* name : {"ITYPEU","ITYPEV","ITYPEW"}) fill(name,real(1.));
        fill("th",real(300.)); fill("qp",real(0.));
        fill("u",real(3.)); fill("v",real(1.)); fill("w",real(0.));
        for (const char* name : {"rhobar","rhobar_up"})
            Kokkos::deep_copy(state.get_field<1>(name).get_mutable_device_data(),real(2.));
        Kokkos::deep_copy(state.get_field<1>("thbar").get_mutable_device_data(),real(300.));
        Kokkos::deep_copy(params.dz_mid.get_mutable_device_data(),real(250.));
        params.max_topo_idx=h+1;
        auto mask=state.get_field<3>("ITYPEU").get_host_data();
        mask(h,h+2,h+3)=real(0.);
        Kokkos::deep_copy(state.get_field<3>("ITYPEU").get_mutable_device_data(),mask);

        // Controlled interface fixture, NOT an admitted moist full-model run:
        // initialize physical moisture explicitly and retain the startup guard.
        auto moist=base;
        moist["physics"]["p3"]["enable_p3"]=true;
        Utils::ConfigurationManager moist_config(save("moist.json",moist));
        bool rejected=false;
        try { Core::validate_model_numerical_configuration(moist_config,1); }
        catch (const std::exception&) { rejected=true; }
        require(rejected,"full-physics guard was bypassed");
        Dynamics::NumericalMethodFactory factory(moist_config,grid,halo,boundary,nullptr);
        const Json advection={{"tendency_terms",{{"advection",{{"enable",true},
            {"spatial_scheme","Takacs"},{"temporal_scheme","ForwardEuler"}}}}}};
        Dynamics::Operators::RegularLatLonScalarTransport transport(grid.geometry());
        Core::Field<3> expected("expected",dims);
        rejected=false;
        try { (void)factory.create("qp",advection,false,true,dims); }
        catch (const std::exception&) { rejected=true; }
        require(rejected,"diagnostic condensate qp must not be independently advected");
        for (const char* name : {"qv","qc","qr","qi","qm","nc","nr","ni","bm"}) {
            fill(name,real(.01));
            const std::string tendency="fe_tendency_"+std::string(name);
            if (!state.has_field(tendency)) state.add_field<3>(tendency,dims);
            auto method=factory.create(name,advection,false,true,dims);
            method->calculate_tendencies(state,grid,params);
#if defined(ENABLE_NCCL)
            // First ordinary execution initializes the shared tendency/halo
            // caches; capture and replay must then use the same live fields.
            if (std::string(name)=="qv") {
                Kokkos::fence();
                cudaGraph_t graph=nullptr;
                cudaGraphExec_t executable=nullptr;
                require(cudaStreamBeginCapture(stream,cudaStreamCaptureModeGlobal)==cudaSuccess,"begin moist transport capture");
                try {
                    method->calculate_tendencies(state,grid,params);
                }
                catch (...) {
                    // End capture before destructing Kokkos allocations, so
                    // cleanup does not conceal the original exception.
                    cudaStreamEndCapture(stream,&graph);
                    if (graph) cudaGraphDestroy(graph);
                    throw;
                }
                require(cudaStreamEndCapture(stream,&graph)==cudaSuccess,"end moist transport capture");
                require(cudaGraphInstantiate(&executable,graph,nullptr,nullptr,0)==cudaSuccess,"instantiate moist transport graph");
                for (const Real mixing_ratio : {real(.02), real(0.), real(.01)}) {
                    fill(name,mixing_ratio);
                    require(cudaGraphLaunch(executable,stream)==cudaSuccess,"replay moist transport graph");
                    require(cudaStreamSynchronize(stream)==cudaSuccess,"complete moist transport graph");
                    const auto replay=state.get_field<3>(tendency).get_host_data();
                    // Independent snapshot: get_host_data may alias host storage.
                    std::vector<Real> snapshot(replay.data(),replay.data()+replay.size());
                    method->calculate_tendencies(state,grid,params);
                    const auto direct=state.get_field<3>(tendency).get_host_data();
                    for (std::size_t n=0;n<snapshot.size();++n)
                        require(snapshot[n]==direct.data()[n],"changed-input graph replay parity");
                }
                cudaGraphExecDestroy(executable);
                cudaGraphDestroy(graph);
            }
#endif
            Kokkos::deep_copy(expected.get_mutable_device_data(),real(0.));
            transport.add_flux_convergence(state.get_field<3>(name),state.get_field<3>("u_mean"),
                state.get_field<3>("v_mean"),state.get_field<3>("w_mean"),params.dz_mid,expected,h,nz-h);
            const auto actual=state.get_field<3>(tendency).get_host_data();
            const auto reference=expected.get_host_data();
            for (int k=h;k<nz-h;++k) for (int j=h;j<ny-h;++j) for (int i=h;i<nx-h;++i)
                require(std::abs(actual(k,j,i)-reference(k,j,i)/real(2.))<real(1e-14),
                    "moist scalar density normalization");
        }
        const auto mass_u=state.get_field<3>("u_mean").get_host_data();
        const auto mass_v=state.get_field<3>("v_mean").get_host_data();
        require(mass_u(h,h+2,h+3)==0 && mass_u(h,h+1,h+1)==6,"native terrain mass-flux mask");
        require(mass_v(h,h-1,h)==0 && mass_v(h,ny-h-1,h)==0,"no normal scalar wall flux");

        auto vapor=state.get_field<3>("qv").get_host_data();
        for (int k=0;k<nz;++k) for (int j=0;j<ny;++j) for (int i=0;i<nx;++i)
            vapor(k,j,i)=real(.01)+real(.0001)*j;
        Kokkos::deep_copy(state.get_field<3>("qv").get_mutable_device_data(),vapor);
        const Json buoyancy={{"tendency_terms",{{"buoyancy",{{"enable",true},
            {"spatial_scheme","Takacs"},{"temporal_scheme","ForwardEuler"}}}}}};
        for (const char* name : {"xi","eta"}) {
            const std::string tendency="fe_tendency_"+std::string(name);
            if (!state.has_field(tendency)) state.add_field<3>(tendency,dims);
            auto method=factory.create(name,buoyancy,false,false,dims);
            method->calculate_tendencies(state,grid,params);
            const auto value=state.get_field<3>(tendency).get_host_data();
            const Real ref=std::string(name)=="xi" ? real(9.806*.608*.0001)/
                (real(6371220.)*grid.horizontal_specification().geometry.dq2) : real(0.);
            require(std::abs(value(h,h+1,h+1)-ref)<real(1e-14),"factory moist buoyancy physical gradient");
            require(value(nz-h-1,h+1,h+1)==0,"buoyancy upper level excluded");
        }
        std::cout << "PASS: RLL moist factory, scalar normalization, terrain flux masks, wall fluxes and buoyancy\n";
    }
    catch (const std::exception& e) { std::cerr << e.what() << '\n'; result=1; }
#if defined(ENABLE_NCCL)
    if (comm) ncclCommDestroy(comm);
#endif
    std::filesystem::remove_all(directory);
    Kokkos::finalize();
    MPI_Finalize();
    return result;
}
