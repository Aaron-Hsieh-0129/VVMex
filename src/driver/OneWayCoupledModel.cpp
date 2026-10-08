#include "driver/OneWayCoupledModel.hpp"
#include "driver/Model.hpp"
#include "driver/FineToCoarseRemapper.hpp"
#include "driver/ConvectiveTransportFeedback.hpp"
#include "driver/ConservativeFluxFeedback.hpp"
#include "core/ModelConfigurationValidation.hpp"
#include "io/history/LegacyHistoryWriter.hpp"
#include "io/bp5/Bp5HistoryWriter.hpp"
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>

namespace VVM::Driver {
namespace {
struct ModelContext {
    Utils::ConfigurationManager config;
    Core::Grid grid;
    Core::Parameters parameters;
    Core::State state;
    Core::HaloExchanger halos;
    Model model;
    std::unique_ptr<IO::HistoryWriter> writer;

    ModelContext(const Utils::ConfigurationManager& configuration, MPI_Comm comm
#if defined(ENABLE_NCCL)
        , ncclComm_t nccl_comm, cudaStream_t stream
#endif
    ) : config(configuration), grid(config,comm), parameters(config,grid),
#if defined(ENABLE_NCCL)
        state(config,parameters,grid,nccl_comm,stream), halos(config,grid,nccl_comm,stream),
#else
        state(config,parameters,grid), halos(grid),
#endif
        model(config,parameters,grid,state,halos) {}

    void open_output() {
        if (config.get_value<std::string>("output.engine","HDF5") == "BP5") {
            writer = std::make_unique<IO::BP5::Bp5HistoryWriter>(config,grid,parameters,state,grid.get_comm());
        } else {
            writer = std::make_unique<IO::LegacyHistoryWriter>(config,grid,parameters,state,grid.get_comm());
        }
        if (config.get_value<bool>("output.output_initial_step",true)) writer->write(0,real(0.));
    }
};

bool close(Real a, Real b) {
    return std::abs(a-b) <= real(32.)*std::numeric_limits<Real>::epsilon()*
        std::max(real(1.),std::max(std::abs(a),std::abs(b)));
}

int coupling_compute_ranks(MPI_Comm comm
#if defined(ENABLE_NCCL)
    , ncclComm_t nccl_comm
#endif
) {
    int size=0;
#if defined(ENABLE_NCCL)
    const auto status=ncclCommCount(nccl_comm,&size);
    if(status!=ncclSuccess) throw std::runtime_error(std::string("NCCL coupling communicator query failed: ")+ncclGetErrorString(status));
#else
    if(MPI_Comm_size(comm,&size)!=MPI_SUCCESS) throw std::runtime_error("MPI coupling communicator query failed.");
#endif
    return size;
}

void validate_pair(const Utils::ConfigurationManager& fine, const Utils::ConfigurationManager& coarse, bool conservative=false) {
    for (const auto* config : {&fine,&coarse}) {
        Core::validate_model_numerical_configuration(*config,1);
        const auto engine = config->get_value<std::string>("output.engine","HDF5");
        if (!Core::is_rll_bubble(*config) || config->get_value<bool>("restart.enable",false) ||
            (engine != "HDF5" && engine != "BP5"))
            throw std::runtime_error("One-way prototype requires fresh rll_bubble models with HDF5 or BP5 output.");
        for (const char* key : {"physics.rrtmgp.enable_rrtmgp","physics.surface_process.enable",
            "dynamics.forcings.random_perturbation.enable","dynamics.forcings.areamn.enable",
            "dynamics.forcings.lateral_boundary_nudging.enable"}) {
            if (config->get_value<bool>(key,false))
                throw std::runtime_error(std::string("Unsupported one-way prototype process: ")+key);
        }
    }
    if (!fine.get_value<bool>("physics.p3.enable_p3",false) ||
        coarse.get_value<bool>("physics.p3.enable_p3",false) != conservative ||
        !fine.get_value<bool>("initial_conditions.bubble.enable",false) ||
        coarse.get_value<bool>("initial_conditions.bubble.enable",false))
        throw std::runtime_error("One-way experiment requires fine P3+bubble and coarse without P3 or bubble.");
    for (const char* key : {"simulation.dt_s","simulation.total_time_s","simulation.output_interval_s"}) {
        if (!close(fine.get_value<Real>(key),coarse.get_value<Real>(key)))
            throw std::runtime_error(std::string("Fine/coarse clocks differ: ")+key);
    }
    const auto f = Core::GridSpecification::from_config(fine);
    const auto c = Core::GridSpecification::from_config(coarse);
    if (f.vertical.nz != c.vertical.nz || f.vertical.type != c.vertical.type ||
        !close(f.vertical.dz,c.vertical.dz) || !close(f.vertical.dz1,c.vertical.dz1))
        throw std::runtime_error("Fine/coarse vertical grids do not match.");
    if (f.horizontal.topology.q1 != c.horizontal.topology.q1 ||
        f.horizontal.topology.q2 != c.horizontal.topology.q2)
        throw std::runtime_error("Fine/coarse boundary topologies do not match.");
    const auto out_f = std::filesystem::absolute(fine.get_value<std::string>("output.output_dir")).lexically_normal();
    const auto out_c = std::filesystem::absolute(coarse.get_value<std::string>("output.output_dir")).lexically_normal();
    if (out_f == out_c) throw std::runtime_error("Fine and coarse output directories must be distinct.");
}

void validate_reference(const ModelContext& fine, const ModelContext& coarse) {
    for (const char* name : {"rhobar","rhobar_up","pbar","pbar_up","dpbar_mid",
        "pibar","pibar_up","thbar","qvbar","Tbar","Tvbar"}) {
        const auto f = fine.state.get_field<1>(name).get_host_data();
        const auto c = coarse.state.get_field<1>(name).get_host_data();
        if (f.extent(0) != c.extent(0)) throw std::runtime_error("Fine/coarse reference dimensions differ.");
        for (size_t k = 0; k < f.extent(0); ++k) {
            const auto tolerance = real(32.)*std::numeric_limits<Real>::epsilon()*
                std::max(real(1.e-30),std::max(std::abs(f(k)),std::abs(c(k))));
            if (!std::isfinite(f(k)) || !std::isfinite(c(k)) || std::abs(f(k)-c(k)) > tolerance)
                throw std::runtime_error(std::string("Fine/coarse reference profiles differ: ")+name);
        }
    }
    for (const auto& pair : {std::make_pair(&fine.parameters.z_mid,&coarse.parameters.z_mid),
        std::make_pair(&fine.parameters.z_up,&coarse.parameters.z_up)}) {
        const auto f = pair.first->get_host_data(), c = pair.second->get_host_data();
        if (f.extent(0) != c.extent(0)) throw std::runtime_error("Fine/coarse vertical coordinates differ.");
        for (size_t k=0;k<f.extent(0);++k) {
            if (!std::isfinite(f(k)) || !std::isfinite(c(k)) || !close(f(k),c(k)))
                throw std::runtime_error("Fine/coarse initialized vertical coordinates differ.");
        }
    }
}
} // namespace

void run_conservative_coupled(const Utils::ConfigurationManager& config, MPI_Comm comm
#if defined(ENABLE_NCCL)
    , ncclComm_t nccl_comm, cudaStream_t stream
#endif
) {
    const int size=coupling_compute_ranks(comm
#if defined(ENABLE_NCCL)
        ,nccl_comm
#endif
    );
    if(size!=1) throw std::runtime_error("Conservative coupling requires one compute rank.");
    const auto ratio=config.get_value<nlohmann::json>("coupling.horizontal_ratio",4);
    if(!ratio.is_number_integer() || ratio!=4) throw std::runtime_error("Conservative coupling requires ratio 4.");
    const Real alpha=config.get_value<Real>("coupling.feedback_strength",real(1.));
    if(!std::isfinite(alpha)||alpha<0) throw std::runtime_error("Invalid coupling strength.");
    const auto variables=config.get_value<nlohmann::json>("coupling.variables",nlohmann::json::object());
    for(const auto& item : variables.items())
        if((item.key()!="theta" && item.key()!="qv" && item.value().get<bool>()) ||
           ((item.key()=="theta" || item.key()=="qv") && !item.value().get<bool>()))
            throw std::runtime_error("Conservative mode couples theta and qv together only.");
    Utils::ConfigurationManager fc(config.get_value<std::string>("coupling.fine_config")),
        cc(config.get_value<std::string>("coupling.coarse_config"));
    validate_pair(fc,cc,true);
    for(const auto* cfg : {&fc,&cc}) {
        if(cfg->get_value<bool>("dynamics.forcings.sponge_layer.enable",false))
            throw std::runtime_error("Residual diagnostics exclude sponge forcing.");
        for(const char* var : {"th","qv","qc","qr","qi","qm","nc","nr","ni","bm"}) {
            const auto key=std::string("dynamics.prognostic_variables.")+var+".tendency_terms.advection.";
            if(cfg->get_value<std::string>(key+"spatial_scheme","")!="MUSCL" ||
                cfg->get_value<std::string>(key+"temporal_scheme","")!="SSPRK2")
                throw std::runtime_error("Conservative mode requires MUSCL/SSPRK2 for every P3 scalar.");
        }
    }
    ModelContext fine(fc,comm
#if defined(ENABLE_NCCL)
        ,nccl_comm,stream
#endif
    ), coarse(cc,comm
#if defined(ENABLE_NCCL)
        ,nccl_comm,stream
#endif
    ), shadow(cc,comm
#if defined(ENABLE_NCCL)
        ,nccl_comm,stream
#endif
    );
    FineToCoarseRemapper remap(fine.grid.geometry(),coarse.grid.geometry(),fine.grid.get_local_physical_points_z());
    fine.model.init(); coarse.model.init(); shadow.model.init(); validate_reference(fine,coarse);
    fine.model.enable_transport_flux_capture(); shadow.model.enable_transport_flux_capture();
    fine.model.enable_physics_feedback_capture(true); shadow.model.enable_physics_feedback_capture(true);
    coarse.model.enable_physics_feedback_capture(true);
    const auto& fg=dynamic_cast<const Core::Geometry::RegularLatLonGeometry&>(fine.grid.geometry());
    const auto& cg=dynamic_cast<const Core::Geometry::RegularLatLonGeometry&>(coarse.grid.geometry());
    const int h=coarse.grid.get_halo_cells(),nz=coarse.grid.get_local_total_points_z(),
        ny=coarse.grid.get_local_total_points_y(),nx=coarse.grid.get_local_total_points_x();
    const Real cp=cc.get_value<Real>("constants.Cp"),lv=cc.get_value<Real>("constants.Lv"),lf=real(3.337e5);
    ConservativeFluxFeedback transport(fg,cg,coarse.parameters,nz-2*h,cp);
    for(const std::string label : {"theta","qv"}) for(const std::string prefix : {
        "feedback_","feedback_transport_","feedback_applied_","feedback_unapplied_",
        "residual_total_","residual_physics_","residual_diffusion_","residual_closure_"})
        coarse.state.add_field<3>(prefix+label,{nz,ny,nx},
            {Core::GridStaggering::Centered,label=="theta"?"K s-1":"kg kg-1 s-1","one-way flux/residual diagnostic"});
    Core::Field<3> mapped("mapped_diagnostic",{nz,ny,nx});
    const Real strength=config.get_value<bool>("coupling.enabled",true)?alpha:real(0.);
    ThermodynamicFeedback feedback{nullptr,nullptr,strength,false,&transport};
    const Real dt=fc.get_value<Real>("simulation.dt_s"),end=fc.get_value<Real>("simulation.total_time_s"),
        interval=fc.get_value<Real>("simulation.output_interval_s");
    if(dt<=0 || end<=0 || interval<dt || !close(end/dt,std::round(end/dt)))
        throw std::runtime_error("Require positive, step-aligned duration and output interval >= dt.");
    fine.open_output(); coarse.open_output();
    const auto file=cc.get_value<std::string>("output.output_dir")+"/conservative_budget.csv";
    std::ofstream budget(file); if(!budget) throw std::runtime_error("Cannot open conservative budget.");
    budget << "time_s,alpha,fine_water_kg,coarse_water_kg,fine_energy_J,coarse_energy_J,fine_qv_min,coarse_qv_min"
        << ",transport_water_kg_s,transport_energy_J_s,transport_water_abs_kg_s,transport_energy_abs_J_s"
        << ",residual_theta_max,residual_qv_max,limited_qv_abs_kg_s,fine_p3_water_kg_s,coarse_p3_water_kg_s,fine_surface_precip_kg_s,coarse_surface_precip_kg_s\n" << std::setprecision(17);
    auto inventory=[&](ModelContext& model,bool energy) {
        const auto& g=dynamic_cast<const Core::Geometry::RegularLatLonGeometry&>(model.grid.geometry());
        const auto area=FineToCoarseRemapper::areas(g),z=model.parameters.z_up.get_device_data();
        const auto rho=model.state.get_field<1>("rhobar").get_device_data(),pi=model.state.get_field<1>("pibar").get_device_data();
        const auto t=model.state.get_field<3>("th").get_device_data(),q=model.state.get_field<3>("qv").get_device_data(),
            c=model.state.get_field<3>("qc").get_device_data(),r=model.state.get_field<3>("qr").get_device_data(),
            ice=model.state.get_field<3>("qi").get_device_data();
        Real sum=0; const int mx=model.grid.get_local_total_points_x(),my=model.grid.get_local_total_points_y();
        Kokkos::parallel_reduce("CoupledInventory",Kokkos::MDRangePolicy<Kokkos::Rank<3>>({h,h,h},{nz-h,my-h,mx-h}),
            KOKKOS_LAMBDA(int k,int j,int i,Real& v) {
                v+=rho(k)*area(j)*(z(k)-z(k-1))*(energy ? cp*pi(k)*t(k,j,i)+lv*q(k,j,i)-lf*ice(k,j,i)
                    : q(k,j,i)+c(k,j,i)+r(k,j,i)+ice(k,j,i));
            },sum); return sum;
    };
    auto minimum=[&](ModelContext& model) {
        const auto q=model.state.get_field<3>("qv").get_device_data(); Real value;
        Kokkos::parallel_reduce("VaporMinimum",Kokkos::MDRangePolicy<Kokkos::Rank<3>>({h,h,h},
            {nz-h,model.grid.get_local_total_points_y()-h,model.grid.get_local_total_points_x()-h}),
            KOKKOS_LAMBDA(int k,int j,int i,Real& v) {if(q(k,j,i)<v) v=q(k,j,i);},Kokkos::Min<Real>(value));return value;
    };
    auto mass_integral=[&](const std::string& name,bool energy,bool absolute) {
        const auto q=coarse.state.get_field<3>(name).get_device_data(),
            v=coarse.state.get_field<3>("feedback_applied_qv").get_device_data();
        const auto area=FineToCoarseRemapper::areas(cg),z=coarse.parameters.z_up.get_device_data();
        const auto rho=coarse.state.get_field<1>("rhobar").get_device_data(),pi=coarse.state.get_field<1>("pibar").get_device_data();
        Real value=0;
        Kokkos::parallel_reduce("CouplingMassIntegral",Kokkos::MDRangePolicy<Kokkos::Rank<3>>({h,h,h},{nz-h,ny-h,nx-h}),
            KOKKOS_LAMBDA(int k,int j,int i,Real& sum) {
                const Real a=energy ? cp*pi(k)*q(k,j,i)+lv*v(k,j,i) : q(k,j,i);
                sum+=rho(k)*area(j)*(z(k)-z(k-1))*(absolute?Kokkos::abs(a):a);
            },value); return value;
    };
    auto water_budget=[&](ModelContext& model,bool precipitation) {
        const auto& geom=dynamic_cast<const Core::Geometry::RegularLatLonGeometry&>(model.grid.geometry());
        const auto area=FineToCoarseRemapper::areas(geom),z=model.parameters.z_up.get_device_data(),
            rho=model.state.get_field<1>("rhobar").get_device_data();
        const int mx=model.grid.get_local_total_points_x(),my=model.grid.get_local_total_points_y();
        Real sum=0;
        if(precipitation) {
            const auto l=model.state.get_field<2>("precip_liq_surf_flux").get_device_data(),
                ice=model.state.get_field<2>("precip_ice_surf_flux").get_device_data();
            Kokkos::parallel_reduce("SurfacePrecipitationBudget",Kokkos::MDRangePolicy<Kokkos::Rank<2>>({h,h},{my-h,mx-h}),
                KOKKOS_LAMBDA(int j,int i,Real& v){v+=area(j)*(l(j,i)+ice(j,i));},sum);
        } else {
            const auto q=model.state.get_field<3>("physics_tendency_qt").get_device_data();
            Kokkos::parallel_reduce("P3TotalWaterBudget",Kokkos::MDRangePolicy<Kokkos::Rank<3>>({h,h,h},{nz-h,my-h,mx-h}),
                KOKKOS_LAMBDA(int k,int j,int i,Real& v){v+=area(j)*rho(k)*(z(k)-z(k-1))*q(k,j,i);},sum);
        }
        return sum;
    };
    auto write_budget=[&]() {
        budget<<fine.state.get_time()<<','<<strength<<','<<inventory(fine,false)<<','<<inventory(coarse,false)
            <<','<<inventory(fine,true)<<','<<inventory(coarse,true)<<','<<minimum(fine)<<','<<minimum(coarse);
        for(bool absolute : {false,true}) {
            budget<<','<<mass_integral("feedback_applied_qv",false,absolute)
                <<','<<mass_integral("feedback_applied_theta",true,absolute);
        }
        for(const std::string label : {"theta","qv"}) {
            const auto f=coarse.state.get_field<3>("residual_closure_"+label).get_device_data(); Real value=0;
            Kokkos::parallel_reduce("ResidualClosure",Kokkos::RangePolicy<>(0,f.size()),
                KOKKOS_LAMBDA(size_t n,Real& v) {const Real a=Kokkos::abs(f.data()[n]);if(a>v)v=a;},Kokkos::Max<Real>(value));
            budget<<','<<value;
        }
        budget<<','<<mass_integral("feedback_unapplied_qv",false,true);
        for(bool precipitation : {false,true}) for(auto* model : {&fine,&coarse}) budget<<','<<water_budget(*model,precipitation);
        budget<<'\n';
        if(!budget) throw std::runtime_error("Conservative budget write failed.");
    };
    write_budget(); Real next=interval;
    while(fine.state.get_time()<end) {
        // The diagnostic shadow is reconstructed exclusively from fine state.
        // Coarse receiver state never enters this path.
        for(const std::string name : {"th","qv","qc","qr","qi","qm","nc","nr","ni","bm"}) {
            remap.remap(fine.state.get_field<3>(name),shadow.state.get_field<3>(name));
            if(fine.state.has_field(name+"_m") && shadow.state.has_field(name+"_m"))
                remap.remap(fine.state.get_field<3>(name+"_m"),shadow.state.get_field<3>(name+"_m"));
        }
        transport.project_wind(fine.state,shadow.state); shadow.model.prepare_projected_state();
        fine.model.run_step(dt); shadow.model.run_step(dt,nullptr,true);
        transport.diagnose(fine.state,shadow.state,coarse.state);
        for(const std::string label : {"theta","qv"}) {
            const std::string name=label=="theta"?"th":"qv";
            for(int component=0;component<3;++component) {
                const std::string source=component==0?name:component==1?"physics_tendency_"+name:"fe_tendency_"+name;
                remap.remap(fine.state.get_field<3>(source),mapped);
                const auto f=mapped.get_device_data(),s=shadow.state.get_field<3>(source).get_device_data();
                const std::string prefix=component==0?"residual_total_":component==1?"residual_physics_":"residual_diffusion_";
                const auto out=coarse.state.get_field<3>(prefix+label).get_mutable_device_data();
                Kokkos::parallel_for("ExactProcessResidual",Kokkos::MDRangePolicy<Kokkos::Rank<3>>({h,h,h},{nz-h,ny-h,nx-h}),
                    KOKKOS_LAMBDA(int k,int j,int i) { out(k,j,i)=(f(k,j,i)-s(k,j,i))/(component==0?dt:real(1.)); });
            }
            const auto total=coarse.state.get_field<3>("residual_total_"+label).get_device_data(),
                adv=coarse.state.get_field<3>("feedback_transport_"+label).get_device_data(),
                phy=coarse.state.get_field<3>("residual_physics_"+label).get_device_data(),
                dif=coarse.state.get_field<3>("residual_diffusion_"+label).get_device_data();
            const auto out=coarse.state.get_field<3>("residual_closure_"+label).get_mutable_device_data();
            Kokkos::parallel_for("ResidualBudgetClosure",Kokkos::MDRangePolicy<Kokkos::Rank<3>>({h,h,h},{nz-h,ny-h,nx-h}),
                KOKKOS_LAMBDA(int k,int j,int i) {out(k,j,i)=total(k,j,i)-adv(k,j,i)-phy(k,j,i)-dif(k,j,i);});
        }
        coarse.model.run_step(dt,&feedback);
        for(auto* model : {&fine,&coarse,&shadow}) {model->state.increment_step();model->state.advance_time(dt);}
        write_budget();
        if(fine.state.get_time()>=next || fine.state.get_time()>=end) {
            fine.writer->write(fine.state.get_step(),fine.state.get_time());
            coarse.writer->write(coarse.state.get_step(),coarse.state.get_time()); next+=interval;
            std::cout<<"[ConservativeOneWay] time="<<fine.state.get_time()<<" alpha="<<strength<<std::endl;
        }
    }
    fine.writer->close();coarse.writer->close();
    fine.model.finalize();coarse.model.finalize();shadow.model.finalize();Kokkos::fence();
}

void run_physics_coupled(const Utils::ConfigurationManager& config, MPI_Comm comm
#if defined(ENABLE_NCCL)
    , ncclComm_t nccl_comm, cudaStream_t stream
#endif
) {
    const int size=coupling_compute_ranks(comm
#if defined(ENABLE_NCCL)
        ,nccl_comm
#endif
    );
    if(size!=1) throw std::runtime_error("Physics feedback requires one compute rank.");
    const auto ratio=config.get_value<nlohmann::json>("coupling.horizontal_ratio",4);
    if(!ratio.is_number_integer() || ratio!=4) throw std::runtime_error("Physics feedback requires ratio 4.");
    const Real alpha=config.get_value<Real>("coupling.feedback_strength",real(1.));
    if(!std::isfinite(alpha) || alpha<0) throw std::runtime_error("Invalid physics feedback strength.");
    const auto variables=config.get_value<nlohmann::json>("coupling.variables",nlohmann::json::object());
    if(!variables.is_object()) throw std::runtime_error("coupling.variables must be an object of boolean switches.");
    for(const auto& item : variables.items()) {
        if(!item.value().is_boolean()) throw std::runtime_error("Coupling variable switches must be boolean.");
        if(item.key()!="theta" && item.key()!="qv" && item.key()!="xi" && item.key()!="eta" && item.key()!="zeta")
            throw std::runtime_error("Physics feedback supports theta, qv, xi, eta and zeta; winds are diagnosed from vorticity.");
    }
    Utils::ConfigurationManager fc(config.get_value<std::string>("coupling.fine_config")),
        cc(config.get_value<std::string>("coupling.coarse_config"));
    validate_pair(fc,cc);
    for(const char* key : {"physics.p3.enable_p3","physics.turbulence.enable_turbulence",
        "physics.rrtmgp.enable_rrtmgp","physics.surface_process.enable"})
        if(cc.get_value<bool>(key,false)) throw std::runtime_error(std::string("Dynamics-only receiver requires disabled ")+key);
    const Real dt=fc.get_value<Real>("simulation.dt_s"),end=fc.get_value<Real>("simulation.total_time_s"),
        interval=fc.get_value<Real>("simulation.output_interval_s");
    if(!std::isfinite(dt) || !std::isfinite(end) || !std::isfinite(interval) || dt<=0 || end<=0 ||
        interval<dt || !close(end/dt,std::round(end/dt)))
        throw std::runtime_error("Require finite positive, step-aligned clocks and output interval >= dt.");
    ModelContext fine(fc,comm
#if defined(ENABLE_NCCL)
        ,nccl_comm,stream
#endif
    ),coarse(cc,comm
#if defined(ENABLE_NCCL)
        ,nccl_comm,stream
#endif
    );
    FineToCoarseRemapper remap(fine.grid.geometry(),coarse.grid.geometry(),fine.grid.get_local_physical_points_z());
    fine.model.init(); coarse.model.init(); validate_reference(fine,coarse);
    fine.model.enable_physics_feedback_capture(); // P3-only output remains a separate diagnostic.
    const int nz=coarse.grid.get_local_total_points_z(),ny=coarse.grid.get_local_total_points_y(),
        nx=coarse.grid.get_local_total_points_x();
    std::vector<std::string> capture{"th"},coupled;
    ThermodynamicFeedback feedback;
    feedback.strength=config.get_value<bool>("coupling.enabled",true)?alpha:real(0.);
    feedback.limit_vapor=config.get_value<bool>("coupling.limit_vapor",true);
    for(const std::string label : {"theta","qv","xi","eta","zeta"}) {
        const std::string name=label=="theta"?"th":label;
        const auto staggering=label=="xi"?Core::GridStaggering::StaggeredYZ:
            label=="eta"?Core::GridStaggering::StaggeredXZ:label=="zeta"?Core::GridStaggering::StaggeredXY:
            Core::GridStaggering::Centered;
        const std::string units=label=="theta"?"K s-1":label=="qv"?"kg kg-1 s-1":"s-2";
        for(const std::string prefix : {"feedback_","feedback_applied_","feedback_unapplied_"})
            coarse.state.add_field<3>(prefix+label,{nz,ny,nx},
                {staggering,units,"fine physical-process forcing; disabled variables remain zero"});
        if(!variables.value(label,label=="theta")) continue;
        coupled.push_back(name);
        if(name!="th") capture.push_back(name);
        auto* field=&coarse.state.get_field<3>("feedback_"+label);
        if(label=="theta") feedback.theta=field;
        else if(label=="qv") feedback.qv=field;
        else feedback.dynamics.push_back({label,field});
    }
    fine.model.enable_physics_tendency_capture(capture);
    fine.open_output(); coarse.open_output();
    Real next=interval;
    while(fine.state.get_time()<end) {
        fine.model.run_step(dt);
        for(const auto& name : coupled) {
            const auto label=name=="th"?"theta":name;
            auto& target=coarse.state.get_field<3>("feedback_"+label);
            remap.remap_staggered(fine.state.get_field<3>("coupling_physics_"+name),target,
                target.get_metadata().grid_staggering);
        }
        coarse.model.run_step(dt,&feedback);
        for(auto* model : {&fine,&coarse}) { model->state.increment_step(); model->state.advance_time(dt); }
        if(fine.state.get_time()>=next || fine.state.get_time()>=end) {
            fine.writer->write(fine.state.get_step(),fine.state.get_time());
            coarse.writer->write(coarse.state.get_step(),coarse.state.get_time()); next+=interval;
            std::cout<<"[OneWayPhysics] time="<<fine.state.get_time()<<" alpha="<<feedback.strength<<std::endl;
        }
    }
    fine.writer->close(); coarse.writer->close(); fine.model.finalize(); coarse.model.finalize(); Kokkos::fence();
}

void run_one_way_coupled(const Utils::ConfigurationManager& config, MPI_Comm comm
#if defined(ENABLE_NCCL)
    , ncclComm_t nccl_comm, cudaStream_t stream
#endif
) {
    if(config.get_value<std::string>("coupling.type","")=="one_way_physics_feedback") {
        run_physics_coupled(config,comm
#if defined(ENABLE_NCCL)
            ,nccl_comm,stream
#endif
        );
        return;
    }
    if(config.get_value<std::string>("coupling.type","")=="one_way_conservative_transport") {
        run_conservative_coupled(config,comm
#if defined(ENABLE_NCCL)
            ,nccl_comm,stream
#endif
        );
        return;
    }
    const int size=coupling_compute_ranks(comm
#if defined(ENABLE_NCCL)
        ,nccl_comm
#endif
    );
    if (size != 1) throw std::runtime_error("One-way coupling requires exactly one compute rank.");
    if (config.get_value<std::string>("coupling.type","") != "one_way_fine_to_coarse")
        throw std::runtime_error("Unsupported coupling type; expected one_way_fine_to_coarse.");
    const Real alpha = config.get_value<Real>("coupling.feedback_strength",real(1.));
    if (!std::isfinite(alpha) || alpha < real(0.))
        throw std::runtime_error("coupling.feedback_strength must be finite and nonnegative.");
    const bool theta_enabled = config.get_value<bool>("coupling.variables.theta",true);
    const bool qv_enabled = config.get_value<bool>("coupling.variables.qv",true);
    const bool transport_enabled=config.get_value<bool>("coupling.convective_transport",false);
    const bool limit_vapor=config.get_value<bool>("coupling.limit_vapor",false);
    const auto ratio = config.get_value<nlohmann::json>("coupling.horizontal_ratio",4);
    if (!ratio.is_number_integer() || ratio != 4)
        throw std::runtime_error("coupling.horizontal_ratio must be the integer 4.");
    const auto variables = config.get_value<nlohmann::json>("coupling.variables", nlohmann::json::object());
    for (const auto& item : variables.items()) {
        if (item.key() != "theta" && item.key() != "qv" && item.value().get<bool>())
            throw std::runtime_error("Only theta and qv feedback are supported.");
    }
    const auto fine_path = config.get_value<std::string>("coupling.fine_config");
    const auto coarse_path = config.get_value<std::string>("coupling.coarse_config");
    const Utils::ConfigurationManager fine_config(fine_path), coarse_config(coarse_path);
    validate_pair(fine_config,coarse_config);
    if(limit_vapor && coarse_config.get_value<std::string>(
        "dynamics.prognostic_variables.qv.tendency_terms.advection.spatial_scheme","")!="MUSCL")
        throw std::runtime_error("Vapor-limited coupling requires MUSCL/SSPRK2 coarse vapor transport.");
    ModelContext fine(fine_config,comm
#if defined(ENABLE_NCCL)
        ,nccl_comm,stream
#endif
    );
    ModelContext coarse(coarse_config,comm
#if defined(ENABLE_NCCL)
        ,nccl_comm,stream
#endif
    );
    FineToCoarseRemapper remapper(fine.grid.geometry(),coarse.grid.geometry(),
        fine.grid.get_local_physical_points_z(),4);
    fine.model.init();
    coarse.model.init();
    validate_reference(fine,coarse);
    fine.model.enable_physics_feedback_capture();
    std::unique_ptr<ConvectiveTransportFeedback> transport;
    if(transport_enabled) transport=std::make_unique<ConvectiveTransportFeedback>(
        dynamic_cast<const Core::Geometry::RegularLatLonGeometry&>(fine.grid.geometry()),
        dynamic_cast<const Core::Geometry::RegularLatLonGeometry&>(coarse.grid.geometry()),
        fine.grid.get_local_physical_points_z());
    const int nz = coarse.grid.get_local_total_points_z();
    for (const auto& name : {"feedback_theta","feedback_qv","feedback_physics_theta","feedback_physics_qv",
        "feedback_transport_theta","feedback_transport_qv","feedback_applied_theta","feedback_applied_qv",
        "feedback_unapplied_theta","feedback_unapplied_qv"}) {
        coarse.state.add_field<3>(name,{nz,coarse.grid.get_local_total_points_y(),coarse.grid.get_local_total_points_x()},
            {Core::GridStaggering::Centered, std::string(name).find("theta")!=std::string::npos ? "K s-1" : "kg kg-1 s-1",
                "fine-to-coarse combined, physics, transport, applied or unapplied tendency"});
    }
    for (const auto& variable : {"theta","qv"}) {
        for (const auto& domain : {"fine","coarse"}) {
            coarse.state.add_field<1>(std::string(domain)+"_domain_integrated_"+variable+"_feedback",{nz},
                {Core::GridStaggering::Centered, std::string(variable)=="theta" ? "K m3 s-1" : "kg kg-1 m3 s-1",
                    "per-level volume integral of unscaled diagnosed feedback"});
        }
    }
    fine.open_output();
    coarse.open_output();
    const auto diagnostics_path = config.get_value<std::string>("coupling.diagnostics_file",
        coarse.config.get_value<std::string>("output.output_dir")+"/coupling_budget.csv");
    const auto parent = std::filesystem::path(diagnostics_path).parent_path();
    if (!parent.empty()) std::filesystem::create_directories(parent);
    std::ofstream budget(diagnostics_path);
    if (!budget) throw std::runtime_error("Cannot open coupling diagnostics: "+diagnostics_path);
    budget << "time_s,alpha,fine_theta,coarse_theta,fine_qv,coarse_qv"
           << ",transport_theta_mass,transport_qv_mass,applied_theta_mass,applied_qv_mass"
           << ",unapplied_theta_mass,unapplied_qv_mass\n" << std::setprecision(17);
    const Real strength = config.get_value<bool>("coupling.enabled",true) ? alpha : real(0.);
    ThermodynamicFeedback feedback{theta_enabled ? &coarse.state.get_field<3>("feedback_theta") : nullptr,
        qv_enabled ? &coarse.state.get_field<3>("feedback_qv") : nullptr,strength,limit_vapor};
    const Real dt = fine.config.get_value<Real>("simulation.dt_s");
    const Real end_time = fine.config.get_value<Real>("simulation.total_time_s");
    const Real output_interval = fine.config.get_value<Real>("simulation.output_interval_s");
    Real next_output = output_interval;
    while (fine.state.get_time() < end_time) {
        // Diagnose pre-step subfilter transport; the producer reads fine state only.
        if(transport) for(const auto& variable : {"theta","qv"}) {
            transport->diagnose(fine.state.get_field<3>(std::string(variable)=="theta" ? "th" : "qv"),
                fine.state.get_field<3>("u"),fine.state.get_field<3>("v"),fine.state.get_field<3>("w"),
                fine.state.get_field<1>("rhobar"),fine.state.get_field<1>("rhobar_up"),fine.parameters.z_up,
                coarse.state.get_field<3>(std::string("feedback_transport_")+variable));
        }
        fine.model.run_step(dt);
        for (const auto& variable : {"theta","qv"}) {
            const auto& source = fine.state.get_field<3>(std::string(variable)=="theta" ? "physics_tendency_th" : "physics_tendency_qv");
            auto& target = coarse.state.get_field<3>(std::string("feedback_")+variable);
            auto& physics = coarse.state.get_field<3>(std::string("feedback_physics_")+variable);
            remapper.remap(source,physics);
            remapper.integrals(source,true,fine.parameters.z_up,
                coarse.state.get_field<1>(std::string("fine_domain_integrated_")+variable+"_feedback"));
            remapper.integrals(physics,false,coarse.parameters.z_up,
                coarse.state.get_field<1>(std::string("coarse_domain_integrated_")+variable+"_feedback"));
            const auto p=physics.get_device_data();
            const auto t=coarse.state.get_field<3>(std::string("feedback_transport_")+variable).get_device_data();
            const auto out=target.get_mutable_device_data();
            Kokkos::parallel_for("CombinePhysicsTransportFeedback",Kokkos::RangePolicy<>(0,out.size()),
                KOKKOS_LAMBDA(size_t n) {out.data()[n]=p.data()[n]+t.data()[n];});
        }
        coarse.model.run_step(dt,&feedback);
        fine.state.increment_step(); fine.state.advance_time(dt);
        coarse.state.increment_step(); coarse.state.advance_time(dt);
        budget << fine.state.get_time() << ',' << strength;
        for (const auto& variable : {"theta","qv"}) {
            for (const auto& domain : {"fine","coarse"}) {
                const auto values = coarse.state.get_field<1>(std::string(domain)+"_domain_integrated_"+variable+"_feedback").get_host_data();
                double sum = 0.;
                for (int k = coarse.grid.get_halo_cells(); k < nz-coarse.grid.get_halo_cells(); ++k) sum += values(k);
                budget << ',' << sum;
            }
        }
        for(const auto& process : {"transport","applied","unapplied"})
            for(const auto& variable : {"theta","qv"}) {
                auto& integral=coarse.state.get_field<1>(std::string("coarse_domain_integrated_")+variable+"_feedback");
                // Reuse scratch only after the physics budgets have been written;
                // restore them below before history output.
                remapper.integrals(coarse.state.get_field<3>(std::string("feedback_")+process+"_"+variable),
                    false,coarse.parameters.z_up,integral);
                const auto values=integral.get_host_data();
                const auto density=coarse.state.get_field<1>("rhobar").get_host_data();
                double sum=0.;
                for(int k=coarse.grid.get_halo_cells();k<nz-coarse.grid.get_halo_cells();++k)
                    sum+=values(k)*density(k);
                budget << ',' << sum;
            }
        for(const auto& variable : {"theta","qv"})
            remapper.integrals(coarse.state.get_field<3>(std::string("feedback_physics_")+variable),
                false,coarse.parameters.z_up,
                coarse.state.get_field<1>(std::string("coarse_domain_integrated_")+variable+"_feedback"));
        budget << '\n';
        if (!budget) throw std::runtime_error("Failed to write coupling diagnostics.");
        if (fine.state.get_time() >= next_output || fine.state.get_time() >= end_time) {
            fine.writer->write(fine.state.get_step(),fine.state.get_time());
            coarse.writer->write(coarse.state.get_step(),coarse.state.get_time());
            std::cout << "[OneWayCoupling] time=" << fine.state.get_time() << " alpha=" << strength << std::endl;
            next_output += output_interval;
        }
    }
    fine.writer->close(); coarse.writer->close();
    fine.model.finalize(); coarse.model.finalize();
    Kokkos::fence();
}
} // namespace VVM::Driver
