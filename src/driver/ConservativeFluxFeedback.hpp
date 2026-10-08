#pragma once
#include "driver/ThermodynamicFeedback.hpp"
#include "driver/FineToCoarseRemapper.hpp"
#include "core/State.hpp"
#include <array>

namespace VVM::Driver {

// Face-integrated numerical flux residual. Only fine and fine-projected shadow
// states enter diagnose(). Receiver moisture is consulted solely by apply().
// Shared donor-limited faces conserve water; there is no cellwise qv clipping.
class ConservativeFluxFeedback final : public ReceiverTransport {
public:
    using Field=Core::Field<3>;
    ConservativeFluxFeedback(const Core::Geometry::RegularLatLonGeometry& fine,
        const Core::Geometry::RegularLatLonGeometry& coarse, const Core::Parameters& parameters,
        int nz, Real cp)
      : f_(fine), c_(coarse), p_(parameters), nz_(nz), cp_(cp),
        area_f_(FineToCoarseRemapper::areas(fine)), area_c_(FineToCoarseRemapper::areas(coarse)),
        donor_("flux_donor",shape()) {
        for(int v=0;v<2;++v) for(int a=0;a<3;++a) {
            flux_[v][a]=std::make_unique<Field>("residual_face",shape());
            applied_[v][a]=std::make_unique<Field>("applied_face",shape());
        }
    }
    std::array<int,3> shape() const {
        const auto l=c_.layout(); return {nz_+2*l.halo,l.local_total_ny(),l.local_total_nx()};
    }
    // Conservative native-face projection, preserving air mass flux with the
    // same spherical quadrature used to restrict numerical scalar fluxes.
    void project_wind(const Core::State& fine, Core::State& shadow) const {
        const int h=c_.layout().halo,nx=c_.layout().global_nx,ny=c_.layout().global_ny,nz=nz_;
        const auto af=area_f_,ac=area_c_;
        const Real sf=std::sin(f_.dq2()/2)/(f_.dq2()/2),sc=std::sin(c_.dq2()/2)/(c_.dq2()/2);
        for(int a=0;a<3;++a) {
            const std::string name=a==0?"u":a==1?"v":"w";
            const auto src=fine.get_field<3>(name).get_device_data();
            auto& field=shadow.get_field<3>(name); field.set_to_zero();
            const auto dst=field.get_mutable_device_data();
            Kokkos::parallel_for("ProjectFineFaceWind",Kokkos::MDRangePolicy<Kokkos::Rank<3>>(
                {h-1,h-1,h-1},{h+nz,h+ny,h+nx}),KOKKOS_LAMBDA(int k,int j,int i) {
                Real sum=0;
                if(a==0 && k>=h && j>=h) {
                    const int fi=h+4*(i-h+1)-1;
                    for(int n=0;n<4;++n) sum+=src(k,h+4*(j-h)+n,fi);
                    dst(k,j,i)=sum*real(.25)*sf/sc;
                } else if(a==1 && k>=h && i>=h) {
                    const int fj=h+4*(j-h+1)-1;
                    for(int n=0;n<4;++n) sum+=src(k,fj,h+4*(i-h)+n);
                    dst(k,j,i)=sum*real(.25)*sf/sc;
                } else if(a==2 && j>=h && i>=h) {
                    for(int jj=0;jj<4;++jj) for(int ii=0;ii<4;++ii) {
                        const int fj=h+4*(j-h)+jj;
                        sum+=af(fj)*src(k,fj,h+4*(i-h)+ii);
                    }
                    dst(k,j,i)=sum/ac(j);
                }
            });
        }
    }
    void diagnose(const Core::State& fine,const Core::State& shadow,Core::State& output) const {
        const int h=c_.layout().halo,nx=c_.layout().global_nx,ny=c_.layout().global_ny,nz=nz_;
        const Real sf=std::sin(f_.dq2()/2)/(f_.dq2()/2),sc=std::sin(c_.dq2()/2)/(c_.dq2()/2);
        const Real fdx=f_.dq1(),fdy=f_.dq2(),cdx=c_.dq1(),cdy=c_.dq2();
        const auto fu=f_.device_view(Core::Geometry::HorizontalLocation::U);
        const auto fv=f_.device_view(Core::Geometry::HorizontalLocation::V);
        const auto cu=c_.device_view(Core::Geometry::HorizontalLocation::U);
        const auto cv=c_.device_view(Core::Geometry::HorizontalLocation::V);
        const auto af=area_f_,ac=area_c_,z=p_.z_up.get_device_data();
        for(int v=0;v<2;++v) {
            const std::string name=v==0?"th":"qv",label=v==0?"theta":"qv";
            for(int a=0;a<3;++a) {
                const std::string axis=a==0?"x":a==1?"y":"z";
                const auto f=fine.get_field<3>("coupling_flux_"+axis+"_"+name).get_device_data();
                const auto c=shadow.get_field<3>("coupling_flux_"+axis+"_"+name).get_device_data();
                flux_[v][a]->set_to_zero();
                const auto out=flux_[v][a]->get_mutable_device_data();
                Kokkos::parallel_for("RestrictNumericalFaceFlux",Kokkos::MDRangePolicy<Kokkos::Rank<3>>(
                    {h-1,h-1,h-1},{h+nz,h+ny,h+nx}),KOKKOS_LAMBDA(int k,int j,int i) {
                    Real sum=0,base=0;
                    if(a==0 && k>=h && j>=h) {
                        const int fi=h+4*(i-h+1)-1;
                        for(int n=0;n<4;++n) {
                            const int fj=h+4*(j-h)+n;
                            sum+=f(k,fj,fi)*fu.sqrt_g(fj,fi)*fdy*sf*(z(k)-z(k-1));
                        }
                        base=c(k,j,i)*cu.sqrt_g(j,i)*cdy*sc*(z(k)-z(k-1));
                    } else if(a==1 && k>=h && i>=h && j>=h && j<h+ny-1) {
                        const int fj=h+4*(j-h+1)-1;
                        for(int n=0;n<4;++n) {
                            const int fi=h+4*(i-h)+n;
                            sum+=f(k,fj,fi)*fv.sqrt_g(fj,fi)*fdx*sf*(z(k)-z(k-1));
                        }
                        base=c(k,j,i)*cv.sqrt_g(j,i)*cdx*sc*(z(k)-z(k-1));
                    } else if(a==2 && k>=h && k<h+nz-1 && j>=h && i>=h) {
                        for(int jj=0;jj<4;++jj) for(int ii=0;ii<4;++ii) {
                            const int fj=h+4*(j-h)+jj;
                            sum+=af(fj)*f(k,fj,h+4*(i-h)+ii);
                        }
                        base=ac(j)*c(k,j,i);
                    }
                    out(k,j,i)=sum-base;
                });
            }
            // Record the exact advection residual before conversion to a
            // conservative sensible-energy flux for application.
            divergence(flux_[v],output,output.get_field<3>("feedback_transport_"+label),false);
            if(v==0) {
                const auto pi=shadow.get_field<1>("pibar").get_device_data();
                const auto piu=shadow.get_field<1>("pibar_up").get_device_data();
                const Real cp=cp_;
                for(int a=0;a<3;++a) {
                    const auto f=flux_[v][a]->get_mutable_device_data();
                    Kokkos::parallel_for("SensibleEnergyFlux",Kokkos::MDRangePolicy<Kokkos::Rank<3>>(
                        {h-1,h-1,h-1},{h+nz,h+ny,h+nx}),KOKKOS_LAMBDA(int k,int j,int i) {
                        f(k,j,i)*=cp*(a==2?piu(k):pi(k));
                    });
                }
            }
            divergence(flux_[v],output,output.get_field<3>("feedback_"+label),v==0);
        }
    }
    void divergence(const std::array<std::unique_ptr<Field>,3>& flux, const Core::State& state,
        Field& output,bool energy) const {
        const auto x=flux[0]->get_device_data(),y=flux[1]->get_device_data(),z=flux[2]->get_device_data();
        const auto rho=state.get_field<1>("rhobar").get_device_data(),pi=state.get_field<1>("pibar").get_device_data();
        const auto area=area_c_,height=p_.z_up.get_device_data();
        const int h=c_.layout().halo,nx=c_.layout().global_nx,ny=c_.layout().global_ny,nz=nz_;
        const Real cp=cp_; const auto out=output.get_mutable_device_data(); output.set_to_zero();
        Kokkos::parallel_for("ResidualFluxDivergence",Kokkos::MDRangePolicy<Kokkos::Rank<3>>(
            {h,h,h},{h+nz,h+ny,h+nx}),KOKKOS_LAMBDA(int k,int j,int i) {
            const Real mass=rho(k)*area(j)*(height(k)-height(k-1));
            out(k,j,i)=-(x(k,j,i)-x(k,j,i-1)+y(k,j,i)-y(k,j-1,i)+z(k,j,i)-z(k-1,j,i))
                /(mass*(energy?cp*pi(k):real(1.)));
        });
    }
    void apply(Core::State& state,Real dt,Real strength) const override {
        const int h=c_.layout().halo,nx=c_.layout().global_nx,ny=c_.layout().global_ny,nz=nz_;
        const auto rho=state.get_field<1>("rhobar").get_device_data(),pi=state.get_field<1>("pibar").get_device_data();
        const auto area=area_c_,height=p_.z_up.get_device_data(); const Real cp=cp_;
        for(int v=0;v<2;++v) {
            const std::string label=v==0?"theta":"qv";
            auto& applied=state.get_field<3>("feedback_applied_"+label); applied.set_to_zero();
            auto& withheld=state.get_field<3>("feedback_unapplied_"+label); withheld.set_to_zero();
            if(strength==real(0.)) continue; // exact standalone control
            const auto q=state.get_field<3>(v==0?"th":"qv").get_mutable_device_data();
            const auto x=flux_[v][0]->get_device_data(),y=flux_[v][1]->get_device_data(),z=flux_[v][2]->get_device_data();
            const auto donor=donor_.get_mutable_device_data();
            Kokkos::parallel_for("ReceiverOutgoingBudget",Kokkos::MDRangePolicy<Kokkos::Rank<3>>(
                {h,h,h},{h+nz,h+ny,h+nx}),KOKKOS_LAMBDA(int k,int j,int i) {
                const Real out=Kokkos::fmax(x(k,j,i),0.)+Kokkos::fmax(-x(k,j,i-1),0.)
                    +Kokkos::fmax(y(k,j,i),0.)+Kokkos::fmax(-y(k,j-1,i),0.)
                    +Kokkos::fmax(z(k,j,i),0.)+Kokkos::fmax(-z(k-1,j,i),0.);
                const Real avail=Kokkos::fmax(q(k,j,i),0.)*rho(k)*area(j)*(height(k)-height(k-1))*(v==0?cp*pi(k):real(1.));
                const Real requested=dt*strength*out;
                donor(k,j,i)=requested>avail ? (requested>0 ? (real(1.)-real(1.e-13))*avail/requested : real(1.)) : real(1.);
            });
            for(int a=0;a<3;++a) {
                const auto f=flux_[v][a]->get_device_data();
                applied_[v][a]->set_to_zero(); const auto out=applied_[v][a]->get_mutable_device_data();
                Kokkos::parallel_for("LimitSharedReceiverFace",Kokkos::MDRangePolicy<Kokkos::Rank<3>>(
                    {h-1,h-1,h-1},{h+nz,h+ny,h+nx}),KOKKOS_LAMBDA(int k,int j,int i) {
                    const Real value=f(k,j,i); if(value==0) return;
                    int dk=k,dj=j,di=i;
                    if(value<0) { if(a==0) ++di; else if(a==1) ++dj; else ++dk; }
                    if(di<h) di=h+nx-1; if(di>=h+nx) di=h;
                    out(k,j,i)=strength*value*donor(dk,dj,di);
                });
            }
            divergence(applied_[v],state,applied,v==0);
            const auto tend=applied.get_device_data(),req=state.get_field<3>("feedback_"+label).get_device_data();
            const auto diff=withheld.get_mutable_device_data();
            Kokkos::parallel_for("ApplyConservativeTransport",Kokkos::MDRangePolicy<Kokkos::Rank<3>>(
                {h,h,h},{h+nz,h+ny,h+nx}),KOKKOS_LAMBDA(int k,int j,int i) {
                q(k,j,i)+=dt*tend(k,j,i);
                diff(k,j,i)=strength*req(k,j,i)-tend(k,j,i);
            });
        }
    }
private:
    const Core::Geometry::RegularLatLonGeometry &f_,&c_;
    const Core::Parameters& p_;
    int nz_; Real cp_;
    Kokkos::View<Real*> area_f_,area_c_;
    mutable Field donor_;
    std::array<std::array<std::unique_ptr<Field>,3>,2> flux_,applied_;
};
} // namespace VVM::Driver
