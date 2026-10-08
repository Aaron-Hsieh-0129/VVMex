#pragma once

#include "driver/FineToCoarseRemapper.hpp"

namespace VVM::Driver {

// Centered, face-filtered covariance transport. This is a conservative
// subfilter-flux closure, not the full discrete Takacs/AB2 residual.
// Horizontal covariance is filtered along each coarse boundary face;
// vertical covariance is filtered over the complete 4x4 block.
class ConvectiveTransportFeedback {
public:
    ConvectiveTransportFeedback(const Core::Geometry::RegularLatLonGeometry& fine,
        const Core::Geometry::RegularLatLonGeometry& coarse, int nz)
        : layout_(coarse.layout()), nz_(nz), south_(fine.latitude_south_edge()),
          dy_(fine.dq2()), dx_(fine.dq1()), radius_(fine.radius()),
          fine_area_(FineToCoarseRemapper::areas(fine)),
          coarse_area_(FineToCoarseRemapper::areas(coarse)),
          fx_("covariance_flux_x",nz+2*layout_.halo,layout_.local_total_ny(),layout_.local_total_nx()),
          fy_("covariance_flux_y",nz+2*layout_.halo,layout_.local_total_ny(),layout_.local_total_nx()),
          fz_("covariance_flux_z",nz+2*layout_.halo,layout_.local_total_ny(),layout_.local_total_nx()) {
        // Reuse all alignment, ratio and geometry checks.
        FineToCoarseRemapper validated(fine,coarse,nz);
    }

    void diagnose(const Core::Field<3>& scalar, const Core::Field<3>& u,
        const Core::Field<3>& v, const Core::Field<3>& w,
        const Core::Field<1>& rho, const Core::Field<1>& rho_up,
        const Core::Field<1>& z_up, Core::Field<3>& output) const {
        const int h=layout_.halo, nx=layout_.global_nx, ny=layout_.global_ny, nz=nz_;
        for(const auto* field : {&scalar,&u,&v,&w}) {
            const auto a=field->get_device_data();
            if(a.extent_int(0)!=nz+2*h || a.extent_int(1)!=4*ny+2*h || a.extent_int(2)!=4*nx+2*h)
                throw std::runtime_error("Convective transport fine field shape mismatch.");
        }
        const auto out=output.get_mutable_device_data();
        if(out.extent_int(0)!=nz+2*h || out.extent_int(1)!=ny+2*h || out.extent_int(2)!=nx+2*h)
            throw std::runtime_error("Convective transport coarse field shape mismatch.");
        for(const auto* field : {&rho,&rho_up,&z_up})
            if(field->get_device_data().extent_int(0)!=nz+2*h)
                throw std::runtime_error("Convective transport vertical shape mismatch.");
        const auto s=scalar.get_device_data(), uu=u.get_device_data(), vv=v.get_device_data(), ww=w.get_device_data();
        const auto r=rho.get_device_data(), ru=rho_up.get_device_data(), z=z_up.get_device_data();
        const auto af=fine_area_, ac=coarse_area_;
        const auto fx=fx_, fy=fy_, fz=fz_;
        const auto radius=radius_, dx=dx_, dy=dy_, south=south_;
        Kokkos::deep_copy(fx,real(0.)); Kokkos::deep_copy(fy,real(0.)); Kokkos::deep_copy(fz,real(0.));
        output.set_to_zero();
        Kokkos::parallel_for("ConvectiveCovarianceFaces",
            Kokkos::MDRangePolicy<Kokkos::Rank<3>>({h,h,h},{h+nz,h+ny,h+nx}),
            KOKKOS_LAMBDA(int k,int j,int i) {
                // Periodic longitude: the last positive face wraps to first cell.
                const int fi=h+4*(i-h)+3, fj0=h+4*(j-h), fi0=h+4*(i-h);
                const int right= i==h+nx-1 ? h : fi+1;
                const auto u0=uu(k,fj0,fi);
                const auto x0=real(.5)*(s(k,fj0,fi)+s(k,fj0,right));
                Real su=0., ss=0., sus=0.;
                for(int n=0;n<4;++n) {
                    const int fj=fj0+n;
                    const auto face_s=real(.5)*(s(k,fj,fi)+s(k,fj,right));
                    const auto du=uu(k,fj,fi)-u0, ds=face_s-x0;
                    su+=du;ss+=ds;sus+=du*ds;
                }
                // Physical u at longitude faces; transverse face length R*dphi.
                fx(k,j,i)=r(k)*radius*dy*(sus-su*ss/real(4.));
                // Bounded latitude walls have zero transport.
                if(j<h+ny-1) {
                    const int fj=fj0+3;
                    const auto v0=vv(k,fj,fi0);
                    const auto y0=real(.5)*(s(k,fj,fi0)+s(k,fj+1,fi0));
                    su=ss=sus=real(0.);
                    for(int n=0;n<4;++n) {
                        const int fi=fi0+n;
                        const auto face_s=real(.5)*(s(k,fj,fi)+s(k,fj+1,fi));
                        const auto dv=vv(k,fj,fi)-v0, ds=face_s-y0;
                        su+=dv;ss+=ds;sus+=dv*ds;
                    }
                    const auto latitude=south+real(fj-h+1)*dy;
                    fy(k,j,i)=r(k)*radius*Kokkos::cos(latitude)*dx*(sus-su*ss/real(4.));
                }
                // Closed lower/upper boundaries. Only internal vertical faces.
                if(k<h+nz-1) {
                    const auto w0=ww(k,fj0,fi0);
                    const auto z0=real(.5)*(s(k,fj0,fi0)+s(k+1,fj0,fi0));
                    Real sw=0., sx=0., swx=0., area=0.;
                    for(int jj=0;jj<4;++jj) for(int ii=0;ii<4;++ii) {
                        const int fj=fj0+jj, fi=fi0+ii;
                        const auto weight=af(fj);
                        const auto face_s=real(.5)*(s(k,fj,fi)+s(k+1,fj,fi));
                        const auto dw=ww(k,fj,fi)-w0, ds=face_s-z0;
                        sw+=weight*dw;sx+=weight*ds;
                        swx+=weight*dw*ds;area+=weight;
                    }
                    fz(k,j,i)=ru(k)*(swx-sw*sx/area);
                }
            });
        Kokkos::parallel_for("ConvectiveCovarianceConvergence",
            Kokkos::MDRangePolicy<Kokkos::Rank<3>>({h,h,h},{h+nz,h+ny,h+nx}),
            KOKKOS_LAMBDA(int k,int j,int i) {
                const int left=i==h ? h+nx-1 : i-1;
                const auto dz=z(k)-z(k-1);
                const auto xdiv=fx(k,j,i)-fx(k,j,left);
                const auto ydiv=fy(k,j,i)-(j==h ? real(0.) : fy(k,j-1,i));
                const auto zdiv=fz(k,j,i)-(k==h ? real(0.) : fz(k-1,j,i));
                out(k,j,i)=-(dz*(xdiv+ydiv)+zdiv)/(r(k)*ac(j)*dz);
            });
    }
private:
    Core::Geometry::HorizontalDomainLayout layout_;
    int nz_;
    Real south_,dy_,dx_,radius_;
    Kokkos::View<Real*> fine_area_,coarse_area_;
    Core::Field<3>::ViewType fx_,fy_,fz_;
};
} // namespace VVM::Driver
