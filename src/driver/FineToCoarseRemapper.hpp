#pragma once

#include "core/Field.hpp"
#include "core/geometry/RegularLatLonGeometry.hpp"
#include <cmath>
#include <algorithm>
#include <limits>

namespace VVM::Driver {

// Exact RLL T-cell areas. Independent of the dycore's pointwise metric
// quadrature: integrated spherical areas are additive across refinement.
class FineToCoarseRemapper {
public:
    FineToCoarseRemapper(const Core::Geometry::HorizontalGeometry& fine,
        const Core::Geometry::HorizontalGeometry& coarse, int nz, int ratio = 4)
        : fine_(fine.layout()), coarse_(coarse.layout()), nz_(nz), ratio_(ratio) {
        using Core::Geometry::RegularLatLonGeometry;
        const auto* f = dynamic_cast<const RegularLatLonGeometry*>(&fine);
        const auto* c = dynamic_cast<const RegularLatLonGeometry*>(&coarse);
        if (!f || !c) throw std::runtime_error("One-way remapping requires RLL geometry.");
        if (ratio != 4 || nz <= 0 || fine_.halo != coarse_.halo ||
            fine_.global_nx != ratio*coarse_.global_nx || fine_.global_ny != ratio*coarse_.global_ny)
            throw std::runtime_error("One-way remapping requires matching halos and an exact 4:1 horizontal ratio.");
        for (const auto& layout : {fine_, coarse_}) {
            if (layout.local_physical_nx != layout.global_nx || layout.local_physical_ny != layout.global_ny ||
                layout.global_start_i != 0 || layout.global_start_j != 0)
                throw std::runtime_error("One-way remapping currently requires one compute rank.");
        }
        const auto close = [](VVM::Real a, VVM::Real b) {
            return std::abs(a-b) <= real(32.)*std::numeric_limits<VVM::Real>::epsilon()*
                std::max(real(1.), std::max(std::abs(a),std::abs(b)));
        };
        if (!close(f->radius(),c->radius()) ||
            !close(f->longitude_west_edge(),c->longitude_west_edge()) ||
            !close(f->latitude_south_edge(),c->latitude_south_edge()) ||
            !close(f->dq1()*ratio,c->dq1()) || !close(f->dq2()*ratio,c->dq2()))
            throw std::runtime_error("Fine/coarse RLL domain edges, Earth radius or cell alignment do not match.");
        fine_area_ = areas(*f);
        coarse_area_ = areas(*c);
        south_=f->latitude_south_edge(); dy_=f->dq2();
    }

    // Native horizontal dual-cell restriction for physical vorticity.
    // Even refinement places coarse faces on fine faces: the r+1 samples
    // span one coarse dual cell, with partial fine dual cells at its edges.
    // Longitude wraps; latitude support is clipped at the channel walls.
    // Vertical staggering is unchanged because both vertical grids coincide.
    void remap_staggered(const Core::Field<3>& source, Core::Field<3>& target,
        Core::GridStaggering staggering) const {
        using Core::GridStaggering;
        if(staggering==GridStaggering::Centered) { remap(source,target); return; }
        const bool x=staggering==GridStaggering::StaggeredXZ || staggering==GridStaggering::StaggeredXY;
        const bool y=staggering==GridStaggering::StaggeredYZ || staggering==GridStaggering::StaggeredXY;
        if(!x && !y) throw std::runtime_error("Unsupported native feedback staggering.");
        check_shape(source,fine_); check_shape(target,coarse_); target.set_to_zero();
        const auto f=source.get_device_data(),c=target.get_mutable_device_data();
        const int h=fine_.halo,r=ratio_,fnx=fine_.global_nx,fny=fine_.global_ny,
            cnx=coarse_.global_nx,cny=coarse_.global_ny;
        const Real south=south_,dy=dy_,north=south+fny*dy;
        Kokkos::parallel_for("NativeFineToCoarsePhysics",Kokkos::MDRangePolicy<Kokkos::Rank<3>>(
            {h,h,h},{h+nz_,h+cny,h+cnx}),KOKKOS_LAMBDA(int k,int j,int i) {
                const int center_j=h+r*(j-h)+(y?r-1:0),center_i=h+r*(i-h)+(x?r-1:0);
                const Real latitude=south+(real(j-h)+(y?real(1.):real(.5)))*r*dy;
                const Real low=Kokkos::fmax(south,latitude-real(.5)*r*dy),
                    high=Kokkos::fmin(north,latitude+real(.5)*r*dy);
                const Real reference=f(k,center_j,center_i);
                Real numerator=0,denominator=0;
                for(int jj=0;jj<r+(y?1:0);++jj) {
                    const int fj=center_j+jj-(y?r/2:0);
                    const Real fine_lat=south+(real(fj-h)+(y?real(1.):real(.5)))*dy;
                    const Real a=Kokkos::fmax(low,fine_lat-dy/2),b=Kokkos::fmin(high,fine_lat+dy/2);
                    if(b<=a) continue;
                    const Real lat_weight=real(2.)*Kokkos::cos((a+b)/2)*Kokkos::sin((b-a)/2);
                    for(int ii=0;ii<r+(x?1:0);++ii) {
                        int fi=center_i+ii-(x?r/2:0);
                        if(fi<h) fi+=fnx; if(fi>=h+fnx) fi-=fnx;
                        const Real weight=lat_weight*((x && (ii==0 || ii==r))?real(.5):real(1.));
                        numerator+=weight*(f(k,fj,fi)-reference); denominator+=weight;
                    }
                }
                c(k,j,i)=reference+numerator/denominator;
            });
    }

    void remap(const Core::Field<3>& source, Core::Field<3>& target) const {
        check_shape(source, fine_);
        check_shape(target, coarse_);
        const auto f = source.get_device_data();
        const auto c = target.get_mutable_device_data();
        const auto weights = fine_area_;
        const int h = fine_.halo, r = ratio_;
        target.set_to_zero();
        Kokkos::parallel_for("FineToCoarsePhysicsFeedback",
            Kokkos::MDRangePolicy<Kokkos::Rank<3>>({h,h,h},
                {h+nz_,h+coarse_.global_ny,h+coarse_.global_nx}),
            KOKKOS_LAMBDA(int k, int j, int i) {
                VVM::Real numerator = real(0.), denominator = real(0.);
                const auto reference = f(k,h+(j-h)*r,h+(i-h)*r);
                for (int jj = 0; jj < r; ++jj) {
                    const int fj = h+(j-h)*r+jj;
                    for (int ii = 0; ii < r; ++ii) {
                        const int fi = h+(i-h)*r+ii;
                        numerator += weights(fj)*(f(k,fj,fi)-reference);
                        denominator += weights(fj);
                    }
                }
                // An exactly constant block has exactly zero deviations, so
                // even nonzero constants survive bitwise in floating point.
                c(k,j,i) = reference+numerator/denominator;
            });
    }

    // Per-level volume-integrated tendencies; only diagnostics cross to host.
    void integrals(const Core::Field<3>& field, bool fine,
        const Core::Field<1>& z_up, Core::Field<1>& result) const {
        const auto layout = fine ? fine_ : coarse_;
        check_shape(field, layout);
        if (z_up.get_device_data().extent_int(0) != nz_+2*layout.halo ||
            result.get_device_data().extent_int(0) != nz_+2*layout.halo)
            throw std::runtime_error("Feedback integral vertical dimensions do not match.");
        const auto areas = fine ? fine_area_ : coarse_area_;
        const auto f = field.get_device_data();
        const auto z = z_up.get_device_data();
        const auto out = result.get_mutable_device_data();
        const int h = layout.halo, nx = layout.global_nx, ny = layout.global_ny;
        result.set_to_zero();
        using Policy = Kokkos::TeamPolicy<>;
        Kokkos::parallel_for("FeedbackVolumeIntegrals", Policy(nz_,Kokkos::AUTO),
            KOKKOS_LAMBDA(const Policy::member_type& team) {
                const int k = h+team.league_rank();
                VVM::Real sum = real(0.);
                Kokkos::parallel_reduce(Kokkos::TeamThreadRange(team,nx*ny),
                    [&](int n, VVM::Real& value) {
                        const int j = h+n/nx, i = h+n%nx;
                        value += areas(j)*f(k,j,i);
                    }, sum);
                Kokkos::single(Kokkos::PerTeam(team), [&]() { out(k) = sum*(z(k)-z(k-1)); });
            });
    }

    // Public because CUDA extended lambdas require a public enclosing function.
    static Kokkos::View<VVM::Real*> areas(const Core::Geometry::RegularLatLonGeometry& geometry) {
        const auto layout = geometry.layout();
        Kokkos::View<VVM::Real*> weights("rll_feedback_cell_area",layout.local_total_ny());
        const auto south = geometry.latitude_south_edge(), dy = geometry.dq2(), dx = geometry.dq1();
        const auto radius = geometry.radius();
        const int h = layout.halo;
        Kokkos::parallel_for("RLLFeedbackAreas", Kokkos::RangePolicy<>(h,h+layout.global_ny),
            KOKKOS_LAMBDA(int j) {
                // Equivalent to sin(north)-sin(south), with less cancellation.
                const auto latitude = south+(real(j-h)+real(.5))*dy;
                weights(j) = radius*radius*dx*real(2.)*Kokkos::cos(latitude)*Kokkos::sin(dy/real(2.));
            });
        return weights;
    }
private:
    void check_shape(const Core::Field<3>& field,
        const Core::Geometry::HorizontalDomainLayout& layout) const {
        const auto& v = field.get_device_data();
        if (v.extent_int(0) != nz_+2*layout.halo || v.extent_int(1) != layout.local_total_ny() ||
            v.extent_int(2) != layout.local_total_nx())
            throw std::runtime_error("Feedback field shape does not match the remapping grid.");
    }
    Core::Geometry::HorizontalDomainLayout fine_, coarse_;
    int nz_, ratio_;
    VVM::Real south_,dy_;
    Kokkos::View<VVM::Real*> fine_area_, coarse_area_;
};

} // namespace VVM::Driver
