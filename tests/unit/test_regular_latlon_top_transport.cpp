#include "core/Field.hpp"
#include "core/geometry/RegularLatLonGeometry.hpp"
#include "dynamics/operators/RegularLatLonTopTransport.hpp"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <limits>

using namespace VVM;
using Core::Field;
using Core::Geometry::HorizontalLocation;
using Dynamics::Operators::RegularLatLonTopTransportDeviceView;

template <class Volume, class Profile, class Plane>
struct Fields {
    Volume u, v, w, zeta;
    Profile rho, rho_up, inverse_spacing_mid;
    Plane f_at_z;
};

// Independent host transcription of the Fortran UP/UM stencil.
double
flux(double previous, double current, double next, double qm, double q0, double q1, double qp) {
    const double up = std::max(current, 0.0), um = std::min(current, 0.0);
    return .5 * current * (q0 + q1) -
           (up * (q1 - q0) - std::sqrt(up * std::max(previous, 0.0)) * (q0 - qm) + um * (q0 - q1) +
               std::sqrt(-um * std::max(-next, 0.0)) * (q1 - qp)) /
               6.;
}

int
main(int argc, char** argv) {
    Kokkos::initialize(argc, argv);
    int failures = 0;
    {
        constexpr int nx = 28, ny = 16, nz = 6, top = 3, h = 2;
        const double radius = 5., dx = 2 * std::acos(-1.) / (nx - 2 * h), dy = .8 / (ny - 2 * h),
                     south = -.4;
        Core::Geometry::HorizontalDomainLayout layout;
        layout.global_nx = layout.local_physical_nx = nx - 2 * h;
        layout.global_ny = layout.local_physical_ny = ny - 2 * h;
        layout.halo = h;
        Core::Geometry::RegularLatLonGeometry geometry(layout, dx, dy, 0., south, radius);
        Field<3> u("u", {nz, ny, nx}), v("v", {nz, ny, nx}), w("w", {nz, ny, nx}),
            z("z", {nz, ny, nx});
        Field<1> rho("rho", {nz}), rho_up("rho_up", {nz}), inv("inv", {nz});
        Field<2> f("f", {ny, nx});
        Field<3> result("result", {3, ny, nx});
        auto uh = u.get_host_data(), vh = v.get_host_data(), wh = w.get_host_data(),
             zh = z.get_host_data();
        auto rh = rho.get_host_data(), ru = rho_up.get_host_data(), ih = inv.get_host_data();
        auto fh = f.get_host_data();
        for (int k = 0; k < nz; ++k) {
            rh(k) = 1.2 + .3 * k;
            ru(k) = 1.4 + .2 * k;
            ih(k) = .7 + .1 * k;
            for (int j = 0; j < ny; ++j) {
                for (int i = 0; i < nx; ++i) {
                    uh(k, j, i) = std::sin((i - h + 1) * dx) + .2 * k;
                    vh(k, j, i) = std::cos((i - h + .5) * dx) - .3 + .01 * j;
                    wh(k, j, i) = k == top ? 0. : .3 * std::sin(i + .4 * j);
                    zh(k, j, i) = (.1 + std::cos((i - h + 1) * dx) * std::sin(.3 * j)) / rh(k);
                    fh(j, i) = .13 * std::sin(south + (j - h + 1) * dy);
                }
            }
        }
        Kokkos::deep_copy(u.get_mutable_device_data(), uh);
        Kokkos::deep_copy(v.get_mutable_device_data(), vh);
        Kokkos::deep_copy(w.get_mutable_device_data(), wh);
        Kokkos::deep_copy(z.get_mutable_device_data(), zh);
        Kokkos::deep_copy(rho.get_mutable_device_data(), rh);
        Kokkos::deep_copy(rho_up.get_mutable_device_data(), ru);
        Kokkos::deep_copy(inv.get_mutable_device_data(), ih);
        Kokkos::deep_copy(f.get_mutable_device_data(), fh);
        Fields<Field<3>::ViewType, Field<1>::ViewType, Field<2>::ViewType> fields{
            u.get_device_data(),
            v.get_device_data(),
            w.get_device_data(),
            z.get_device_data(),
            rho.get_device_data(),
            rho_up.get_device_data(),
            inv.get_device_data(),
            f.get_device_data()};
        Kokkos::View<RegularLatLonTopTransportDeviceView> op("op");
        auto host_op = Kokkos::create_mirror_view(op);
        host_op() = Dynamics::Operators::make_regular_lat_lon_top_transport_device_view(geometry);
        Kokkos::deep_copy(op, host_op);
        auto output = result.get_mutable_device_data();
        for (bool planetary : {false, true}) {
            Kokkos::parallel_for("top transport",
                Kokkos::MDRangePolicy<Kokkos::Rank<2>>({h, h}, {ny - h, nx - h}),
                KOKKOS_LAMBDA(int j, int i) {
                    const auto t = op().calculate_at_z(fields, top, j, i, planetary);
                    output(0, j, i) = t.q1;
                    output(1, j, i) = t.q2;
                    output(2, j, i) = t.vertical;
                });
            auto actual = result.get_host_data();
            auto q = [&](int k, int j, int i) {
                return planetary ? fh(j, i) / rh(k) : zh(k, j, i);
            };
            auto mass = [&](int j, int i, int direction) {
                double sum = 0.;
                for (int jj = j; jj <= j + 1; ++jj) {
                    for (int ii = i; ii <= i + 1; ++ii) {
                        sum += direction == 0 ? uh(top, jj, ii) /
                                                    (radius * std::cos(south + (jj - h + .5) * dy))
                                              : vh(top, jj, ii) / radius;
                    }
                }
                double phi = south + (j - h + (direction == 0 ? 1. : 1.5)) * dy;
                return .25 * rh(top) * sum * radius * radius * std::cos(phi);
            };
            auto face = [&](int j, int i, int direction) {
                int di = direction == 0, dj = direction == 1;
                return flux(mass(j - dj, i - di, direction),
                    mass(j, i, direction),
                    mass(j + dj, i + di, direction),
                    q(top, j - dj, i - di),
                    q(top, j, i),
                    q(top, j + dj, i + di),
                    q(top, j + 2 * dj, i + 2 * di));
            };
            auto wm = [&](int k, int j, int i) {
                return .25 * ru(k) *
                       (wh(k, j, i) + wh(k, j, i + 1) + wh(k, j + 1, i) + wh(k, j + 1, i + 1));
            };
            for (int j = h; j < ny - h; ++j) {
                for (int i = h; i < nx - h; ++i) {
                    const double jac = radius * radius * std::cos(south + (j - h + 1) * dy);
                    double expected[3] = {(face(j, i, 0) - face(j, i - 1, 0)) / (-jac * dx),
                        (face(j, i, 1) - face(j - 1, i, 1)) / (-jac * dy),
                        0.};
                    const double velocity = wm(top - 1, j, i), old = wm(top - 2, j, i);
                    expected[2] = .5 * velocity * (q(top, j, i) + q(top - 1, j, i));
                    if (velocity >= 0.) {
                        expected[2] -= (velocity * (q(top, j, i) - q(top - 1, j, i)) -
                                           std::sqrt(velocity * std::max(old, 0.)) *
                                               (q(top - 1, j, i) - q(top - 2, j, i))) /
                                       6.;
                    }
                    expected[2] *= ih(top);
                    for (int n = 0; n < 3; ++n) {
                        const double tolerance = 512 * std::numeric_limits<Real>::epsilon() *
                                                 std::max(1., std::abs(expected[n]));
                        if (std::abs(actual(n, j, i) - expected[n]) > tolerance) {
                            ++failures;
                        }
                    }
                }
            }
        }
    }
    Kokkos::finalize();
    std::printf("Independent CVVM top transport: %d failures\n", failures);
    return failures ? 1 : 0;
}
