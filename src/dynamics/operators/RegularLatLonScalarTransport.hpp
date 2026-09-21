#ifndef VVM_DYNAMICS_OPERATORS_REGULAR_LAT_LON_SCALAR_TRANSPORT_HPP
#define VVM_DYNAMICS_OPERATORS_REGULAR_LAT_LON_SCALAR_TRANSPORT_HPP

#include <stdexcept>

#include "core/geometry/GeometryKind.hpp"
#include "dynamics/operators/GeneralizedScalarTransport.hpp"

namespace VVM::Dynamics::Operators {

// Source compatibility for existing RLL field-level callers.
// Production owns GeneralizedScalarTransport directly. This wrapper keeps
// the old geometry guard and API, but owns no stencil or kernel.
class RegularLatLonScalarTransport {
public:
    explicit RegularLatLonScalarTransport(const Core::Geometry::HorizontalGeometry& geometry,
        Real alpha = real(1.0))
        : generalized_(require_rll_geometry(geometry), alpha) {}

    static void
    prepare_execution() {
        GeneralizedScalarTransport::prepare_execution();
    }

    void
    add_flux_convergence(const Core::Field<3>& scalar_q,
        const Core::Field<3>& contravariant_mass_flux_q1,
        const Core::Field<3>& contravariant_mass_flux_q2,
        const Core::Field<3>& vertical_mass_flux,
        const Core::Field<1>& vertical_cell_spacing,
        Core::Field<3>& out_flux_convergence,
        int k_begin,
        int k_end) const {
        generalized_.add_flux_convergence(scalar_q,
            contravariant_mass_flux_q1,
            contravariant_mass_flux_q2,
            vertical_mass_flux,
            vertical_cell_spacing,
            out_flux_convergence,
            k_begin,
            k_end);
    }

private:
    static const Core::Geometry::HorizontalGeometry&
    require_rll_geometry(const Core::Geometry::HorizontalGeometry& geometry) {
        if (geometry.kind() != Core::Geometry::GeometryKind::RegularLatLon) {
            throw std::invalid_argument("RegularLatLonScalarTransport requires "
                                        "regular latitude-longitude geometry.");
        }

        return geometry;
    }

    GeneralizedScalarTransport generalized_;
};

} // namespace VVM::Dynamics::Operators

#endif
