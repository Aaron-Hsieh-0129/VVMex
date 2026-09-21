#ifndef VVM_DYNAMICS_OPERATORS_REGULAR_LAT_LON_VORTICITY_TENDENCY_HPP
#define VVM_DYNAMICS_OPERATORS_REGULAR_LAT_LON_VORTICITY_TENDENCY_HPP

#include "dynamics/operators/RegularLatLonVorticityTendencyBoundary.hpp"

namespace VVM::Dynamics::Operators {

// Source compatibility only. Production owns GeneralizedVorticityTendency
// directly. This wrapper has no stencil, launch functor or device operator.
// Remove after remaining external callers/tests migrate.
class RegularLatLonVorticityTendency {
public:
    using Term = GeneralizedVorticityTendency::Term;

    explicit RegularLatLonVorticityTendency(const Core::Geometry::HorizontalGeometry& geometry)
        : generalized_(require_rll_geometry(geometry)) {}

    static void
    prepare_execution() {
        GeneralizedVorticityTendency::prepare_execution();
    }

    void
    add_from_canonical_state(const Core::State& state,
        const Core::Grid& grid,
        const Core::Parameters& params,
        Core::Field<3>& output,
        const std::string& variable,
        Term term) const {
        add_regular_lat_lon_physical_vorticity_tendency(generalized_,
            state,
            grid,
            params,
            output,
            variable,
            term);
    }

private:
    static const Core::Geometry::HorizontalGeometry&
    require_rll_geometry(const Core::Geometry::HorizontalGeometry& geometry) {
        if (geometry.kind() != Core::Geometry::GeometryKind::RegularLatLon) {
            throw std::invalid_argument(
                "RegularLatLonVorticityTendency compatibility requires RLL geometry.");
        }

        return geometry;
    }

    GeneralizedVorticityTendency generalized_;
};

} // namespace VVM::Dynamics::Operators
#endif
