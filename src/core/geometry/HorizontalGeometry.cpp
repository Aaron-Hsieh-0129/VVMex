#include "core/geometry/HorizontalGeometry.hpp"
#include "core/geometry/HorizontalGeometryBuilders.hpp"

#include <cmath>
#include <limits>
#include <stdexcept>
#include <utility>

namespace VVM::Core::Geometry {
namespace {

constexpr int coefficient_count = 10;

KOKKOS_INLINE_FUNCTION bool
finite_value(const Real value) {
    constexpr Real largest = std::numeric_limits<Real>::max();
    return value == value && value <= largest && value >= -largest;
}

// All four metric entries participate independently. B is the physical
// coordinate basis supplied by the mapping; it fixes information (orientation)
// that cannot be recovered from g. Neither matrix is assumed symmetric here.
KOKKOS_INLINE_FUNCTION bool
derive_coefficients(const Real a,
    const Real b,
    const Real c,
    const Real d,
    const Real b11,
    const Real b12,
    const Real b21,
    const Real b22,
    Real (&v)[coefficient_count]) {
    if (!finite_value(a) || !finite_value(b) || !finite_value(c) || !finite_value(d) ||
        a <= real(0) || d <= real(0) || !finite_value(b11) || !finite_value(b12) ||
        !finite_value(b21) || !finite_value(b22)) {
        return false;
    }
    const Real determinant = a * d - b * c;
    const Real basis_determinant = b11 * b22 - b12 * b21;
    if (!finite_value(determinant) || determinant <= real(0) || !finite_value(basis_determinant) ||
        basis_determinant <= real(0)) {
        return false;
    }
    const Real J = Kokkos::sqrt(determinant);
    v[0] = J;
    v[1] = real(1) / J;
    v[2] = d / J;
    v[3] = -b / J;
    v[4] = -c / J;
    v[5] = a / J;
    v[6] = b22 / basis_determinant;
    v[7] = -b12 / basis_determinant;
    v[8] = -b21 / basis_determinant;
    v[9] = b11 / basis_determinant;
    for (int m = 0; m < coefficient_count; ++m) {
        if (!finite_value(v[m])) {
            return false;
        }
    }
    return finite_value(v[2] / J) && finite_value(v[3] / J) && finite_value(v[4] / J) &&
           finite_value(v[5] / J);
}

void
validate_field(const GeometryField2D& field, const HorizontalDomainLayout& layout) {
    const auto nx = static_cast<std::size_t>(layout.local_total_nx());
    const auto ny = static_cast<std::size_t>(layout.local_total_ny());
    switch (field.layout) {
    case GeometryFieldLayout::Constant:
        if (std::isfinite(field.constant)) {
            return;
        }
        break;
    case GeometryFieldLayout::VaryingI:
        if (field.one_dimensional.extent(0) >= nx) {
            return;
        }
        break;
    case GeometryFieldLayout::VaryingJ:
        if (field.one_dimensional.extent(0) >= ny) {
            return;
        }
        break;
    case GeometryFieldLayout::Full2D:
        if (field.two_dimensional.extent(0) >= ny && field.two_dimensional.extent(1) >= nx) {
            return;
        }
        break;
    }
    throw std::invalid_argument(
        "Geometry field must be finite when constant and cover local halos when stored.");
}

bool
same_field(const GeometryField2D& a, const GeometryField2D& b) {
    if (a.layout != b.layout) {
        return false;
    }
    switch (a.layout) {
    case GeometryFieldLayout::Constant:
        return a.constant == b.constant;
    case GeometryFieldLayout::VaryingI:
    case GeometryFieldLayout::VaryingJ:
        return a.one_dimensional == b.one_dimensional;
    case GeometryFieldLayout::Full2D:
        return a.two_dimensional == b.two_dimensional;
    }
    return false;
}

bool
same_metric(const HorizontalGeometryInput& a, const HorizontalGeometryInput& b) {
    return same_field(a.g_cov.a11, b.g_cov.a11) && same_field(a.g_cov.a12, b.g_cov.a12) &&
           same_field(a.g_cov.a21, b.g_cov.a21) && same_field(a.g_cov.a22, b.g_cov.a22) &&
           same_field(a.contravariant_to_physical.a11, b.contravariant_to_physical.a11) &&
           same_field(a.contravariant_to_physical.a12, b.contravariant_to_physical.a12) &&
           same_field(a.contravariant_to_physical.a21, b.contravariant_to_physical.a21) &&
           same_field(a.contravariant_to_physical.a22, b.contravariant_to_physical.a22);
}

GeometryFieldLayout
coefficient_layout(const HorizontalGeometryInput& input) {
    GeometryFieldLayout result = GeometryFieldLayout::Constant;
    for (const auto& field : {input.g_cov.a11,
             input.g_cov.a12,
             input.g_cov.a21,
             input.g_cov.a22,
             input.contravariant_to_physical.a11,
             input.contravariant_to_physical.a12,
             input.contravariant_to_physical.a21,
             input.contravariant_to_physical.a22}) {
        if (field.layout == GeometryFieldLayout::Full2D) {
            return GeometryFieldLayout::Full2D;
        }
        if (field.layout != GeometryFieldLayout::Constant) {
            if (result != GeometryFieldLayout::Constant && result != field.layout) {
                return GeometryFieldLayout::Full2D;
            }
            result = field.layout;
        }
    }
    return result;
}

void
initialize_metric(HorizontalGeometryDeviceView& output,
    const HorizontalGeometryInput& input,
    const HorizontalDomainLayout& layout,
    const std::string& label) {
    std::array<GeometryField2D, coefficient_count> fields;
    const auto storage_layout = coefficient_layout(input);
    const auto g = input.g_cov;
    const auto basis = input.contravariant_to_physical;
    int invalid = 0;
    if (storage_layout == GeometryFieldLayout::Constant) {
        Real values[coefficient_count] = {};
        if (!derive_coefficients(g.a11.constant,
                g.a12.constant,
                g.a21.constant,
                g.a22.constant,
                basis.a11.constant,
                basis.a12.constant,
                basis.a21.constant,
                basis.a22.constant,
                values)) {
            invalid = 1;
        }
        for (int m = 0; m < coefficient_count; ++m) {
            fields[m] = GeometryField2D::constant_value(values[m]);
        }
    }
    else if (storage_layout == GeometryFieldLayout::Full2D) {
        Kokkos::Array<Kokkos::View<Real**>, coefficient_count> storage;
        for (int m = 0; m < coefficient_count; ++m) {
            storage[m] = Kokkos::View<Real**>(label + "_" + std::to_string(m),
                layout.local_total_ny(),
                layout.local_total_nx());
            fields[m] = GeometryField2D::full_2d(storage[m]);
        }
        Kokkos::parallel_reduce("DeriveHorizontalMetric2D",
            Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0},
                {layout.local_total_ny(), layout.local_total_nx()}),
            KOKKOS_LAMBDA(const int j, const int i, int& count) {
                Real values[coefficient_count] = {};
                if (!derive_coefficients(g.a11(j, i),
                        g.a12(j, i),
                        g.a21(j, i),
                        g.a22(j, i),
                        basis.a11(j, i),
                        basis.a12(j, i),
                        basis.a21(j, i),
                        basis.a22(j, i),
                        values)) {
                    ++count;
                }
                for (int m = 0; m < coefficient_count; ++m) {
                    storage[m](j, i) = values[m];
                }
            },
            invalid);
    }
    else {
        Kokkos::Array<Kokkos::View<Real*>, coefficient_count> storage;
        const bool varies_i = storage_layout == GeometryFieldLayout::VaryingI;
        const int size = varies_i ? layout.local_total_nx() : layout.local_total_ny();
        for (int m = 0; m < coefficient_count; ++m) {
            storage[m] = Kokkos::View<Real*>(label + "_" + std::to_string(m), size);
            fields[m] = varies_i ? GeometryField2D::varying_i(storage[m])
                                 : GeometryField2D::varying_j(storage[m]);
        }
        Kokkos::parallel_reduce("DeriveHorizontalMetric1D",
            size,
            KOKKOS_LAMBDA(const int index, int& count) {
                const int j = varies_i ? 0 : index, i = varies_i ? index : 0;
                Real values[coefficient_count] = {};
                if (!derive_coefficients(g.a11(j, i),
                        g.a12(j, i),
                        g.a21(j, i),
                        g.a22(j, i),
                        basis.a11(j, i),
                        basis.a12(j, i),
                        basis.a21(j, i),
                        basis.a22(j, i),
                        values)) {
                    ++count;
                }
                for (int m = 0; m < coefficient_count; ++m) {
                    storage[m](index) = values[m];
                }
            },
            invalid);
    }
    if (invalid != 0) {
        throw std::invalid_argument(
            "Geometry requires finite matrices, positive metric diagonal entries, "
            "positive determinants, and finite derived coefficients, including halos.");
    }
    // Preserve exact constant zeros and constant diagonal inverses. Boundary
    // and restart adapters inspect these handles without reading device data.
    const auto is_zero = [](const GeometryField2D& field) {
        return field.layout == GeometryFieldLayout::Constant && field.constant == real(0);
    };
    const auto zero = GeometryField2D::constant_value(real(0));
    if (is_zero(g.a12)) {
        fields[3] = zero;
    }
    if (is_zero(g.a21)) {
        fields[4] = zero;
    }
    if (is_zero(basis.a12)) {
        fields[7] = zero;
    }
    if (is_zero(basis.a21)) {
        fields[8] = zero;
    }
    if (is_zero(basis.a12) && is_zero(basis.a21)) {
        if (basis.a11.layout == GeometryFieldLayout::Constant) {
            fields[6] = GeometryField2D::constant_value(real(1) / basis.a11.constant);
        }
        if (basis.a22.layout == GeometryFieldLayout::Constant) {
            fields[9] = GeometryField2D::constant_value(real(1) / basis.a22.constant);
        }
    }
    output.sqrt_g = fields[0];
    output.inv_sqrt_g = fields[1];
    output.sqrt_g_g_contra = {fields[2], fields[3], fields[4], fields[5]};
    output.contravariant_to_physical = basis;
    output.physical_to_contravariant = {fields[6], fields[7], fields[8], fields[9]};
}

} // namespace

void
validate_horizontal_geometry_layout(const HorizontalDomainLayout& layout, Real dq1, Real dq2) {
    if (layout.global_nx <= 0 || layout.global_ny <= 0 || layout.local_physical_nx <= 0 ||
        layout.local_physical_ny <= 0 || layout.halo < 0 || layout.global_start_i < 0 ||
        layout.global_start_j < 0 || layout.local_physical_nx > layout.global_nx ||
        layout.local_physical_ny > layout.global_ny ||
        layout.global_start_i > layout.global_nx - layout.local_physical_nx ||
        layout.global_start_j > layout.global_ny - layout.local_physical_ny ||
        layout.halo > (std::numeric_limits<int>::max() - layout.local_physical_nx) / 2 ||
        layout.halo > (std::numeric_limits<int>::max() - layout.local_physical_ny) / 2) {
        throw std::invalid_argument(
            "Horizontal geometry requires a valid local domain and halo extent.");
    }
    if (!std::isfinite(dq1) || dq1 <= real(0) || !std::isfinite(dq2) || dq2 <= real(0) ||
        !std::isfinite(real(1) / dq1) || !std::isfinite(real(1) / dq2)) {
        throw std::invalid_argument(
            "Horizontal geometry requires finite positive spacings and reciprocals.");
    }
}

HorizontalGeometry::HorizontalGeometry(HorizontalDomainLayout layout,
    Real dq1,
    Real dq2,
    const std::array<HorizontalGeometryInput, 4>& inputs,
    GeometryKind kind,
    std::string name)
    : layout_(layout), dq1_(dq1), dq2_(dq2), kind_(kind), name_(std::move(name)) {
    validate_horizontal_geometry_layout(layout_, dq1_, dq2_);
    for (std::size_t l = 0; l < inputs.size(); ++l) {
        const auto& input = inputs[l];
        for (const auto& field : {input.g_cov.a11,
                 input.g_cov.a12,
                 input.g_cov.a21,
                 input.g_cov.a22,
                 input.q1,
                 input.q2,
                 input.longitude,
                 input.latitude,
                 input.contravariant_to_physical.a11,
                 input.contravariant_to_physical.a12,
                 input.contravariant_to_physical.a21,
                 input.contravariant_to_physical.a22}) {
            validate_field(field, layout_);
        }
        // Existing Cartesian shortcuts are valid only for the identity metric
        // in the unrotated physical frame.
        if (kind_ == GeometryKind::Cartesian) {
            for (const auto& field : {input.g_cov.a11,
                     input.g_cov.a22,
                     input.contravariant_to_physical.a11,
                     input.contravariant_to_physical.a22}) {
                if (field.layout != GeometryFieldLayout::Constant || field.constant != real(1)) {
                    throw std::invalid_argument("Cartesian geometry requires the identity metric.");
                }
            }
            for (const auto& field : {input.g_cov.a12,
                     input.g_cov.a21,
                     input.contravariant_to_physical.a12,
                     input.contravariant_to_physical.a21}) {
                if (field.layout != GeometryFieldLayout::Constant || field.constant != real(0)) {
                    throw std::invalid_argument(
                        "Cartesian geometry requires an unrotated identity metric.");
                }
            }
        }
        auto& output = locations_[l];
        bool shared = false;
        for (std::size_t previous = 0; previous < l; ++previous) {
            if (same_metric(input, inputs[previous])) {
                output = locations_[previous];
                shared = true;
                break;
            }
        }
        if (!shared) {
            initialize_metric(output, input, layout_, name_ + "_metric_" + std::to_string(l));
        }
        output.g_cov = input.g_cov;
        output.q1 = input.q1;
        output.q2 = input.q2;
        output.longitude = input.longitude;
        output.latitude = input.latitude;
        output.dq1 = dq1_;
        output.dq2 = dq2_;
        output.rdq1 = real(1) / dq1_;
        output.rdq2 = real(1) / dq2_;
    }
}

std::size_t
HorizontalGeometry::location_index(HorizontalLocation location) {
    switch (location) {
    case HorizontalLocation::T:
        return 0;
    case HorizontalLocation::U:
        return 1;
    case HorizontalLocation::V:
        return 2;
    case HorizontalLocation::Z:
        return 3;
    }
    throw std::invalid_argument("HorizontalGeometry received an invalid HorizontalLocation.");
}

std::array<Real, 2>
HorizontalGeometry::physical_spacing(
    const HorizontalLocation location, const int j, const int i) const {
    const auto geometry = device_view(location);
    Kokkos::View<Real*> lengths("horizontal physical spacing", 2);
    Kokkos::parallel_for("HorizontalPhysicalSpacing", 1, KOKKOS_LAMBDA(const int) {
        lengths(0) = geometry.physical_spacing_q1(j, i);
        lengths(1) = geometry.physical_spacing_q2(j, i);
    });
    const auto host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), lengths);
    return {host(0), host(1)};
}

} // namespace VVM::Core::Geometry
