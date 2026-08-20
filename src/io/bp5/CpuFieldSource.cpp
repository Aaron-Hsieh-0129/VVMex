#include "CpuFieldSource.hpp"

#include <variant>

namespace VVM::IO::BP5 {

FieldInput CpuFieldSource::prepare(
    const std::string& field_name,
    const FieldSelection& selection,
    CpuBufferMode mode,
    OutputElementType element_type) const {
    auto it = state_.begin();
    while (it != state_.end() && it->first != field_name) ++it;
    if (it == state_.end()) {
        throw std::invalid_argument("BP5 output field '" + field_name + "' is not registered.");
    }

    FieldInput result;
    std::visit(
        [&](const auto& field) {
            using FieldType = std::decay_t<decltype(field)>;
            if constexpr (std::is_same_v<FieldType, std::monostate>) {
                throw std::invalid_argument("BP5 output field '" + field_name + "' is empty.");
            } 
            else if constexpr (FieldType::DimValue == 0) {
                throw std::invalid_argument(
                    "BP5 history output does not support 0-D state field '" + field_name + "'.");
            } 
            else {
                if (selection.dimensions != FieldType::DimValue) {
                    throw std::logic_error(
                        "BP5 schema dimension mismatch for field '" + field_name + "'.");
                }
                const auto& view = field.get_device_data();
                for (std::size_t d = 0; d < selection.memory_count.size(); ++d) {
                    if (view.extent(d) != selection.memory_count[d]) {
                        throw std::invalid_argument(
                            "BP5 field '" + field_name + "' extent " + std::to_string(d) +
                            " does not match the grid-derived memory extent.");
                    }
                }

                result.elements = selection.elements();
                if (selection.empty()) return;

                if (mode == CpuBufferMode::Direct) {
#if defined(KOKKOS_ENABLE_CUDA)
                    throw std::logic_error(
                        "BP5 direct mode cannot expose CUDA device memory for field '" +
                        field_name + "'; the writer must resolve CUDA output to host packing.");
#else
                    if (!output_element_matches_real(element_type)) {
                        throw std::logic_error(
                            "BP5 direct mode was selected for field '" + field_name +
                            "' while output precision requires conversion; the writer "
                            "should have resolved this to packing.");
                    }
                    using Layout = typename std::decay_t<decltype(view)>::array_layout;
                    if constexpr (!std::is_same_v<Layout, Kokkos::LayoutRight>) {
                        throw std::invalid_argument(
                            "BP5 direct mode requires LayoutRight for field '" + field_name +
                            "'; select output.bp5.buffer_mode='pack'.");
                    } 
                    else {
                        result.data = view.data();
                        result.location = FieldMemoryLocation::Host;
                        result.packed = false;
                    }
#endif
                } 
                else {
#if defined(KOKKOS_ENABLE_CUDA)
                    const auto source = field.get_host_data();
#else
                    Kokkos::fence("bp5_cpu_field_source");
                    const auto& source = view;
#endif
                    if (element_type == OutputElementType::Float32) {
                        auto& buffer =
                            buffers_.require<float>(field_name, selection.elements());
                        pack_view(source, selection, buffer);
                        result.data = buffer.data();
                    } 
                    else {
                        auto& buffer =
                            buffers_.require<double>(field_name, selection.elements());
                        pack_view(source, selection, buffer);
                        result.data = buffer.data();
                    }
                    result.location = FieldMemoryLocation::Host;
                    result.packed = true;
                }
            }
        },
        it->second);
    return result;
}

} // namespace VVM::IO::BP5
