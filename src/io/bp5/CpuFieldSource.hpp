#ifndef VVM_IO_BP5_CPU_FIELD_SOURCE_HPP
#define VVM_IO_BP5_CPU_FIELD_SOURCE_HPP

#include <cstddef>
#include <stdexcept>
#include <string>
#include <type_traits>

#include <Kokkos_Core.hpp>

#include "Bp5BufferSet.hpp"
#include "Bp5FieldSchema.hpp"
#include "Bp5OutputConfig.hpp"
#include "FieldInput.hpp"
#include "core/State.hpp"

namespace VVM::IO::BP5 {

class CpuFieldSource {
public:
    CpuFieldSource(const Core::State& state, Bp5BufferSet& buffers)
        : state_(state), buffers_(buffers) {}

    FieldInput prepare(const std::string& field_name,
                       const FieldSelection& selection,
                       CpuBufferMode mode,
                       OutputElementType element_type) const;

    template <typename Out, typename View>
    static void pack_view(const View& view,
                          const FieldSelection& selection,
                          std::vector<Out>& output) {
        constexpr std::size_t dimensions = View::rank;
        static_assert(dimensions >= 1 && dimensions <= 4,
                      "BP5 packing supports 1-D through 4-D views.");
        if (selection.dimensions != dimensions) {
            throw std::invalid_argument("BP5 selection/view dimension mismatch.");
        }
        output.resize(selection.elements());
        if (selection.empty()) return;

        std::size_t out = 0;
        if constexpr (dimensions == 1) {
            for (std::size_t k = 0; k < selection.count[0]; ++k) {
                output[out++] = static_cast<Out>(
                    view(selection.memory_start[0] + k));
            }
        } 
        else if constexpr (dimensions == 2) {
            for (std::size_t j = 0; j < selection.count[0]; ++j) {
                for (std::size_t i = 0; i < selection.count[1]; ++i) {
                    output[out++] = static_cast<Out>(
                        view(selection.memory_start[0] + j,
                             selection.memory_start[1] + i));
                }
            }
        } 
        else if constexpr (dimensions == 3) {
            for (std::size_t k = 0; k < selection.count[0]; ++k) {
                for (std::size_t j = 0; j < selection.count[1]; ++j) {
                    for (std::size_t i = 0; i < selection.count[2]; ++i) {
                        output[out++] = static_cast<Out>(
                            view(selection.memory_start[0] + k,
                                 selection.memory_start[1] + j,
                                 selection.memory_start[2] + i));
                    }
                }
            }
        } 
        else {
            for (std::size_t c = 0; c < selection.count[0]; ++c) {
                for (std::size_t k = 0; k < selection.count[1]; ++k) {
                    for (std::size_t j = 0; j < selection.count[2]; ++j) {
                        for (std::size_t i = 0; i < selection.count[3]; ++i) {
                            output[out++] = static_cast<Out>(
                                view(selection.memory_start[0] + c,
                                     selection.memory_start[1] + k,
                                     selection.memory_start[2] + j,
                                     selection.memory_start[3] + i));
                        }
                    }
                }
            }
        }
    }

private:
    const Core::State& state_;
    Bp5BufferSet& buffers_;
};

} // namespace VVM::IO::BP5

#endif
