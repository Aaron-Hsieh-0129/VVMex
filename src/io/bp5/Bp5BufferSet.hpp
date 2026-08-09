#ifndef VVM_IO_BP5_BUFFER_SET_HPP
#define VVM_IO_BP5_BUFFER_SET_HPP

#include <cstddef>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

#include "core/vvm_types.hpp"

namespace VVM::IO::BP5 {

// Persistent per-field staging buffers, allocated on first use and reused for
// every later step. Kept in one map per element type: output precision is fixed
// for the lifetime of a dataset, so exactly one of these is ever populated.
class Bp5BufferSet {
public:
    template <typename T>
    std::vector<T>& require(const std::string& field_name,
                            std::size_t elements) {
        auto& buffers = select<T>();
        auto [it, inserted] = buffers.try_emplace(field_name, elements);
        if (!inserted && it->second.size() != elements) {
            throw std::logic_error(
                "BP5 persistent buffer size changed for field '" + field_name + "'.");
        }
        return it->second;
    }

    std::size_t bytes() const noexcept {
        std::size_t total = 0;
        for (const auto& item : float_buffers_) {
            total += item.second.size() * sizeof(float);
        }
        for (const auto& item : double_buffers_) {
            total += item.second.size() * sizeof(double);
        }
        return total;
    }

private:
    template <typename T>
    std::map<std::string, std::vector<T>>& select();

    std::map<std::string, std::vector<float>> float_buffers_;
    std::map<std::string, std::vector<double>> double_buffers_;
};

template <>
inline std::map<std::string, std::vector<float>>&
Bp5BufferSet::select<float>() {
    return float_buffers_;
}

template <>
inline std::map<std::string, std::vector<double>>&
Bp5BufferSet::select<double>() {
    return double_buffers_;
}

} // namespace VVM::IO::BP5

#endif
