#ifndef VVM_IO_BP5_BUFFER_SET_HPP
#define VVM_IO_BP5_BUFFER_SET_HPP

#include <cstddef>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

#include "core/vvm_types.hpp"

namespace VVM::IO::BP5 {

class Bp5BufferSet {
public:
    std::vector<VVM::Real>& require(const std::string& field_name,
                                    std::size_t elements) {
        auto [it, inserted] = buffers_.try_emplace(field_name, elements);
        if (!inserted && it->second.size() != elements) {
            throw std::logic_error(
                "BP5 persistent buffer size changed for field '" + field_name + "'.");
        }
        return it->second;
    }

    std::size_t bytes() const noexcept {
        std::size_t total = 0;
        for (const auto& item : buffers_) {
            total += item.second.size() * sizeof(VVM::Real);
        }
        return total;
    }

private:
    std::map<std::string, std::vector<VVM::Real>> buffers_;
};

} // namespace VVM::IO::BP5

#endif
