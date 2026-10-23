#pragma once
#include "io/hdf5/CompressedHdf5Writer.hpp"
#include "io/OutputCompression.hpp"
#include "utils/ConfigurationManager.hpp"
#include <adios2.h>

namespace VVM::IO {
inline Hdf5CompressionOptions hdf5_compression_options(const Utils::ConfigurationManager& config) {
    Hdf5CompressionOptions options;
    const auto settings = configured_output_compression(config, "output.hdf5_compression");
    // Validate HDF5 overrides with the same schema as the common block.
    merge_output_compression(settings, nlohmann::json::object());
    options.enabled = settings.value("enabled", false);
    options.level = settings.value("level", 1);
    const auto minimum = settings.value("minimum_bytes", int64_t(1048576));
    if (options.level < 0 || options.level > 9 || minimum < 1)
        throw std::invalid_argument("output.hdf5_compression requires level 0–9 and positive minimum_bytes");
    options.minimum_bytes = static_cast<size_t>(minimum);
    return options;
}

// Preserve the same root attributes as the ADIOS2 HDF5 sink. All currently
// emitted numeric attributes use model float/double precision.
inline void copy_hdf5_attributes(adios2::IO& io, CompressedHdf5Writer& writer) {
    for (const auto& item : io.AvailableAttributes()) {
        const auto& name = item.first;
        if (auto a = io.InquireAttribute<std::string>(name)) {
            const auto values = a.Data();
            if (values.size() != 1) throw std::runtime_error("Unsupported string attribute array: " + name);
            writer.attribute(name, values.front());
        } else if (auto a = io.InquireAttribute<float>(name)) {
            writer.attribute(name, a.Data());
        } else if (auto a = io.InquireAttribute<double>(name)) {
            writer.attribute(name, a.Data());
        } else {
            throw std::runtime_error("Unsupported HDF5 output attribute type: " + name);
        }
    }
}
} // namespace VVM::IO
