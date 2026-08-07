#ifndef VVM_IO_HISTORY_LEGACY_HISTORY_WRITER_HPP
#define VVM_IO_HISTORY_LEGACY_HISTORY_WRITER_HPP

#include <memory>
#include <stdexcept>

#include "io/OutputManager.hpp"
#include "io/history/HistoryWriter.hpp"

namespace VVM::IO {

class LegacyHistoryWriter final : public HistoryWriter {
public:
    LegacyHistoryWriter(
        const Utils::ConfigurationManager& config,
        const Core::Grid& grid,
        const Core::Parameters& parameters,
        Core::State& state,
        MPI_Comm comm)
        : manager_(std::make_unique<OutputManager>(
              config, grid, parameters, state, comm)) {}

    void write(std::size_t step, VVM::Real time) override {
        if (!manager_) {
            throw std::runtime_error("Legacy history writer is already closed.");
        }
        manager_->write(step, time);
    }

    void close() override { manager_.reset(); }

private:
    std::unique_ptr<OutputManager> manager_;
};

} // namespace VVM::IO

#endif
