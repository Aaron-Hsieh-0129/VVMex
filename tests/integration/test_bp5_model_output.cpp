#include "core/vvm_types.hpp"

#include <adios2.h>
#include <mpi.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

namespace {
int failures = 0;

void check(bool condition, const char* message) {
    if (condition) return;
    ++failures;
    std::fprintf(stderr, "FAIL: %s\n", message);
}
} // namespace

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    if (argc != 2) {
        std::fprintf(stderr, "usage: test_bp5_model_output DATASET\n");
        MPI_Finalize();
        return 2;
    }
    try {
        adios2::ADIOS adios(MPI_COMM_SELF);
        auto io = adios.DeclareIO("VVM_BP5_MODEL_SMOKE_READER");
        auto reader = io.Open(argv[1], adios2::Mode::Read, MPI_COMM_SELF);
        int steps = 0;
        while (reader.BeginStep() == adios2::StepStatus::OK) {
            auto time_var = io.InquireVariable<VVM::Real>("time");
            auto step_var = io.InquireVariable<std::int64_t>("model_step");
            auto thbar_var = io.InquireVariable<VVM::Real>("thbar");
            auto topo_var = io.InquireVariable<VVM::Real>("topo");
            auto u_var = io.InquireVariable<VVM::Real>("u");
            check(time_var && step_var && thbar_var && topo_var && u_var,
                  "model BP5 variables exist");
            if (time_var && step_var && thbar_var && topo_var && u_var) {
                check(thbar_var.Shape() == adios2::Dims({33}), "model thbar shape");
                check(topo_var.Shape() == adios2::Dims({32, 32}), "model topo shape");
                check(u_var.Shape() == adios2::Dims({33, 32, 32}), "model u shape");
                VVM::Real time = VVM::real(-1);
                std::int64_t model_step = -1;
                reader.Get(time_var, time, adios2::Mode::Sync);
                reader.Get(step_var, model_step, adios2::Mode::Sync);
                check(time == VVM::real(steps), "model time advances across BP5 output");
                check(model_step == steps, "model step advances across BP5 output");
                std::vector<VVM::Real> values(33 * 32 * 32);
                reader.Get(u_var, values.data(), adios2::Mode::Sync);
                bool finite = true;
                for (const auto value : values) finite = finite && std::isfinite(value);
                check(finite, "model u output is finite");
            }
            reader.EndStep();
            ++steps;
        }
        reader.Close();
        check(steps == 11, "initial plus ten computed BP5 steps are readable");

        const auto units = io.InquireAttribute<std::string>("units", "u");
        const auto staggering =
            io.InquireAttribute<std::string>("grid_staggering", "u");
        check(units && units.Data().at(0) == "m s-1", "model field units metadata");
        check(staggering && staggering.Data().at(0) == "staggered_x",
              "model field staggering metadata");
    } catch (const std::exception& e) {
        std::fprintf(stderr, "exception: %s\n", e.what());
        ++failures;
    }
    if (failures == 0) std::puts("test_bp5_model_output: PASS");
    MPI_Finalize();
    return failures == 0 ? 0 : 1;
}
