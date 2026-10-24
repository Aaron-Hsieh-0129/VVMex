// Standalone ADIOS2 lossless roundtrip, including empty writer ranks.
#include <adios2.h>
#include <mpi.h>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank = 0, ranks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);
    try {
        if (argc != 2) throw std::invalid_argument("usage: test_bp5_compression OUTPUT.bp");
        // Probe the loaded library, not feature macros from potentially
        // overlapping dependency include paths. Destroy ADIOS before any
        // skip path calls MPI_Finalize, and agree on skipping across ranks.
        int available = 1;
        {
            adios2::ADIOS probe(MPI_COMM_WORLD);
            try {
                probe.DefineOperator("capability_check", "blosc");
            } catch (const std::exception& e) {
                if (std::string(e.what()).find("didn't compile with blosc library") == std::string::npos)
                    throw;
                available = 0;
            }
        }
        int all_available = 0;
        MPI_Allreduce(&available, &all_available, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
        if (!all_available) {
            if (rank == 0) std::puts("SKIP: loaded ADIOS2 library was built without Blosc support");
            MPI_Finalize();
            return 77;
        }
        const size_t plane = 128 * 512;
        const size_t first = 2 * size_t(rank) / size_t(ranks);
        const size_t end = 2 * size_t(rank + 1) / size_t(ranks);
        adios2::ADIOS adios(MPI_COMM_WORLD);
        auto io = adios.DeclareIO("compressed");
        io.SetEngine("BP5");
        io.SetParameter("StatsLevel", "0");
        auto op = adios.DefineOperator("lossless", "blosc", {{"compressor", "zstd"},
            {"clevel", "1"}, {"nthreads", "1"}, {"threshold", "128"},
            {"doshuffle", "BLOSC_SHUFFLE"}});
        auto f32 = io.DefineVariable<float>("rain", {2,128,512}, {first,0,0}, {end-first,128,512});
        auto f64 = io.DefineVariable<double>("u", {2,128,512}, {first,0,0}, {end-first,128,512});
        f32.AddOperation(op, {});
        f64.AddOperation(op, {});
        auto clock = io.DefineVariable<int64_t>("model_step");
        auto writer = io.Open(argv[1], adios2::Mode::Write);
        for (int step = 0; step < 2; ++step) {
            std::vector<float> rain((end-first)*plane);
            std::vector<double> u(rain.size());
            for (size_t i=0; i<rain.size(); ++i) {
                rain[i] = float((first*plane+i)%301)*0.125f + step;
                u[i] = double((first*plane+i)%301)*0.125 + step;
            }
            writer.BeginStep();
            if (end > first) {
                writer.Put(f32, rain.data(), adios2::Mode::Sync);
                writer.Put(f64, u.data(), adios2::Mode::Deferred);
            }
            const int64_t value = INT64_C(9007199254740993)+step;
            if (rank == 0) writer.Put(clock, value);
            writer.EndStep();
        }
        writer.Close();
        MPI_Barrier(MPI_COMM_WORLD);
        if (rank == 0) {
            adios2::ADIOS read_adios(MPI_COMM_SELF);
            auto read_io = read_adios.DeclareIO("readback");
            read_io.SetEngine("BP5");
            auto reader = read_io.Open(argv[1], adios2::Mode::Read);
            for (int step=0; step<2; ++step) {
                if (reader.BeginStep() != adios2::StepStatus::OK)
                    throw std::runtime_error("Missing BP5 step");
                std::vector<float> rain(2*plane), expected_rain(2*plane);
                std::vector<double> u(2*plane), expected_u(2*plane);
                int64_t value=0;
                reader.Get(read_io.InquireVariable<float>("rain"), rain.data(), adios2::Mode::Sync);
                reader.Get(read_io.InquireVariable<double>("u"), u.data(), adios2::Mode::Sync);
                reader.Get(read_io.InquireVariable<int64_t>("model_step"), &value, adios2::Mode::Sync);
                for (size_t i=0; i<rain.size(); ++i) {
                    expected_rain[i] = float(i%301)*0.125f + step;
                    expected_u[i] = double(i%301)*0.125 + step;
                }
                if (std::memcmp(rain.data(), expected_rain.data(), rain.size()*sizeof(float)) ||
                    std::memcmp(u.data(), expected_u.data(), u.size()*sizeof(double)) ||
                    value != INT64_C(9007199254740993)+step)
                    throw std::runtime_error("Compressed BP5 values differ");
                reader.EndStep();
            }
            reader.Close();
            std::puts("PASS: compressed BP5 bitwise float/double readback and exact clocks");
        }
        MPI_Barrier(MPI_COMM_WORLD);
    } catch (const std::exception& e) {
        std::fprintf(stderr,"[rank %d] %s\n",rank,e.what());
        MPI_Abort(MPI_COMM_WORLD,1);
    }
    MPI_Finalize();
    return 0;
}
