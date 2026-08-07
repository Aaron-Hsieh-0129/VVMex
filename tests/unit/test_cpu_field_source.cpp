#include "io/bp5/CpuFieldSource.hpp"

#include <cstdio>
#include <vector>

using VVM::IO::BP5::CpuFieldSource;
using VVM::IO::BP5::FieldSelection;

namespace {
int failures = 0;
void check(bool condition, const char* message) {
    if (!condition) {
        std::fprintf(stderr, "FAIL: %s\n", message);
        ++failures;
    }
}
} // namespace

int main(int argc, char** argv) {
    Kokkos::initialize(argc, argv);
    {
        std::vector<VVM::Real> packed;

        Kokkos::View<VVM::Real*, Kokkos::LayoutRight, Kokkos::HostSpace> one("one", 7);
        for (std::size_t k = 0; k < 7; ++k) one(k) = 10 + k;
        FieldSelection s1{1, {3}, {0}, {3}, {2}, {7}};
        CpuFieldSource::pack_view(one, s1, packed);
        check(packed == std::vector<VVM::Real>({12, 13, 14}), "1-D pack");

        Kokkos::View<VVM::Real**, Kokkos::LayoutRight, Kokkos::HostSpace> two("two", 4, 5);
        for (std::size_t j = 0; j < 4; ++j)
            for (std::size_t i = 0; i < 5; ++i) two(j, i) = 100 * j + i;
        FieldSelection s2{2, {2, 3}, {0, 0}, {2, 3}, {1, 1}, {4, 5}};
        CpuFieldSource::pack_view(two, s2, packed);
        check(packed == std::vector<VVM::Real>({101, 102, 103, 201, 202, 203}), "2-D pack order");

        Kokkos::View<VVM::Real***, Kokkos::LayoutRight, Kokkos::HostSpace> three("three", 4, 4, 5);
        for (std::size_t k = 0; k < 4; ++k)
            for (std::size_t j = 0; j < 4; ++j)
                for (std::size_t i = 0; i < 5; ++i) three(k, j, i) = 10000 * k + 100 * j + i;
        FieldSelection s3{3, {2, 2, 2}, {0, 0, 0}, {2, 2, 2}, {1, 1, 2}, {4, 4, 5}};
        CpuFieldSource::pack_view(three, s3, packed);
        check(packed == std::vector<VVM::Real>({10102, 10103, 10202, 10203,
                                                20102, 20103, 20202, 20203}),
              "3-D pack order and halo exclusion");

        Kokkos::View<VVM::Real****, Kokkos::LayoutRight, Kokkos::HostSpace> four("four", 2, 3, 3, 4);
        for (std::size_t c = 0; c < 2; ++c)
            for (std::size_t k = 0; k < 3; ++k)
                for (std::size_t j = 0; j < 3; ++j)
                    for (std::size_t i = 0; i < 4; ++i)
                        four(c, k, j, i) = 1000000 * c + 10000 * k + 100 * j + i;
        FieldSelection s4{4, {2, 1, 1, 2}, {0, 0, 0, 0}, {2, 1, 1, 2},
                          {0, 1, 1, 1}, {2, 3, 3, 4}};
        CpuFieldSource::pack_view(four, s4, packed);
        check(packed == std::vector<VVM::Real>({10101, 10102, 1010101, 1010102}),
              "4-D component-major pack order");

        FieldSelection empty{3, {2, 2, 2}, {0, 0, 0}, {2, 0, 2}, {0, 0, 0}, {4, 4, 5}};
        CpuFieldSource::pack_view(three, empty, packed);
        check(packed.empty(), "empty selection pack");
    }
    Kokkos::finalize();
    if (failures == 0) std::puts("test_cpu_field_source: PASS");
    return failures == 0 ? 0 : 1;
}
