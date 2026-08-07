#include "io/bp5/Bp5FieldSchema.hpp"

#include <cstdio>
#include <stdexcept>

using namespace VVM::IO::BP5;

namespace {
int failures = 0;
void check(bool condition, const char* message) {
    if (!condition) {
        std::fprintf(stderr, "FAIL: %s\n", message);
        ++failures;
    }
}
} // namespace

int main() {
    const InclusiveBounds bounds{2, 7, 1, 6, 1, 4};
    const GridRegion upper_left{
        10, 8, 6,
        0, 0, 0,
        5, 4, 6,
        2};
    Bp5FieldSchema schema(bounds, upper_left, 0);

    const auto one = schema.selection(1);
    check(one.shape == adios2::Dims{6}, "1-D global shape");
    check(one.start == adios2::Dims{1}, "1-D start");
    check(one.count == adios2::Dims{4}, "1-D count");
    check(one.memory_start == adios2::Dims{3}, "1-D halo offset");
    check(one.memory_count == adios2::Dims{10}, "1-D memory extent");

    const auto two = schema.selection(2);
    check(two.shape == (adios2::Dims{8, 10}), "2-D y,x shape");
    check(two.start == (adios2::Dims{1, 2}), "2-D global start");
    check(two.count == (adios2::Dims{3, 3}), "2-D intersection");
    check(two.memory_start == (adios2::Dims{3, 4}), "2-D memory start");
    check(two.memory_count == (adios2::Dims{8, 9}), "2-D ghosted extent");

    const auto three = schema.selection(3);
    check(three.shape == (adios2::Dims{6, 8, 10}), "3-D z,y,x shape");
    check(three.count == (adios2::Dims{4, 3, 3}), "3-D count");
    check(three.elements() == 36, "3-D elements");

    const auto four = schema.selection(4, 3);
    check(four.shape == (adios2::Dims{3, 6, 8, 10}), "4-D c,z,y,x shape");
    check(four.start == (adios2::Dims{0, 1, 1, 2}), "4-D start");
    check(four.count == (adios2::Dims{3, 4, 3, 3}), "4-D count");
    check(four.memory_start == (adios2::Dims{0, 3, 3, 4}), "4-D memory start");

    const GridRegion outside{
        10, 8, 6,
        8, 7, 0,
        2, 1, 6,
        2};
    Bp5FieldSchema empty_schema(bounds, outside, 3);
    check(empty_schema.selection(3).empty(), "empty rank intersection");
    check(empty_schema.selection(3).elements() == 0, "empty rank elements");
    check(empty_schema.selection(1).empty(), "non-root replicated 1-D field");

    std::size_t coverage = 0;
    int rank = 0;
    for (std::size_t y0 : {std::size_t{0}, std::size_t{4}}) {
        for (std::size_t x0 : {std::size_t{0}, std::size_t{5}}) {
            const GridRegion region{10, 8, 6, x0, y0, 0, 5, 4, 6, 2};
            coverage += Bp5FieldSchema(bounds, region, rank++).selection(2).elements();
        }
    }
    check(coverage == bounds.nx() * bounds.ny(), "synthetic decomposition coverage");

    try {
        (void)schema.selection(0);
        check(false, "0-D selection accepted");
    } catch (const std::invalid_argument&) {
    }
    try {
        (void)schema.selection(4, 0);
        check(false, "zero-component 4-D selection accepted");
    } catch (const std::invalid_argument&) {
    }

    if (failures == 0) std::puts("test_bp5_field_schema: PASS");
    return failures == 0 ? 0 : 1;
}
