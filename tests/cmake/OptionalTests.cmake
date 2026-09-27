# --- physics tier: GPU only, ~80 s. P3, RRTMGP and the Noah land model -----
# Forced OFF on a CPU build by the backend guard above: tests/references_cpu/
# has no digests for these cases, so they would fail for the wrong reason.
if(VVM_TEST_PHYSICS)
    add_case_test(evergreen          1 physics)
    add_case_test(grass              1 physics)
    add_case_test(urban              1 physics)
    add_case_test(sea_grass_mountain 1 physics)
    add_case_test(sea_urban_mountain 1 physics)
    add_case_test(rcemip             1 physics --enable-random-perturbation)
endif()

# --- multirank: needs 4 real GPUs, ~30 s ----------------------------------
# 4 ranks gives a 2x2 decomposition. This is the tier that would have caught the
# initialiser bug where each rank stamped its own perturbation from local indices;
# 2 ranks splits one direction only and misses it.
#
# Two cases, not three: every rank-invariance failure is a halo or a global-index
# bug, and what decides whether a case can expose one is which terms touch the
# halo, not which baseline it came from. twisting carries all three vorticity
# components and their cross-derivatives; 2dbubble adds buoyancy, thermodynamics
# and the full elliptic wind solve, the widest halo dependence in the model. The
# x-only advection of advection_u is a strict subset of both, so its pair of
# 4-GPU runs bought no coverage the other two did not already have.
#
# On a CPU build these are the slowest tests in the tree by a wide margin: the
# wind solver exchanges halos many times per step, and at 4 ranks over MPI on a
# 32^2 grid that is pure latency with no work to hide it (measured on blaze:
# twisting 1 s at 1 rank, 20 min at 4). Keep them out of any per-commit gate.
if(VVM_TEST_MULTIRANK)
    add_rank_invariance_test(twisting 4)
    add_rank_invariance_test(2dbubble 4)
endif()
