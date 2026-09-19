find_package(MPI REQUIRED)
find_package(Python3 COMPONENTS Interpreter REQUIRED)

set(TEST_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(CASE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../rundata/input_configs/default_cases)
# one_gpu_per_rank.sh only assigns CUDA_VISIBLE_DEVICES, so it has nothing to do
# in a CPU build.
if(VVM_ENABLE_GPU)
    set(GPU_WRAP ${TEST_DIR}/scripts/one_gpu_per_rank.sh)
else()
    set(GPU_WRAP "")
endif()

# Reference data is backend-specific. CPU and GPU agree only to a few ulp, which
# the 1e-6 tolerance absorbs for the advection cases but not for 2dbubble (a
# convective case amplifies it to ~2e-2 over 120 steps), and never for the
# SHA-256 digests. So each backend is gated against references generated on that
# backend; cross-backend agreement is a separate question, and one that
# -DVVM_DETERMINISTIC_FP=ON settles for the dry cases -- see
# docs/developer-guides/reproducibility.md.
if(VVM_ENABLE_GPU)
    set(VVM_BASELINE_DIR  ${TEST_DIR}/baselines)
    set(VVM_REFERENCE_DIR ${TEST_DIR}/references)
else()
    set(VVM_BASELINE_DIR  ${TEST_DIR}/baselines_cpu)
    set(VVM_REFERENCE_DIR ${TEST_DIR}/references_cpu)
endif()
message(STATUS "Test reference data: ${VVM_BASELINE_DIR}, ${VVM_REFERENCE_DIR}")

# OpenMPI binds each rank to a single core by default. On a CPU build that pins
# every OpenMP thread of a rank onto one CPU, so raising OMP_NUM_THREADS makes
# the run slower rather than faster (measured: advection_u 20 s -> 45 s at 16
# threads, Cpus_allowed_list: 0). core_run.sh passes --bind-to none for the same
# reason. The GPU path is one thread per rank, so leave its binding alone.
if(VVM_ENABLE_GPU)
    set(VVM_MPI_BIND_ARGS "")
else()
    set(VVM_MPI_BIND_ARGS --bind-to none)
endif()

# Use VVM_ROOT as the runtime working directory for CTest tests.
# Root CMakeLists.txt already requires VVM_ROOT, so this should normally exist.
set(VVM_TEST_WORKDIR "$ENV{VVM_ROOT}")
if(VVM_TEST_WORKDIR STREQUAL "")
  message(FATAL_ERROR "VVM_ROOT is not set. Please export VVM_ROOT=/path/to/VVMex before configuring tests.")
endif()

file(TO_CMAKE_PATH "${VVM_TEST_WORKDIR}" VVM_TEST_WORKDIR)

# How long each default_cases test integrates. Short enough to stay a test,
# long enough that a numerical regression has shown up in the output.
set(VVM_TEST_SECONDS 120 CACHE STRING "Simulated seconds for default_cases tests")

# Optional suites are omitted from CTest until enabled; labels then filter them.
# See tests/README.md for selection and resource requirements.
option(VVM_TEST_RLL "Register RLL dynamics model runs" OFF)
option(VVM_TEST_RLL_PHYSICS "Register RLL turbulence, surface and P3 model runs" OFF)
option(VVM_TEST_REGRESSION "Register baseline and configuration-equivalence model runs" ON)
option(VVM_TEST_BP5       "Register direct BP5 output tests (CPU and GPU builds)"     OFF)
option(VVM_TEST_SST       "Register the SST relay precision test (2 ranks, network)"  OFF)
option(VVM_TEST_PHYSICS   "Register physics-tier tests (GPU builds only, ~80 s)"      OFF)
option(VVM_TEST_MULTIRANK "Register rank-invariance tests (4 ranks, ~30 s)"           OFF)

# ---------------------------------------------------------------------------
# What a test occupies while it runs.
#
# On a GPU build the scarce resource is the device. Several NCCL ranks sharing
# one GPU is unsupported and silently changes results, so anything that touches
# a device takes a hard lock and the suite runs one model at a time.
#
# On a CPU build the scarce resource is cores, and there is no reason two small
# cases cannot run side by side: they are separate processes writing separate
# output directories, and thread count does not change results. So instead of a
# lock, each test declares how many cores it wants (ranks x threads) and
# `ctest -j <cores>` packs them without oversubscribing. The default cases are
# 32x32x33 and 512x16x74 grids -- far too small to keep 224 cores busy one at a
# time, which is what the lock forced.
# ---------------------------------------------------------------------------
set(VVM_TEST_CPU_THREADS 64 CACHE STRING
    "OMP_NUM_THREADS per rank for CPU tests")


# A tier the current backend cannot run is switched off here, with the reason.
# Silently registering nothing would look like the tier passed.
if(VVM_TEST_PHYSICS AND NOT VVM_ENABLE_GPU)
    message(WARNING
        "VVM_TEST_PHYSICS=ON ignored: no CPU reference data exists under "
        "tests/references_cpu/ for the physics cases.")
    set(VVM_TEST_PHYSICS OFF)
endif()

# Report the tiers the way the rest of the configure output reports its options,
# including which backend each one needs and how to switch it on.
function(_report_test_tier LABEL STATE NEEDS OPT_NAME)
    if(STATE)
        message(STATUS "  Test tier ${LABEL}: ON   (${NEEDS})")
    else()
        message(STATUS "  Test tier ${LABEL}: OFF  (${NEEDS}) -- enable with -D${OPT_NAME}=ON")
    endif()
endfunction()

message(STATUS "Test tiers (native digests are bit-for-bit; RCEMIP uses canonical float32; simulated seconds per case: ${VVM_TEST_SECONDS}):")
if(VVM_ENABLE_GPU)
    message(STATUS "  Test tier default  : ON   (unit and I/O tests) -- 1 GPU")
else()
    message(STATUS "  Test tier default  : ON   (unit and I/O tests)")
endif()
_report_test_tier("bp5      " ${VVM_TEST_BP5}       "CPU or GPU build"  VVM_TEST_BP5)
_report_test_tier("sst      " ${VVM_TEST_SST}       "2 ranks, SST"      VVM_TEST_SST)
_report_test_tier("physics  " ${VVM_TEST_PHYSICS}   "GPU build only"    VVM_TEST_PHYSICS)
_report_test_tier("multirank" ${VVM_TEST_MULTIRANK} "4 ranks / 4 GPUs"  VVM_TEST_MULTIRANK)
_report_test_tier("rll      " ${VVM_TEST_RLL} "CPU or GPU model runs" VVM_TEST_RLL)
_report_test_tier("rll-physics" ${VVM_TEST_RLL_PHYSICS} "CPU or GPU model runs" VVM_TEST_RLL_PHYSICS)
_report_test_tier("regression" ${VVM_TEST_REGRESSION} "baseline model runs" VVM_TEST_REGRESSION)
message(STATUS "  Select the test device with VVM_TEST_GPUS (e.g. VVM_TEST_GPUS=6 ctest --test-dir build)")

