# Tests

Enable tests with `-DBUILD_TESTS=ON` on your usual machine preset, build, then
run CTest. Source your normal build environment first (`VVM_ROOT`, libraries,
MPI, and Python dependencies).

```sh
cmake --preset blaze -DBUILD_TESTS=ON
cmake --build build -j 8
ctest --test-dir build --output-on-failure
```

## Choose suites

CMake flags control which tests are registered. Changing a flag requires
reconfiguring; CMake remembers flag values in the build directory.

| Flag | Default | Coverage |
| --- | --- | --- |
| `BUILD_TESTS` | OFF | Build and register project tests |
| `VVM_TEST_REGRESSION` | ON | Cartesian baselines, mountain digest, configuration equivalence |
| `VVM_TEST_RLL` | OFF | Jung and RLL mountain model runs; RLL BP5 history when BP5 is also enabled |
| `VVM_TEST_RLL_PHYSICS` | OFF | Eight RLL turbulence, surface, and P3 model checks |
| `VVM_TEST_BP5` | OFF | Direct BP5 unit and integration tests, including multiple ranks |
| `VVM_TEST_SST` | OFF | SST relay precision; needs working SST networking |
| `VVM_TEST_PHYSICS` | OFF | GPU reference cases for P3, radiation, and land |
| `VVM_TEST_MULTIRANK` | OFF | Additional decomposition and rank-invariance checks |

The two RLL flags are independent and support CPU and GPU builds. Small RLL
geometry, solver, transport, and physics unit checks remain in the default
suite. No numerical coverage was deleted. `VVM_TEST_PHYSICS` still controls the
older GPU reference cases; it does not enable RLL physics model runs.

```sh
# Enable both RLL model suites.
cmake --preset blaze -DBUILD_TESTS=ON -DVVM_TEST_RLL=ON -DVVM_TEST_RLL_PHYSICS=ON
ctest --test-dir build --output-on-failure -L '^rll'

# Disable model regressions and both RLL model suites, then select unit checks.
cmake --preset blaze -DBUILD_TESTS=ON -DVVM_TEST_REGRESSION=OFF \
  -DVVM_TEST_RLL=OFF -DVVM_TEST_RLL_PHYSICS=OFF
ctest --test-dir build --output-on-failure -L '^unit$'

# Preview without running; select one registered test by name.
ctest --test-dir build -N
ctest --test-dir build --output-on-failure -R '^test_regular_latlon_surface$'
```

Labels filter registered tests: `-L '^rll$'` selects dynamics model runs,
`-L '^rll-physics$'` selects RLL physics, and `-L '^rll'` selects both.
Many unit tests use MPI and Kokkos and need a GPU in GPU builds.
Set `VVM_TEST_GPUS=6` when running CTest to select a device. Multi-rank GPU
tests need distinct physical devices. CPU builds use
`-DVVM_TEST_CPU_THREADS=<n>` per rank and CTest `PROCESSORS` weights;
`ctest -j <cores>` schedules within that budget.

## Add a test

`CMakeLists.txt` lists the suites under `cmake/`. Add the registration to the
matching suite; shared setup belongs in `TestHelpers.cmake`, and runtime
settings are applied last by `TestEnvironment.cmake`.

For `unit/test_example.cpp`, add to `cmake/UnitTests.cmake`:

```cmake
add_vvm_unit_test(test_example)

# Or, for a test requiring libraries, one MPI rank, and device initialization:
add_vvm_unit_test(test_example DEVICE MPI TIMEOUT 120
    LIBRARIES vvm_core vvm_utils Kokkos::kokkos MPI::MPI_CXX)
```

Choose one registration. `DEVICE` applies the GPU wrapper and GPU lock, or
CPU processor budget. `MPI` launches one rank. For multiple configurations,
call `vvm_add_test_executable(name LIBRARIES ...)` once and use `add_test()` for
each invocation, declaring labels, timeout, and resources as the nearby tests do.

For a Python model check accepting the model executable, output directory,
`--wrapper`, and `--launcher`, add inside the appropriate flag block in
`cmake/RLLTests.cmake`:

```cmake
vvm_add_model_test(test_rll_example
    SCRIPT check_rll_example.py OUTPUT rll_example_tests
    TIMEOUT 300 LABELS rll ARGS --example-option)
```

The helper sets the working directory, integration label, timeout, launcher,
and backend resource policy. Give each model check a unique output directory.
Existing baseline and digest cases use `add_vvm_test()` and `add_case_test()`
in `cmake/RegressionTests.cmake`.

Check registration without compiling or running a model:

```sh
python3 tests/scripts/check_test_registry.py
python3 tests/scripts/test_cmake_registration.py
```

The second command needs CMake, CTest, and a C++ compiler. It uses placeholder
libraries to check suite selection and resource settings on both backends;
it does not validate compilation or numerical results.
