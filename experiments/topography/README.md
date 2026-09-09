# Centered RLL mountain: implementation handoff

Base revision: `84f90094b32c249959d1fea01c13e373e2f4aa6d`, branch `feat/generalized-coordinate`.
Worktree changes include pre-existing user formatting in `Initializer.cpp`; do not discard it.

`rundata/input_configs/topography.json` now selects `rll_mountain`: 400×100×16 physical points, radius6371220m, longitude0–360°, latitude±45°, dz1000m, dt10s, duration3600s, output600s. A spherical Gaussian mountain is centered at180°E,0°N, height2000m and e-folding half-width500km, truncated at3 widths and rounded to existing vertical interfaces. The height/width are example terrain settings, not constants from Jung2019. The background is the case1 jet with its perturbation, unit density and inert theta300K. This is a neutral terrain-flow experiment, not a mountain-wave benchmark or a reproduction of the flat-paper case.

The existing VVMex full-column, fixed-iteration terrain method is retained. RLL metric factors enter the curl of masked physical winds: xi atV, eta (negative northward component) atU. Fluid-cell vorticity is retained; adjusted fields feed vertical inversion, column zeta diagnosis and wind recovery. The prognostic top zeta is not overwritten by a separate terrain curl. There is no new solver matrix, convergence stopping, Cartesian mean subtraction, or density normalization. Existing CUDA preparation/capture/replay remains enabled. Raw diagnostic winds inside terrain need not vanish; `u_topo/v_topo/w_topo` are the explicitly masked fields, as in the Cartesian method. This is not a cut-cell solver.

Fixed counts: initial horizontal1000, subsequent horizontal200, vertical10; WRXMU8105.694691387022. Periodic longitude, existing bounded-latitude channel walls and southern circulation preservation. Moisture, thermodynamic transport/buoyancy, Coriolis, external terrain input and restart remain unsupported in this deliberately narrow RLL mode.

```sh
export VVM_ROOT=/home/mog/CVVMex
export LD_LIBRARY_PATH=/home/mog/libs_VVMex/lib:$LD_LIBRARY_PATH
export OMP_NUM_THREADS=4 VVM_TEST_GPUS=0
cmake --build build --clean-first -j32
mpiexec --bind-to none -n 1 tests/scripts/one_gpu_per_rank.sh build/vvm rundata/input_configs/topography.json
ctest --test-dir build -R 'test_rll_mountain_shared_model|test_jung2019_shared_model|mountain' --output-on-failure
```

Choose a fresh output directory before rerunning. Default RLL output: `output/topo_rll` (HDF5); the earlier Cartesian run at `output/topo/vvm_output.bp` is preserved. That earlier run reached600s with finite fields and passed its three original mountain regressions, but is not evidence for RLL terrain.

New integrated test: `tests/scripts/check_rll_mountain.py`. It checks rest, nonzero terrain response, exact masks and height, independent physical spherical curl (rtol2e-13/atol2e-15), exact zero-height equivalence to the original flat mode, circulation (1e-11m/s), finite histories and invalid terrain rejection. Each launch uses a unique directory and writes its full configuration and log. `--ranks 4` enables a distributed execution check; it does not by itself compare decompositions.

## Validation results

Clean CPU and CUDA builds passed (NVHPC24.9, CUDA12.6, Kokkos4.7.2, Release/double; presets blaze and blaze-cpu). Final source adjustment rebuilt in each backend. Logs: `build{,_cpu}/rll-terrain-{build,rebuild}.log`.

The one-hour400×100×16 CUDA run completed all360 steps and seven snapshots in13.73s of reported model runtime. All outputs passed the independent terrain/curl/mask checks and remained finite. Final maximum|w|0.8785760676m/s; raw inside-solid maximum|w|0.0004101537m/s; masked inside-solid w exactly0. Southern circulation drift2.93e-18m/s. The nonzero raw residual is explicitly retained, not hidden by the plot. Output `output/topo_rll` is488MiB, with `terrain_validation.json` and `terrain.png`. Total experiments+output remains below8GiB. Earlier Cartesian output is untouched.

```sh
python experiments/topography/analyze.py rundata/input_configs/topography.json
python experiments/topography/compare.py REFERENCE_OUTPUT CANDIDATE_OUTPUT --report REPORT.json
```

Short80×20×12 tests (dt10s,120s, width1000km) passed CPU/CUDA serial rest, terrain response, independent curl/masks, exact zero-height equivalence and capability rejection. CPU four-rank suite passed; CUDA four-rank rest/terrain checks and history comparison passed, with its remaining flat/rejection checks in progress. Paths:

- CPU serial: `build_cpu/rll_mountain_tests/run-q9fxbei1`.
- CPU four ranks: `build_cpu/rll_mountain_mpi_tests/run-vxcqh_z5`.
- CUDA serial: `build/rll_mountain_tests/run-vwjvgr2r`.
- CUDA four ranks: `build/rll_mountain_mpi_tests/run-37snrmr1`.

Cross-backend max wind error6.99e-14m/s; one/four-rank CPU5.56e-17m/s and CUDA1.12e-16m/s. Exact masks/height/time checks pass. Reports saved in `output/topo_rll/{cpu_cuda,cpu_mpi,cuda_mpi}_comparison.json`. Tolerances fixed before comparison: absolute1e-10m/s for winds and1e-15s^-1 for vorticity, including adapted scratch.

Full original suites are running with unchanged reference tolerances (`build{,_cpu}/rll-terrain-ctest.log`). CPU original Cartesian bubble and mountain already passed. Next step: finish those suites and record complete pass/fail counts, then make a narrowly scoped local commit excluding user formatting. No long-duration stability, vertical/horizontal refinement, stratified mountain-wave benchmark, or CVVM terrain-equivalence claim is made. Those would require additional experiments, not a relaxed test threshold.
