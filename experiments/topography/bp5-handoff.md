# RLL BP5 and shifted-jet continuation

The user requested a tunable jet aligned with the mountain, BP5 as the experiment default with asynchronous packed output, and possibly higher latitude. The preset `rundata/input_configs/topography.json` now places both centers at20°N (mountain longitude180°E), within the existing±45° channel. Both latitude keys remain independent. The user's duration360000s and output interval6000s are preserved; peak wind80m/s and mountain height2000m/width500km remain unchanged. This is not an exact Williamson experiment.

The new jet-center option shifts only the wind profile. Background relative vorticity and channel streamfunction are recomputed consistently at physical spherical metric locations. An omitted option now bypasses degree/radian conversion entirely, preserving the original jet arithmetic for both Jung cases. A jet crossing the channel walls is rejected.

BP5 is selected explicitly in this experiment, not globally substituted for other configurations. Requested options: TwoLevelShm, one subfile, stats0, async_write=true, buffer_mode=pack, overwrite=true. Fresh output directory `output/topo_rll_rotating_bp5` protects earlier results; subsequent runs into this same BP5 dataset WILL overwrite it as requested. RLL BP5 axes now contain centered longitude/latitude in degrees, with proper units and geometry/radius metadata. Cartesian coordinate arithmetic and metadata are unchanged. Only the mountain RLL preflight admits BP5; flat Jung reproduction guards and restart restrictions remain unchanged.

Validation: clean GPU build of all targets passed (`build/rll-bp5-build.log`). Six existing no-history tests passed: GrADS axes, RLL geometry, model configuration validation, BP5 option parsing, BP5 field schema, and RLL top transport (`build/rll-bp5-component-tests.log`). The new optional-BP5-tier CTest `test_rll_bp5_history` is registered and runs120s on80×40×8, comparing all output fields bitwise against HDF5. It independently checks longitude/latitude axes, units, native-Z f, shifted background jet, terrain heights, finite fields and final time. It uses the exact requested packed/asynchronous options and unique output directories. Python syntax/API inspection passed; the integration test has not run yet. CPU rebuild, MPI BP5 parity and full regression reruns were not performed for these changes.

```sh
export VVM_ROOT=/home/mog/CVVMex
export LD_LIBRARY_PATH=/home/mog/libs_VVMex/lib:$LD_LIBRARY_PATH
export OMP_NUM_THREADS=4 VVM_TEST_GPUS=0
cmake --build build -j32
ctest --test-dir build -R '^test_rll_bp5_history$' --output-on-failure
```

Storage is a live blocker for history-producing tests: preserved experiments6.8GiB plus output8.8GiB exceeds the earlier10GB total cap. Asked user to allow up to100MiB extra for validation; no answer yet. Nothing deleted, and no new simulation history generated. The current full-duration preset would produce substantially more data than the small validation test; do not launch it under that small allowance.

Read-only inspection of the user's existing rotating history at360000s found finite u/v/w/zeta (maxima107.70m/s,50.61m/s,2.54m/s,9.88e-4s^-1). An in-memory plot of top-zeta change and top northward wind shows downstream wave-like undulations, with sharp small-scale features. This does not establish Rossby-wave identification or convergence. No Karman-vortex experiment was introduced: that is a distinct wake problem, and no evidence yet justifies substituting it. No plot/history was saved by this inspection.

Next: obtain storage allowance and run the BP5/HDF5 comparison before claiming RLL BP5 runtime validation. The new configuration is prepared and compiled but not runtime-qualified. Preserve unrelated Initializer.cpp formatting, .clang-format and earlier Cartesian additions. Source/config/test changes are recorded as a local preparation milestone; nothing pushed.
