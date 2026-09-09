# Minimal rotating mountain smoke setup

User requested only enabling Coriolis and moving the mountain north, without a new solver or a full Williamson benchmark. Starting implementation revision: `7ba0a9e` (earlier terrain CPU93/93 and GPU175/175 passed; those counts do not validate these new edits).

`rundata/input_configs/topography.json`: mountain180°E,10°N; unchanged height2000m and width500km. This latitude remains inside the existing eastward jet; the jet is zero at30°N. Rotation7.292e-5s^-1, with Takacs/AB2 Coriolis terms for xi, eta and zeta. The unrelated Jung perturbation is disabled. dt10s, duration600s, two snapshots in a fresh `output/topo_rll_rotating_smoke` directory (under70MiB). This is a startup test, not enough time to demonstrate developed Rossby waves. For eventual multi-day analysis, inspect meridional wind and horizontal relative-vorticity anomalies rather than only vertical velocity.

Minimal wiring: existing RegularLatLonVorticityTendency::Planetary is called by the three Coriolis dispatch methods and admitted by the shared numerical factory. Its separate planetary transport/deformation contributions are not added to the existing relative-vorticity transport a second time. Initializer supplies physical f=2*OMEGA*sin(phi_Z), including analytic halos; f_2d metadata uses Z staggering only for RLL. Mountain latitude is optional and validated so its three-width support stays inside the channel. Nonrotating mountain and flat Jung paths retain their defaults; nonzero mountain rotation requires all three Coriolis terms. Solver arithmetic, fixed iteration counts and CUDA graphs are unchanged.

```sh
export VVM_ROOT=/home/mog/CVVMex
export LD_LIBRARY_PATH=/home/mog/libs_VVMex/lib:$LD_LIBRARY_PATH
export OMP_NUM_THREADS=4 VVM_TEST_GPUS=0
cmake --build build -j32
mpiexec --bind-to none -n1 tests/scripts/one_gpu_per_rank.sh build/vvm rundata/input_configs/topography.json
```

Validation: clean CUDA build passed (`build/rll-rotation-build.log`); final metadata adjustment and all existing test executables rebuilt successfully (`build/rll-rotation-test-build.log`). Three existing no-history CUDA checks passed: test_regular_latlon_geometry, test_model_configuration_validation, test_regular_latlon_top_transport, including its independent planetary-transport formulas (`build/rll-rotation-operator-tests.log`). JSON switches/latitude checked; f at mountain center2.5324850e-5s^-1, background wind16.21324m/s. Solver code is unchanged. No complete CPU/MPI suite was rerun for this quick change, and these component results do not establish a coupled rotating-mountain run or Rossby-wave development.

No new simulation run yet: existing experiments+output total about12.4GiB, above the earlier10GB cap. User was asked whether to allow at most70MiB extra for this smoke test; no output is deleted or overwritten. User formatting in Initializer.cpp and .clang-format remains untouched. Next step: run the short configuration only if storage allowance is supplied, then verify finite final winds/vorticity, native-Z f, shifted height and exactly600s final time. Longer Rossby-wave development and full CPU/MPI regression are outside this quick change.
