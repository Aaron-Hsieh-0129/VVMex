# Jung 2019 Section 4.2: RLL results

## Assessment

Both paper-wall RLL cases have completed the five-resolution study through their required comparison times. All saved native vorticity fields are finite, and relative L2 decreases strictly with refinement at **every saved time** for both cases. This establishes stable joint space/time self-convergence for this shared-model implementation under the protocol in [rll-section42-validation.md](rll-section42-validation.md).

This completes the requested Section 4.2 reproduction as a stable, self-convergent RLL realization with the paper's walls and the user-approved own-reference assessment. It is not a claim of identical CVVM trajectories, a pure spatial order, or an exact continuum solution. The supplied archive uses different lateral halo handling. Final original GPU/MPI and CPU CTest qualification both pass.

## Scientific configuration

Source: local Jung 2019 PDF, Section 4.2, equations (47)–(50), Figures 6–8; `refs/RUN_BAR/RUN/a.setup`, `refs/CODE_BAR`, and `refs/DATA_BAR/Y_TST4`. The active setup script selects cubed sphere; its commented RLL settings and original source were inspected, not executed.

| Setting | Value |
| --- | --- |
| Radius / domain | 6371220 m; longitude 0–360° periodic; latitude −45–45° |
| Latitude walls | Impermeable/free-slip, as specified in the paper |
| Jet | Maximum 80 m/s; equation (47), zero outside its support |
| CASE 1 jet / bump latitude | −π/8 to π/8; π/24 |
| CASE 2 jet / bump latitude | −π/16 to π/8; π/16 |
| Vorticity bump | 1e−6 cos(φ) exp[−9λ²−900(φ−φ₂)²] s⁻¹, wrapped λ in [−π,π) |
| Dynamics | Shared Takacs transport, stretching/twisting and AB2; no Coriolis or explicit diffusion |
| Fixed solvers | Horizontal 200; initial horizontal 1000; vertical 10; shift 2/(Δλ)² |
| Vertical embedding | Four uniform physical levels, Δz=1000 m; unit reference density; inert 300 K θ |
| End / history times | CASE 1: 168 h; CASE 2: 120 h; histories every 24 h, including initialization |

The inactive vertical embedding is an implementation choice, not an additional paper constant. The model retains its original buoyancy declaration/range, density normalization, field meanings, native staggering, and shared diagnostic architecture. Total streamfunction wall difference retains the jet; a separate covariant southern circulation degree is preserved. There is no Cartesian mean-wind subtraction.

## Five-resolution L2 results

Reference: each case's own 6400×1600 solution, approximately 6.25 km at the equator, dt=18.75 s. Native Z points coincide exactly across these grids; sample reference indices `r-1::r` in both horizontal dimensions. Compute the full-band cosine-latitude weighted relative L2, with no fitted shift, smoothing, or latitude crop. The constant angular-area and radius factors cancel.

| Grid | Approximate spacing | dt (s) | CASE 1 L2 at 168 h | CASE 2 L2 at 120 h |
| --- | --- | ---: | ---: | ---: |
| 400×100 | 100 km | 300 | 0.19351646 | 0.29255996 |
| 800×200 | 50 km | 150 | 0.11810059 | 0.17116543 |
| 1600×400 | 25 km | 75 | 0.06651746 | 0.09106368 |
| 3200×800 | 12.5 km | 37.5 | 0.02941360 | 0.03790632 |
| 6400×1600 | 6.25 km | 18.75 | Reference | Reference |

Final adjacent log₂ error ratios: CASE 1, 0.7124 / 0.8282 / 1.1772; CASE 2, 0.7733 / 0.9104 / 1.2644. These are observed joint-refinement rates against a finite reference, not formal spatial orders. There are no monotonicity violations at the saved 24-hour times. They do not prove monotonicity between saved times.

Results are under `experiments/jung2019/convergence_case{1,2}/`: `convergence.json`, `l2_history.png`, `final_vorticity.png`, and a smaller preview. The native-field plots were visually inspected: major vortex positions agree progressively across resolution, while filament thickness and small roll-up structure become better resolved. Plot latitude limits are ±30° for readability; **the quantitative norm always uses the entire ±45° band**. The paper describes eastward wave-packet growth and improved resolution of the instability; the results exhibit that behavior. No digitized paper-figure error is claimed.

CASE 1 input directories: `l2_case1_n400`, `l2_case1_n800`, `case1_1600x400_dt75`, `l2_case1_n3200`, `l2_case1_n6400`. CASE 2 uses `l2_case2_n{400,800,1600,3200,6400}`. All paths are relative to `experiments/jung2019`.

## Stability, circulation, and uncertainty

- The original-comment dt600 at400 schedule, scaled linearly with spacing, makes CASE 1 at1600/dt150 nonfinite between144 and168 h. Increasing the fixed horizontal count to800 does not cure it. Both failed histories are preserved. Halving dt to75 with200 iterations is finite through168 h; this selects the final constant-Courant schedule above. Refinement timesteps are measured stability choices, not attributed to undocumented paper settings.
- At1600×400, halving CASE 1 dt75 to37.5 changes final vorticity by L2=0.01305697, compared with0.06651746 for dt75 against the6400 spatial/time reference. This is useful temporal sensitivity evidence, not a temporal-error bound. Report: `case1_1600x400_dt75/difference_from_l2_case1_n1600_dt37p5.json` and PNG.
- At400×100/dt600, changing horizontal200 to1000 iterations gives final CASE 1 L2 difference0.04453316. Fixed-iteration error is not proven negligible and remains part of the reproduced numerical method. No convergence-driven solver was introduced.
- The full-field CASE 1 1600/dt75 history retains exactly zero north-wall normal wind, horizontal vorticity and vertical wind. The maximum saved southern covariant-circulation-mean change is1.283e−9 m²/s (roughly3e−16 m/s when divided by its metric factor). Compact high-resolution histories save only an exact top-Z snapshot, so they cannot independently establish long-run wind or energy diagnostics there.
- No separate finer-than6.25km reference, highest-resolution timestep-halving run, enstrophy/energy conservation qualification, or pure spatial-order study was performed. Those are not inferred from the L2 pass.

## CVVM archive comparison

`refs/DATA_BAR/Y_TST4` contains CASE 1 snapshots at96/120/144/168 h. Lower `z3dz` is streamfunction by design in `CODE_BAR/ldoutput_nc.F`; only its upper slice is relative vorticity. Native VVMex Z is averaged to T for this archive comparison. Longitude columns are sorted by their recorded geographic coordinates (the archive starts near225°); no best-fit phase shift is used.

At400×100/dt600, vorticity relative L2 against the archive is0.01487154 at96 h and0.56188749 at168 h; final zonal/meridional wind L2 is0.26071668/0.52896477. See `case1_400x100/comparison*.json` and PNG. These are descriptive disagreements, not a tolerance pass.

The original `bound_extra.F` uniformly copies latitude halos, including streamfunction. The archive's boundary-adjacent T-row normal wind reaches about22 m/s at168 h and boundary-row streamfunction varies by7.8e7–9.4e7 m²/s. This differs materially from the paper's stated impermeable/free-slip channel, which this implementation retains. Boundary audit: `experiments/jung2019/reference_boundary_audit.json`. Nonlinearity alone is not used to dismiss this configuration difference. No distinct CASE 2 archive was identified; own-reference CASE 2 results are available.

## Regression qualification

- Final CPU full CTest: **92/92 passed**,731.46 s, `build/rll-cpu-qualified-ctest-console.log`.
- Final GPU full CTest: **174/174 passed**,1193.09 s, `build/rll-qualified-ctest-console.log`. Earlier full modified-source suites passed173/173 GPU and91/91 CPU, before final integration additions.
- Shared-model rest/jet/coupled tests and ten unsupported-configuration rejection cases pass on CPU and GPU. Rest stays exactly zero; unperturbed jet maximum wind drift2.85e−14 m/s, zeta unchanged exactly. Predetermined bounds remain1e−11 m/s and1e−15 s⁻¹.
- Short coupled one-/four-rank CPU/GPU comparisons pass1e−10 m/s wind and1e−15 s⁻¹ vorticity parity bounds. Report: `coupled_smoke/backend_comparison.json`. These are short-run backend checks, not full-duration multi-backend comparisons.
- Production diagnostic CUDA graphs are prepared, captured for both AB2 backing allocations, and replayed. Long GPU runs use this path; no graph-disable fallback was introduced.
- RLL top transport has an independent host evaluation of the original Fortran flux formula, including varying density and velocity signs. New compact output is checked for exact equality to prognostic top zeta.
- Analysis tests:4/4 passed in33.618s; analytic relative norms on nonuniform native points, missing-final rejection, nonfinite-reference rejection, and failure of the monotonicity criterion for deliberately nonconvergent data. `build/rll-analysis-tests.log`.
- A transient earlier GPU integration failure was caused by launching during a rebuild that temporarily removed a shared library; rerun after build completion passed. A performance trial's warmup counters were unusable because existing `TimingManager::set_step` is not called; interval/reset timing was used instead. Neither is concealed as a scientific success.

## Revisions, environment, and reproducibility

Branch `feat/generalized-coordinate`; initial inspected HEAD `16e0355a8db89c817c1d4594e351514ddc7aceb4` (not reset to the older synchronized revision). Shared implementation commit `beef1cf4f94245a6b89e04e962bd59f0803b62ad`. RLL-only unused Cartesian scratch removal: `0e51d91056b6921d77dfcad1b16dcdc178a19062`. This optimization produced bit-identical short-run saved fields before final backend qualification. The6400/3200 references ran the former binary; later lower-grid runs used the numerically equivalent optimization. Each launcher run records actual HEAD, dirty diff hash, executable/configuration/cache hashes, environment, command and exit code in `provenance.json`; do not relabel old binary provenance as a later commit.

Release FP64; NVHPC24.9, CUDA12.6, HPC-X2.20 MPI, GCC at `/home/mog/gcc11`; GPU Kokkos4.7.2 at `/home/mog/libs_GPUVVM`, ADIOS2 at `/home/mog/libs_VVMex`, NCCL; eight H200 GPUs. CPU Kokkos/ADIOS2 prefix `/raid/mog/libs_VVM_cpu`, I/O dependencies in its `tpl` directory. Actual CPU cache uses the HPC-X MPI wrapper; the generic CPU installation guide's separate-MPI example is not its exact environment. Existing presets: `blaze` and `blaze-cpu`; caches in `build` and `build_cpu`.

```sh
# Run separately, with no tests/launches overlapping these builds.
cmake --preset blaze -DBUILD_TESTS=ON
cmake --build build --clean-first -j64
cmake --preset blaze-cpu -DBUILD_TESTS=ON
cmake --build build_cpu --clean-first -j64

export VVM_ROOT=/home/mog/CVVMex
export LD_LIBRARY_PATH=/home/mog/libs_VVMex/lib:$LD_LIBRARY_PATH
export OMP_NUM_THREADS=4 VVM_TEST_GPUS=0,1,2,3
ctest --test-dir build --output-on-failure -j1
# In a CPU environment prepend the CPU libraries instead:
LD_LIBRARY_PATH=/raid/mog/libs_VVM_cpu/lib:/raid/mog/libs_VVM_cpu/tpl/lib:$LD_LIBRARY_PATH \
  ctest --test-dir build_cpu --output-on-failure -j1
python -m unittest discover -s tools/jung2019 -p test_convergence.py -v
```

Header dependency scanning is disabled in the existing root CMake, so header changes require a clean rebuild. CPU CTest declares64 OpenMP threads; shell `OMP_NUM_THREADS` does not override that test property. Tests were serialized, within224 cores. One GPU/rank per full experiment was faster than four-rank decomposition in measured windowed timing; largest run used approximately32GB GPU memory.

To regenerate the numerical hierarchy, choose fresh names and enough remaining storage; **do not run a duplicate full hierarchy alongside the retained7.2GB output under the10GB limit**. This example configures/launches one selected member:

```sh
python tools/jung2019/experiment.py configure NEW_CASE1_6400 \
  --case 1 --nx 6400 --dt 18.75 --iterations 200 --initial-iterations 1000 --compact
python tools/jung2019/experiment.py launch experiments/jung2019/NEW_CASE1_6400 \
  --ranks 1 --threads 4 --gpus 4
python tools/jung2019/experiment.py analyze experiments/jung2019/NEW_CASE1_6400

python tools/jung2019/convergence.py \
  --reference experiments/jung2019/l2_case1_n6400 \
  --runs experiments/jung2019/l2_case1_n400 experiments/jung2019/l2_case1_n800 \
    experiments/jung2019/case1_1600x400_dt75 experiments/jung2019/l2_case1_n3200 \
  --output experiments/jung2019/convergence_case1
python tools/jung2019/convergence.py \
  --reference experiments/jung2019/l2_case2_n6400 \
  --runs experiments/jung2019/l2_case2_n400 experiments/jung2019/l2_case2_n800 \
    experiments/jung2019/l2_case2_n1600 experiments/jung2019/l2_case2_n3200 \
  --output experiments/jung2019/convergence_case2
```

Generator defaults retain the source-comment dt600 at400 schedule; explicitly pass the validated dt values in the table for refinement. CASE 2 is generated with `--case 2` and automatically ends at120 h. All actual configurations, histories, logs and diagnostic hashes remain under the project. Final experiment storage is7,202,515,993 bytes (below10GB). Original user work, reference data, failed runs and prior results were preserved; no push or merge.

## Section 4.3 remaining work

Section 4.3 was not enabled. It requires a separately verified balanced vertically sheared jet and thermodynamic state; the paper's0.5K potential-temperature perturbation centered atπ/28, constant below10km and tapered to zero by15km; the shifted latitude-dependent Coriolis `2Ω sin(φ+π/4)`; planetary contributions exactly once in full3D tendencies/diagnostics; compatible divergent wind recovery, vertical grid/boundaries and fixed-iteration solver qualification. Verify its vertical-domain, timestep, resolution and reference-output settings directly from its source configurations before implementing. Add thermal-wind/rest checks, coupled3D CPU/GPU/MPI/graph checks and a separate scientific reference/refinement study. Moisture, terrain and other unsupported RLL configurations remain rejected.
