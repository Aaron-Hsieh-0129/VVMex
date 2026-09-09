# RLL Section 4.2 handoff

## Asset relocation (2026-09-09)

All experiment configuration, documentation, tools and experiment-specific tests now live under `experiments/jung2019`; see [README](README.md). The user's initial moves were retained and old import/configuration/documentation paths repaired. `tests/CMakeLists.txt` includes the experiment's `tests/register.cmake`, preserving `test_jung2019_shared_model` and its backend/resource settings. New integration evidence goes to `testing_output/build` or `testing_output/build_cpu` here. Historical run provenance and old build output remain unchanged. Shared numerical runtime code and generic RLL operator tests were not moved or modified.

Relocation validation: both CMake presets regenerated; GPU integration1/1 passed (81.07s), CPU integration1/1 passed (110.65s), analysis4/4 passed (30.999s), launcher CASE2 generation/root/template/output checks passed. Logs: `build/jung-relocation-ctest.log`, `build_cpu/jung-relocation-ctest.log`, `build/jung-relocation-analysis-tests.log`. Full model CTest suites were not rerun for this path-only change; their earlier results below remain historical. Small relocated assets are explicitly tracked despite the user's experiment ignore rule; generated data stay ignored. No numerical configuration values changed and no scientific run was repeated.

## Scientific handoff

Updated 2026-09-09, Asia/Taipei. Section4.2 is complete as a stable, self-convergent paper-wall RLL realization under the user-approved own-reference criterion. Both cases complete all five resolutions. Final CPU92/92 and GPU/MPI174/174 CTest pass. Read [results](rll-section42-results.md) for quantitative evidence, commands and limitations, and [validation protocol](rll-section42-validation.md) for the pre-set assessment rule. Exact archive identity and a pure spatial order are not claimed.

## Checkout and authority

- Repository `/home/mog/CVVMex`, branch `feat/generalized-coordinate`. Initial inspected HEAD `16e0355a8db89c817c1d4594e351514ddc7aceb4`; implementation `beef1cf4f94245a6b89e04e962bd59f0803b62ad`; numerically equivalent RLL performance change `0e51d91056b6921d77dfcad1b16dcdc178a19062`.
- Preserve the user's pre-existing `.gitignore` addition of `refs`. All reference data, failed runs and existing results retained. Direct edits/builds/local commits authorized; no push or merge. All governing documents read; old propose-only workflow superseded by user. No applicable AGENTS.md; no subagents used.
- Limits:224 cores/ranks, GPUs0–7, experiment output below10GB. Current experiments approximately7.2GB. Do not launch a duplicate hierarchy without checking its projected storage.

## Completed implementation and science

Shared RLL transport/deformation, numerical factory and tendency wiring, narrowly guarded production model path, both analytic initializations, channel/circulation wind recovery through existing fixed-iteration solvers, prepared/captured/replayed CUDA diagnostics, geographic output and exact compact top-Z snapshots. Cartesian arithmetic, field meanings, density normalization, buoyancy range and unsupported-configuration rejection retained. See implementation commit for source diff.

Both cases ran400×100,800×200,1600×400,3200×800,6400×1600 through168h/120h respectively, dt300/150/75/37.5/18.75s. Horizontal200, initial1000, vertical10 fixed iterations. All saved native vorticity finite; relative L2 decreases at every saved24-hour time against each case's own6400 reference.

| Final relative L2 | 400 | 800 | 1600 | 3200 |
| --- | ---: | ---: | ---: | ---: |
| CASE 1,168h | .19351646 | .11810059 | .06651746 | .02941360 |
| CASE 2,120h | .29255996 | .17116543 | .09106368 | .03790632 |

Results/plots: `experiments/jung2019/convergence_case{1,2}`. CASE1's1600 member is `case1_1600x400_dt75`; other hierarchy members use `l2_case{case}_n{nx}`. Every launcher run has config, log and provenance. Full-field long-run invariants and short CPU/GPU1-/4-rank parity pass. Compact reference histories do not record winds/energy independently.

At1600, CASE1 dt150 fails even with800 fixed iterations; halved dt75 succeeds. Further dt37.5 sensitivity L2=.01305697 at168h. At400/dt600, horizontal200 versus1000 difference=.04453316. Do not claim negligible timestep/solver error or a pure spatial order.

User accepted own-reference L2 in place of identical nonlinear archive trajectories. Keep paper impermeable/free-slip walls: original CVVM archive uniform-copy halos differ materially. Archive CASE1 L2=.01487154 at96h,.56188749 at168h; no separate CASE2 archive identified. This disagreement remains explicit, not dismissed solely as nonlinearity.

## Tests and current activity

- Final full CPU92/92 passed,731.46s: `build/rll-cpu-qualified-ctest-console.log`.
- Final full GPU174/174 passed,1193.09s; GPU0–3, serialized: `build/rll-qualified-ctest-console.log`. Earlier full GPU173/173 andCPU91/91 passed before final additions. No tests remain active.
- Production rest/jet/coupled, ten configuration rejections, compact exactness, independent top transport and1-/4-rank backend tests pass. CUDA graphs are enabled, not bypassed.
- Four independent analysis tests pass, including rejection of nonconvergent data: `build/rll-analysis-tests.log`. Run `python -m unittest discover -s experiments/jung2019/tests -p test_convergence.py -v`.
- No scientific model runs remain active. Original user bubble output was preserved at `build/pre-rll-testing_output_2dbubble`.

## Environment and operational pitfalls

GPU preset `blaze`: NVHPC24.9/CUDA12.6/HPC-X2.20, FP64, Kokkos4.7.2, NCCL, eightH200. CPU preset `blaze-cpu`: OpenMP, `/raid/mog/libs_VVM_cpu` plus`tpl`; actual compiler wrapper remainsHPC-X. Exact configure/run/test commands are in the results document and per-run provenance. CPU CTest explicitly uses64threads, overriding shell OMP_NUM_THREADS.

Root CMake disables header dependency scanning: header changes require clean rebuilds. Sandbox startup fails with bwrap loopback error; reviewed escalated execution works. Local edits use `command apply_patch` there. One-/four-rank6400 windowed benchmarks favor one GPU per full run; a previous warmup timing option was invalid because existing TimingManager::set_step is never called. Preserve benchmarks but do not use their empty timing counters.

## Handoff and subsequent work

No required Section4.2 implementation or planned validation remains. Analysis scripts/tests/documentation are delivered in a separate logical local commit; use `git log -3 --oneline` for implementation, performance and analysis commits. Preserve `.gitignore` and untracked experimental data; large outputs are not staged. No push or merge. Further temporal/solver sensitivity, energy diagnostics or archive-wall matching would be distinct follow-up validation, not silently claimed here.

Section4.3 remains separate and unimplemented: balanced sheared jet/thermodynamics,0.5K perturbation/taper, shifted planetary terms exactly once, full3D divergent integration, vertical/solver qualification and its own reference/refinement study. See the results document for the detailed checklist; moisture/terrain remain unsupported.
