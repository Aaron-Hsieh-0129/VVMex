# Section 4.2 validation protocol

## Scientific target and comparison

The user confirmed on 2026-09-09 that exact agreement with the supplied nonlinear CVVM trajectory is not required, and requested Jung's high-resolution-reference L2 study. Retain the paper's impermeable/free-slip latitude walls; do not silently change them to the archive's uniform-copy halo policy. Archive comparisons remain descriptive and their boundary discrepancy remains documented.

Use separate CASE 1 and CASE 2 reference solutions at 6400×1600 (approximately 6.25 km at the equator). Compare 400×100, 800×200, 1600×400 and 3200×800 against the corresponding case's reference at identical physical output times, through 168 h and 120 h respectively. This follows the paper's use of each grid system's own highest-resolution solution rather than an assumed exact analytic nonlinear solution.

Native relative vertical vorticity is point-valued at Z. The grids are nested with coincident positive-face Z points. Sample the fine solution at those exact locations: for refinement factor r, indices r-1, 2r-1, etc. No fitted spatial shift, pattern matching, latitude cropping, amplitude normalization, or smoothing is applied. Compute

`L2 = sqrt(sum(cos(phi_Z) * (zeta_coarse - zeta_reference)^2) / sum(cos(phi_Z) * zeta_reference^2))`.

The common radius and constant angular cell factors cancel. Wall vorticity is zero; half weighting the boundary node therefore would not change this norm. Plot all coarse-grid error histories against the same 6400×1600 reference. Pairwise adjacent-grid differences are supporting diagnostics, not substitutes for that reference study.

## Qualification before interpretation

- Require finite saved vorticity and the actual configured final output on every admitted run. A successful process exit alone does not pass this check.
- Require original Cartesian regressions, backend/MPI checks, production CUDA graph replay, stationary rest/jet tests and configuration rejection tests. Predetermined short-run bounds remain 1e-11 m/s and 1e-15 s^-1 for stationary invariants; backend parity uses 1e-10 m/s and 1e-15 s^-1. These are not nonlinear reproduction tolerances.
- Establish timestep stability separately from solver sensitivity. The provisional constant-Courant schedule dt=600*(400/nx) failed for CASE 1 at nx1600 between144h and168h. Preserve that failed result. Test halved dt while retaining200 iterations, and separately800 fixed iterations at the original dt. Never replace fixed counts by convergence-based stopping.
- The final refinement timestep schedule must be explicitly recorded after these stability diagnostics, not attributed to the paper: the supplied sources only establish dt600 at400×100 and a separate commented8000×2000 example withdt20. The user authorizes reducing dt at high resolution.
- Compare a representative run at a smaller dt and report its change separately. Since dt changes with resolution, the resulting curves measure combined spatial/time-discretization convergence; do not claim a pure spatial convergence order without a separate temporal-error study.

## Assessment rule fixed before the reference runs

The stability diagnostics selected dt=300, 150, 75, 37.5 and18.75s atnx400,800,1600,3200,6400, with200 fixed horizontal iterations and1000 initial iterations. CASE1 at1600×400/dt75 remains finite through168h, whereas dt150 fails even with800 iterations. A further dt37.5 run at1600×400 is retained as a temporal sensitivity check. No convergence-driven solver is used.

The paper supplies qualitative convergence evidence, not a numerical acceptance threshold for this implementation. Do not invent a matching-percent tolerance from the computed results. Report final and time-dependent L2 values, observed adjacent-resolution rates, whether errors decrease monotonically with refinement at each saved time, and all violations. Demonstrated self-convergence requires a finite highest-resolution reference and decreasing final L2 under refinement for both cases, supported by the full error histories. Failure of that criterion remains a scientific failure to resolve, not a reason to weaken it.

Even a successful self-convergence study does not establish identical solutions for different lateral boundary policies. Separate the claims: shared-model execution and regressions; stable/self-convergent paper-wall experiments; and agreement or disagreement with the supplied CVVM archive.

## Output budget

High-resolution histories may select `rll_zeta_top`, an exact native-Z snapshot copied from the existing prognostic zeta field after the shared wind diagnostic. This uses ordinary State and HDF5 history output, not a separate dynamical subsystem. Full low-/intermediate-resolution outputs retain winds, horizontal vorticity and background fields for circulation/vertical-homogeneity diagnostics. Compact histories do not claim to record those omitted diagnostics.

The experiment launcher checks resource limits, records source/build/configuration hashes and run commands, refuses pre-existing output paths, and estimates storage before launch. Keep total experiment data below10GB; preserve failed and prior runs.
