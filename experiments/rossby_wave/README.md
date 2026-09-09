# Williamson test request: scope audit and handoff

Update: the user clarified that the desired quick experiment is rotating zonal flow over a north-shifted mountain, not exact Williamson reproduction. The minimal configuration/wiring and its validation status are documented in `../topography/rotation-smoke.md`. The audit below records the earlier scope assessment; its statement that rotation is not wired is superseded by that update.

Requested after RLL terrain implementation commit `363e262985cc47565afc7570e3d4b75a93b1711e`.

The user requested replacing the user-facing topography test with a Rossby-wave test like Williamson's. An asynchronous clarification asks whether they mean test6 (Rossby–Haurwitz wave, no mountain) or test5 (zonal flow over an isolated mountain). Do not silently substitute one for the other.

Primary reference: Williamson et al., *A standard test set for numerical approximations to the shallow water equations in spherical geometry*, JCP102 (1992),211–224, DOI10.1016/S0021-9991(05)80016-6. The authors' 1991 technical report is available at https://digital.library.unt.edu/ark:/67531/metadc1069483/m2/1/high_res_d/5232139.pdf (sections3.5/3.6). Check the published case parameters before implementing, especially if selecting test5.

The report specifies a wavenumber4 Rossby–Haurwitz initial condition for test6, a balanced shallow-water height field, and comparison at days0,7,14 against a high-resolution shallow-water solution. The shape-preserving analytic propagation applies to the nondivergent barotropic vorticity equation, not to the full shallow-water equations. Test5 instead contains an isolated mountain and comparisons at days5,10,15. A mountain-free wave and mountain-induced Rossby response are distinct tests.

Current implementation constraints:

- `src/core/GridSpecification.cpp` and `src/core/geometry/RegularLatLonGeometry.cpp` reject domains/halos reaching the poles. Current RLL is a channel, not a global sphere.
- `src/core/RegularLatLonModelConfiguration.hpp` admits only the flat Jung and neutral mountain modes, with no Coriolis tendency or free-surface/height evolution.
- `src/core/InitializerJung2019.cpp` initializes unit density, inert theta, and the Jung jet, not Williamson's balanced height/winds.
- `src/dynamics/solvers/WindSolverRegularLatLon.cpp` retains the rigid-lid diagnostic and channel circulation. Its potential walls are constant in longitude. Restricting an unmodified Rossby–Haurwitz wave to latitude±45° conflicts with its nonzero normal wind there.
- RLL planetary-vorticity operators already exist in `RegularLatLonVorticityTendency.cpp`, but enabling them requires correct native-Z f initialization and exactly-once planetary contributions through the full model. The Cartesian initialization formula must not be reused with angular dx/dy.

Thus simply replacing the initializer cannot establish Williamson reproduction. A clearly labeled barotropic/channel adaptation would require the user's agreement on modified equations and boundaries; exact global shallow-water reproduction requires broader pole and free-surface support. Do not remove guards or add an unrelated shallow-water subsystem as a hidden configuration change.

The validated mountain fixture is preserved in `experiments/topography/configs/mountain.json`; `tests/scripts/check_rll_mountain.py` now uses this stable fixture rather than the user-facing file. The user independently extended `rundata/input_configs/topography.json` to36000s. Preserve that edit pending the test choice.

Storage check after the user's longer run: experiments6.8GiB plus output5.6GiB, above the earlier10GB total cap. Preserve all results; do not launch further history-producing experiments until storage scope/budget is resolved. Final36000s snapshot was read-only checked: all six wind/vorticity fields finite. This is not a full longer-run validation.

Next steps: finish the original CPU/GPU regression reporting, obtain the intended Williamson case and adaptation/global scope, resolve output budget, then implement the selected experiment through the shared architecture with a new reproducible configuration and independent scientific tests.
