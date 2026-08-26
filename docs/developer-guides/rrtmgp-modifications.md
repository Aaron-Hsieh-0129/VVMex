# RRTMGP single-precision modifications

VVMex uses the Kokkos-based **RRTMGP radiation scheme** from E3SM EAMxx, vendored under
`src/physics/rrtmgp/external/`. Two kernels there carry constants and expression groupings
that are only valid in double precision. They are unreachable in a default (FP64) build, but
a single-precision build hits both: the first aborts the run, the second corrupts the
shortwave silently.

Both changes are selected at compile time on `sizeof(RealT)`. **A double-precision build
compiles to byte-identical device code**, verified by comparing `cuobjdump -sass` output for
`eamxx_rrtmgp_interface.cpp` before and after. Nothing about the FP64 reference path moves.

## Modification comparison

| Kernel | VVMex modification | Original RRTMGP / RTE behavior |
| ------ | ------------------ | ------------------------------ |
| `gas_optical_depths_minor` | Minor-gas density scaling groups the O(1) ratio first in single precision. | `scaling * col_gas * vmr_fact * dry_fact`, evaluated left to right. |
| `sw_two_stream` | Two-stream eigenvalue floor `k_min` is `1e-5` in single precision. | `k_min` hard-coded to `1.e-12`. |

## `gas_optical_depths_minor` — optical depth overflows to `inf`

In `mo_gas_optics_kernels.h`, a minor gas that scales with the density of a second gas is
scaled as

```cpp
scaling = scaling * mycol_gas_imnr * vmr_fact * dry_fact;
```

C++ evaluates this left to right. `scaling` is already a column amount of order 1e23, and
`mycol_gas_imnr` is another one, so the first product is around **1e46** before `vmr_fact`
(≈ 3e-24) brings it back to a physical value. That intermediate is unremarkable in double
and past `FLT_MAX` in single precision.

The consequence is not subtle. On the first radiation call roughly 60 M of 73 M longwave
`tau` values become non-finite, the temperature field goes NaN, and the run aborts inside
`p3_main_part1` on the `is_nan_t_atm` kernel check — far from the actual cause.

VVMex groups the O(1) density ratio first, which is mathematically identical and stays in
range. The double path keeps the original association.

## `sw_two_stream` — thin layers reflect rounding noise

In `mo_rte_solver_kernels.h`, the two-stream eigenvalue is floored to avoid a division by
zero for conservative scattering:

```cpp
RealT k = sqrt(Kokkos::fmax((gamma1 - gamma2) * (gamma1 + gamma2), 1.e-12));
```

`Rdir` and `Tdir` are then formed from differences that cancel two O(1) terms down to a
remainder of order `k * mu0`. With `k_min = 1e-12` that remainder is about 5e-7 of the terms
being subtracted: double still has around nine significant digits left, single precision has
none. Optically thin, near-conservative layers then reflect rounding noise instead of light.

The upper atmosphere of **every** column is made of exactly such layers — thin and
Rayleigh-dominated, so `w0` is at the conservative limit — which is where the damage is done.
Measured on a 256x256 TaiwanVVM case with fixed `mu0 = 0.5` and `TSI = 1360.9 W m-2`, against
a double run from the same state, the bias in `sw_heating` (the tendency actually applied to
`th`) is largest aloft and does not care about terrain at all:

| Level (68 = model top) | Double, K s-1 | Single, `k_min = 1e-12` | Bias |
| ---------------------- | ------------- | ----------------------- | ---- |
| 68 | 8.85e-06 | 1.48e-05 | +67% |
| 66 | 4.68e-06 | 9.14e-06 | +95% |
| 62 | 4.77e-06 | 7.36e-06 | +54% |
| 58 | 9.39e-06 | 1.14e-05 | +22% |

Column-integrated, that is a **+19% shortwave heating bias**, and it is flat across the
domain — 19.6% over ocean columns with no pseudo-layers at all, 19.0% over the highest
terrain. Surface downward shortwave, which drives the land model, is 2.8% high.
With `k_min = 1e-5` every one of those numbers drops below 0.01%.

Nothing warns about this. `check_range_k` is applied to `tau`, `ssa`, and `g`, and every one
of them stays inside its valid range while the fluxes are wrong.

### A note on terrain

The error is easiest to *see* in the top-of-model `swdn` diagnostic, where it correlates with
the topography index at 1.00 and the worst columns report more downward shortwave than the
incident top-of-atmosphere flux. That correlation is a property of the diagnostic, not of the
bug. Terrain columns are padded with `num_dummy = topo - 1` thin pseudo-layers above the model
top (`VVM_rrtmgp_process_interface.cpp`), and the unpack reads the flux at RRTMGP level `hx`,
so a mountain column samples the beam after it has passed through more layers than a sea
column does. The diagnostic therefore scales with terrain height while the underlying error
does not:

| Build | mean | min | max | spread |
| ----- | ---- | --- | --- | ------ |
| Double (reference) | 667.4102 | 667.1001 | 667.5475 | 0.45 |
| Single, `k_min = 1e-12` | 675.7683 | 669.4679 | 723.6589 | 54.19 |
| Single, `k_min = 1e-5` | 667.4111 | 667.1010 | 667.5490 | 0.45 |

Reading the terrain correlation as the cause would be a mistake: the pseudo-layers' own
heating rates are discarded on unpack, but the beam that reaches the real atmosphere below
them is not, and neither is the error it carries.

### Choosing `k_min`

The floor was swept against the double reference at two solar zenith angles, `mu0 = 0.5` and
`mu0 = 0.05`. Low sun is the harder case, because the surviving remainder scales with
`k * mu0`. Worst single-column error, relative to the double top-of-model flux:

| `k_min` | `k` | error at `mu0 = 0.5` | error at `mu0 = 0.05` |
| ------- | --- | -------------------- | --------------------- |
| `1e-12` (original) | 1e-6 | 8.5e-2 | 7.4e-2 |
| `1e-6` | 1e-3 | 1.4e-4 | 1.0e-4 |
| **`1e-5`** | 3.2e-3 | **1.6e-5** | **1.1e-5** |
| `1e-4` | 1e-2 | 7.0e-6 | 4.6e-6 |

Everything from `1e-5` up is converged to roughly 1e-5 relative, which is about a hundred
times float epsilon and is as close as the accumulated arithmetic gets. `1e-6` is a further
order of magnitude worse, so `1e-5` is the smallest floor that works.

Smallest is what we want, because the floor is not free. It engages whenever
`2*(1 - w0)*(gamma1 + gamma2) < k_min`, which for cloud droplet single-scattering albedo in
the visible is essentially always, and it then pushes the layer away from its conservative
scattering solution once `k*tau` stops being small. For a thick cloud layer with `tau = 50`,
`k_min = 1e-4` biases `Rdif` low by about 1.2%, while `k_min = 1e-5` biases it by about
0.13%. In a cloud-resolving model that difference matters more than the remaining 1e-5 in
clear sky, so `1e-5` is used.

## Finding the changes

The minor-gas grouping is in
`src/physics/rrtmgp/external/cpp/rrtmgp/kernels/mo_gas_optics_kernels.h`.
The two-stream floor is in
`src/physics/rrtmgp/external/cpp/rte/kernels/mo_rte_solver_kernels.h`.
Neither changes a double-precision build. See
[Reproducibility](reproducibility.md) for other precision-dependent behavior.
