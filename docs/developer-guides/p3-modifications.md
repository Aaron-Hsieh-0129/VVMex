# VVMex P3 Modifications from Fortran P3

VVMex uses the Kokkos-based **P3 microphysics scheme** from E3SM EAMxx as the implementation framework, but modifies the process coupling to follow the original Fortran P3 behavior needed by the Vector Vorticity Model (VVM). The main change is that VVMex handles vapor-cloud-rain-ice phase adjustment inside the P3 microphysics step instead of relying on EAMxx macrophysics/SHOC to provide those tendencies externally.

In EAMxx, several processes are split across P3, SHOC, and other safeguards. In VVMex, these processes are coupled inside P3 so the microphysics can update vapor, hydrometeor mass, number, and thermodynamic state consistently during the model step.

## Process Comparison

| Feature / process | VVMex modification following Fortran P3 | EAMxx P3 behavior |
| ----------------- | --------------------------------------- | ----------------- |
| Cloud fraction | All-or-nothing treatment. | Cloud fraction is decided by SHOC. |
| Air density | Full non-hydrostatic calculation, $\rho = P / (R T)$. | Hydrostatic approximation, $\rho = -\frac{1}{g}\frac{dP}{dz}$. |
| Saturation vapor pressure (`qvs`, `qis`) | Uses `Polysvp1`, combining Flatau et al. (1992) style polynomial saturation vapor pressure with Goff-Gratch behavior. | Uses Murphy-Koop or `Polysvp1`; the `Polysvp1` path is Flatau-only. |
| Droplet activation | Performed after `ice_nucleation` and before `cloud_water_autoconversion`, together with the initial-step saturation adjustment. | Coupled to SHOC or prescribed external inputs. |
| Rain droplet size distribution | Marshall-Palmer distribution with $\mu_r = 0$. | Generalized Gamma distribution with $\mu_r = 1$. |
| In-cloud limits | No artificial in-cloud or precipitation thresholds are applied. | Applies explicit `incloud_limit = 5.1e-3` and `precip_limit = 1.0e-2` thresholds. |
| Ice lookup table | Uses v6.4. The table resolution differs, so the `lookup_table_1a_dum1_c` coefficient is changed accordingly. | Uses v4.1.1. |
| Condensation, evaporation, deposition, sublimation | Partitions vapor phase change among $q_c$, $q_r$, and $q_i$ using relaxation rates and saturation state. | SHOC handles condensation and evaporation of $q_c$. Deposition, sublimation, and evaporation of $q_r$ are handled separately in `ice_deposition_sublimation()` and `evaporate_rain()`. |
| Bergeron process | Included within the coupled mixed-phase vapor-adjustment step. | Treated as an independent process tendency. |
| Mass conservation | Applies coupled vapor-hydrometeor mass and number updates. | Processes are updated separately. |
| Supersaturation limiter | No extra EAMxx-style supersaturation limiter; saturation behavior is controlled through the coupled vapor-adjustment and conservation update. | Additional safeguards prevent overshooting saturation. |

## Restored and Modified Kernels

The Fortran-P3-style logic is integrated into the Kokkos pack-oriented kernels under `src/physics/p3/impl/`.

### `p3_calc_cond_evap_dep_sub_impl.hpp`

This file restores the semi-analytic integration for thermodynamic phase changes and combines competing vapor adjustment pathways according to their calculated timescales. It handles:

- **Cloud condensation/evaporation:** explicit tendencies for $q_v \rightarrow q_c$ (`qv2qc_conden_tend`) and $q_c \rightarrow q_v$ (`qc2qv_evap_tend`).
- **Rain condensation/evaporation:** vapor exchange involving $q_r$.
- **Ice deposition/sublimation:** restored processes for $q_v \rightarrow q_i$ and $q_i \rightarrow q_v$.
- **Mixed-phase competition:** coupled vapor adjustment among $q_c$, $q_r$, and $q_i$ based on relaxation rates and saturation state.

### `p3_droplet_activation_impl.hpp`

This file restores droplet activation inside P3. It computes new droplets from environmental supersaturation and initializes both cloud droplet number tendency (`nc_nuclet_tend`) and associated mass tendency (`qv2qc_nucleat_tend`). In the VVMex sequence, activation occurs after ice nucleation and before cloud-water autoconversion.

### `p3_limit_cond_evap_dep_sub_impl.hpp`

This file limits the coupled condensation, evaporation, deposition, and sublimation tendencies so the update does not over-deplete available water or violate the coupled mass/number update. In VVMex this is part of the Fortran-P3-style coupled vapor adjustment, not the same as the additional EAMxx supersaturation safeguard listed in the comparison table.

## Main Loop Integration

These processes are integrated into the `p3_main` execution flow. In `p3_main_impl_part2.hpp`, the sequence includes:

1. **Phase-change tendency calculation:** `calc_cond_evap_dep_sub` computes coupled condensation, evaporation, deposition, and sublimation tendencies.
2. **Droplet activation:** `droplet_activation` computes supersaturation-driven new droplet mass and number.
3. **Coupled limiting and conservation:** `limit_cond_evap_dep_sub` scales the coupled tendencies before mapping back to cell averages.
4. **Prognostic updates:** tendencies such as `qv2qc_conden_tend`, `qc2qv_evap_tend`, `qv2qi_depos_tend`, and related number tendencies are injected into the liquid and ice update paths to advance $q_v$, $q_c$, $q_r$, $q_i$, temperature, and potential temperature consistently.

## Precision-Specific Lookup Tables

`p3_init` caches the four non-ice tables (`mu_r`, `vn`, `vm`, `revap`) as raw binary dumps of
the working scalar type, so the filename carries the element size: `*_v2.dat8` for a
double-precision build and `*_v2.dat4` for a single-precision one. `rundata/p3/` therefore
has to hold **both sets**, and both are required inputs, not optional caches.

Nothing checks that the file was actually read. `read_computed_tables` streams the bytes
straight into a zero-initialized Kokkos view, so a missing or truncated file leaves every
table at zero and the run continues. With `vn_table_vals` zeroed the rain fallspeed is zero:
$q_c$ and $q_r$ still form and look normal, but rain never reaches the ground and surface
precipitation is effectively zero for the whole run. This is silent — there is no warning and
no abort.

Regenerate a missing set by running any case once with the matching build and

```json
"physics": { "p3": { "make_lookup_table": true } }
```

which computes the tables in memory and writes them to `rundata/p3/`. Use a single rank —
every rank writes the same paths. Set the option back to `false` afterwards.

The tables are computed by integrating the drop size distribution over 10,000 bins in the
working precision, so a `.dat4` set generated this way sits up to 2.3e-5 (relative) from the
corresponding `.dat8` values. That is well inside single precision's own error budget for the
model, but it does mean the two sets are not simply rounded copies of each other.

## Tracking the Code Changes

For developers navigating the codebase, the VVMex-specific P3 changes are marked with comments such as `// Aaron - evporation/condensation/deposition/sublimation` and `// Aaron - limit saturation adjustment` within `p3_functions.hpp` and related `_impl.hpp` files.
