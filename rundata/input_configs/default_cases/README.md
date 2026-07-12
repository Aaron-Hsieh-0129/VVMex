# Default paper experiment configurations

This directory contains the VVMex configuration files used for the experiments reported in the VVMex v1.0 paper.

These files specify the model setup for the verification, validation, and performance experiments. 

| Paper experiment | Configuration file(s) | Purpose | Related figure |
|---|---|---|---|
| Bubble with shear | `p3_bubble_shear.json` | Diagnostic full-physics test with P3 microphysics | Fig. 3 |
| Component verification | `advection_u.json`, `advection_v.json`, `advection_w.json`, `stretching.json`, `twisting.json` | Difference-based verification of individual dynamical-core components | Fig. 4 |
| Dry mountain-wave test | `mountain.json` | Dynamical-core verification using the Wu and Arakawa mountain-wave case | Fig. 5 |
| Ocean–land–mountain case | `sea_grass_mountain.json` | Validation of thermally driven circulations over heterogeneous terrain | Fig. 6 |
| Homogeneous diurnal land cases | `grass.json`, `urban.json` | Validation of land–atmosphere coupling over idealized land surfaces | Fig. 7 |
| RCEMIP radiative-convective equilibrium | `rcemip.json` | Long-duration full-physics validation under idealized tropical conditions | Fig. 8 |
| Performance benchmark | `taiwanvvm_2048.json` | Strong-scaling and wall-clock performance evaluation | Fig. 9 |

## Notes

- These configurations are intended to document the model setup used in the paper.