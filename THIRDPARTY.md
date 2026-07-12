# Third-party components

VVMex incorporates or adapts several third-party software components.

| Component | Source | Role in VVMex | Local modification | License / citation |
|---|---|---|---|---|
| Kokkos | Kokkos project | Performance-portable C++ programming model | Used as an external dependency | See upstream project |
| ADIOS2 | ADIOS2 project | Scalable model output | Used as an external dependency | See upstream project |
| P3 microphysics | E3SM/EAMxx | Cloud microphysics | Adapted for VVMex; vapor–cloud water conversion restored for LES use without SHOC | See upstream project and cited papers |
| RRTMGP radiation | E3SM/EAMxx / RRTMGP | Longwave and shortwave radiation | Adapted for VVMex coupling | See upstream project and cited papers |
| Noah land surface model | CWA / WRF-related implementation | Land surface coupling | Integrated into the VVMex workflow | This module is owned by the Central Weather Administration (CWA) of Taiwan and is not publicly released by CWA. Please contact before modifying or redistributing this component. |