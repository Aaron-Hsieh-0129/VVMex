# Reproducibility

What VVMex guarantees about repeating a run, and what it takes to make a
CPU-only build agree with a GPU build bit-for-bit.

## What is already bit-for-bit

| Repeat the run with… | Result |
|---|---|
| the same binary, same rank count, same case | identical, run after run |
| a different `output.engine` (HDF5 / SST / BP5) | identical values and metadata — see [Output](../user-guides/output.md) |
| a different `output.precision` | identical to the model state, then rounded once on write |
| a different backend (GPU vs CPU-only) | **not** identical by default — see below |

Random perturbation is off in every regression case, which is what makes the
first row an exact statement rather than a statistical one.

## Why GPU and CPU-only differ

Both backends run the same source. Three things can still make them disagree,
and the measurement below separates them — one binary, the same input bits, the
same expression, executed on `Kokkos::Cuda` and on `Kokkos::OpenMP` (double
precision, 2^20 values, nvcc device pass vs nvc++ `-fast` host pass):

| Operation | GPU vs CPU |
|---|---|
| `+ - /`, `sqrt`, `exp` | identical |
| `a*b + c` | identical |
| **`a*b + c*d`** | **25% of inputs differ by 1 ULP** |
| **`parallel_reduce` sum** | **differs, ~10 ULP over a million values** |
| the same sum in a fixed order | identical |
| `log` | 29% differ by 1 ULP |
| `pow` | 0.7% differ by 1 ULP |
| `tanh` | 19% differ by up to 2 ULP |

So it is **not** fast-math, and not FMA as such: both compilers contract
`a*b + c` into an FMA and agree. What they do not agree on is *which* multiply to
fuse once an expression has more than one product, and a fused product keeps a
wider intermediate than a rounded one. That single ULP is the first thing that
moves: in `advection_u`, the earliest difference between a GPU and a CPU-only run
is the θ advection tendency `d_th_0` after one step, at 27 points inside the
Takacs flux, at 1e-14. Every later difference grows from there — 1e-13 after 120
steps for advection, 1e-2 for `2dbubble`, where a convective case amplifies it.

The reduction row matters separately: CUDA's shuffle tree and OpenMP's
per-thread partials combine partial sums in different orders. `State`'s
horizontal mean can switch to summing each row sequentially and then summing the
rows in order — the same fixed-order discipline already used for the per-rank
sums — which removes the backend and thread-count dependence. Both changes live
behind one switch, because either one alone leaves the backends apart.

## Making a CPU build match a GPU build

Configure both builds with:

```bash
cmake --preset <gpu-preset> -DVVM_DETERMINISTIC_FP=ON
cmake --preset <cpu-preset> -DVVM_DETERMINISTIC_FP=ON
```

The option adds `--fmad=false` to the device compile and `-Mnofma` to the host
compile so neither backend contracts, and defines `VVM_DETERMINISTIC_FP`, which
switches the horizontal mean to its fixed-order form.

It is **off by default**, and with it off nothing changes: a default build
reproduces the stored baselines for its backend bit-for-bit, on both GPU and
CPU. Turning it on changes results on *both* backends in the last ULP — that is
the point, since agreement means meeting in the middle rather than one backend
adopting the other's rounding. A build with the option on therefore needs its
own reference data.

Measured on blaze, one rank:

| Case | GPU vs CPU-only, default | with `VVM_DETERMINISTIC_FP=ON` |
|---|---|---|
| `advection_u` | 4 fields differ, up to 6e-13 | **bit-for-bit identical** |
| `advection_v` | 4 fields differ, up to 6e-13 | **bit-for-bit identical** |
| `advection_w` | 4 fields differ, up to 7e-12 | **bit-for-bit identical** |
| `twisting` | 3 fields differ, up to 2e-13 | **bit-for-bit identical** |
| `stretching` | 4 fields differ, up to 5e-15 | identical values; `w` holds `-0.0` where the CPU holds `+0.0` |
| `2dbubble` | 4 fields differ, up to 2e-2 | **bit-for-bit identical** |

`2dbubble` is the informative one: a convective case that amplifies a single
ULP into 2e-2 over 120 steps agrees exactly once neither backend contracts,
which is what confirms contraction was the whole story for the dry core.

Cost of switching it on: **~0.2%** wall time on GPU (`2dbubble`) and **~1.2%** on
CPU (`advection_u`). Contraction is disabled, not fast-math — accuracy against
the exact result is marginally *lower* without FMA, which is the usual trade for
reproducibility.

Signed zero is worth calling out: `-0.0` and `+0.0` compare equal and behave
identically in every arithmetic operation the model performs, so a field that
differs only there is numerically identical. A byte-level comparison still
reports it.

### What this does not cover

- **Physics.** P3, RRTMGP, and Noah lean on `log`, `pow`, and `tanh`, whose
  device and host implementations differ by 1–2 ULP. No compiler flag fixes
  that; it would take one math library shared by both backends.
- **Rank count.** A different decomposition partitions the sums differently, so
  1-rank and 4-rank runs are not expected to be bit-identical to each other —
  that is a separate property, checked by the multirank tier.
- **Different hardware or compiler versions.** Everything above was measured
  with NVHPC 24.9 on sm_90.

## Reproducing the measurement

The per-operation table comes from running one Kokkos binary over both
execution spaces. The end-to-end table comes from running the same case with
both builds and comparing the HDF5 output byte for byte:

```bash
h5diff -c build/testing_output_advection_u/vvm_output_000001.h5 \
          build_cpu/testing_output_advection_u/vvm_output_000001.h5
```

Both baseline sets (`tests/baselines/`, `tests/baselines_cpu/`) exist because of
what this page describes: at the default the two backends need separate
references for the dry cases, and with physics on they need them regardless.
Neither set changes when the option is merely available — only a build that
turns it on produces different numbers, and that configuration is not gated by
stored references today.
