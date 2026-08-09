# VVMex Installation Guide
This guide provides step-by-step instructions for building the dependencies required for VVMex from source.

## 0. Preparation
To make this guide easy to copy and paste, please define your target installation directory first. All libraries will be installed under this directory.

```bash
# Replace this with your desired installation path
export INSTALL_DIR=/path/to/your/custom/libs
mkdir -p $INSTALL_DIR

export PATH=$INSTALL_DIR/bin:$PATH
export C_INCLUDE_PATH=$INSTALL_DIR/include:$C_INCLUDE_PATH
export LIBRARY_PATH=$INSTALL_DIR/lib64:$INSTALL_DIR/$LIB/lib:$LIBRARY_PATH
export LD_LIBRARY_PATH=$INSTALL_DIR/lib64:$INSTALL_DIR/lib:$LD_LIBRARY_PATH
```
Before compiling the base compiler (GCC), it is recommended to clear your library paths to avoid linking conflicts with system libraries:

```bash
unset LIBRARY_PATH LD_LIBRARY_PATH
```

---

## 1. Compiler & Core Tools

### GCC 11.4
VVMex requires C++17 support. If your system GCC is too old, build GCC 11.4:

```bash
wget https://ftp.gnu.org/gnu/gcc/gcc-11.4.0/gcc-11.4.0.tar.gz
tar -zxvf gcc-11.4.0.tar.gz
cd gcc-11.4.0
./contrib/download_prerequisites

mkdir build && cd build
../configure --prefix=$INSTALL_DIR/gcc11 \
             --enable-languages=c,c++,fortran \
             --disable-multilib \
             --disable-bootstrap
make -j$(nproc)
make install
cd ../..

# Add to current session for subsequent builds
export PATH=$INSTALL_DIR/gcc11/bin:$PATH
export LD_LIBRARY_PATH=$INSTALL_DIR/gcc11/lib64:$INSTALL_DIR/gcc11/lib:$LD_LIBRARY_PATH

```
*(Note: If your system has a higher version of GCC but you still want to force CMake to use this GCC 11, you will need to add `-DCMAKE_C_FLAGS="--gcc-toolchain=$INSTALL_DIR/gcc11"` and `-DCMAKE_CXX_FLAGS="--gcc-toolchain=$INSTALL_DIR/gcc11"` during the configuration.)*

### NVIDIA HPC SDK (NVHPC 24.9)
Download and install NVHPC 24.9 from the official NVIDIA website. Assuming it is installed at `/path/to/nvhpc_24_9`, export its path so we can use its MPI and CUDA wrappers for the rest of the installation:

```bash
export NVHPC_DEFAULT_CUDA=13.0
export NVHPC_VERSION=25.9
export NVHPC_HPCX_VERSION=2.24
export NVHPC_ROOT=/path/to/nvhpc/Linux_x86_64/${NVHPC_VERSION}

# Just give info above

export CUDA_HOME=${NVHPC_ROOT}/cuda/${NVHPC_DEFAULT_CUDA}
export C_INCLUDE_PATH=$CUDA_HOME/include:$C_INCLUDE_PATH
export LIBRARY_PATH=$CUDA_HOME/lib64:$LIBRARY_PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export PATH=${NVHPC_ROOT}/compilers/bin:$PATH
export LIBRARY_PATH=${NVHPC_ROOT}/compilers/lib:$LIBRARY_PATH
export LD_LIBRARY_PATH=${NVHPC_ROOT}/compilers/lib:$LD_LIBRARY_PATH
export C_INCLUDE_PATH=${NVHPC_ROOT}/math_libs/include:$C_INCLUDE_PATH
export LIBRARY_PATH=${NVHPC_ROOT}/math_libs/lib64:$LIBRARY_PATH
export LD_LIBRARY_PATH=${NVHPC_ROOT}/math_libs/lib64:$LD_LIBRARY_PATH

export HPCX_HOME=${NVHPC_ROOT}/comm_libs/${NVHPC_DEFAULT_CUDA}/hpcx/hpcx-${NVHPC_HPX_VERSION}
export PATH=${HPCX_HOME}/ompi/bin:${HPCX_HOME}/ucx/bin:${PATH}
export LD_LIBRARY_PATH=${HPCX_HOME}/ompi/lib:${HPCX_HOME}/ucx/lib:${HPCX_HOME}/sharp/lib:${HPCX_HOME}/nccl_rdma_sharp_plugin/lib:${LD_LIBRARY_PATH}
export LIBRARY_PATH=${HPCX_HOME}/ompi/lib:${HPCX_HOME}/ucx/lib:${HPCX_HOME}/sharp/lib:${HPCX_HOME}/nccl_rdma_sharp_plugin/lib:${LIBRARY_PATH}
export C_INCLUDE_PATH=${HPCX_HOME}/ompi/include:${HPCX_HOME}/ucx/include:${C_INCLUDE_PATH}
export CPLUS_INCLUDE_PATH=${HPCX_HOME}/ompi/include:${HPCX_HOME}/ucx/include:${CPLUS_INCLUDE_PATH}
export OPAL_PREFIX=${HPCX_HOME}/ompi
```

### CMake 4.2.0

```bash
wget https://github.com/Kitware/CMake/releases/download/v4.2.0/cmake-4.2.0.tar.gz
tar -zxvf cmake-4.2.0.tar.gz
cd cmake-4.2.0
./configure --prefix=$INSTALL_DIR/cmake
make -j$(nproc)
make install
cd ..

export PATH=$INSTALL_DIR/cmake/bin:$PATH

```

---

## 2. I/O Libraries

### HDF5 1.14.5
Must be compiled with MPI wrappers for parallel I/O.

```bash
wget https://github.com/HDFGroup/hdf5/releases/download/hdf5_1.14.5/hdf5-1.14.5.tar.gz
tar -zxvf hdf5-1.14.5.tar.gz
cd hdf5-1.14.5
./configure --prefix=$INSTALL_DIR \
            --enable-parallel --enable-shared --enable-cxx --enable-unsupported \
            CC="mpicc" CXX="mpic++" FC="mpifort" LIBS="-lm"
make -j$(nproc)
make install
cd ..
```

### ZLIB 1.3.1 (Optional)
*(Skip this step if zlib is already available on your system.)*

```bash
tar -zxvf zlib-1.3.1.tar.gz
cd zlib-1.3.1
./configure --prefix=$INSTALL_DIR
make -j$(nproc)
make install
cd ..
```

### NetCDF-C 4.4.1.1
IMPORTANT WARNING: Do NOT use MPI wrappers (mpicc, mpic++) to compile NetCDF-C and NetCDF-Fortran. Using MPI wrappers here will cause errors in the RRTMGP NetCDF reader. Use standard serial compilers (gcc, g++, gfortran) instead.
```bash
wget https://github.com/Unidata/netcdf-c/archive/refs/tags/v4.4.1.1.tar.gz
tar -zxvf v4.4.1.1.tar.gz
cd netcdf-c-4.4.1.1
./configure --prefix=$INSTALL_DIR \
            --enable-netcdf-4 \
            CC=gcc CXX=g++ FC=gfortran \
            CFLAGS="-fPIC -O2" CXXFLAGS="-fPIC -O2" FCFLAGS="-fPIC -O2"
make -j$(nproc)
make install
cd ..
```

### PnetCDF 1.14.1
PnetCDF provides parallel I/O for classic NetCDF files. This **must** be compiled with MPI wrappers.

```bash
wget https://parallel-netcdf.github.io/Release/pnetcdf-1.14.1.tar.gz
tar -zxvf pnetcdf-1.14.1.tar.gz
cd pnetcdf-1.14.1
./configure --prefix=$INSTALL_DIR \
            --with-netcdf4=$INSTALL_DIR \
            --disable-shared \
            CC=mpicc CXX=mpic++ FC=mpifort \
            CFLAGS="-fPIC -O2" CXXFLAGS="-fPIC -O2" FFLAGS="-fPIC -O2" FCFLAGS="-fPIC -O2"
make -j$(nproc)
make install
cd ..

```

### NetCDF-Fortran 4.4.1
IMPORTANT WARNING: Again, use standard serial compilers (gcc, gfortran), NOT MPI wrappers. The -fallow-argument-mismatch flag is required for modern GCC.
```bash
wget https://github.com/Unidata/netcdf-fortran/archive/refs/tags/v4.4.1.tar.gz
tar -zxvf v4.4.1.tar.gz
cd netcdf-fortran-4.4.1
export FFLAGS="-g -O2 -fallow-argument-mismatch" 
export FCFLAGS="-g -O2 -fallow-argument-mismatch"

./configure --prefix=$INSTALL_DIR \
            --enable-shared \
            CC=gcc FC=gfortran
make -j$(nproc)
make install
cd ..

```

### Kokkos 4.7.01
Note: Replace -DKokkos_ARCH_HOPPER90=ON with the appropriate architecture flag for your GPU (e.g., AMPERE80, VOLTA70, ADA89).
```bash
wget https://github.com/kokkos/kokkos/releases/download/4.7.01/kokkos-4.7.01.tar.gz
tar -zxvf kokkos-4.7.01.tar.gz
cd kokkos-4.7.01
mkdir build && cd build
cmake .. \
    -DCMAKE_INSTALL_PREFIX=$INSTALL_DIR \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CXX_STANDARD=17 \
    -DCMAKE_CXX_COMPILER=mpic++ \
    -DKokkos_ENABLE_SERIAL=ON \
    -DKokkos_ENABLE_OPENMP=ON \
    -DKokkos_ENABLE_CUDA=ON \
    -DKokkos_ENABLE_CUDA_LAMBDA=ON \
    -DKokkos_ARCH_HOPPER90=ON \
    -DBUILD_SHARED_LIBS=TRUE
make -j$(nproc)
make install
cd ../..

```

Verify that CUDA really was enabled and that the architecture matches your GPU:

```bash
grep -E "Kokkos_DEVICES|Kokkos_ARCH" \
     $INSTALL_DIR/lib/cmake/Kokkos/KokkosConfigCommon.cmake
#   set(Kokkos_DEVICES CUDA;OPENMP;SERIAL)
#   set(Kokkos_ARCH HOPPER90)

ldd $INSTALL_DIR/lib/libkokkoscore.so.4.7 | grep -i cudart   # expect a hit
```

An architecture mismatch does not fail the build. It produces a library that
either JITs every kernel at first launch — a large one-off startup cost — or
fails at launch on a device the generated code does not target. Check this
before building VVMex rather than debugging it later.

### libfabric 1.22.0
libfabric provides the OpenFabrics Interfaces used by high-performance communication transports. Build it before ADIOS2 so ADIOS2 can find it when enabling SST/RDMA-capable transports.

```bash
wget https://github.com/ofiwg/libfabric/releases/download/v1.22.0/libfabric-1.22.0.tar.bz2
tar -xjf libfabric-1.22.0.tar.bz2
cd libfabric-1.22.0
./configure --prefix=$INSTALL_DIR \
            --enable-shared \
            CC=gcc CXX=g++
make -j$(nproc)
make install
cd ..

export PKG_CONFIG_PATH=$INSTALL_DIR/lib/pkgconfig:$INSTALL_DIR/lib64/pkgconfig:$PKG_CONFIG_PATH
export LD_LIBRARY_PATH=$INSTALL_DIR/lib:$INSTALL_DIR/lib64:$LD_LIBRARY_PATH
```


### ADIOS2 2.11.0

Build ADIOS2 **without** Kokkos support (`-DADIOS2_USE_Kokkos=OFF`).

An ADIOS2 built with Kokkos calls `Kokkos::initialize()` itself, from
`adios2::helper::KokkosInit()`. That matters for the SST IO server: VVMex
deliberately does not initialize Kokkos on IO server ranks, because they only move
host buffers between ADIOS2 and HDF5 and never compute. If ADIOS2 initializes
Kokkos behind their back, every IO rank opens a CUDA context it never uses --
measured at 520 MB per rank, plus a device slot that a compute rank could have had
under CUDA exclusive mode.

Turning it off costs nothing: VVMex never hands ADIOS2 a Kokkos view. Every
`Put`/`Get` receives a raw host pointer, because `OutputManager` stages fields into
`Kokkos::HostSpace` buffers first.

```bash
git clone https://github.com/ornladios/ADIOS2.git
cd ADIOS2
git checkout tags/v2.11.0
mkdir build && cd build
cmake .. \
    -DCMAKE_INSTALL_PREFIX=$INSTALL_DIR \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_C_COMPILER=mpicc \
    -DCMAKE_CXX_COMPILER=mpic++ \
    -DCMAKE_PREFIX_PATH=$INSTALL_DIR \
    -DADIOS2_USE_MPI=ON \
    -DADIOS2_USE_HDF5=ON \
    -DADIOS2_USE_Kokkos=OFF \
    -DADIOS2_USE_CUDA=OFF \
    -DHDF5_ROOT=$INSTALL_DIR
make -j$(nproc)
make install
cd ../..

```

Verify the result before moving on:

```bash
grep -E "ADIOS2_HAVE_(Kokkos|CUDA|SST|MPI) " \
     $INSTALL_DIR/lib/cmake/adios2/adios2-config-common.cmake
#   set(ADIOS2_HAVE_MPI TRUE)
#   set(ADIOS2_HAVE_SST TRUE)
#   set(ADIOS2_HAVE_CUDA )        <- empty
#   set(ADIOS2_HAVE_Kokkos )      <- empty
```

VVMex re-checks this at configure time and prints one of:

```
-- ADIOS2 has no Kokkos support -- IO server ranks stay off the GPU
```
```
CMake Warning: ADIOS2 at ... was built with Kokkos support.
   IO server ranks will each occupy a GPU even though they never compute.
```

If you see the warning, the `ADIOS2_DIR` in your CMake preset is pointing at a
Kokkos-enabled build. Note that an `ADIOS2_DIR` naming a path that does not exist
does **not** fail: CMake falls back to whatever else is on the search path, so this
warning is the only signal that the IO server will land on a GPU.

Already have a Kokkos-enabled ADIOS2 you would rather not disturb? Build the
Kokkos-free one into a second prefix and point `ADIOS2_DIR` at that; it layers over
the first, taking `libfabric`, `libhdf5` and `libz` from it. `submit.py` puts
`ADIOS2_DIR` ahead of the other library directories on `LD_LIBRARY_PATH`, which it
must -- both prefixes carry `libadios2_*.so` with identical SONAMEs, so whichever
comes first is the one that loads.

Warning: If using less than 2.11.0, some errors may appear when compiling VVM. You need to modify adios2/cxx/KokkosView.h to adios2/cxx11/KokkosView.h in the code and cmakelist.

---

## 4. Environment Setup Script
To avoid cluttering your `.bashrc`, create a file named `env_setup.sh` in your workspace. Source this file (`source env_setup.sh`) every time before compiling or running VVMex.
**env_setup.sh:**

```bash
#!/bin/bash

# --- 1. Base Paths ---
export INSTALL_DIR=/path/to/your/custom/libs
export NVHPC_ROOT=/path/to/nvhpc_24_9/Linux_x86_64/24.9

# --- 2. NVHPC & MPI & CUDA ---
export CUDA_HOME=$NVHPC_ROOT/cuda/12.6
export OPAL_PREFIX=$NVHPC_ROOT/comm_libs/openmpi/openmpi-3.1.5

export PATH=$CUDA_HOME/bin:$OPAL_PREFIX/bin:$NVHPC_ROOT/compilers/bin:$PATH
export C_INCLUDE_PATH=$CUDA_HOME/include:$OPAL_PREFIX/include:$NVHPC_ROOT/math_libs/include:$C_INCLUDE_PATH
export LIBRARY_PATH=$CUDA_HOME/lib64:$OPAL_PREFIX/lib:$NVHPC_ROOT/compilers/lib:$NVHPC_ROOT/math_libs/lib64:$LIBRARY_PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$OPAL_PREFIX/lib:$NVHPC_ROOT/compilers/lib:$NVHPC_ROOT/math_libs/lib64:$LD_LIBRARY_PATH

# --- 3. GCC 11 & CMake ---
export PATH=$INSTALL_DIR/gcc11/bin:$INSTALL_DIR/cmake/bin:$PATH
export C_INCLUDE_PATH=$INSTALL_DIR/gcc11/include:$C_INCLUDE_PATH
export LIBRARY_PATH=$INSTALL_DIR/gcc11/lib64:$INSTALL_DIR/gcc11/lib:$LIBRARY_PATH
export LD_LIBRARY_PATH=$INSTALL_DIR/gcc11/lib64:$INSTALL_DIR/gcc11/lib:$LD_LIBRARY_PATH

# --- 4. I/O & Framework Libraries ---
export PATH=$INSTALL_DIR/bin:$PATH
export C_INCLUDE_PATH=$INSTALL_DIR/include:$C_INCLUDE_PATH
export LIBRARY_PATH=$INSTALL_DIR/lib64:$INSTALL_DIR/$LIB/lib:$LIBRARY_PATH
export LD_LIBRARY_PATH=$INSTALL_DIR/lib64:$INSTALL_DIR/lib:$LD_LIBRARY_PATH

echo "VVMex Environment Loaded Successfully!"

```

---

## 5. Configure and Build VVMex

Add a GPU preset to `CMakePresets.json` describing your machine, then:

```bash
source env_setup.sh
export VVM_ROOT=/path/to/VVMex

cmake --preset <your-gpu-preset> -DBUILD_TESTS=ON
cmake --build build -j$(nproc)
```

Expected configure output:

```text
-- Detected VVM_ROOT: /path/to/VVMex
-- VVM Execution Backend: GPU (Kokkos CUDA)
-- VVM Precision: DOUBLE (FP64)
-- Building with NCCL support
-- MPI found: /path/to/hpcx/ompi/bin/mpic++
-- ADIOS2 has no Kokkos support -- IO server ranks stay off the GPU
```

The last two lines are the ones worth reading. `Building with Standard MPI
support` on a GPU build means `ENABLE_NCCL` was turned off, and a warning about
ADIOS2 being Kokkos-enabled means each I/O rank will occupy a GPU it never uses
(see section 3).

Confirm the binary picked up the intended stack:

```bash
ldd build/vvm | grep -E 'libkokkoscore|libadios2_cxx11|libnccl'
nm -D --undefined-only build/vvm | grep -c nccl     # non-zero with NCCL on
```

`libkokkoscore` must resolve to **your** prefix. CPU and CUDA Kokkos share a
SONAME, so a stray CPU prefix earlier on `LD_LIBRARY_PATH` silently produces a
model that never touches the GPU.

---

## 6. Running

Use `submit.py`; it reads the same preset that built the code, so the launcher
cannot disagree with the binary about the backend.

```bash
# Local run on specific physical GPUs
VVM_GPU_LIST=0,1,2,3 ./submit.py --local \
    -c rundata/input_configs/default_cases/advection_u.json \
    --preset <your-gpu-preset> --compute 4 --nodes 1

# SLURM, one rank per GPU
./submit.py --preset <your-gpu-preset> \
    -c rundata/input_configs/default_cases/sea_grass_mountain.json \
    --compute 16 --nodes 1 --gpus 16 -t 24:00:00
```

The usual mapping is one MPI rank per GPU. `--gpus` covers compute ranks only —
I/O ranks are host-only and take no device. Leave `--cpus` unset so the wrapper
fills the node and pins each rank NUMA-local to its GPU; the `[GPUMap]` lines
report what it decided.

GPU builds support `HDF5`, `SST`, and `BP5`. BP5 writes directly from
compute ranks without I/O-server ranks; CUDA fields are synchronized to host
and packed before ADIOS2 sees them. See [Output](../user-guides/output.md).

---

## 7. Tests

```bash
ctest --test-dir build --output-on-failure
```

The default tier is seven bit-for-bit regression cases plus the unit tests, and
needs one GPU. Heavier tiers are opt-in at configure time:

| Tier | Needs | Enable |
|---|---|---|
| default | 1 GPU | always registered |
| bp5 | up to 4 GPUs | `-DVVM_TEST_BP5=ON` |
| physics | 1 GPU | `-DVVM_TEST_PHYSICS=ON` |
| multirank | 4 **physical** GPUs | `-DVVM_TEST_MULTIRANK=ON` |
| large | 8 GPUs + the 961 MB `taiwanvvm_2048.nc` | `-DVVM_TEST_LARGE=ON` |

Multi-rank tiers need that many real devices. Several NCCL ranks sharing one GPU
is unsupported and silently changes results, so they cannot be faked on a
smaller machine — see `tests/scripts/one_gpu_per_rank.sh`.

The BP5 tier includes exact readback, `float32` conversion, async output, and
a bit-for-bit comparison with HDF5 from the same deterministic GPU run.

GPU results are gated against `tests/baselines/` and `tests/references/`. The
CPU backend has its own data under `*_cpu/`, because the two agree only to a few
ulp and the SHA-256 digests tolerate nothing.

---

## 8. Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `Could not find NVIDIA CPU Math Library` | `NVHPC_DIR` unset or wrong | VVMex links `libnvcpumath` unconditionally; set `NVHPC_DIR` in the preset |
| RRTMGP fails with `NetCDF: Invalid dimension ID or name` | NetCDF-C was built with the MPI wrappers | Rebuild NetCDF-C with plain `gcc`/`g++` (section 2), then rebuild PnetCDF and VVMex |
| `[Rank N] ERROR: no visible CUDA device` | `CUDA_VISIBLE_DEVICES` empty, or fewer GPUs than compute ranks | Check `--gpus` and `VVM_GPU_LIST`; the model aborts rather than silently sharing a device |
| Warning that ranks may share GPUs | `--gpus` below compute ranks per node | Request `>= ceil(compute / nodes)` |
| I/O ranks each hold ~520 MB of GPU memory | ADIOS2 was built with Kokkos, so it calls `Kokkos::initialize()` on I/O ranks | Rebuild ADIOS2 with `-DADIOS2_USE_Kokkos=OFF`, or point `ADIOS2_DIR` at a Kokkos-free prefix |
| Model runs but never uses the GPU | A CPU Kokkos with the same SONAME loaded first | `ldd build/vvm \| grep libkokkoscore`; keep the CUDA prefix ahead on `LD_LIBRARY_PATH` |
| Very slow first time step, then normal | Kokkos CUDA architecture does not match the device, so kernels JIT at launch | Rebuild Kokkos with the correct `Kokkos_ARCH_*` |
| `version 'GLIBCXX_3.4.30' not found` starting `vvm` | NVHPC compiled against a newer GCC than the `libstdc++` loaded at run time | Pin GCC via `makelocalrc`, or pass `--gcc-toolchain` for **C, C++ and Fortran** |
| `undefined reference to std::ios_base_library_init()` | Same GCC mismatch, at link time | Add `--gcc-toolchain=$INSTALL_DIR/gcc11` to that library's build |
| Low GPU utilization, no error | One core per rank cannot carry the launch loop, MPI, NCCL and the driver threads | Leave `--cpus` unset; see [CPU allocation](../user-guides/job-submission.md#cpu-allocation) |
| SST run segfaults in the data-plane read handler at ~400+ writers | Known SST scaling limit on this class of system | Reduce writer count, or use direct `HDF5`/`BP5` output |

To check which GCC ABI a finished binary needs:

```bash
objdump -T build/vvm | grep -oE 'GLIBCXX_3\.4\.[0-9]+' | sort -uV | tail -1
```

GCC 11 provides up to `GLIBCXX_3.4.29`. Anything higher will not start against a
GCC 11 runtime.
