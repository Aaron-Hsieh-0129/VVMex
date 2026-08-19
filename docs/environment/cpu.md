# VVMex CPU-Only Installation Guide

This is a complete, self-contained guide to building VVMex and every dependency it
needs for **CPU-only execution**. Nothing here depends on having built the GPU
stack first; everything is installed into a single prefix of its own.

In a CPU build, Kokkos runs on its OpenMP backend, the standard-MPI communication
path replaces NCCL, and the Noah land model's OpenACC directives compile as
comments. No GPU is used at run time.

Why NVHPC is still required: VVMex links `libnvcpumath` unconditionally and the
Noah land model is written for `nvfortran` flags (`-Mallocatable=03`, `-Mfreeform`,
`-Mextend`, `-r8`). Configuration fails without it. The SDK is only being used as a
compiler and MPI stack here — its CUDA backend stays unused. A pure GCC/gfortran
build is **not** currently supported.

Build order matters: each library below is used by the ones after it.

---

## 0. Preparation

Everything goes into one directory. Keeping the CPU stack in a prefix of its own is
what makes the rest of this guide simple: if a CUDA Kokkos ever shares a prefix
with this one, the two `libkokkoscore.so` have the same SONAME and the loader can
pick the wrong one.

```bash
# Replace with your desired installation path
export VVM_CPU_DIR=/path/to/your/cpu/libs
mkdir -p $VVM_CPU_DIR
```

Before compiling the base compiler, clear inherited include and library search
paths. This is especially important after loading Intel oneAPI: its `CPATH` can
make the bootstrap GCC select Intel's `float.h` instead of GCC's own header,
causing the bundled MPFR build to fail because `DBL_MAX` is not defined.

```bash
unset CPATH C_INCLUDE_PATH CPLUS_INCLUDE_PATH
unset LIBRARY_PATH LD_LIBRARY_PATH
```

Run these commands before `configure`, not after a failed build. If a configure or
compile attempt was made with the contaminated environment, delete and recreate
the build directory so cached detection results are not reused.

---

## 1. Compiler & Core Tools

### GCC 11.4

VVMex requires C++17. NVHPC also needs a GCC to supply its C++ standard library, and
pinning that to a known version avoids a class of ABI failures described in
[Troubleshooting](#11-troubleshooting).

```bash
wget https://ftp.gnu.org/gnu/gcc/gcc-11.4.0/gcc-11.4.0.tar.gz
tar -zxvf gcc-11.4.0.tar.gz
cd gcc-11.4.0
./contrib/download_prerequisites

mkdir build && cd build
../configure --prefix=$VVM_CPU_DIR/gcc11 \
             --enable-languages=c,c++,fortran \
             --disable-multilib \
             --disable-bootstrap
make -j$(nproc)
make install
cd ../..

export PATH=$VVM_CPU_DIR/gcc11/bin:$PATH
export LD_LIBRARY_PATH=$VVM_CPU_DIR/gcc11/lib64:$VVM_CPU_DIR/gcc11/lib:$LD_LIBRARY_PATH
```

### NVIDIA HPC SDK (NVHPC 24.9)

Download and install NVHPC 24.9 from the NVIDIA website. It provides `nvc`,
`nvc++`, `nvfortran` and `libnvcpumath`. Its bundled HPC-X MPI is not used —
section 2 builds Open MPI instead.

```bash
export NVHPC_VERSION=24.9
export NVHPC_ROOT=/path/to/nvhpc/Linux_x86_64/${NVHPC_VERSION}

export PATH=${NVHPC_ROOT}/compilers/bin:$PATH
export LIBRARY_PATH=${NVHPC_ROOT}/compilers/lib:$LIBRARY_PATH
export LD_LIBRARY_PATH=${NVHPC_ROOT}/compilers/lib:$LD_LIBRARY_PATH
export C_INCLUDE_PATH=${NVHPC_ROOT}/math_libs/include:$C_INCLUDE_PATH
export LIBRARY_PATH=${NVHPC_ROOT}/math_libs/lib64:$LIBRARY_PATH
export LD_LIBRARY_PATH=${NVHPC_ROOT}/math_libs/lib64:$LD_LIBRARY_PATH
```

IMPORTANT: point NVHPC at the GCC 11 you just built. NVHPC selects a GCC through
its `localrc` file, and by default that is whatever system GCC it found at install
time. If that GCC is newer than the one whose `libstdc++` you load at run time, the
build produces binaries that fail to start. Either regenerate `localrc`:

```bash
${NVHPC_ROOT}/compilers/bin/makelocalrc ${NVHPC_ROOT}/compilers/bin \
    -gcc $VVM_CPU_DIR/gcc11/bin/gcc \
    -gpp $VVM_CPU_DIR/gcc11/bin/g++ \
    -g77 $VVM_CPU_DIR/gcc11/bin/gfortran -x
```

or pass `--gcc-toolchain=$VVM_CPU_DIR/gcc11` explicitly to every compiler
invocation. This guide does the latter, because it is explicit and does not modify
the SDK. Whichever you choose, apply it to **C, C++ and Fortran** — Fortran is the
one most easily forgotten, and VVMex links a Fortran library.

### CMake 4.2.0

```bash
wget https://github.com/Kitware/CMake/releases/download/v4.2.0/cmake-4.2.0.tar.gz
tar -zxvf cmake-4.2.0.tar.gz
cd cmake-4.2.0
./configure --prefix=$VVM_CPU_DIR/cmake
make -j$(nproc)
make install
cd ..

export PATH=$VVM_CPU_DIR/cmake/bin:$PATH
```

---

## 2. MPI — Open MPI 4.1.6

VVMex is an MPI program in every configuration; the CPU build uses MPI for all halo
exchanges and global reductions in place of NCCL. You need working `mpicc`,
`mpic++` and `mpifort` wrappers on `PATH` before building any of the I/O libraries
below.

Build Open MPI yourself rather than using a vendor-bundled one. Two reasons:

- **No CUDA.** A CUDA-aware MPI keeps a link-time dependency on `libcuda.so.1`
  through `libfabric`, so the final binary will not start on a machine with no
  NVIDIA driver, even though no GPU is used. `--without-cuda` removes that.
- **Control.** The wrappers must invoke the NVHPC compilers, because the Noah land
  model needs `nvfortran` flags (`-Mallocatable=03`, `-Mfreeform`, `-r8`). Building
  it yourself makes that explicit instead of inherited.

```bash
wget https://download.open-mpi.org/release/open-mpi/v4.1/openmpi-4.1.6.tar.gz
tar -zxvf openmpi-4.1.6.tar.gz
cd openmpi-4.1.6

./configure --prefix=$VVM_CPU_DIR \
            --without-cuda \
            --without-ucx \
            --enable-mpi-fortran=usempi \
            --enable-mca-no-build=fs-gpfs \
            CC=nvc CXX=nvc++ FC=nvfortran \
            CFLAGS="--gcc-toolchain=$VVM_CPU_DIR/gcc11" \
            CXXFLAGS="--gcc-toolchain=$VVM_CPU_DIR/gcc11" \
            FCFLAGS="--gcc-toolchain=$VVM_CPU_DIR/gcc11"
make -j$(nproc)
make install
cd ..

export OPAL_PREFIX=$VVM_CPU_DIR
export PATH=$VVM_CPU_DIR/bin:$PATH
export LD_LIBRARY_PATH=$VVM_CPU_DIR/lib:$LD_LIBRARY_PATH
```

`CC`/`CXX`/`FC` are what the wrappers will call afterwards, so `mpicc` becomes
`nvc`, `mpic++` becomes `nvc++`, and `mpifort` becomes `nvfortran` — which is what
the rest of this guide and the VVMex preset assume.

`--enable-mpi-fortran=usempi` builds the `mpif.h` and `use mpi` interfaces used by
VVMex without the unnecessary, much larger `use mpi_f08` library. The GPFS
component is disabled because some cluster GPFS headers are incompatible with
Open MPI 4.1.6; generic MPI-IO remains available.

`--without-ucx` keeps the build self-contained. On a single node, Open MPI's
shared-memory transport is used regardless. If you are running across nodes over
InfiniBand, drop that flag and point `--with-ucx=` at a UCX install instead.

Verify the wrappers resolve to the NVHPC compilers and that no CUDA is linked:

```bash
command -v mpicc mpic++ mpifort mpirun
mpicc --showme:command           # nvc
mpic++ --showme:command          # nvc++
mpifort --showme:command         # nvfortran
mpicc --version | head -1        # nvc
mpifort --version | head -1      # nvfortran
mpirun --version | head -1       # mpirun (Open MPI) 4.1.6

ldd $VVM_CPU_DIR/lib/libmpi.so | grep -i cuda    # expect no output
```

All four paths printed by `command -v` must belong to this installation. Do not
compile with one MPI implementation's wrapper and launch with another
implementation's `mpirun`.

Quick C++ functional check:

```bash
cat > hello.cpp <<'EOF'
#include <cstdio>
#include <mpi.h>

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    std::printf("rank %d of %d\n", rank, size);
    MPI_Finalize();
    return 0;
}
EOF

mpic++ hello.cpp -o hello
mpirun --mca btl self,vader,tcp -np 2 ./hello
```

Expect one `rank 0 of 2` line and one `rank 1 of 2` line. The explicit BTL list
is appropriate for this single-node check and avoids harmless
`unknown link speed 0x80` warnings from probing cluster network interfaces.

Do not use shell `printf '...%d...'` to generate this source: shell `printf`
interprets `%d` itself and, with no corresponding arguments, writes literal
zeros into the C++ file.

---

## 3. Compression and I/O Libraries

### ZLIB 1.3.1 (optional)

Skip if zlib is already available on your system.

```bash
wget https://www.zlib.net/zlib-1.3.1.tar.gz
tar -zxvf zlib-1.3.1.tar.gz
cd zlib-1.3.1
./configure --prefix=$VVM_CPU_DIR
make -j$(nproc)
make install
cd ..
```

### HDF5 1.14.5

Must be built with MPI wrappers for parallel I/O.

```bash
wget https://github.com/HDFGroup/hdf5/releases/download/hdf5_1.14.5/hdf5-1.14.5.tar.gz
tar -zxvf hdf5-1.14.5.tar.gz
cd hdf5-1.14.5

# Vendor include paths (especially Intel oneAPI) can shadow GCC's float.h.
unset CPATH C_INCLUDE_PATH CPLUS_INCLUDE_PATH
unset LIBRARY_PATH

# DBL_EPSILON must expand to a number, not remain as the literal token.
printf '#include <float.h>\nDBL_EPSILON\n' | mpicc -E -x c - | tail -1

./configure --prefix=$VVM_CPU_DIR \
            --enable-parallel --enable-shared --enable-cxx --enable-unsupported \
            --disable-nonstandard-feature-float16 \
            CC="mpicc" CXX="mpic++" FC="mpifort" LIBS="-lm"
make -j$(nproc)
make install
cd ..
```

### NetCDF-C 4.4.1.1

NetCDF-C must be compiled by GCC, **not** by the NVHPC-backed `mpicc`/`mpic++`
wrappers. NetCDF-C 4.4.1.1 built with `nvc -O2` can build successfully but then
fail to open the classic RRTMGP coefficient files with `NetCDF: Invalid dimension
ID or name`. This is the same restriction as in the GPU environment.

The installed HDF5 is parallel, so its public header includes `mpi.h`. Supply the
Open MPI include and library directories explicitly while keeping GCC as the
actual compiler. NetCDF-C must still detect parallel HDF5 and install
`netcdf_par.h`; PnetCDF needs that API to open VVMex's NetCDF-4/HDF5 initial
conditions. Do not enable NetCDF-C's separate PnetCDF dispatch feature.

```bash
wget https://github.com/Unidata/netcdf-c/archive/refs/tags/v4.4.1.1.tar.gz
tar -zxvf v4.4.1.1.tar.gz
cd netcdf-c-4.4.1.1

# Use the same Open MPI installation used to build HDF5. This works whether MPI
# was installed directly in $VVM_CPU_DIR or in a separate sub-prefix.
export MPI_ROOT=$(dirname "$(dirname "$(command -v mpicc)")")

mkdir build-cpu && cd build-cpu
../configure --prefix=$VVM_CPU_DIR \
            --enable-netcdf-4 \
            --enable-parallel-tests \
            --disable-pnetcdf \
            CC=$VVM_CPU_DIR/gcc11/bin/gcc \
            CXX=$VVM_CPU_DIR/gcc11/bin/g++ \
            CFLAGS="-fPIC -O2" CXXFLAGS="-fPIC -O2" \
            CPPFLAGS="-I$VVM_CPU_DIR/include -I$MPI_ROOT/include" \
            LDFLAGS="-L$VVM_CPU_DIR/lib64 -L$VVM_CPU_DIR/lib -L$MPI_ROOT/lib -Wl,-rpath,$MPI_ROOT/lib" \
            LIBS="-lmpi"
make -j$(nproc)
make check

# Test the file used by RRTMGP before installing. It must print "classic".
./ncdump/ncdump -k \
    $VVM_ROOT/rundata/rrtmgp/rrtmgp-data-sw-g112-210809.nc

# Both parallel values must be 1, and the generated header must exist.
grep NC_HAS_PARALLEL include/netcdf_meta.h
test -f include/netcdf_par.h

make install
cd ../..

# Confirm that the installed tool also reads the RRTMGP data.
$VVM_CPU_DIR/bin/ncdump -k \
    $VVM_ROOT/rundata/rrtmgp/rrtmgp-data-sw-g112-210809.nc
grep NC_HAS_PARALLEL $VVM_CPU_DIR/include/netcdf_meta.h
test -f $VVM_CPU_DIR/include/netcdf_par.h
```

Before configuring PnetCDF, make the headers and shared libraries already
installed in the CPU prefix visible. Add these lines to `~/.zshrc` after the
definitions of `VVM_CPU_DIR` and `NVHPC_ROOT`, then reload the shell
configuration. Reset these paths instead of appending inherited Intel oneAPI
paths:

```bash
export CPATH=$VVM_CPU_DIR/include
export LIBRARY_PATH=$VVM_CPU_DIR/lib64:$VVM_CPU_DIR/lib:$NVHPC_ROOT/compilers/lib:$VVM_CPU_DIR/gcc11/lib64:$VVM_CPU_DIR/gcc11/lib
export LD_LIBRARY_PATH=$VVM_CPU_DIR/lib64:$VVM_CPU_DIR/lib:$NVHPC_ROOT/compilers/lib:$VVM_CPU_DIR/gcc11/lib64:$VVM_CPU_DIR/gcc11/lib

source ~/.zshrc
```

`LIBRARY_PATH` is used while linking, and `LD_LIBRARY_PATH` is used while running
shared-library executables; do not use `LDPATH`.

### PnetCDF 1.14.1

This **must** use MPI wrappers and `--with-netcdf4`. The latter enables the
driver that lets `ncmpi_open()` read the NetCDF-4/HDF5 initial-condition files
used by VVMex. Do not use NetCDF-C's `--enable-pnetcdf`; that is a different
feature and is not supported by this PnetCDF NetCDF-4 driver.

Build shared libraries because VVMex links `libpnetcdf.so`. Do not use
`--disable-shared`: if an older shared library is already installed, a
static-only rebuild updates `libpnetcdf.a` but leaves the stale `.so` in place,
and VVMex continues loading the old feature-disabled library.

```bash
wget https://parallel-netcdf.github.io/Release/pnetcdf-1.14.1.tar.gz
tar -zxvf pnetcdf-1.14.1.tar.gz
cd pnetcdf-1.14.1

# Both tokens must expand to numeric expressions. Literal names indicate that a
# vendor float.h is still shadowing GCC's header.
printf '#include <float.h>\nFLT_EPSILON DBL_EPSILON\n' | mpicc -E -x c - | tail -1

mkdir build-cpu && cd build-cpu
../configure --prefix=$VVM_CPU_DIR \
            --with-netcdf4=$VVM_CPU_DIR \
            --enable-shared \
            CC=mpicc CXX=mpic++ FC=mpifort \
            CFLAGS="-fPIC -O2" CXXFLAGS="-fPIC -O2" FFLAGS="-fPIC -O2" FCFLAGS="-fPIC -O2"
make -j$(nproc)
make check
make install
cd ../..

$VVM_CPU_DIR/bin/pnetcdf-config --all | grep -E 'NetCDF4|PnetCDF Version'
```

If NetCDF-C is replaced or rebuilt, delete PnetCDF's old `build-cpu` directory
and rebuild PnetCDF against the corrected headers and libraries. Rebuild VVMex
afterwards as well. HDF5 does not need to be rebuilt for this change.

### NetCDF-Fortran 4.4.1

Again: serial compilers, not MPI wrappers. `-fallow-argument-mismatch` is required
for modern GCC.

```bash
wget https://github.com/Unidata/netcdf-fortran/archive/refs/tags/v4.4.1.tar.gz
tar -zxvf v4.4.1.tar.gz
cd netcdf-fortran-4.4.1
export FFLAGS="-g -O2 -fallow-argument-mismatch"
export FCFLAGS="-g -O2 -fallow-argument-mismatch"

./configure --prefix=$VVM_CPU_DIR \
            --enable-shared \
            CC=gcc FC=gfortran
make -j$(nproc)
make install
cd ..
unset FFLAGS FCFLAGS
```

---

## 4. Kokkos 4.7.02 (CPU)

This is the first place the CPU build genuinely differs from a GPU one. CUDA is
off, and OpenMP becomes the execution space. There is no architecture flag to set.

```bash
wget https://github.com/kokkos/kokkos/releases/download/4.7.02/kokkos-4.7.02.tar.gz
tar -zxvf kokkos-4.7.02.tar.gz
cd kokkos-4.7.02
mkdir build && cd build
cmake .. \
    -DCMAKE_INSTALL_PREFIX=$VVM_CPU_DIR \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CXX_STANDARD=17 \
    -DCMAKE_CXX_COMPILER=mpic++ \
    -DCMAKE_CXX_FLAGS=--gcc-toolchain=$VVM_CPU_DIR/gcc11 \
    -DKokkos_ENABLE_SERIAL=ON \
    -DKokkos_ENABLE_OPENMP=ON \
    -DKokkos_ENABLE_CUDA=OFF \
    -DBUILD_SHARED_LIBS=TRUE
make -j$(nproc)
make install
cd ../..
```

Verify that no CUDA leaked in:

```bash
grep Kokkos_DEVICES $VVM_CPU_DIR/lib/cmake/Kokkos/KokkosConfigCommon.cmake
#   set(Kokkos_DEVICES OPENMP;SERIAL)

ldd $VVM_CPU_DIR/lib/libkokkoscore.so.4.7 | grep -i cuda      # expect no output
```

---

## 5. libfabric 1.22.0

Provides the OpenFabrics Interfaces used by ADIOS2's SST transport. Build it before
ADIOS2 so ADIOS2 can find it.

```bash
wget https://github.com/ofiwg/libfabric/releases/download/v1.22.0/libfabric-1.22.0.tar.bz2
tar -xjf libfabric-1.22.0.tar.bz2
cd libfabric-1.22.0
./configure --prefix=$VVM_CPU_DIR \
            --enable-shared \
            CC=gcc CXX=g++
make -j$(nproc)
make install
cd ..

export PKG_CONFIG_PATH=$VVM_CPU_DIR/lib/pkgconfig:$VVM_CPU_DIR/lib64/pkgconfig:$PKG_CONFIG_PATH
export LD_LIBRARY_PATH=$VVM_CPU_DIR/lib:$VVM_CPU_DIR/lib64:$LD_LIBRARY_PATH
```

---

## 6. ADIOS2 2.12.1 (without Kokkos)

`ADIOS2_USE_Kokkos=ON` builds `libadios2_core_kokkos.so`, which links Kokkos and
pulls `libcudart`/`libcuda` into every process that uses ADIOS2. VVMex never hands
ADIOS2 a Kokkos view — every `Put`/`Get` passes a raw host pointer — so this
support is simply turned off.

```bash
git clone https://github.com/ornladios/ADIOS2.git
cd ADIOS2
git checkout tags/v2.12.1
mkdir build && cd build
cmake .. \
    -DCMAKE_INSTALL_PREFIX=$VVM_CPU_DIR \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_C_COMPILER=mpicc \
    -DCMAKE_CXX_COMPILER=mpic++ \
    -DCMAKE_CXX_FLAGS=--gcc-toolchain=$VVM_CPU_DIR/gcc11 \
    -DCMAKE_C_FLAGS=--gcc-toolchain=$VVM_CPU_DIR/gcc11 \
    -DCMAKE_PREFIX_PATH=$VVM_CPU_DIR \
    -DHDF5_ROOT=$VVM_CPU_DIR \
    -DADIOS2_USE_MPI=ON \
    -DADIOS2_USE_HDF5=ON \
    -DADIOS2_USE_SST=ON \
    -DADIOS2_USE_Kokkos=OFF \
    -DADIOS2_USE_CUDA=OFF \
    -DBUILD_TESTING=OFF
make -j$(nproc)
make install
cd ../..
```

Verify:

```bash
grep -E "ADIOS2_HAVE_(Kokkos|CUDA|SST|MPI) " \
     $VVM_CPU_DIR/lib64/cmake/adios2/adios2-config-common.cmake
#   set(ADIOS2_HAVE_MPI TRUE)
#   set(ADIOS2_HAVE_SST TRUE)
#   set(ADIOS2_HAVE_CUDA )        <- empty
#   set(ADIOS2_HAVE_Kokkos )      <- empty
```

That file does not record HDF5, so check the generated header instead. HDF5
support is what installs `bp2h5`, the converter described in
[Output](../user-guides/output.md#converting-to-hdf5):

```bash
grep -E "ADIOS2_HAVE_HDF5|ADIOS2_FEATURE_LIST" \
     $VVM_CPU_DIR/include/adios2/common/ADIOSConfig.h
#   #define ADIOS2_HAVE_HDF5
#   #define ADIOS2_FEATURE_LIST ... "HDF5", ...

test -x $VVM_CPU_DIR/bin/bp2h5 && echo "bp2h5 present"
```

Read the `#define`, not the `/* CMake Option: ADIOS2_USE_HDF5=OFF */` comment
directly above it — that comment reports CMake's default, not the value used for
this build.

The CPU BP5 implementation was validated with the unmodified 2.12.1 release.
SST remains enabled so existing SST configurations can still be selected; the
direct BP5 path does not use SST or its transports.

---

## 7. Environment Setup Script

Create `env_setup_cpu.sh` in your workspace and source it before every build or
run.

```bash
#!/bin/bash

# --- 1. Base paths ---
export VVM_CPU_DIR=/path/to/your/cpu/libs
export NVHPC_ROOT=/path/to/nvhpc/Linux_x86_64/24.9

# --- 2. NVHPC compilers (CUDA backend unused) ---
export PATH=$NVHPC_ROOT/compilers/bin:$PATH

# --- 3. MPI (Open MPI built into the CPU prefix) ---
export OPAL_PREFIX=$VVM_CPU_DIR

# --- 4. GCC 11 and CMake ---
export PATH=$VVM_CPU_DIR/gcc11/bin:$VVM_CPU_DIR/cmake/bin:$PATH

# --- 5. All CPU libraries (Kokkos, ADIOS2, HDF5, NetCDF, PnetCDF, libfabric) ---
export PATH=$VVM_CPU_DIR/bin:$PATH
export CPATH=$VVM_CPU_DIR/include
export LIBRARY_PATH=$VVM_CPU_DIR/lib64:$VVM_CPU_DIR/lib:$NVHPC_ROOT/compilers/lib:$NVHPC_ROOT/math_libs/lib64:$VVM_CPU_DIR/gcc11/lib64:$VVM_CPU_DIR/gcc11/lib
export LD_LIBRARY_PATH=$VVM_CPU_DIR/lib64:$VVM_CPU_DIR/lib:$NVHPC_ROOT/compilers/lib:$NVHPC_ROOT/math_libs/lib64:$VVM_CPU_DIR/gcc11/lib64:$VVM_CPU_DIR/gcc11/lib

echo "VVMex CPU Environment Loaded Successfully!"
```

---

## 8. Configure and Build VVMex

Add a CPU preset to `CMakePresets.json`. Substitute your real paths.

```json
{
  "name": "cpu",
  "displayName": "CPU-only (Kokkos OpenMP)",
  "generator": "Unix Makefiles",
  "binaryDir": "${sourceDir}/build_cpu",
  "cacheVariables": {
    "CMAKE_BUILD_TYPE": "Release",

    "NVHPC_DIR": "/path/to/nvhpc/Linux_x86_64/24.9",

    "CMAKE_CXX_COMPILER":     "/path/to/your/cpu/libs/bin/mpic++",
    "CMAKE_C_COMPILER":       "/path/to/your/cpu/libs/bin/mpicc",
    "CMAKE_Fortran_COMPILER": "/path/to/your/cpu/libs/bin/mpifort",

    "CMAKE_CXX_FLAGS":     "--gcc-toolchain=/path/to/your/cpu/libs/gcc11",
    "CMAKE_C_FLAGS":       "--gcc-toolchain=/path/to/your/cpu/libs/gcc11",
    "CMAKE_Fortran_FLAGS": "--gcc-toolchain=/path/to/your/cpu/libs/gcc11",

    "Kokkos_DIR": "/path/to/your/cpu/libs/lib/cmake/Kokkos",
    "ADIOS2_DIR": "/path/to/your/cpu/libs/lib/cmake/adios2",

    "HDF5_DIR":           "/path/to/your/cpu/libs",
    "NETCDF_C_DIR":       "/path/to/your/cpu/libs",
    "NETCDF_Fortran_DIR": "/path/to/your/cpu/libs",
    "PNETCDF_DIR":        "/path/to/your/cpu/libs",

    "VVM_ENABLE_GPU": "OFF",
    "ENABLE_NCCL": "OFF",
    "EAMXX_ENABLE_GPU": "OFF",
    "Kokkos_ENABLE_CUDA": "OFF",

    "VVM_USE_DOUBLE_PRECISION": "ON",
    "SCREAM_DOUBLE_PRECISION": "ON",
    "RRTMGP_USE_DOUBLE_PRECISION": "ON",
    "SCREAM_PACK_SIZE": "1",
    "SCREAM_SMALL_PACK_SIZE": "1",
    "SCREAM_P3_SMALL_KERNELS": "ON"
  }
}
```

The four `*_ENABLE_GPU` / `ENABLE_NCCL` / `Kokkos_ENABLE_CUDA` settings must agree.
`VVM_ENABLE_GPU=OFF` is the master switch; the others are set explicitly so a stale
CMake cache cannot leave one of them on.

Build into `build_cpu`, separate from any GPU tree:

```bash
source env_setup_cpu.sh
export VVM_ROOT=/path/to/VVMex

cmake --preset cpu -DBUILD_TESTS=ON && cmake --build build_cpu -j$(nproc)
```

Expected configure output:

```
-- VVM Execution Backend: CPU (Kokkos OpenMP)
-- Building with Standard MPI support
-- Enabled Kokkos devices: OPENMP;SERIAL
```

Confirm the binary is GPU-free:

```bash
nm -D --undefined-only build_cpu/vvm | grep -ciE 'nccl|cudaMalloc|acc_'   # expect 0
ldd build_cpu/vvm | grep libkokkoscore                                    # your CPU prefix
```

---

## 9. Running

Same workflow as a GPU run apart from the preset. `--cpus` sets the OpenMP threads
per rank and is the main parallelism control:

```bash
./submit.py --local -c "rundata/input_configs/default_cases/<case>.json" \
    --preset "cpu" --compute 2 --nodes 1 --io 2 --cpus 16
```

- `--compute N` — MPI ranks
- `--cpus M` — OpenMP threads per rank
- `--io N` — IO ranks; only meaningful when `output.engine` is `SST`
- Do **not** set `VVM_GPU_LIST`; no GPU mapping is performed

`submit.py` reads the backend and binary path from the preset, so nothing else
changes. A CPU run reports:

```
 Backend: cpu | Binary: /path/to/VVMex/build_cpu/vvm
 [GPUMap] ... source=cpu_backend_no_mapping OMP_NUM_THREADS=16
```

Run length and output cadence come from the case JSON
(`simulation.total_time_s`, `simulation.output_interval_s`), not the command line.

---

## 10. Tests

```bash
ctest --test-dir build_cpu --output-on-failure
```

Expected: `100% tests passed, 0 tests failed out of 27` in about six minutes.

The CPU build is gated against CPU-generated reference data
(`tests/baselines_cpu/`, `tests/references_cpu/`), selected automatically from
`VVM_ENABLE_GPU`.

Threads default to **64 per rank**; change with `-DVVM_TEST_CPU_THREADS=<n>`.
This is a configure-time option and nothing else: CTest stamps
`OMP_NUM_THREADS=set:<n>` onto every CPU test, which overrides whatever the
calling shell exports, so setting `OMP_NUM_THREADS` before `ctest` has no
effect.

Thread count is not one of the axes that changes CPU arithmetic, so raising it
is free. Measured on a 224-core host:

| | 16 threads | 64 threads |
|---|---|---|
| `Run_2dbubble` | 283 s | 108 s |
| `Run_mountain` | 405 s | 134 s |
| default tier, serial | ~14 min | 5.8 min |
| default tier, `ctest -j 224` | — | **3.5 min** |

The 64-thread output is bit-for-bit identical to the 16-thread output over all
272710 values of `2dbubble`, and `mountain` still matches its SHA-256 digest.
On a CPU build each test also declares a `PROCESSORS` weight (ranks x threads)
instead of taking the GPU resource lock, so `ctest -j <cores>` runs them side by
side without oversubscribing — that is where the last third of the speedup comes
from. Lower the thread count if you enable the multirank tier: it runs 4 ranks at
once, so 64 threads per rank peaks at 256. CMake warns when the widest enabled
tier would exceed the host's core count.

Optional tiers are opt-in at configure time:

```bash
cmake --preset cpu -DBUILD_TESTS=ON -DVVM_TEST_MULTIRANK=ON
```

Note: CPU has its own reference data because CPU and GPU results agree only to a
few ulp. The advection cases stay near 1e-13 and would pass the 1e-6 tolerance, but
a convective case amplifies the same seed to ~2e-2 over 120 steps, and the SHA-256
digest checks tolerate nothing. Each backend is gated against itself, and both
remain strict bit-for-bit gates within a backend. The physics tier
(`-DVVM_TEST_PHYSICS=ON`) does not yet ship CPU references.

---

## 11. Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| MPFR fails with `DBL_MAX undeclared` while building GCC | An inherited `CPATH` selects a vendor `float.h` (commonly Intel oneAPI's) instead of GCC's header | Unset `CPATH`, `C_INCLUDE_PATH`, `CPLUS_INCLUDE_PATH`, `LIBRARY_PATH`, and `LD_LIBRARY_PATH`, then recreate the GCC build directory |
| HDF5's `H5timer.c` fails with `DBL_EPSILON is undefined` | A vendor include path selects a non-GCC `float.h`; Intel oneAPI commonly causes this | Unset `CPATH`, `C_INCLUDE_PATH`, `CPLUS_INCLUDE_PATH`, and `LIBRARY_PATH`, run the HDF5 preflight check, then clean and rebuild HDF5 |
| HDF5 links fail with `__extendhfxf2` or `__truncxfhf2` undefined | HDF5's optional `_Float16` support is incompatible with the NVHPC 24.9 and GCC 11 runtime combination | Clean HDF5 and configure with `--disable-nonstandard-feature-float16`; normal `float` and `double` support is unaffected |
| NetCDF or RRTMGP fails with `Invalid dimension ID or name` when opening an RRTMGP coefficient file | NetCDF-C 4.4.1.1 was compiled by the NVHPC-backed MPI wrapper; this local build misreads valid classic NetCDF files | Rebuild NetCDF-C with GCC and the explicit MPI include/link paths from section 3; require the RRTMGP `ncdump -k` check to print `classic`, then rebuild PnetCDF and VVMex |
| NetCDF-C configure finds `-lhdf5` but reports that `hdf5.h` cannot be found | Parallel HDF5's `H5public.h` includes `mpi.h`, but plain GCC was not given the Open MPI include directory | Set `MPI_ROOT` from `command -v mpicc` and add `-I$MPI_ROOT/include`, `-L$MPI_ROOT/lib`, and `LIBS=-lmpi` while retaining `CC=gcc` |
| PnetCDF configure fails because `netcdf_par.h` is missing | GCC-built NetCDF-C did not detect parallel HDF5, so its parallel NetCDF-4 API was not installed | Reconfigure NetCDF-C with the explicit MPI paths, `--enable-netcdf-4`, and parallel HDF5; verify both `NC_HAS_PARALLEL` values are `1`, then configure PnetCDF with `--with-netcdf4=$VVM_CPU_DIR` |
| `pnetcdf-config` says NetCDF-4 is enabled but VVMex still reports `Attempt to use feature that was not turned on` | PnetCDF was rebuilt with `--disable-shared`; the new static archive was installed but VVMex kept loading an older `libpnetcdf.so` | Rebuild PnetCDF with `--enable-shared`, confirm the shared-library timestamp changes, and test the NetCDF-4 input with `ncmpidump -h` before submitting |
| PnetCDF `ncmpidump/vardata.c` fails with `FLT_EPSILON` or `DBL_EPSILON` undefined | An inherited Intel oneAPI path in `CPATH` selects Intel's `float.h` | Reset `CPATH` to `$VVM_CPU_DIR/include`, verify the epsilon preflight, and configure in a new `build-cpu` directory |
| `libmpi_usempif08.la` fails with `Nonrepresentable section on output` | The optional MPI F08 binding produces a linker-incompatible library with this NVHPC/system-linker combination | Configure with `--enable-mpi-fortran=usempi`; VVMex uses the `mpi` module and does not require `mpi_f08` |
| `fs_gpfs_file_set_info.c` fails because a GPFS structure has no `reserved` field | The installed GPFS headers are incompatible with Open MPI 4.1.6's optional GPFS component | Configure with `--enable-mca-no-build=fs-gpfs`; generic MPI-IO remains available |
| `Could not find NVIDIA CPU Math Library` | `NVHPC_DIR` unset or wrong | NVHPC is mandatory even for CPU builds; set `NVHPC_DIR` in the preset |
| Segfault on first `std::cout` in a small test binary | NVHPC compiled against a newer GCC than the `libstdc++` loaded at run time | Pin GCC via `makelocalrc` or `--gcc-toolchain` |
| `undefined reference to std::ios_base_library_init()` when linking a library | Same GCC mismatch, at link time | Add `--gcc-toolchain=$VVM_CPU_DIR/gcc11` to that library's build |
| `version 'GLIBCXX_3.4.30' not found` when starting `vvm` | `CMAKE_Fortran_FLAGS` missing `--gcc-toolchain` | Set it for Fortran too, then rebuild from a deleted tree |
| `#error ... __CUDACC__ macro as expected` | A CUDA Kokkos header directory is on the include path | Keep the CPU prefix separate; do not add a GPU prefix `include/` |
| Tests occupy a GPU | The binary loaded a CUDA Kokkos with the same SONAME | Ensure only the CPU prefix is on `LD_LIBRARY_PATH` |
| `Kokkos::abort: Requested Team Size is too large!` | A CUDA-sized team requested on the OpenMP backend | Handled in VVMex; check local edits that pin team sizes |
| More threads runs slower | MPI bound each rank to a single core | `core_run.sh` gives each rank a `PE=` slice; check for `[Affinity]` messages |
| `libcuda.so.1 => not found` on a driver-less machine | A CUDA-aware MPI pulls `libcuda` via `libfabric` | Rebuild Open MPI with `--without-cuda` (section 2) |
| `make` fails before linking because CMake reports missing `GLIBCXX_3.4.29` or `CXXABI_1.3.13` from `/lib64/libstdc++.so.6` | The bundled CMake loaded the old system C++ runtime | Load the CPU environment first and keep `$VVM_CPU_DIR/gcc11/lib64` ahead of system paths in `LD_LIBRARY_PATH` |
| Unit-test links report many undefined `nc_*` references from `libpnetcdf.so` | This PnetCDF build exposes NetCDF-4 compatibility calls, but the imported CMake target did not propagate NetCDF-C | Link `PnetCDF::pnetcdf` transitively to `NetCDF::netcdf`; the repository CMake target now does this |

To check which GCC ABI a finished binary needs:

```bash
objdump -T build_cpu/vvm | grep -oE 'GLIBCXX_3\.4\.[0-9]+' | sort -uV | tail -1
```

GCC 11 provides up to `GLIBCXX_3.4.29`. Anything higher will fail to start against
a GCC 11 runtime.
