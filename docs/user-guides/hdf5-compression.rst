Lossless compressed HDF5 output
==============================

Enable in the existing ``output`` section of a run configuration::

    "compression": {
      "enabled": true,
      "level": 1,
      "minimum_bytes": 1048576
    }

This shared ``output.compression`` block also applies when the output
engine is BP5. The older ``output.hdf5_compression`` block is still
accepted; explicit fields in it override the shared values.

This option applies to both direct HDF5 output and the SST-to-HDF5 I/O
server. ADIOS2 still handles SST transport and the field definitions;
the optional sink uses the native parallel HDF5 API because ADIOS2's
HDF5 engine does not accept compression operators. No ADIOS2 rebuild
or external HDF5 filter plugin is needed.

The sink writes shuffle, gzip, and Fletcher32 chunks immediately. Arrays
with at least two dimensions and at least ``minimum_bytes`` bytes are
compressed; scalars, coordinates and smaller datasets retain contiguous
storage. The default chunk shape is (1,128,512) in three dimensions and
(128,512) in two dimensions, clipped to the global shape. Higher-rank
arrays use one cell in each additional leading dimension. Filters are
lossless with respect to the configured output precision; choosing
float32 output for a float64 simulation is still a separate conversion.

Names, clocks, root metadata, field metadata and dimension scales are
preserved for the existing restart reader and GrADS descriptors. Reads
decompress transparently. Compression is disabled by default. Rebuild
VVMex to use the new sink; an already running process keeps its old writer.

All ranks use collective data and metadata operations, including ranks
with an empty selection. On the tested Open MPI 4.1.7 stack, OMPIO stalls
on successive filtered writes, whereas ROMIO completes. ``submit.py``
and ``tools/core_run.sh`` select ROMIO automatically when this option is
enabled. For a manual multi-rank Open MPI launch, use
``mpirun --mca io romio321 ...``. The writer checks this setting before
opening output. Other MPI implementations use their normal MPI-IO backend.

Validation
----------

The standalone ``test_compressed_hdf5`` checks bitwise float32/float64
readback, the filter pipeline, nested coordinates, a 64-bit clock above
2^53, chunks shared across ranks, and ranks with no selected cells. Its
CTest registrations cover one and four ranks. Full model restart and
SST streaming integration should also be checked before production use.

A 2026-10-23 benchmark sampled eight 128x512 chunks from each of 63
numeric fields in completed output 780 (126 MiB raw). Direct gzip-1
and post-processing gzip-1 both produced about 38 MiB, with byte-identical
readback. Gzip-4 saved only another 0.72 MiB on that sample. The sample
contains many near-zero fields and is not a full-output size estimate;
the previous full-file result of approximately 50 percent savings is
the better capacity estimate. Direct compression avoids the intermediate
uncompressed file, rather than improving the underlying compression ratio.
