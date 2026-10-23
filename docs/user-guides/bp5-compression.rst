Lossless BP5 compression
=======================

Add the shared settings to the ``output`` section::

    "compression": {
      "enabled": true,
      "level": 1,
      "minimum_bytes": 1048576
    }

These settings also apply to HDF5 and SST-to-HDF5 output. Optional BP5
codec and thread overrides belong in the existing ``output.bp5`` section::

    "compression": {
      "codec": "zstd",
      "threads": 1
    }

Explicit fields in ``output.bp5.compression`` override the shared values
field by field. Existing engine-specific configurations remain supported.

This uses ADIOS2's native ``blosc`` operator, backed by Blosc2, with byte
shuffle and lossless Zstd. ``lz4`` and ``zlib`` are also supported codec
choices, provided the Blosc2 library was built with them. Level must be
0--9 (0 disables compression inside Blosc); threads must be 1--32767.
Compression defaults to disabled. Output precision remains a separate
setting: converting float64 state to float32 output still loses precision.

The operator is attached to two-dimensional and higher-dimensional field
datasets. Coordinates, one-dimensional profiles, time and the exact 64-bit
model step remain uncompressed. Compressed fields use contiguous host
staging buffers on both CPU and GPU runs. Threads are per MPI writer rank;
start with one to avoid multiplying CPU demand across all ranks.

``minimum_bytes`` is the threshold for each MPI rank's field block, rather
than the global dataset size used by HDF5 output. Blosc stores smaller
blocks without compression, inside its normal operator framing. Ranks with
empty selections keep using the existing BP5 empty-rank behaviour.

ADIOS2 must be built with ``ADIOS2_USE_Blosc2=ON`` and a Blosc2 library
supporting the selected codec. Rebuild VVMex against that installation.
The current installation under ``/data/Aaron/libs_VVMex_269`` has
``ADIOS2_HAVE_BLOSC2`` disabled. Enabling the setting on that build raises
a clear startup error before the existing output dataset can be replaced.

ADIOS2 automatically decompresses BP5 variables when they are read, so the
existing BP5 restart code does not need a manual extraction step. Readers,
including an ADIOS2-backed GrADS build, must also have Blosc2 support.
The BP5 format and its reading requirements remain different from HDF5.

Compression settings are included in the existing cross-rank consistency
check and printed in the writer startup log. Append keeps the previous
steps and applies the selected operator to newly written blocks. Existing
compression defaults and simulation configurations are unchanged.

Validation
----------

``test_bp5_output_config`` covers parsing, invalid values, the packing
requirement, defaults and the parameters passed to ADIOS2. With a
compression-enabled ADIOS2 build, also run the compressed BP5 integration
test and the normal restart/append tests before production use. HDF5 size
and timing measurements do not establish the ratio or throughput of BP5
Blosc2; a representative BP5 benchmark is still required.
