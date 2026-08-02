#!/bin/bash
# Give each MPI rank exactly one visible GPU, then exec the real command.
#
# src/main.cpp calls Kokkos with set_device_id(0) unconditionally and relies on
# the launcher to constrain CUDA_VISIBLE_DEVICES; outside submit.py/core_run.sh
# nothing does that, so every rank would land on the same physical device.
# Several NCCL ranks per device is unsupported and does not merely run slowly --
# it changes results (measured: 4 ranks on 1 GPU differed from 4 ranks on 4 GPUs,
# which was bit-for-bit with 1 rank). Hence the multi-rank tests need real GPUs.
#
# VVM_TEST_GPUS may hold a comma-separated device list (default: 0,1,2,...).
set -u
LOCAL_RANK="${OMPI_COMM_WORLD_LOCAL_RANK:-${SLURM_LOCALID:-${PMI_RANK:-0}}}"

if [ -n "${VVM_TEST_GPUS:-}" ]; then
    IFS=',' read -ra _gpus <<< "$VVM_TEST_GPUS"
    export CUDA_VISIBLE_DEVICES="${_gpus[$(( LOCAL_RANK % ${#_gpus[@]} ))]}"
else
    export CUDA_VISIBLE_DEVICES="$LOCAL_RANK"
fi

exec "$@"
