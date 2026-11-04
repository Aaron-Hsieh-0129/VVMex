"""MPI binding arguments shared with the CMake test launchers."""

import os
import shlex


def mpi_bind_args():
    return shlex.split(os.environ.get("VVM_TEST_MPI_BIND_ARGS", "--bind-to none"))
