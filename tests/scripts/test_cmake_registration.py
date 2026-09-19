#!/usr/bin/env python3
"""Exercise suite switches and runtime policy without building the model.

Imported placeholder dependencies let CMake generate the real test registry;
this checks registration only, not linking or numerical model behavior.
"""
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]


class RegistrationTests(unittest.TestCase):
    def test_suite_selection_and_resources(self):
        with tempfile.TemporaryDirectory(prefix="vvm-test-registration-") as tmp:
            source = Path(tmp)
            (source / "FindMPI.cmake").write_text(
                'set(MPI_FOUND TRUE)\nset(MPIEXEC_EXECUTABLE /usr/bin/mpiexec)\n'
                'set(MPIEXEC_NUMPROC_FLAG -n)\n')
            (source / "CMakeLists.txt").write_text(f'''
cmake_minimum_required(VERSION 3.22)
project(TestRegistration LANGUAGES CXX)
list(PREPEND CMAKE_MODULE_PATH "${{CMAKE_CURRENT_SOURCE_DIR}}")
enable_testing()
foreach(dep Kokkos::kokkos MPI::MPI_CXX HDF5::HL PnetCDF::pnetcdf
            NetCDF::netcdf adios2::adios2)
    add_library(${{dep}} INTERFACE IMPORTED)
endforeach()
add_executable(vvm IMPORTED)
set_target_properties(vvm PROPERTIES IMPORTED_LOCATION /usr/bin/true)
set(CMAKE_SOURCE_DIR "{ROOT.as_posix()}")
add_subdirectory("{ROOT.as_posix()}/tests" tests)
''')
            build = source / "build"

            def registry(gpu=False, **switches):
                flags = dict(RLL=False, RLL_PHYSICS=False, REGRESSION=True,
                             BP5=False, SST=False, PHYSICS=False, MULTIRANK=False)
                flags.update(switches)
                command = ["cmake", "-S", str(source), "-B", str(build),
                           f"-DVVM_ENABLE_GPU={'ON' if gpu else 'OFF'}",
                           "-DVVM_TEST_CPU_THREADS=2"]
                command += [f"-DVVM_TEST_{k}={'ON' if v else 'OFF'}"
                            for k, v in flags.items()]
                result = subprocess.run(command, env={**os.environ, "VVM_ROOT": str(ROOT)},
                                        capture_output=True, text=True)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                data = subprocess.check_output(
                    ["ctest", "--test-dir", str(build), "--show-only=json-v1"], text=True)
                return {t["name"]: t for t in json.loads(data)["tests"]}

            def props(test):
                return {p["name"]: p["value"] for p in test["properties"]}

            default = registry()
            self.assertNotIn("test_jung2019_shared_model", default)
            self.assertFalse(any(n.startswith("test_rll_") for n in default))
            self.assertIn("test_regular_latlon_turbulence", default)
            self.assertIn("Run_advection_u", default)
            vertical = props(default["test_vertical_elliptic_solver_rll_r1"])
            self.assertIn("OMP_NUM_THREADS=set:2", vertical["ENVIRONMENT_MODIFICATION"])

            dynamics = registry(RLL=True)
            self.assertEqual(set(dynamics) - set(default),
                             {"test_jung2019_shared_model", "test_rll_mountain_shared_model"})
            for name in set(dynamics) - set(default):
                p = props(dynamics[name])
                self.assertIn("rll", p["LABELS"])
                self.assertEqual(p["PROCESSORS"], 2)
                self.assertNotIn("RESOURCE_LOCK", p)

            physics = registry(RLL_PHYSICS=True)
            added = set(physics) - set(default)
            self.assertEqual(len(added), 8)
            self.assertIn("test_rll_periodic_rest_model", added)
            for name in added:
                self.assertIn("rll-physics", props(physics[name])["LABELS"])
            self.assertEqual(physics["test_rll_periodic_rest_model"]["command"][-2:],
                             ["--periodic", "--rest"])

            bp5 = registry(BP5=True)
            self.assertNotIn("test_rll_bp5_history", bp5)
            both = registry(RLL=True, BP5=True)
            self.assertIn("test_rll_bp5_history", both)

            all_gpu = registry(gpu=True, RLL=True, RLL_PHYSICS=True,
                               BP5=True, SST=True, PHYSICS=True, MULTIRANK=True)
            for name in added | (set(dynamics) - set(default)):
                self.assertEqual(props(all_gpu[name])["RESOURCE_LOCK"], ["gpu"])
            device = all_gpu["test_horizontal_scalar_gradient"]
            self.assertTrue(device["command"][0].endswith("one_gpu_per_rank.sh"))
            self.assertEqual(props(device)["RESOURCE_LOCK"], ["gpu"])
            self.assertNotIn("RESOURCE_LOCK", props(all_gpu["test_process_scheduling"]))
            self.assertIn("test_vertical_elliptic_solver_rll_r4", all_gpu)
            self.assertIn("Run_physics_rcemip", all_gpu)

            # Reconfigure the same build back to OFF: no stale registrations.
            small = registry(REGRESSION=False)
            self.assertNotIn("Run_advection_u", small)
            self.assertNotIn("Prep_grid_configuration_equivalence", small)
            self.assertIn("test_regular_latlon_turbulence", small)
            self.assertFalse(any(n.startswith("test_rll_") for n in small))


if __name__ == "__main__":
    unittest.main()
