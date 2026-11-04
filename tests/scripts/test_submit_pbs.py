#!/usr/bin/env python3
"""Check PBS resource requests and PALS launching without submitting jobs."""

import contextlib
import importlib.util
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("vvm_submit", ROOT / "submit.py")
submit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(submit)


class PBSSubmissionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="vvm pbs '")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / "tools").mkdir()
        (self.root / "tools/core_run.sh").touch()
        self.config = self.root / "case.json"

    def run_main(self, options, backend="gpu", engine="HDF5"):
        self.config.write_text(json.dumps({
            "output": {"engine": engine, "output_dir": str(self.root / "output")},
        }))
        env = {"VVM_ROOT": str(self.root), "VVM_BACKEND": backend}
        old_cwd = os.getcwd()
        self.addCleanup(os.chdir, old_cwd)
        with patch.object(sys, "argv", ["submit.py", "-c", str(self.config),
                                        "--preset", "derecho", *options]), \
                patch.object(submit, "setup_environment", return_value=env), \
                patch.object(submit, "create_code_snapshot"), \
                patch.object(submit, "write_submit_command"), \
                patch.object(submit, "resolve_partition", return_value=("normal", None)) as resolve, \
                patch.object(submit, "query_node_resources") as query, \
                patch.object(submit.subprocess, "run") as run, \
                contextlib.redirect_stdout(io.StringIO()):
            run.return_value = subprocess.CompletedProcess([], 0, "12345.server\n", "")
            submit.main()
        self.scheduler_calls = run.call_args_list
        return run.call_args_list[0], resolve, query

    def test_pbs_sst_requests_compute_and_io_cpus_but_only_compute_gpus(self):
        call, resolve, query = self.run_main([
            "--scheduler", "pbs", "-A", "PROJECT", "--queue", "develop",
            "--nodes", "2", "--compute", "8", "--io", "4", "--cpus", "10",
            "--pbs-arg=-l", "--pbs-arg=mem=200GB",
            "--out", str(self.root / "stdout.log"),
            "--err", str(self.root / "stderr.log"), "-t", "01:00:00",
        ], engine="SST")
        cmd = call.args[0]
        self.assertEqual(cmd[0], "qsub")
        self.assertIn("-V", cmd)
        self.assertEqual(cmd[cmd.index("-A") + 1], "PROJECT")
        self.assertEqual(cmd[cmd.index("-q") + 1], "develop")
        self.assertIn("select=2:ncpus=60:mpiprocs=6:ompthreads=10:ngpus=4", cmd)
        self.assertIn("place=scatter", cmd)
        self.assertEqual(cmd[-2:], ["-l", "mem=200GB"])
        self.assertEqual(cmd[cmd.index("-o") + 1], str(self.root / "stdout.log"))
        self.assertEqual(cmd[cmd.index("-e") + 1], str(self.root / "stderr.log"))
        self.assertIn("walltime=01:00:00", cmd)
        env = call.kwargs["env"]
        self.assertEqual(env["VVM_TOTAL_TASKS"], "12")
        self.assertEqual(env["VVM_CPUS_PER_NODE"], "60")
        self.assertEqual(env["VVM_MPI_FLAVOR"], "pals")
        self.assertNotIn("sbatch", cmd)
        resolve.assert_not_called()
        query.assert_not_called()
        subprocess.run(["bash", "-n"], input=call.kwargs["input"], text=True, check=True)
        # Execute the generated script as well: the root contains a literal quote.
        subprocess.run(["bash"], input=call.kwargs["input"], text=True, check=True)

    def test_pbs_cpu_has_no_gpu_request_and_supports_shared_nodes(self):
        for flags, placement in [([], "scatter"), (["--exclusive"], "scatter:excl"),
                                 (["--no-exclusive"], "scatter:shared")]:
            with self.subTest(flags=flags):
                call, _, _ = self.run_main([
                    "--scheduler", "pbs", "--compute", "2", *flags,
                ], backend="cpu", engine="BP5")
                cmd = call.args[0]
                self.assertEqual(cmd[cmd.index("-A") + 1], "NMMM0004")
                self.assertEqual(cmd[cmd.index("-q") + 1], "main")
                self.assertIn("walltime=12:00:00", cmd)
                self.assertNotIn("-o", cmd)
                self.assertNotIn("-e", cmd)
                self.assertEqual(call.kwargs["cwd"], str(self.root / "output"))
                self.assertIn("-h", cmd)
                self.assertEqual(self.scheduler_calls[1].args[0], [
                    "qalter", "-o", str(self.root / "output/pbs_12345.out"),
                    "-e", str(self.root / "output/pbs_12345.err"), "12345.server"])
                self.assertEqual(self.scheduler_calls[2].args[0],
                                 ["qrls", "-h", "u", "12345.server"])
                self.assertIn("select=1:ncpus=2:mpiprocs=2:ompthreads=1", cmd)
                self.assertIn(f"place={placement}", cmd)
                self.assertFalse(any("ngpus" in arg for arg in cmd))
                self.assertEqual(call.kwargs["env"]["VVM_GPUS"], "0")

    def test_pbs_log_templates_use_job_number_and_preserve_exit_status(self):
        out_path = self.root / "logs_%j" / "pbs_%j.out"
        err_path = self.root / "logs_%j" / "pbs_%j.log"
        call, _, _ = self.run_main([
            "--scheduler", "pbs", "--out", str(out_path), "--err", str(err_path),
        ])
        self.assertFalse((self.root / "logs_%j").exists())
        (self.root / "tools/core_run.sh").write_text(
            "printf 'model output\\n'\nprintf 'model error\\n' >&2\nexit 17\n")
        qalter = self.scheduler_calls[1].args[0]
        self.assertEqual(qalter, ["qalter", "-o", str(self.root / "logs_12345/pbs_12345.out"),
                                "-e", str(self.root / "logs_12345/pbs_12345.log"),
                                "12345.server"])
        # Emulate PBS opening exactly the paths configured before job release.
        with open(qalter[2], "w") as stdout, open(qalter[4], "w") as stderr:
            result = subprocess.run(["bash"], input=call.kwargs["input"], text=True,
                                    env=dict(os.environ, PBS_JOBID="12345.server"),
                                    stdout=stdout, stderr=stderr)
        self.assertEqual(result.returncode, 17)
        self.assertEqual((self.root / "logs_12345/pbs_12345.out").read_text(), "model output\n")
        self.assertEqual((self.root / "logs_12345/pbs_12345.log").read_text(), "model error\n")
        self.assertFalse((self.root / "logs_%j").exists())

    def test_pbs_mixed_log_paths_only_forward_literal_stream_to_qsub(self):
        for template_option, literal_option in [("--out", "--err"), ("--err", "--out")]:
            with self.subTest(template=template_option):
                literal_path = str(self.root / "literal.log")
                call, _, _ = self.run_main([
                    "--scheduler", "pbs",
                    template_option, str(self.root / "pbs_%j.log"),
                    literal_option, literal_path,
                ])
                cmd = call.args[0]
                template_flag = "-o" if template_option == "--out" else "-e"
                literal_flag = "-o" if literal_option == "--out" else "-e"
                self.assertNotIn(template_flag, cmd)
                self.assertEqual(cmd[cmd.index(literal_flag) + 1], literal_path)
                qalter = self.scheduler_calls[1].args[0]
                self.assertEqual(qalter[qalter.index(template_flag) + 1],
                                 str(self.root / "pbs_12345.log"))
                self.assertEqual(qalter[qalter.index(literal_flag) + 1], literal_path)

    def test_pbs_literal_log_paths_submit_without_hold_or_followup(self):
        call, _, _ = self.run_main([
            "--scheduler", "pbs", "--out", str(self.root / "stdout.log"),
            "--err", str(self.root / "stderr.log"),
        ])
        self.assertNotIn("-h", call.args[0])
        self.assertEqual(len(self.scheduler_calls), 1)

    def test_pbs_explicit_user_hold_is_preserved(self):
        self.run_main(["--scheduler", "pbs", "--pbs-arg=-h"])
        self.assertEqual([call.args[0][0] for call in self.scheduler_calls],
                         ["qsub", "qalter"])

    def test_pbs_log_setup_failure_does_not_release_job(self):
        qalter = ["qalter", "-o", str(self.root / "pbs_12345.out"),
                  "-e", str(self.root / "pbs_12345.err"), "12345.server"]
        with patch.object(submit.subprocess, "run", side_effect=[
            subprocess.CompletedProcess([], 0, "12345.server\n", ""),
            subprocess.CalledProcessError(1, qalter),
        ]) as run, contextlib.redirect_stdout(io.StringIO()), \
                contextlib.redirect_stderr(io.StringIO()) as error:
            with self.assertRaises(subprocess.CalledProcessError):
                submit.run_pbs_submission(["qsub", "-h"], "#!/bin/bash\n", {},
                                          str(self.root),
                                          (str(self.root / "pbs_%j.out"),
                                           str(self.root / "pbs_%j.err")))
        self.assertEqual([call.args[0][0] for call in run.call_args_list],
                         ["qsub", "qalter"])
        self.assertIn("12345.server", error.getvalue())

    def test_slurm_default_remains_sbatch_with_original_account(self):
        call, resolve, _ = self.run_main(["--cpus", "2"])
        cmd = call.args[0]
        self.assertEqual(cmd[0], "sbatch")
        self.assertIn("--account=" + submit.DEFAULT_ACCOUNT, cmd)
        self.assertIn("--time=24:00:00", cmd)
        self.assertIn("--exclusive", cmd)
        self.assertNotIn("input", call.kwargs)
        self.assertNotIn("cwd", call.kwargs)
        resolve.assert_called_once()

    def test_pbs_rejects_empty_account_and_slurm_only_options(self):
        for extra, message in [(["-A", ""], "requires --account"),
                               (["-A", "PROJECT", "--slurm-arg=--mem=0"], "SLURM-only")]:
            with self.subTest(extra=extra), self.assertRaisesRegex(SystemExit, message):
                self.run_main(["--scheduler", "pbs", *extra])

    def test_preset_selects_configured_pals_launcher(self):
        (self.root / "CMakePresets.json").write_text(json.dumps({
            "configurePresets": [{"name": "derecho", "cacheVariables": {
                "MPIEXEC_EXECUTABLE": "/opt/cray/pals/default/bin/mpiexec",
            }}],
        }))
        with patch.object(submit, "get_vvm_root", return_value=str(self.root)), \
                contextlib.redirect_stdout(io.StringIO()):
            env = submit.setup_environment("derecho")
        self.assertEqual(env["VVM_MPI_FLAVOR"], "pals")
        self.assertEqual(env["VVM_MPI_LAUNCHER"], "/opt/cray/pals/default/bin/mpiexec")

    def test_pbs_wizard_replay_retains_scheduler_queue_and_account(self):
        def ask(prompt, default):
            if "Batch scheduler" in prompt:
                return "pbs"
            if "project account" in prompt:
                return "PROJECT"
            if "PBS queue" in prompt:
                return "develop"
            return str(default)

        with patch.object(submit, "get_vvm_root", return_value=str(self.root)), \
                patch.object(submit, "get_available_presets", return_value=["derecho"]), \
                patch.object(submit, "ask", side_effect=ask), \
                patch.object(submit, "ask_path", return_value=str(self.config)), \
                patch.object(submit, "peek_io_engine", return_value="HDF5"), \
                patch.object(submit, "peek_output_dir", return_value="output"), \
                patch.object(sys, "argv", [str(ROOT / "submit.py")]), \
                contextlib.redirect_stdout(io.StringIO()):
            args = submit.interactive_wizard()
        with patch.object(sys, "argv", args.equivalent_argv):
            replay = submit.parse_args()
        self.assertEqual((replay.scheduler, replay.queue, replay.account),
                         ("pbs", "develop", "PROJECT"))
        self.assertTrue(replay.out.endswith("/pbs_%j.out"))
        self.assertTrue(replay.err.endswith("/pbs_%j.err"))
        self.assertEqual(replay.time, "12:00:00")
        self.assertIsNone(replay.exclusive)
        self.assertEqual(replay.cpus, 1)


class PALSLauncherTests(unittest.TestCase):
    def test_pals_launch_and_rank_wrapper_use_pmi_and_pals_ids(self):
        with tempfile.TemporaryDirectory(prefix="vvm pals '") as temp:
            root = Path(temp)
            launcher = root / "mpiexec"
            report = root / "launch.json"
            nodefile = root / "pbs_nodes"
            nodefile.write_text("deg0054\n" * 8 + "deg0072\n" * 8)
            launcher.write_text(
                "#!" + sys.executable + "\nimport json,os,sys\n"
                "with open(os.environ['LAUNCH_REPORT'], 'w') as f:\n"
                " json.dump({'argv':sys.argv[1:], 'env':dict(os.environ)}, f)\n")
            launcher.chmod(0o755)
            env = dict(os.environ, VVM_MPI_FLAVOR="pals", VVM_MPI_LAUNCHER=str(launcher),
                       PBS_JOBID="123.server", PBS_NODEFILE=str(nodefile),
                       VVM_CPUS_PER_NODE="64", VVM_NODES="2",
                       VVM_BACKEND="gpu", VVM_COMPUTE_PER_NODE="4", VVM_IO_PER_NODE="2",
                       VVM_COMPUTE_TASKS="8", VVM_IO_TASKS="4", VVM_TOTAL_TASKS="12",
                       VVM_IO_ENGINE="SST", VVM_GPUS="4", VVM_IO_CPUS="1",
                       VVM_CONFIG_FILE="case with spaces.json", LAUNCH_REPORT=str(report))
            subprocess.run(["bash", str(ROOT / "tools/core_run.sh")], env=env,
                           check=True, capture_output=True, text=True)
            actual = json.loads(report.read_text())
            argv = actual["argv"]
            self.assertEqual(argv[:9], ["--genvall", "--cpu-bind", "none", "--gpu-bind", "none",
                                       "--ppn", "0", "-n", "12"])
            self.assertEqual(argv[9:11], ["--hosts", "deg0054,deg0072"])
            self.assertEqual(argv[-3:], ["case with spaces.json", "--io-tasks", "4"])
            self.assertEqual(actual["env"]["OMP_NUM_THREADS"], "15")
            self.assertFalse(any(arg in argv for arg in ("--mca", "--oversubscribe", "-x")))

            # Run the actual rank wrapper against a fake binary for both roles.
            binary = root / "vvm"
            binary.write_text('#!/bin/bash\nprintf "%s|%s" "$CUDA_VISIBLE_DEVICES" "$OMP_NUM_THREADS"\n')
            binary.chmod(0o755)
            rank_env = dict(actual["env"], VVM_BINARY=str(binary), VVM_GPU_LIST="0,1,2,3")
            fake_bin = root / "bin"
            fake_bin.mkdir()
            (fake_bin / "ldd").write_text("#!/bin/bash\nexit 0\n")
            (fake_bin / "ldd").chmod(0o755)
            rank_env["PATH"] = str(fake_bin) + os.pathsep + rank_env.get("PATH", "")
            for key in ("OMPI_COMM_WORLD_LOCAL_RANK", "OMPI_COMM_WORLD_RANK", "SLURM_LOCALID",
                        "SLURM_PROCID", "PMI_LOCAL_RANK", "PMI_RANK"):
                rank_env.pop(key, None)
            for prefix, local_key, global_key in [
                ("PMI", "PMI_LOCAL_RANK", "PMI_RANK"),
                ("PALS", "PALS_LOCAL_RANKID", "PALS_RANKID"),
            ]:
                for local, global_rank, expected in [(2, 4, "2|"), (4, 8, "|1")]:
                    with self.subTest(source=prefix, rank=global_rank):
                        result = subprocess.run(
                            ["bash", "-c", argv[argv.index("-c") + 1], "--", *argv[-3:]],
                            env=dict(rank_env, **{local_key: str(local), global_key: str(global_rank)}),
                            capture_output=True, text=True, check=True)
                        self.assertTrue(result.stdout.startswith(expected), result.stdout)


if __name__ == "__main__":
    unittest.main()
