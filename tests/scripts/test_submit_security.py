#!/usr/bin/env python3
"""Check submission replay quoting and snapshot confidentiality in temporary repos."""
import contextlib
import importlib.util
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("vvm_submit", ROOT / "submit.py")
submit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(submit)


class SubmissionSecurityTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="vvm submission '")
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)
        self.repo = self.directory / "repo"
        self.repo.mkdir()

    def git(self, *args):
        return subprocess.check_output(
            ["git", "--literal-pathspecs", "-c", "core.hooksPath=/dev/null", "-c", "user.name=Test",
             "-c", "user.email=test@example.invalid", *args], cwd=self.repo,
            stderr=subprocess.DEVNULL, text=True)

    def snapshot(self, inputs=(), external=False):
        output = self.directory / "snapshot"
        config = self.repo / "case.json"
        config.write_text("{}\n")
        with contextlib.redirect_stdout(io.StringIO()):
            submit.create_code_snapshot(str(self.repo), str(output), str(config),
                                        [str(p) for p in inputs], "output",
                                        include_external_inputs=external)
        return output

    def initialize_repo(self):
        self.git("init", "-q")
        (self.repo / "model.cpp").write_text("original\n")
        self.git("add", "model.cpp")
        self.git("commit", "-qm", "initial")

    def test_snapshot_keeps_local_edits_without_untracked_or_sensitive_files(self):
        self.initialize_repo()
        (self.repo / ".env").write_text("synthetic secret\n")
        (self.repo / "credentials.json").write_text("synthetic credentials\n")
        self.git("add", "-f", ".env", "credentials.json")
        self.git("commit", "-qm", "sensitive fixture")
        (self.repo / ".env").write_text("changed synthetic secret\n")
        (self.repo / "untracked.txt").write_text("untracked private content\n")
        (self.repo / "model.cpp").write_text("local edit\n")
        self.git("config", "remote.origin.url", "https://user:synthetic-token@example.invalid/repo")
        output = self.snapshot()
        self.assertEqual((output / "model.cpp").read_text(), "local edit\n")
        self.assertTrue((output / "case.json").is_file())
        for name in [".env", "credentials.json", "untracked.txt"]:
            self.assertFalse((output / name).exists(), name)
        metadata = (output / submit.SNAPSHOT_VERSION_FILE).read_text()
        self.assertNotIn("synthetic-token", metadata)
        history = subprocess.check_output(
            ["git", "-C", str(output), "show", "HEAD:model.cpp"], text=True)
        self.assertEqual(history, "original\n")
        patch_text = (output / submit.SNAPSHOT_PATCH_FILE).read_text()
        self.assertIn("local edit", patch_text)
        self.assertNotIn("synthetic secret", patch_text)

    def test_snapshot_does_not_follow_tracked_symlinks(self):
        self.initialize_repo()
        outside = self.directory / "private.txt"
        outside.write_text("synthetic private data\n")
        (self.repo / "linked.txt").symlink_to(outside)
        self.git("add", "linked.txt")
        self.git("commit", "-qm", "link fixture")
        output = self.snapshot()
        self.assertFalse((output / "linked.txt").exists())

    def test_snapshot_treats_git_pathspec_syntax_and_whitespace_literally(self):
        self.initialize_repo()
        for name in [" leading.cpp", ":(glob)**"]:
            (self.repo / name).write_text("source\n")
            self.git("add", name)
        (self.repo / ".env").write_text("synthetic credential\n")
        self.git("add", "-f", ".env")
        self.git("commit", "-qm", "literal path fixture")
        (self.repo / ".env").write_text("changed synthetic credential\n")
        (self.repo / " leading.cpp").write_text("local source\n")
        output = self.snapshot()
        self.assertEqual((output / " leading.cpp").read_text(), "local source\n")
        self.assertTrue((output / ":(glob)**").is_file())
        self.assertNotIn("synthetic credential", (output / submit.SNAPSHOT_PATCH_FILE).read_text())

    def test_external_inputs_require_opt_in_and_cannot_overwrite_source(self):
        self.initialize_repo()
        outside = self.directory / "profile.txt"
        outside.write_text("synthetic profile\n")
        collision = self.directory / "model.cpp"
        collision.write_text("replacement content\n")
        credential = self.directory / "credentials.json"
        credential.write_text("synthetic credentials\n")
        output = self.snapshot([outside])
        self.assertFalse((output / outside.name).exists())
        output = self.snapshot([outside, collision, credential], external=True)
        self.assertEqual((output / outside.name).read_text(), outside.read_text())
        self.assertEqual((output / "model.cpp").read_text(), "original\n")
        self.assertFalse((output / credential.name).exists())

    def test_missing_git_checkout_fails_before_removing_existing_snapshot(self):
        output = self.directory / "snapshot"
        output.mkdir()
        (output / "keep").write_text("keep")
        with self.assertRaisesRegex(RuntimeError, "Git checkout"):
            self.snapshot()
        self.assertTrue((output / "keep").exists())

    def test_replay_preserves_literals_for_cli_and_interactive_arguments(self):
        script = self.repo / "submit '$ unusual.py"
        report = self.directory / "argv.json"
        script.write_text(
            "#!" + sys.executable + "\nimport json,os,sys\n"
            "with open(os.environ['SECURITY_ARGV_REPORT'], 'w') as f:\n"
            " json.dump({'argv':sys.argv[1:], 'gpu':os.environ.get('VVM_GPU_LIST')},f)\n")
        script.chmod(0o755)
        marker = self.directory / "must-not-exist"
        hostile = f"case ' $(touch \"{marker}\") `printf bad` ; name.json"
        env = dict(os.environ, SECURITY_ARGV_REPORT=str(report))
        output = self.directory / "output"
        output.mkdir()
        for interactive in [False, True]:
            with self.subTest(interactive=interactive):
                args = SimpleNamespace(equivalent_command="untrusted legacy string")
                if interactive:
                    def ask(prompt, default=None):
                        if "Run locally" in prompt:
                            return "Y"
                        return str(default)
                    with patch.object(submit, "get_vvm_root", return_value=str(self.repo)), \
                            patch.object(submit, "get_available_presets", return_value=[]), \
                            patch.object(submit, "peek_io_engine", return_value="HDF5"), \
                            patch.object(submit, "peek_output_dir", return_value="output"), \
                            patch.object(submit, "ask_path", return_value=hostile), \
                            patch.object(submit, "ask", side_effect=ask), \
                            patch.dict(os.environ, {"VVM_GPU_LIST": hostile}), \
                            patch.object(sys, "argv", [str(script)]), \
                            contextlib.redirect_stdout(io.StringIO()):
                        args = submit.interactive_wizard()
                with patch.object(sys, "argv", [str(script), "-c", hostile]), \
                        contextlib.redirect_stdout(io.StringIO()):
                    submit.write_submit_command(str(output), str(self.repo), args)
                subprocess.run(["bash", str(output / submit.SUBMIT_COMMAND_FILE)],
                               env=env, check=True, capture_output=True, text=True)
                actual = json.loads(report.read_text())
                self.assertEqual(actual["argv"][actual["argv"].index("-c") + 1], hostile)
                if interactive:
                    self.assertEqual(actual["gpu"], hostile)
                self.assertFalse(marker.exists())


if __name__ == "__main__":
    unittest.main()
