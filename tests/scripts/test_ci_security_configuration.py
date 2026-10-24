#!/usr/bin/env python3
"""Check dependency pinning, workflow parsing and Pages artifact boundaries."""
from pathlib import Path
import os
import subprocess
import tempfile
import unittest

import yaml

ROOT = Path(__file__).resolve().parents[2]


class UniqueKeysLoader(yaml.SafeLoader):
    def construct_mapping(self, node, deep=False):
        keys = set()
        for key, _ in node.value:
            name = self.construct_object(key, deep=deep)
            if name in keys:
                raise ValueError(f"Duplicate YAML key: {name}")
            keys.add(name)
        return super().construct_mapping(node, deep=deep)


class WorkflowSecurityTests(unittest.TestCase):
    def workflows(self):
        for path in (ROOT / ".github/workflows").glob("*.yml"):
            yield path.name, yaml.load(path.read_text(), Loader=UniqueKeysLoader)

    def test_duplicate_workflow_keys_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "Duplicate YAML key"):
            yaml.load("with:\n  a: 1\nwith:\n  b: 2\n", Loader=UniqueKeysLoader)

    def test_external_actions_are_pinned_and_checkout_credentials_not_persisted(self):
        for name, workflow in self.workflows():
            for job in workflow["jobs"].values():
                for step in job.get("steps", []):
                    ref = step.get("uses", "")
                    if ref and not ref.startswith("./"):
                        self.assertRegex(ref, r"^[^@]+@[0-9a-f]{40}$", name)
                    if ref.startswith("actions/checkout@"):
                        self.assertIs(step.get("with", {}).get("persist-credentials"), False, name)

    def test_pages_build_does_not_receive_deployment_permissions(self):
        workflow = dict(self.workflows())["docs.yml"]
        self.assertEqual(workflow["permissions"], {"contents": "read"})
        self.assertEqual(workflow["jobs"]["deploy"]["permissions"]["pages"], "write")

    def test_pages_archive_accepts_regular_files_and_rejects_symlinks(self):
        workflow = dict(self.workflows())["docs.yml"]
        command = next(step["run"] for step in workflow["jobs"]["build"]["steps"]
                       if step.get("name") == "Archive Pages artifact")
        for symlink in [False, True]:
            with self.subTest(symlink=symlink), tempfile.TemporaryDirectory() as directory:
                base = Path(directory)
                site = base / "site"
                site.mkdir()
                outside = base / "private"
                outside.write_text("synthetic private file")
                if symlink:
                    (site / "index.html").symlink_to(outside)
                else:
                    (site / "index.html").write_text("<html>public</html>")
                result = subprocess.run(["bash", "-e", "-o", "pipefail", "-c", command],
                                        cwd=base, env={**os.environ, "RUNNER_TEMP": str(base)},
                                        capture_output=True, text=True)
                self.assertEqual(result.returncode == 0, not symlink, result.stderr)
                self.assertEqual((base / "artifact.tar").exists(), not symlink)


if __name__ == "__main__":
    unittest.main()
