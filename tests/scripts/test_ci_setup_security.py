#!/usr/bin/env python3
"""Exercise the runner installer's checksum gate without downloads or registration."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
PAYLOAD = "synthetic runner archive"


class RunnerChecksumTests(unittest.TestCase):
    def run_installer(self, version, checksum=None):
        with tempfile.TemporaryDirectory(prefix="vvm-runner-checksum-") as directory:
            base = Path(directory)
            project = base / "project"
            project.mkdir()
            (project / "CMakePresets.json").write_text(json.dumps({
                "configurePresets": [{"name": "fixture", "cacheVariables": {"VVM_ENABLE_GPU": "OFF"}}]
            }))
            tools = base / "bin"
            tools.mkdir()
            stubs = {
                "curl": '''#!/bin/sh
printf called > "$CURL_MARKER"
while [ "$#" -gt 0 ]; do
  if [ "$1" = "-o" ]; then output="$2"; shift 2; else shift; fi
done
printf '%s' "$FIXTURE_PAYLOAD" > "$output"
''',
                "tar": '''#!/bin/sh
printf called > "$TAR_MARKER"
while [ "$#" -gt 0 ]; do
  if [ "$1" = "-C" ]; then output="$2"; shift 2; else shift; fi
done
printf '#!/bin/sh\nexit 42\n' > "$output/config.sh"
chmod +x "$output/config.sh"
''',
                "df": '''#!/bin/sh
printf 'Filesystem 1024-blocks Used Available Capacity Mounted on\n'
printf 'fixture 1073741824 0 1073741824 0%% /\n'
''',
            }
            for name, text in stubs.items():
                path = tools / name
                path.write_text(text)
                path.chmod(0o755)
            env = dict(os.environ, VVM_ROOT=str(project), FIXTURE_PAYLOAD=PAYLOAD,
                       PATH=str(tools) + os.pathsep + os.environ["PATH"],
                       CURL_MARKER=str(base / "curl-called"), TAR_MARKER=str(base / "tar-called"))
            command = ["bash", str(ROOT / "tools/ci/setup_ci_runner.sh"),
                       "--preset", "fixture", "--dir", str(base / "runner"),
                       "--url", "https://example.invalid/fixture", "--token", "synthetic-token",
                       "--runner-version", version, "--no-service"]
            if checksum is not None:
                command += ["--runner-sha256", checksum]
            result = subprocess.run(command, env=env, capture_output=True, text=True, timeout=20)
            return result, (base / "curl-called").exists(), (base / "tar-called").exists()

    def test_bad_checksum_prevents_extraction(self):
        result, downloaded, extracted = self.run_installer("9.9.9", "0" * 64)
        self.assertEqual(result.returncode, 1, result.stderr)
        self.assertTrue(downloaded)
        self.assertFalse(extracted)
        self.assertIn("checksum mismatch", result.stderr)

    def test_matching_checksum_reaches_configuration(self):
        digest = hashlib.sha256(PAYLOAD.encode()).hexdigest()
        result, downloaded, extracted = self.run_installer("9.9.9", digest)
        # The fake config.sh deliberately stops before registration/env setup.
        self.assertEqual(result.returncode, 42, result.stderr)
        self.assertTrue(downloaded)
        self.assertTrue(extracted)

    def test_custom_version_requires_checksum_before_download(self):
        result, downloaded, extracted = self.run_installer("9.9.9")
        self.assertEqual(result.returncode, 2, result.stderr)
        self.assertFalse(downloaded)
        self.assertFalse(extracted)

    def test_invalid_version_is_rejected_before_download(self):
        result, downloaded, extracted = self.run_installer("../other", "0" * 64)
        self.assertEqual(result.returncode, 2, result.stderr)
        self.assertFalse(downloaded)
        self.assertFalse(extracted)


if __name__ == "__main__":
    unittest.main()
