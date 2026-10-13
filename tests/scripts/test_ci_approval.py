#!/usr/bin/env python3
"""Exercise the hosted role check without credentials or GitHub writes."""
import json
from pathlib import Path
import subprocess
import unittest

import yaml

ROOT = Path(__file__).resolve().parents[2]
WORKFLOW = yaml.safe_load((ROOT / ".github/workflows/ci-approval.yml").read_text())
SCRIPT = WORKFLOW["jobs"]["approval"]["steps"][0]["with"]["script"]

# Execute the workflow's actual github-script body against a stub API.
HARNESS = r"""
const fs = require('node:fs');
const input = JSON.parse(fs.readFileSync(0, 'utf8'));
const outputs = {}, calls = [];
const github = { rest: { repos: { getCollaboratorPermissionLevel: async (args) => {
  calls.push(args);
  if (input.error) throw { status: input.error };
  return { data: { permission: input.permission } };
} } } };
const core = { setOutput: (key, value) => { outputs[key] = value; },
               info: () => {}, warning: () => {} };
const AsyncFunction = Object.getPrototypeOf(async function () {}).constructor;
const run = new AsyncFunction('github', 'core', 'context', 'process', input.script);
run(github, core, { repo: { owner: 'base-owner', repo: 'base-repo' } },
    { env: { RUN_ACTOR: input.actor } }).then(() => {
  process.stdout.write(JSON.stringify({ outputs, calls }));
}).catch((error) => { console.error(error); process.exitCode = 1; });
"""


class ApprovalTests(unittest.TestCase):
    def run_policy(self, permission=None, actor="contributor", error=None):
        data = dict(script=SCRIPT, actor=actor, permission=permission, error=error)
        result = subprocess.run(["node", "-e", HARNESS], input=json.dumps(data),
                                capture_output=True, text=True, check=True)
        return json.loads(result.stdout)

    def test_verified_admin_runs_automatically(self):
        result = self.run_policy("admin", actor="administrator")
        self.assertEqual(result["outputs"]["admin_actor"], "administrator")
        self.assertEqual(result["calls"], [dict(owner="base-owner", repo="base-repo",
                                               username="administrator")])

    def test_all_non_admin_roles_require_approval(self):
        for permission in ("write", "maintain", "read", "triage", "none", None):
            with self.subTest(permission=permission):
                self.assertEqual(self.run_policy(permission)["outputs"]["admin_actor"], "")

    def test_api_failures_require_approval(self):
        for status in (403, 404, 429, 500):
            with self.subTest(status=status):
                self.assertEqual(self.run_policy(error=status)["outputs"]["admin_actor"], "")

    def test_missing_actor_requires_approval_without_api_call(self):
        result = self.run_policy("admin", actor="")
        self.assertEqual(result["outputs"]["admin_actor"], "")
        self.assertEqual(result["calls"], [])


if __name__ == "__main__":
    unittest.main()
