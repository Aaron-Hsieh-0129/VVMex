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
run(github, core, { repo: { owner: 'base-owner', repo: 'base-repo' },
                   eventName: input.event, payload: input.payload },
    { env: { RUN_ACTOR: input.actor, RUNNER_SETTING: input.runner_setting } }).then(() => {
  process.stdout.write(JSON.stringify({ outputs, calls }));
}).catch((error) => { console.error(error); process.exitCode = 1; });
"""


class ApprovalTests(unittest.TestCase):
    def run_policy(self, permission=None, actor="contributor", error=None,
                   runner_setting="true", fork=False, event="pull_request"):
        payload = {"pull_request": {"head": {"repo": {
            "full_name": "fork-owner/base-repo" if fork else "base-owner/base-repo",
        }}}} if event == "pull_request" else {}
        data = dict(script=SCRIPT, actor=actor, permission=permission, error=error,
                    runner_setting=runner_setting, event=event, payload=payload)
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

    def test_first_contributor_fork_reaches_approval_without_variables(self):
        result = self.run_policy("none", fork=True, runner_setting="")
        self.assertEqual(result["outputs"]["runner_enabled"], "true")
        self.assertEqual(result["outputs"]["admin_actor"], "")

    def test_fork_permission_lookup_failure_still_reaches_required_review(self):
        result = self.run_policy(error=403, fork=True, runner_setting="")
        self.assertEqual(result["outputs"]["runner_enabled"], "true")
        self.assertEqual(result["outputs"]["admin_actor"], "")

    def test_administrator_rerun_of_fork_is_enabled_without_variables(self):
        result = self.run_policy("admin", actor="administrator", fork=True,
                                 runner_setting="")
        self.assertEqual(result["outputs"]["runner_enabled"], "true")
        self.assertEqual(result["outputs"]["admin_actor"], "administrator")

    def test_explicit_disabled_setting_never_launches_a_runner(self):
        for setting in ("false", "FALSE", "invalid"):
            for fork in (True, False):
                with self.subTest(setting=setting, fork=fork):
                    result = self.run_policy("admin", fork=fork, runner_setting=setting)
                    self.assertEqual(result["outputs"]["runner_enabled"], "false")
                    self.assertEqual(result["calls"], [])

    def test_missing_setting_only_enables_fork_pull_requests(self):
        for event in ("pull_request", "push", "workflow_dispatch"):
            with self.subTest(event=event):
                result = self.run_policy("admin", event=event, runner_setting="")
                self.assertEqual(result["outputs"]["runner_enabled"], "false")
                self.assertEqual(result["calls"], [])

    def test_both_backends_gate_on_hosted_policy_output(self):
        for backend, setting in (("cpu", "VVMEX_RUNNER"),
                                 ("gpu", "VVMEX_GPU_RUNNER")):
            with self.subTest(backend=backend):
                workflow = yaml.safe_load(
                    (ROOT / f".github/workflows/{backend}-tests.yml").read_text())
                policy = workflow["jobs"]["approval"]
                job = workflow["jobs"][backend]
                self.assertNotIn("if", policy)
                self.assertEqual(policy["with"]["runner_setting"],
                                 "${{ vars." + setting + " }}")
                self.assertEqual(job["needs"], "approval")
                self.assertEqual(job["if"],
                                 "needs.approval.outputs.runner_enabled == 'true'")
                self.assertIn("'ci-self-hosted'", job["environment"]["name"])
                self.assertIn("== github.triggering_actor", job["environment"]["name"])

    def test_both_backends_run_cleo_and_multirank_tests(self):
        for backend in ("cpu", "gpu"):
            with self.subTest(backend=backend):
                workflow = yaml.safe_load(
                    (ROOT / f".github/workflows/{backend}-tests.yml").read_text())
                configure = next(step["run"] for step in workflow["jobs"][backend]["steps"]
                                 if step.get("name") == "Configure")
                self.assertIn("-DVVM_TEST_CLEO=ON", configure)
                self.assertIn("-DVVM_TEST_MULTIRANK=ON", configure)


if __name__ == "__main__":
    unittest.main()
