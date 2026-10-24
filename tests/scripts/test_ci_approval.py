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
                   actor: input.source_actor, ref: input.ref,
                   eventName: input.event, payload: input.payload },
    { env: { RUN_ACTOR: input.actor, RUNNER_SETTING: input.runner_setting } }).then(() => {
  process.stdout.write(JSON.stringify({ outputs, calls }));
}).catch((error) => { console.error(error); process.exitCode = 1; });
"""


class ApprovalTests(unittest.TestCase):
    def run_policy(self, permission=None, actor="contributor", error=None,
                   runner_setting="true", fork=False, event="pull_request",
                   author="contributor", source_actor="contributor", ref="refs/heads/topic"):
        payload = {"pull_request": {"user": {"login": author}, "head": {"repo": {
            "full_name": "fork-owner/base-repo" if fork else "base-owner/base-repo",
        }}}} if event == "pull_request" else {}
        data = dict(script=SCRIPT, actor=actor, permission=permission, error=error,
                    runner_setting=runner_setting, event=event, payload=payload,
                    source_actor=source_actor, ref=ref)
        result = subprocess.run(["node", "-e", HARNESS], input=json.dumps(data),
                                capture_output=True, text=True, check=True)
        return json.loads(result.stdout)

    def test_dependabot_pr_skips_model_tests_even_when_admin_reruns_it(self):
        for actor in ["dependabot[bot]", "administrator"]:
            with self.subTest(actor=actor):
                result = self.run_policy("admin", actor=actor, author="dependabot[bot]")
                self.assertEqual(result["outputs"]["runner_enabled"], "false")
                self.assertEqual(result["outputs"]["admin_actor"], "")
                self.assertEqual(result["calls"], [])

    def test_dependabot_pushes_skip_model_tests(self):
        for source_actor, ref in [("dependabot[bot]", "refs/heads/topic"),
                                  ("administrator", "refs/heads/dependabot/pip/python")]:
            with self.subTest(source_actor=source_actor, ref=ref):
                result = self.run_policy("admin", event="push", source_actor=source_actor, ref=ref)
                self.assertEqual(result["outputs"]["runner_enabled"], "false")
                self.assertEqual(result["calls"], [])

    def test_manual_dispatch_can_test_a_dependabot_branch(self):
        result = self.run_policy("admin", actor="administrator", event="workflow_dispatch",
                                 ref="refs/heads/dependabot/pip/python")
        self.assertEqual(result["outputs"]["runner_enabled"], "true")
        self.assertEqual(result["outputs"]["admin_actor"], "administrator")

    def test_normal_pushes_still_run_model_tests(self):
        result = self.run_policy("admin", actor="administrator", event="push")
        self.assertEqual(result["outputs"]["runner_enabled"], "true")

    def test_dependabot_updates_are_grouped_and_limit_routine_open_prs(self):
        config = yaml.safe_load((ROOT / ".github/dependabot.yml").read_text())
        self.assertEqual({u["package-ecosystem"] for u in config["updates"]},
                         {"github-actions", "pip"})
        for update in config["updates"]:
            with self.subTest(ecosystem=update["package-ecosystem"]):
                self.assertEqual(update["open-pull-requests-limit"], 1)
                groups = list(update["groups"].values())
                for kind in ["version-updates", "security-updates"]:
                    self.assertTrue(any(group.get("applies-to", "version-updates") == kind
                                        and group["patterns"] == ["*"] for group in groups))
        pip = next(u for u in config["updates"] if u["package-ecosystem"] == "pip")
        self.assertEqual(set(pip["directories"]), {"/", "/tools/ci"})

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
