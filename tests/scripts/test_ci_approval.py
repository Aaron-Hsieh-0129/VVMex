#!/usr/bin/env python3
"""Exercise backend selection and owner approval without credentials or GitHub writes."""
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
    def run_policy(self, actor="contributor", runner_setting="true", fork=False,
                   event="pull_request", author="contributor",
                   source_actor="contributor", ref="refs/heads/topic"):
        payload = {"pull_request": {"user": {"login": author}, "head": {"repo": {
            "full_name": "fork-owner/base-repo" if fork else "base-owner/base-repo",
        }}}} if event == "pull_request" else {}
        data = dict(script=SCRIPT, actor=actor, runner_setting=runner_setting,
                    event=event, payload=payload, source_actor=source_actor, ref=ref)
        result = subprocess.run(["node", "-e", HARNESS], input=json.dumps(data),
                                capture_output=True, text=True, check=True)
        return json.loads(result.stdout)

    def test_dependabot_pr_skips_model_tests_even_when_owner_reruns_it(self):
        for actor in ["dependabot[bot]", "base-owner"]:
            with self.subTest(actor=actor):
                result = self.run_policy(actor=actor, author="dependabot[bot]")
                self.assertEqual(result["outputs"], {"runner_enabled": "false"})
                self.assertEqual(result["calls"], [])

    def test_dependabot_pushes_skip_model_tests(self):
        for actor, ref in [("dependabot[bot]", "refs/heads/topic"),
                           ("base-owner", "refs/heads/dependabot/pip/python")]:
            with self.subTest(actor=actor, ref=ref):
                result = self.run_policy(event="push", source_actor=actor, ref=ref)
                self.assertEqual(result["outputs"], {"runner_enabled": "false"})
                self.assertEqual(result["calls"], [])

    def test_manual_dispatch_can_request_a_dependabot_branch(self):
        result = self.run_policy(actor="base-owner", event="workflow_dispatch",
                                 ref="refs/heads/dependabot/pip/python")
        self.assertEqual(result["outputs"], {"runner_enabled": "true"})

    def test_actor_and_pr_author_never_create_an_approval_bypass(self):
        for actor in ["base-owner", "administrator", "contributor", ""]:
            for author in ["base-owner", "contributor"]:
                with self.subTest(actor=actor, author=author):
                    result = self.run_policy(actor=actor, author=author)
                    self.assertEqual(result["outputs"], {"runner_enabled": "true"})
                    self.assertEqual(result["calls"], [])

    def test_fork_with_unavailable_variables_can_request_owner_review(self):
        result = self.run_policy(fork=True, runner_setting="")
        self.assertEqual(result["outputs"], {"runner_enabled": "true"})
        self.assertEqual(result["calls"], [])

    def test_explicit_disabled_setting_never_launches_a_runner(self):
        for setting in ["false", "FALSE", "invalid"]:
            for fork in [True, False]:
                with self.subTest(setting=setting, fork=fork):
                    result = self.run_policy(fork=fork, runner_setting=setting)
                    self.assertEqual(result["outputs"], {"runner_enabled": "false"})

    def test_missing_setting_only_allows_fork_review_requests(self):
        for event in ["pull_request", "push", "workflow_dispatch"]:
            with self.subTest(event=event):
                result = self.run_policy(event=event, runner_setting="")
                self.assertEqual(result["outputs"], {"runner_enabled": "false"})

    def test_dependabot_updates_are_grouped_and_limit_routine_open_prs(self):
        config = yaml.safe_load((ROOT / ".github/dependabot.yml").read_text())
        self.assertEqual({u["package-ecosystem"] for u in config["updates"]},
                         {"github-actions", "pip"})
        for update in config["updates"]:
            self.assertEqual(update["open-pull-requests-limit"], 1)
            for kind in ["version-updates", "security-updates"]:
                self.assertTrue(any(group.get("applies-to", "version-updates") == kind
                                    and group["patterns"] == ["*"]
                                    for group in update["groups"].values()))

    def test_both_backends_always_require_the_same_owner_review_environment(self):
        for backend, setting in [("cpu", "VVMEX_RUNNER"), ("gpu", "VVMEX_GPU_RUNNER")]:
            with self.subTest(backend=backend):
                workflow = yaml.safe_load(
                    (ROOT / f".github/workflows/{backend}-tests.yml").read_text())
                policy = workflow["jobs"]["approval"]
                job = workflow["jobs"][backend]
                self.assertEqual(policy["with"]["runner_setting"], "${{ vars." + setting + " }}")
                self.assertEqual(workflow["jobs"][backend]["environment"],
                                 {"name": "ci-self-hosted"})
                self.assertEqual(job["needs"], "approval")
                self.assertEqual(job["if"], "needs.approval.outputs.runner_enabled == 'true'")
                self.assertNotIn("admin_actor", str(workflow))
                self.assertNotIn("ci-self-hosted-admin", str(workflow))
                events = workflow.get("on", workflow.get(True))
                self.assertIn("pull_request", events)
                self.assertNotIn("pull_request_target", events)

    def test_basic_checks_stay_on_github_hosted_runners_without_heavy_approval(self):
        checks = yaml.safe_load((ROOT / ".github/workflows/checks.yml").read_text())
        events = checks.get("on", checks.get(True))
        self.assertIn("pull_request", events)
        for job in checks["jobs"].values():
            self.assertEqual(job["runs-on"], "ubuntu-latest")
            self.assertNotIn("environment", job)
        self.assertEqual(checks["permissions"], {"contents": "read"})

    def test_only_scheduled_or_manual_nightly_jobs_can_omit_owner_review(self):
        for path in (ROOT / ".github/workflows").glob("*.yml"):
            workflow = yaml.safe_load(path.read_text())
            for job in workflow["jobs"].values():
                runners = job.get("runs-on", [])
                if "self-hosted" not in runners:
                    continue
                if path.name == "nightly.yml":
                    events = workflow.get("on", workflow.get(True))
                    self.assertEqual(set(events), {"schedule", "workflow_dispatch"})
                else:
                    self.assertEqual(job.get("environment"), {"name": "ci-self-hosted"}, path.name)

    def test_both_backends_keep_cleo_and_multirank_coverage(self):
        for backend in ["cpu", "gpu"]:
            workflow = yaml.safe_load((ROOT / f".github/workflows/{backend}-tests.yml").read_text())
            configure = next(step["run"] for step in workflow["jobs"][backend]["steps"]
                             if step.get("name") == "Configure")
            self.assertIn("-DVVM_TEST_CLEO=ON", configure)
            self.assertIn("-DVVM_TEST_MULTIRANK=ON", configure)


if __name__ == "__main__":
    unittest.main()
