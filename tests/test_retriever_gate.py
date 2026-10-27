import json
import contextlib
import datetime as dt
import io
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import types
import unittest
from unittest.mock import patch

import retriever_gate as gate


def experiment(objective="latency", minimum_gain=0.15):
    ids = [f"q{i}" for i in range(10)]
    manifest = {
        "version": 2, "prepared_at": "2026-10-27T10:00:00+00:00", "plan_commit": "committed",
        "plan": {"candidate": "rerank-new", "production": "v2-qwen", "production_reranker": "none",
                 "objective": objective, "minimum_gain": minimum_gain, "rounds": 4},
        "expected_ids": ids, "semantic_ids": ids,
    }
    artifacts = []
    for rd in range(1, 5):
        for arm in ("production", "candidate"):
            rows = [{"id": id, "category": "hard_fact", "passed": True, "answer_passed": True,
                     "response": "A finished answer", "reasons": [], "judge": {"pass": True},
                     "seconds": 10 if arm == "production" else 8, "tool_calls": [],
                     "usage": {"input_tokens": 100}, "cost_usd": 0.01,
                     "serving_usage": {"input_tokens": 100}, "serving_cost_usd": 0.01} for id in ids]
            artifact = {"meta": {"partial": False, "tier": "full", "judge": True,
                                  "started": "2026-10-27T11:00:00+00:00", "model": "same-model",
                                  "prompt_sha256": "same-prompt"},
                        "results": rows, "summary": {"passed": 10, "total": 10}}
            artifacts.append(({"arm": arm, "round": rd}, artifact))
    return manifest, artifacts


def alter(artifacts, arm, update):
    for spec, artifact in artifacts:
        if spec["arm"] == arm:
            update(artifact)
            artifact["summary"]["passed"] = sum(r["passed"] for r in artifact["results"])


class PromotionTests(unittest.TestCase):
    def test_repeatable_latency_gain_with_quality_held_wins(self):
        result = gate.evaluate(*experiment())
        self.assertTrue(result["passed"])
        self.assertAlmostEqual(result["observed_gain"], 0.2)

    def test_tie_and_smaller_gain_do_not_justify_extra_stage(self):
        for seconds in (10, 9.9, 9):
            manifest, artifacts = experiment()
            alter(artifacts, "candidate", lambda a: [r.update(seconds=seconds) for r in a["results"]])
            self.assertFalse(gate.evaluate(manifest, artifacts)["passed"])

    def test_faster_but_worse_quality_loses(self):
        manifest, artifacts = experiment()
        alter(artifacts, "candidate", lambda a: a["results"][0].update(passed=False, answer_passed=False,
              reasons=["missing 'required fact'"], judge={"pass": False}))
        result = gate.evaluate(manifest, artifacts)
        self.assertFalse(result["passed"])
        self.assertTrue(any("regressed" in b for b in result["blockers"]))

    def test_one_fallback_blocks_an_otherwise_winning_candidate(self):
        manifest, artifacts = experiment()
        artifacts[1][1]["results"][0].update(response="Here are fallback links",
                                             reasons=["answer did not finish (finish reason 'fallback')"])
        result = gate.evaluate(manifest, artifacts)
        self.assertFalse(result["passed"])
        self.assertEqual(result["metrics"]["candidate"]["completion_failures"], 1)

    def test_tail_regression_blocks_median_gain(self):
        manifest, artifacts = experiment()
        alter(artifacts, "candidate", lambda a: a["results"][-1].update(seconds=20))
        self.assertFalse(gate.evaluate(manifest, artifacts)["passed"])

    def test_benefit_must_hold_in_each_round(self):
        manifest, artifacts = experiment()
        for row in artifacts[1][1]["results"]:
            row["seconds"] = 10
        result = gate.evaluate(manifest, artifacts)
        self.assertFalse(result["passed"])
        self.assertTrue(any("every paired round" in b for b in result["blockers"]))

    def test_quality_gain_requires_semantic_improvement(self):
        manifest, artifacts = experiment("quality", 1)
        self.assertFalse(gate.evaluate(manifest, artifacts)["passed"])
        alter(artifacts, "production", lambda a: a["results"][0].update(passed=False, answer_passed=False,
              reasons=["judge: incomplete explanation"], judge={"pass": False}))
        self.assertTrue(gate.evaluate(manifest, artifacts)["passed"])

    def test_reliability_objective_counts_completions(self):
        manifest, artifacts = experiment("reliability", 1)
        alter(artifacts, "production", lambda a: a["results"][0].update(passed=False, answer_passed=False,
              reasons=["answer did not finish (finish reason 'fallback')"]))
        self.assertTrue(gate.evaluate(manifest, artifacts)["passed"])

    def test_zero_or_missing_cost_telemetry_cannot_win_cost_comparison(self):
        manifest, artifacts = experiment("cost", 0.15)
        alter(artifacts, "candidate", lambda a: [r.update(serving_cost_usd=0, serving_usage={}) for r in a["results"]])
        with self.assertRaisesRegex(ValueError, "metered costs"):
            gate.evaluate(manifest, artifacts)
        manifest, artifacts = experiment("cost", 0.15)
        alter(artifacts, "candidate", lambda a: [r.update(serving_cost_usd=0.008) for r in a["results"]])
        self.assertTrue(gate.evaluate(manifest, artifacts)["passed"])

    def test_judge_cost_cannot_be_mistaken_for_serving_cost(self):
        manifest, artifacts = experiment("cost", 0.15)
        for _, artifact in artifacts:
            for row in artifact["results"]:
                row.pop("serving_cost_usd")
                row.pop("serving_usage")
        with self.assertRaisesRegex(ValueError, "metered costs"):
            gate.evaluate(manifest, artifacts)

    def test_missing_or_incomparable_evidence_is_rejected(self):
        for change in (
            lambda m, a: a.pop(),
            lambda m, a: a[1][0].update(arm="production"),
            lambda m, a: a[1][1]["results"].pop(),
            lambda m, a: a[1][1]["results"][0].update(id="q1"),
            lambda m, a: a[1][1]["meta"].update(partial=True),
            lambda m, a: a[1][1]["meta"].update(judge=False),
            lambda m, a: a[1][1]["results"][0].update(judge=None),
            lambda m, a: a[1][1]["meta"].update(prompt_sha256="changed"),
            lambda m, a: a[1][1]["meta"].update(model="changed"),
            lambda m, a: a[1][1]["meta"].update(started="2026-10-26T10:00:00+00:00"),
            lambda m, a: a[1][1]["results"][0].update(seconds=float("nan")),
            lambda m, a: a[1][1]["summary"].update(passed=0),
            lambda m, a: m.pop("version"),
        ):
            manifest, artifacts = experiment()
            change(manifest, artifacts)
            with self.subTest(change=change), self.assertRaises((ValueError, KeyError)):
                gate.evaluate(manifest, artifacts)

    def test_rounds_and_gain_cannot_be_zero_or_trivial(self):
        plan = experiment()[0]["plan"]
        for key, value in (("rounds", 2), ("rounds", 3), ("rounds", True),
                           ("minimum_gain", 0), ("minimum_gain", 0.001), ("minimum_gain", float("nan"))):
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                gate.validate_plan({**plan, key: value})

    def test_artifact_change_invalidates_replay(self):
        manifest, artifacts = experiment()
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            manifest["runs"] = []
            for i, (spec, artifact) in enumerate(artifacts):
                path = root / f"run{i}.json"
                gate.save(path, artifact)
                manifest["runs"].append({**spec, "artifact": str(path), "sha256": gate.digest(path)})
            path = root / "manifest.json"
            manifest.update(root=str(root), plan_path="plan.json")
            gate.save(path, manifest)
            with patch.object(gate, "git", return_value=json.dumps(manifest["plan"])):
                self.assertTrue(gate.evaluate_file(path)["passed"])
                gate.save(root / "run1.json", {"edited": True})
                with self.assertRaisesRegex(ValueError, "measurement changed"):
                    gate.evaluate_file(path)

    def test_posthoc_manifest_plan_is_rejected(self):
        manifest, _ = experiment()
        original = json.dumps(manifest["plan"])
        manifest.update(root="/irrelevant", plan_path="plan.json", runs=[])
        manifest["plan"]["minimum_gain"] = 0.05
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "manifest.json"
            gate.save(path, manifest)
            with patch.object(gate, "git", return_value=original):
                with self.assertRaisesRegex(ValueError, "committed declaration"):
                    gate.evaluate_file(path)

    def test_prepare_freezes_committed_plan_and_refuses_uncommitted_edits(self):
        plan = experiment()[0]["plan"]
        committed = json.dumps(plan)
        current = {"production": {"path": "/models/v2-qwen"}, "production_reranker": None,
                   "candidate": {"path": "/models/new"}}
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "evals").mkdir()
            (root / "evals/golden.yaml").write_text('[{"id":"q0","expected_summary":"A supported answer"}]')
            path = root / "plan.json"
            path.write_text(committed)
            def git(_, *args):
                return committed if args[0] == "show" else "committed"
            with patch.object(gate, "git", side_effect=git), patch.object(gate, "state", return_value=current), \
                 patch.dict("sys.modules", {"yaml": types.SimpleNamespace(safe_load=json.loads)}):
                manifest = gate.load(gate.prepare(root, root, path, plan["candidate"]))
                self.assertEqual(manifest["plan"], plan)
                self.assertEqual(manifest["runs"], [])
                self.assertEqual(manifest["semantic_ids"], ["q0"])
                path.write_text(json.dumps({**plan, "minimum_gain": 0.05}))
                with self.assertRaisesRegex(ValueError, "commit the exact plan"):
                    gate.prepare(root, root, path, plan["candidate"])

    def test_changed_setup_cannot_authorize_switch(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            plan = root / "plan.json"
            gate.save(plan, experiment()[0]["plan"])
            manifest = root / "manifest.json"
            gate.save(manifest, {"plan": gate.load(plan), "plan_path": str(plan),
                                 "plan_sha256": gate.digest(plan), "state": {"model": "old"}})
            result = root / "verdict.json"
            gate.save(result, {"version": 2, "passed": True, "manifest": str(manifest)})
            with patch("sys.argv", ["retriever_gate.py", "check-switch", str(result)]), \
                 patch.object(gate, "state", return_value={"model": "changed"}):
                with self.assertRaisesRegex(ValueError, "changed; measure again"):
                    gate.main()

    def test_committed_plan_to_switch_decision_without_model_calls(self):
        template, artifacts = experiment()
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp).resolve()
            for args in (("init", "-q"), ("config", "user.email", "test@example.invalid"),
                         ("config", "user.name", "Gate test")):
                subprocess.run(["git", "-C", str(root), *args], check=True, capture_output=True)
            plan = root / "plan.json"
            gate.save(plan, template["plan"])
            subprocess.run(["git", "-C", str(root), "add", "plan.json"], check=True)
            subprocess.run(["git", "-C", str(root), "-c", "commit.gpgsign=false", "-c", "core.hooksPath=/dev/null",
                            "commit", "-qm", "Declare target before measurement"], check=True)
            (root / "evals").mkdir()
            golden = [{"id": id, "expected_summary": "A supported answer"} for id in template["expected_ids"]]
            (root / "evals/golden.yaml").write_text(json.dumps(golden))
            current = {"production": {"path": "/models/v2-qwen"}, "production_reranker": None,
                       "candidate": {"path": "/models/new"}}
            with patch.object(gate, "state", return_value=current), \
                 patch.dict("sys.modules", {"yaml": types.SimpleNamespace(safe_load=json.loads)}), \
                 patch.object(Path, "cwd", return_value=root):
                manifest = gate.prepare(root, root, plan, template["plan"]["candidate"])
                for i, (spec, artifact) in enumerate(artifacts):
                    artifact["meta"]["started"] = dt.datetime.now(dt.timezone.utc).isoformat()
                    path = root / f"run{i}.json"
                    gate.save(path, artifact)
                    argv = ["retriever_gate.py", "add", str(manifest), "--arm", spec["arm"],
                            "--round", str(spec["round"]), "--artifact", str(path)]
                    with patch("sys.argv", argv):
                        gate.main()
                result = gate.evaluate_file(manifest)
                self.assertTrue(result["passed"])
                result_path = root / "verdict.json"
                gate.save(result_path, result)
                with patch("sys.argv", ["retriever_gate.py", "check-switch", str(result_path)]), contextlib.redirect_stdout(io.StringIO()):
                    gate.main()

    def test_legacy_pass_cannot_authorize_switch(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "legacy.json"
            gate.save(path, {"candidate": "rerank-old", "production": "v2-qwen", "passed": True})
            with patch("sys.argv", ["retriever_gate.py", "check-switch", str(path)]):
                with self.assertRaisesRegex(ValueError, "legacy tolerance-only"):
                    gate.main()

    def test_model_changes_are_part_of_switch_identity(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "config.json").write_text("{}")
            weights = root / "model.safetensors"
            weights.write_text("before")
            before = gate.model_identity(root)
            weights.write_text("changed weights")
            self.assertNotEqual(before, gate.model_identity(root))

    def test_unknown_error_is_not_treated_as_successful_completion(self):
        row = experiment()[1][0][1]["results"][0]
        row["reasons"] = ["upstream stream disconnected"]
        self.assertFalse(gate.completed(row))
        row["reasons"] = ["missing 'word'", "search_sections not called (called: ['search'])"]
        self.assertTrue(gate.completed(row))


class ShellRollbackTests(unittest.TestCase):
    def run_rollback(self, healthy=True, extra=()):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            shutil.copyfile(Path(__file__).resolve().parents[1] / "gate_retriever.sh", root / "gate_retriever.sh")
            (root / "models/reranker-v1").mkdir(parents=True)
            (root / "models/reranker-v1/config.json").write_text("{}")
            link = root / "models/production-rerank"
            link.symlink_to("reranker-v1")
            (root / "bin").mkdir()
            calls = root / "calls"
            scripts = {"systemctl": '#!/bin/sh\nprintf "%s\\n" "$*" >> "$MOCK_CALLS"\n',
                       "curl": f"#!/bin/sh\nexit {0 if healthy else 1}\n", "sleep": "#!/bin/sh\nexit 0\n"}
            for name, script in scripts.items():
                p = root / "bin" / name
                p.write_text(script)
                p.chmod(0o755)
            env = {**os.environ, "PATH": str(root / "bin") + os.pathsep + os.environ["PATH"], "MOCK_CALLS": str(calls)}
            result = subprocess.run(["bash", str(root / "gate_retriever.sh"), "--disable-rerank", *extra],
                                    env=env, capture_output=True, text=True)
            return result.returncode, link.is_symlink(), (root / "models/reranker-v1/config.json").is_file(), calls.read_text() if calls.exists() else ""

    def test_healthy_rollback_removes_only_activation(self):
        code, linked, weights, calls = self.run_rollback()
        self.assertEqual(code, 0)
        self.assertFalse(linked)
        self.assertTrue(weights)
        self.assertEqual(calls.count("restart riverguide-retriever"), 1)

    def test_failed_health_restores_previous_link(self):
        code, linked, weights, calls = self.run_rollback(healthy=False)
        self.assertEqual(code, 1)
        self.assertTrue(linked)
        self.assertTrue(weights)
        self.assertEqual(calls.count("restart riverguide-retriever"), 2)

    def test_mixed_rollback_and_promotion_flags_do_nothing(self):
        code, linked, weights, calls = self.run_rollback(extra=("--switch",))
        self.assertEqual(code, 2)
        self.assertTrue(linked)
        self.assertTrue(weights)
        self.assertEqual(calls, "")


if __name__ == "__main__":
    unittest.main()
