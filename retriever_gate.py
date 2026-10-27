"""Predeclared, end-to-end promotion decisions for gate_retriever.sh.

The shell owns services; this module owns data and decisions. No model calls here.
Prepare freezes a committed plan before any measurements. Promotion replays the
raw results and checks that the measured production and candidate still match.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
from pathlib import Path
import statistics
import subprocess
import uuid

VERSION = 2
OBJECTIVES = {"quality", "latency", "cost", "reliability"}
GRADE_REASONS = ("missing ", "contains ", "no canonical link", "expected no tool calls",
                 "answered without searching", "called ", "judge: ")


def load(path: Path) -> dict:
    return json.loads(path.read_text())


def save(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n")


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git(root: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(root), *args], text=True).strip()


def number(value, name: str, minimum: float = 0) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < minimum:
        raise ValueError(f"{name} must be a finite number >= {minimum}")
    return float(value)


def validate_plan(plan: dict) -> None:
    keys = {"candidate", "production", "production_reranker", "objective", "minimum_gain", "rounds"}
    if set(plan) != keys:
        raise ValueError(f"plan must contain exactly {sorted(keys)}")
    for key in ("candidate", "production", "production_reranker"):
        name = plan[key]
        if not isinstance(name, str) or not name or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-" for c in name) or name in {".", ".."}:
            raise ValueError(f"invalid {key} name")
    if plan["objective"] not in OBJECTIVES:
        raise ValueError(f"objective must be one of {sorted(OBJECTIVES)}")
    rounds = plan["rounds"]
    if isinstance(rounds, bool) or not isinstance(rounds, int) or rounds < 4 or rounds % 2:
        raise ValueError("rounds must be even and at least 4, to balance arm order")
    gain = number(plan["minimum_gain"], "minimum_gain")
    if plan["objective"] in {"latency", "cost"}:
        if not 0.05 <= gain < 1:
            raise ValueError("latency/cost minimum_gain must be a fraction >= 0.05 and < 1")
    elif gain < 1 or not gain.is_integer():
        raise ValueError("quality/reliability minimum_gain must be a whole question per round, at least 1")


def model_identity(path: Path) -> dict | None:
    if not path.exists():
        return None
    path = path.resolve()
    if not (path / "config.json").is_file():
        raise ValueError(f"no model config in {path}")
    # Reading GB of weights would add work to every switch. Bind their file
    # identities, sizes and modification times, plus the actual config bytes.
    files = [[p.name, p.stat().st_size, p.stat().st_mtime_ns]
             for p in sorted(path.iterdir()) if p.is_file()]
    return {"path": str(path), "config_sha256": digest(path / "config.json"), "files": files}


def state(root: Path, riverguide: Path, candidate: str) -> dict:
    if git(riverguide, "rev-parse", "HEAD") != git(riverguide, "rev-parse", "main"):
        raise ValueError("RiverGuide's working checkout is not main")
    subprocess.run(["git", "-C", str(riverguide), "diff", "--exit-code", "HEAD", "--", "evals", "web", "prompt.md"],
                   check=True, stdout=subprocess.DEVNULL)
    model = candidate.removeprefix("rerank-") if candidate.startswith("rerank-") else candidate
    return {
        "production": model_identity(root / "models/production"),
        "production_reranker": model_identity(root / "models/production-rerank"),
        "candidate": model_identity(root / "models" / model),
        "corpus_commit": load(root / "index/meta.json")["corpus_commit"],
        "riverguide_main": git(riverguide, "rev-parse", "main"),
        "engine_sha256": digest(root / "lab/production.env"),
        "source_sha256": {p: digest(root / p) for p in
                          ("retriever_service.py", "rerank.py", "retriever_gate.py", "gate_retriever.sh")},
    }


def prepare(root: Path, riverguide: Path, plan_path: Path, candidate: str) -> Path:
    import yaml  # Already used by RiverGuide's golden harness.

    root, riverguide = root.resolve(), riverguide.resolve()
    plan_path = plan_path.resolve()
    relative = str(plan_path.relative_to(root))
    git(root, "ls-files", "--error-unmatch", relative)
    committed = git(root, "show", f"HEAD:{relative}")
    if committed != plan_path.read_text().strip():
        raise ValueError("commit the exact plan before measuring; staged or untracked plans do not qualify")
    plan = load(plan_path)
    validate_plan(plan)
    current = state(root, riverguide, candidate)
    if plan["candidate"] != candidate or current["candidate"] is None:
        raise ValueError("plan candidate does not match the model being measured")
    if current["production"] is None or Path(current["production"]["path"]).name != plan["production"]:
        raise ValueError("plan production does not match the active retriever")
    rr = current["production_reranker"]
    if (Path(rr["path"]).name if rr else "none") != plan["production_reranker"]:
        raise ValueError("plan production_reranker does not match production")
    if candidate.startswith("rerank-") and rr:
        raise ValueError("this harness compares an added reranker to retriever-only; an existing reranker needs a separate fair comparison")
    golden = yaml.safe_load((riverguide / "evals/golden.yaml").read_text())
    judged = [row["id"] for row in golden if row.get("expected_summary")]
    if not judged:
        raise ValueError("no semantic checks declared in the golden set")
    folder = root / "runs/retriever-gates" / (dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + candidate + "-" + uuid.uuid4().hex[:8])
    folder.mkdir(parents=True)
    manifest = folder / "manifest.json"
    save(manifest, {"version": VERSION, "prepared_at": dt.datetime.now(dt.timezone.utc).isoformat(),
                    "plan": plan, "plan_path": relative, "plan_sha256": digest(plan_path),
                    "plan_commit": git(root, "rev-parse", "HEAD"), "root": str(root), "state": current,
                    "expected_ids": [row["id"] for row in golden], "semantic_ids": judged, "runs": []})
    return manifest


def completed(row: dict) -> bool:
    if not isinstance(row.get("response"), str) or not row["response"].strip() or row.get("error"):
        return False
    # Older API artifacts did not serialize error separately. All unknown reasons
    # are operational failures, rather than silently treating an error as success.
    return all(reason.startswith(GRADE_REASONS) or " not called (called:" in reason
               for reason in row["reasons"])


def metrics(rows: list[dict], semantic_ids: set[str]) -> dict:
    times = sorted(number(r["seconds"], "seconds", 0.000001) for r in rows)
    # The harness's ordinary usage/cost fields include the semantic judge. They
    # cannot establish serving cost. Require separate, explicitly metered fields.
    measured_cost = all(isinstance(r.get("serving_usage"), dict) and any(
        isinstance(v, (int, float)) and not isinstance(v, bool) and v > 0 for v in r["serving_usage"].values())
        and isinstance(r.get("serving_cost_usd"), (int, float)) and not isinstance(r["serving_cost_usd"], bool)
        and math.isfinite(r["serving_cost_usd"]) and r["serving_cost_usd"] > 0 for r in rows)
    categories = sorted({r["category"] for r in rows})
    return {"total": len(rows), "passed": sum(r["passed"] for r in rows),
            "answer_passed": sum(r["answer_passed"] for r in rows),
            "semantic_passed": sum(completed(r) and isinstance(r.get("judge"), dict) and r["judge"].get("pass") is True
                                   for r in rows if r["id"] in semantic_ids),
            "completion_failures": sum(not completed(r) for r in rows),
            "median_seconds": statistics.median(times), "p95_seconds": times[math.ceil(.95 * len(times)) - 1],
            "mean_seconds": statistics.mean(times), "tool_calls": sum(len(r["tool_calls"]) for r in rows),
            "cost_usd": sum(r["serving_cost_usd"] for r in rows) if measured_cost else None,
            "categories": {cat: {key: sum(r[key] for r in rows if r["category"] == cat)
                                  for key in ("passed", "answer_passed")} for cat in categories}}


def evaluate(manifest: dict, artifacts: list[tuple[dict, dict]]) -> dict:
    if manifest.get("version") != VERSION or not manifest.get("prepared_at") or not manifest.get("plan_commit"):
        raise ValueError("no predeclared version-2 experiment; legacy tolerance passes cannot authorize promotion")
    plan = manifest["plan"]
    validate_plan(plan)
    expected = set(manifest["expected_ids"])
    semantic = set(manifest["semantic_ids"])
    if not expected or len(expected) != len(manifest["expected_ids"]) or not semantic or not semantic <= expected:
        raise ValueError("invalid expected question IDs or semantic checks")
    rounds = plan["rounds"]
    seen, signature, categories = set(), None, None
    per_round, all_rows = {}, {"production": [], "candidate": []}
    prepared = dt.datetime.fromisoformat(manifest["prepared_at"])
    for spec, artifact in artifacts:
        arm, rd = spec["arm"], spec["round"]
        if arm not in all_rows or not isinstance(rd, int) or isinstance(rd, bool) or not 1 <= rd <= rounds or (arm, rd) in seen:
            raise ValueError("duplicate or invalid arm/round")
        seen.add((arm, rd))
        meta, rows = artifact["meta"], artifact["results"]
        if meta.get("partial") is not False or meta.get("tier") != "full" or meta.get("judge") is not True:
            raise ValueError("every run must be full, non-partial, and semantically judged")
        if dt.datetime.fromisoformat(meta["started"]) < prepared:
            raise ValueError("measurement predates the frozen plan")
        sig = (meta.get("model"), meta.get("prompt_sha256"))
        if not all(isinstance(v, str) and v for v in sig) or signature is not None and sig != signature:
            raise ValueError("model or prompt differs between runs")
        signature = sig
        ids = [r["id"] for r in rows]
        if len(ids) != len(expected) or set(ids) != expected:
            raise ValueError("missing, duplicate, or unexpected questions")
        cats = {r["id"]: r["category"] for r in rows}
        if categories is not None and cats != categories:
            raise ValueError("question categories differ between arms")
        categories = cats
        for r in rows:
            if not isinstance(r.get("passed"), bool) or not isinstance(r.get("answer_passed"), bool):
                raise ValueError("missing boolean grades")
            if not isinstance(r.get("reasons"), list) or not all(isinstance(v, str) for v in r["reasons"]) or not isinstance(r.get("tool_calls"), list):
                raise ValueError("missing reasons or tool trace")
            if r["id"] in semantic and completed(r) and (not isinstance(r.get("judge"), dict) or not isinstance(r["judge"].get("pass"), bool)):
                raise ValueError("missing semantic judgment")
        m = metrics(rows, semantic)
        if artifact["summary"].get("passed") != m["passed"] or artifact["summary"].get("total") != len(expected):
            raise ValueError("summary does not match raw results")
        per_round[(arm, rd)] = m
        all_rows[arm].extend(rows)
    if len(seen) != 2 * rounds:
        raise ValueError("incomplete experiment")
    totals = {arm: metrics(rows, semantic) for arm, rows in all_rows.items()}
    base, cand = totals["production"], totals["candidate"]
    blockers = []
    for key in ("passed", "answer_passed", "semantic_passed"):
        if cand[key] < base[key]:
            blockers.append(f"{key} regressed: {cand[key]} vs {base[key]}")
    for cat in base["categories"]:
        for key in ("passed", "answer_passed"):
            if cand["categories"][cat][key] < base["categories"][cat][key]:
                blockers.append(f"{cat} {key} regressed")
    if cand["completion_failures"]:
        blockers.append(f"candidate has {cand['completion_failures']} incomplete/error answers")
    for key in ("median_seconds", "p95_seconds"):
        if cand[key] > base[key] * 1.05:
            blockers.append(f"{key} regressed by more than 5%")
    if base["cost_usd"] is not None and cand["cost_usd"] is not None and cand["cost_usd"] > base["cost_usd"] * 1.05:
        blockers.append("measured cost regressed by more than 5%")
    if (base["cost_usd"] is None) != (cand["cost_usd"] is None):
        blockers.append("serving cost telemetry is missing from one arm")
    objective, minimum = plan["objective"], plan["minimum_gain"]
    def gain(b, c):
        if objective == "quality":
            return c["semantic_passed"] - b["semantic_passed"]
        if objective == "reliability":
            return b["completion_failures"] - c["completion_failures"]
        key = "median_seconds" if objective == "latency" else "cost_usd"
        if b[key] is None or c[key] is None or b[key] <= 0:
            raise ValueError("cost objective requires positive, metered costs and token usage for every answer; zero placeholders do not qualify")
        return 1 - c[key] / b[key]
    observed = gain(base, cand)
    if objective in {"quality", "reliability"}:
        observed /= rounds
    if observed + 1e-12 < minimum:
        blockers.append(f"predeclared {objective} benefit not met: {observed:.4g} vs required {minimum}")
    gains = [gain(per_round[("production", rd)], per_round[("candidate", rd)]) for rd in range(1, rounds + 1)]
    if any(g <= 0 for g in gains):
        blockers.append("target benefit was not positive in every paired round")
    return {"version": VERSION, "candidate": plan["candidate"], "production": plan["production"],
            "production_reranker": plan["production_reranker"], "passed": not blockers,
            "verdict": "demonstrated end-to-end benefit" if not blockers else "no demonstrated end-to-end benefit; retain incumbent",
            "summary": "; ".join(blockers) if blockers else f"predeclared {objective} benefit met ({observed:.4g} >= {minimum}) with guardrails held",
            "objective": objective, "minimum_gain": minimum, "observed_gain": observed,
            "round_gains": gains, "blockers": blockers, "metrics": totals,
            "runs": [{"arm": arm, "round": rd, **m} for (arm, rd), m in sorted(per_round.items())]}


def evaluate_file(manifest_path: Path) -> dict:
    manifest = load(manifest_path)
    committed_plan = json.loads(git(Path(manifest["root"]), "show", f"{manifest['plan_commit']}:{manifest['plan_path']}"))
    if committed_plan != manifest["plan"]:
        raise ValueError("frozen plan does not match its committed declaration")
    artifacts = []
    for spec in manifest["runs"]:
        path = Path(spec["artifact"])
        if digest(path) != spec["sha256"]:
            raise ValueError(f"measurement changed: {path}")
        artifacts.append((spec, load(path)))
    result = evaluate(manifest, artifacts)
    result["manifest"] = str(manifest_path.resolve())
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["prepare", "add", "evaluate", "check-switch"])
    parser.add_argument("path", type=Path)
    parser.add_argument("--riverguide", type=Path, default=Path("/home/irsc/Projects/riverguide"))
    parser.add_argument("--candidate")
    parser.add_argument("--arm", choices=["production", "candidate"])
    parser.add_argument("--round", type=int)
    parser.add_argument("--artifact", type=Path)
    args = parser.parse_args()
    root = Path.cwd().resolve()
    if args.action == "prepare":
        if not args.candidate:
            parser.error("prepare needs --candidate")
        print(prepare(root, args.riverguide, args.path, args.candidate))
    elif args.action == "add":
        if not args.arm or not args.round or not args.artifact:
            parser.error("add needs --arm, --round, --artifact")
        manifest = load(args.path)
        manifest["runs"].append({"arm": args.arm, "round": args.round,
                                 "artifact": str(args.artifact.resolve()), "sha256": digest(args.artifact)})
        save(args.path, manifest)
    elif args.action == "evaluate":
        print(json.dumps(evaluate_file(args.path), indent=2))
    else:
        stored = load(args.path)
        if stored.get("version") != VERSION or not stored.get("manifest"):
            raise ValueError("legacy tolerance-only result cannot authorize promotion; run a predeclared experiment")
        if stored.get("passed") is not True:
            raise ValueError("candidate did not demonstrate the required benefit")
        manifest_path = Path(stored["manifest"])
        manifest = load(manifest_path)
        if digest(root / manifest["plan_path"]) != manifest["plan_sha256"]:
            raise ValueError("plan changed since measurement")
        if load(root / manifest["plan_path"]) != manifest["plan"]:
            raise ValueError("frozen plan differs from its declaration")
        if state(root, args.riverguide, manifest["plan"]["candidate"]) != manifest["state"]:
            raise ValueError("production, candidate, corpus, engine, or source changed; measure again")
        result = evaluate_file(manifest_path)
        if not result["passed"]:
            raise ValueError(result["summary"])
        if stored["candidate"] != result["candidate"] or stored["production"] != result["production"]:
            raise ValueError("stored result identity does not match the experiment")
        print(result["summary"])


if __name__ == "__main__":
    try:
        main()
    except (ValueError, KeyError, TypeError, OSError, subprocess.CalledProcessError) as exc:
        raise SystemExit(f"retriever gate: {exc}")
