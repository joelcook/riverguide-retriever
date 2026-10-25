"""Pick the engine that survives the bake-off.

    python3 bakeoff_verdict.py ollama-p1 ollama-p4 llamacpp vllm sglang    # writes runs/bakeoff.json

The incumbent (lab/production.env's ENGINE) keeps its place unless a challenger is clearly
better where students feel it:
  - it started (fundraising's extraction call is tried and reported, not required: the app is
    shelved as of 2026-10-24 and follows whatever engine wins when it comes back);
  - quality held: golden passes over both rounds at least the incumbent's minus 2 (run-to-run noise);
  - it finished as many answers at 4 students (River Guide's real peak is 3) as the incumbent, give or take 1;
  - p90 seconds at 4 students at least 15% lower, and one student's median no more than 10% higher.
Among challengers that clear all of that, the lowest p90 at 4 students wins; questions a minute
at 16 students breaks a tie.
"""
import json
import sys
from pathlib import Path

RUNS = Path("runs")


def production_engine() -> str:
    for line in Path("lab/production.env").read_text().splitlines() if Path("lab/production.env").exists() else []:
        if line.startswith("ENGINE="):
            return line.split("=", 1)[1].strip()
    return "ollama-p1"


def result(engine: str) -> dict:
    meta = json.loads((RUNS / f"bake-{engine}.json").read_text())
    if not meta.get("started"):
        return {"engine": engine, "started": False, "why": meta.get("why", "")[-200:]}
    golden = [json.loads((RUNS / f"golden-bake-{engine}-r{r}.json").read_text())["summary"] for r in (1, 2)]
    load = {lv["students"]: lv for lv in json.loads((RUNS / f"load-bake-{engine}.json").read_text())["levels"]}
    return {
        "engine": engine, "started": True, "settings": meta["settings"], "memory_mib": meta["memory_mib"],
        "fundraising_ok": meta["fundraising_ok"], "golden": sum(g["passed"] for g in golden), "golden_of": sum(g["total"] for g in golden),
        "median_1": load[1]["median_seconds"], "p90_4": load[4]["p90_seconds"], "unfinished_4": load[4]["unfinished"],
        "qpm_4": load[4]["questions_per_minute"], "qpm_16": load[16]["questions_per_minute"], "p90_16": load[16]["p90_seconds"],
    }


def main() -> int:
    engines = sys.argv[1:]
    table = {e: result(e) for e in engines}
    incumbent = production_engine()
    base = table.get(incumbent)
    if not base or not base["started"]:
        print(f"the incumbent {incumbent} has no result; keeping it")
        winner, reason = incumbent, "incumbent unmeasured"
    else:
        def clears(r: dict) -> bool:
            return (r["started"] and r["golden"] >= base["golden"] - 2
                    and r["unfinished_4"] <= base["unfinished_4"] + 1
                    and r["p90_4"] <= 0.85 * base["p90_4"] and r["median_1"] <= 1.10 * base["median_1"])
        challengers = [r for e, r in table.items() if e != incumbent and clears(r)]
        if challengers:
            best = min(challengers, key=lambda r: (r["p90_4"], -r["qpm_16"]))
            winner = best["engine"]
            reason = (f"p90 at 4 students {best['p90_4']} s vs {base['p90_4']} s, golden {best['golden']} vs {base['golden']}, "
                      f"{best['qpm_16']} vs {base['qpm_16']} questions/min at 16")
        else:
            winner, reason = incumbent, "no challenger cleared the bar"
    out = {"incumbent": incumbent, "winner": winner, "reason": reason, "engines": table}
    (RUNS / "bakeoff.json").write_text(json.dumps(out, indent=2))
    print(f"{'engine':10} {'golden':>7} {'median@1':>8} {'p90@4':>6} {'q/min@4':>7} {'q/min@16':>8} {'MiB':>6} fundraising")
    for r in table.values():
        if r["started"]:
            print(f"{r['engine']:10} {r['golden']:>3}/{r['golden_of']:<3} {r['median_1']:>8} {r['p90_4']:>6} {r['qpm_4']:>7} "
                  f"{r['qpm_16']:>8} {r['memory_mib']:>6} {'ok' if r['fundraising_ok'] else 'FAILED'}")
        else:
            print(f"{r['engine']:10} did not start: {r['why'][-80:]!r}")
    print(f"winner: {winner} ({reason})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
