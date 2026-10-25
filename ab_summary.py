"""Summarize model_ab.sh's runs and pick the survivor.

    python3 ab_summary.py PROD CHALLENGER     # prints one summary line, then "WINNER <model>"

The incumbent keeps ties: the challenger wins only with at least MARGIN more golden passes over
all rounds, at no more than SLOWER times production's median seconds.
"""
import json
import statistics
import sys
from pathlib import Path

MARGIN, SLOWER = 3, 1.5


def runs(model: str) -> list[dict]:
    return [json.loads(p.read_text())["summary"]
            for p in sorted(Path("runs").glob(f"golden-model-{model.replace(':', '-')}-r*.json"))]


prod, challenger = sys.argv[1:3]
result = {m: runs(m) for m in (prod, challenger)}
print("; ".join(f"{m} passed {', '.join(str(r['passed']) for r in rs)} of {rs[0]['total']} "
                f"(median {', '.join(str(r['median_seconds']) for r in rs)} s)" for m, rs in result.items() if rs)
      or "no runs found")
p, c = result[prod], result[challenger]
wins = (len(p) == len(c) > 0
        and sum(r["passed"] for r in c) >= sum(r["passed"] for r in p) + MARGIN
        and statistics.median(r["median_seconds"] for r in c) <= SLOWER * statistics.median(r["median_seconds"] for r in p))
print("WINNER", challenger if wins else prod)
