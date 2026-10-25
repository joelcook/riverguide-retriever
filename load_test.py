"""How many students at once can a model server answer? RiverGuide's own loop, under load.

    cd ../riverguide && .venv/bin/python ../riverguide-retriever/load_test.py --name ollama-p1 \
        --base http://127.0.0.1:11434 --model qwen3.8:27b --levels 1 4 8 16

Each level is N students asking at once: N threads, each answering golden questions one after
another through evals/search_ab.py's loop (prompt.md, the three tools, keyword search, up to 8
steps), until the level's questions are done. Per question it records seconds, model calls,
whether the answer finished, and the golden grade; per level, median and p90 seconds, questions
a minute, and pass rate. --base is any OpenAI-compatible server (Ollama, llama.cpp, vLLM, SGLang),
so one script compares them all. Questions wait out RiverGuide's :47 canary (:44 to :50), and
that pause is left out of the level's throughput. Results: runs/load-<name>.json.
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import os
import statistics
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

RIVERGUIDE = Path("/home/irsc/Projects/riverguide")
sys.path.insert(0, str(RIVERGUIDE))
ROOT = Path(__file__).resolve().parent

import yaml  # noqa: E402

from evals import run  # noqa: E402
from evals.search_ab import answer  # noqa: E402
from trained_search import MAINTENANCE, wait_for_canary  # noqa: E402

CANARY = (44, 50)  # minutes past the hour


def canary_overlap(start: float, end: float) -> float:
    """Seconds of [start, end] inside a :44-:50 window."""
    total, t = 0.0, start
    while t < end:
        hour = t - (t % 3600)
        lo, hi = hour + CANARY[0] * 60, hour + CANARY[1] * 60
        total += max(0.0, min(end, hi) - max(t, lo))
        t = hour + 3600
    return total


def level(n: int, count: int, items: list[dict], base: str, model: str) -> dict:
    queue = iter(itertools.islice(itertools.cycle(items), count))
    lock = threading.Lock()
    rows: list[dict] = []

    def student() -> None:
        system = run.v2_system()  # one per thread: the tools keep their own connections
        while True:
            with lock:
                item = next(queue, None)
            if item is None:
                return
            wait_for_canary()
            t = time.perf_counter()
            try:
                result = answer(system, item["question"], model, base, max_steps=8)
            except Exception as exc:  # noqa: BLE001  a failed request is a result, not a crash
                result = {"response": "", "tool_calls": [], "error": f"error: {exc}"[:200], "notes": []}
            seconds = time.perf_counter() - t
            reasons = system.grade(item, result)
            if result.get("error"):
                reasons.insert(0, result["error"])
            with lock:
                rows.append({"id": item["id"], "seconds": round(seconds, 2), "model_calls": len(result["tool_calls"]) + 1,
                             "finished": not result.get("error"), "passed": not reasons, "error": result.get("error")})

    start = time.time()
    with ThreadPoolExecutor(n) as pool:
        for _ in range(n):
            pool.submit(student)
    wall = time.time() - start
    # In a maintenance window the canary is off and wait_for_canary doesn't wait: nothing to leave out.
    busy = max(1e-9, wall - (0.0 if MAINTENANCE.exists() else canary_overlap(start, start + wall)))
    secs = sorted(r["seconds"] for r in rows)
    return {
        "students": n, "questions": len(rows), "wall_seconds": round(wall, 1), "canary_pause_seconds": round(wall - busy, 1),
        "questions_per_minute": round(len(rows) / busy * 60, 2),
        "median_seconds": round(statistics.median(secs), 1) if secs else None,
        "p90_seconds": round(secs[max(0, math.ceil(0.9 * len(secs)) - 1)], 1) if secs else None,  # nearest rank
        "max_seconds": round(secs[-1], 1) if secs else None,
        "passed": sum(r["passed"] for r in rows), "unfinished": sum(not r["finished"] for r in rows),
        "model_calls_per_question": round(statistics.mean(r["model_calls"] for r in rows), 2) if rows else None,
        "rows": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True)
    parser.add_argument("--base", default=os.environ.get("PROD_BASE", "http://127.0.0.1:11434"))
    parser.add_argument("--model", default="qwen3.8:27b")
    parser.add_argument("--levels", type=int, nargs="+", default=[1, 4, 8, 16])
    parser.add_argument("--per-level", type=int, default=20, help="questions per level (at least 2 per student)")
    args = parser.parse_args()

    items = yaml.safe_load((RIVERGUIDE / "evals/golden.yaml").read_text(encoding="utf-8"))
    out = {"name": args.name, "base": args.base, "model": args.model,
           "started": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "levels": []}
    print(f"{'students':>8} {'questions':>9} {'q/min':>6} {'median s':>8} {'p90 s':>6} {'max s':>6} {'passed':>7} {'unfinished':>10} {'calls/q':>7}", flush=True)
    for n in args.levels:
        result = level(n, max(args.per_level, 2 * n), items, args.base, args.model)
        out["levels"].append(result)
        r = result
        print(f"{r['students']:>8} {r['questions']:>9} {r['questions_per_minute']:>6} {r['median_seconds']:>8} {r['p90_seconds']:>6} "
              f"{r['max_seconds']:>6} {r['passed']:>3}/{r['questions']:<3} {r['unfinished']:>10} {r['model_calls_per_question']:>7}", flush=True)
        (ROOT / "runs" / f"load-{args.name}.json").write_text(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
