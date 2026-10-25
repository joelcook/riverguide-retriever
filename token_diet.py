"""Can RiverGuide read less per question and answer as well? Fewer or shorter search results.

    cd ../riverguide && .venv/bin/python ../riverguide-retriever/token_diet.py [--repeats 3] [VARIANT...]

Production's path, as fewer_calls.py --prefetch full measures it: the retriever searches the
question before Qwen's first step, and Qwen can search again. The search results are 62% of a
first call's prompt (3,300 of 5,300 tokens; the instructions and tools, 2,000, are already cached),
so they are what students wait for Qwen to read. Each variant changes only how many results a
search returns by default and how long each excerpt is, everywhere search runs:
    base     8 results, whole excerpts (today, about 1,230 characters each)
    r5       5 results
    x600     8 results, excerpts cut to 600 characters at a word
    r5x800   5 results, cut to 800
The golden set runs --repeats times per variant (one run varies by several questions, so a
single pass can't tell a small loss from noise). Results: runs/diet-<variant>.json; the summary
compares each variant with base. Golden questions only; nothing from students.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fewer_calls import RIVERGUIDE, ROOT, answer  # noqa: E402  (also puts riverguide on sys.path)

import yaml  # noqa: E402

from evals import run  # noqa: E402
from trained_search import wait_for_canary  # noqa: E402

VARIANTS = {"base": (8, None), "r5": (5, None), "x600": (8, 600), "r5x800": (5, 800)}
# web/src/search.ts adds this to every result the Worker hands the model.
FOUND = ("Excerpts from official College pages, all re-checked against the live site on the "
         "'checked' date. Quote dates, amounts, and contacts exactly as written.")


def retriever_search(base: str, query: str, limit: int) -> dict:
    req = urllib.request.Request(f"{base}/search", json.dumps({"query": query, "limit": limit}).encode(),
                                 {"content-type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=30))


def cut(text: str, chars: int | None) -> str:
    if chars is None or len(text) <= chars:
        return text
    head = text[:chars]
    return (head[: head.rfind(" ")] if " " in head else head) + "…"


def searcher(base: str, default_limit: int, chars: int | None):
    def search(query: str, limit: int | None = None) -> dict:
        n = max(1, min(int(limit or default_limit), 10))
        found = retriever_search(base, query, n)
        results = [{**r, "excerpt": cut(r["excerpt"], chars)} for r in found.get("results", [])]
        return {"checked": found.get("checked"), "results": results, "note": FOUND}
    return search


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("variants", nargs="*", default=list(VARIANTS))
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--model", default="qwen3.8:27b")
    parser.add_argument("--ollama", default=os.environ.get("PROD_BASE", "http://127.0.0.1:11434"))
    parser.add_argument("--retriever", default="http://127.0.0.1:8791")
    args = parser.parse_args()
    items = yaml.safe_load((RIVERGUIDE / "evals/golden.yaml").read_text(encoding="utf-8"))
    summaries = {}
    for name in args.variants:
        limit, chars = VARIANTS[name]
        system = run.v2_system()
        dispatch, search = system.dispatch, searcher(args.retriever, limit, chars)
        system.dispatch = lambda n, a, d=dispatch, s=search: s(a.get("query", ""), a.get("limit")) if n == "search" else d(n, a)
        rows = []
        for rep in range(args.repeats):
            for item in items:
                wait_for_canary()
                t = time.perf_counter()
                try:
                    result = answer(system, item["question"], args.model, args.ollama, lambda q, s=search: s(q))
                except Exception as exc:  # noqa: BLE001
                    result = {"response": "", "tool_calls": [], "model_calls": 0, "tokens": {"prompt": 0, "completion": 0},
                              "error": f"error: {exc}", "notes": []}
                reasons = system.grade(item, result)
                if result["error"]:
                    reasons.insert(0, result["error"])
                rows.append({"id": item["id"], "repeat": rep, "passed": not reasons, "reasons": reasons,
                             "model_calls": result["model_calls"], "tokens": result["tokens"],
                             "seconds": round(time.perf_counter() - t, 1)})
            print(f"{name} repeat {rep + 1}: {sum(r['passed'] for r in rows if r['repeat'] == rep)}/{len(items)}", flush=True)
        per_repeat = [sum(r["passed"] for r in rows if r["repeat"] == k) for k in range(args.repeats)]
        summary = {"variant": name, "results": limit, "excerpt_chars": chars, "repeats": args.repeats,
                   "passed_per_repeat": per_repeat, "passed_mean": round(statistics.mean(per_repeat), 1), "total": len(items),
                   "model_calls_per_question": round(statistics.mean(r["model_calls"] for r in rows), 2),
                   "prompt_tokens_per_question": round(statistics.mean(r["tokens"]["prompt"] for r in rows)),
                   "completion_tokens_per_question": round(statistics.mean(r["tokens"]["completion"] for r in rows)),
                   "median_seconds": statistics.median(r["seconds"] for r in rows)}
        (ROOT / "runs" / f"diet-{name}.json").write_text(json.dumps({"summary": summary, "rows": rows}, indent=2))
        summaries[name] = summary
        print(json.dumps(summary), flush=True)
    if "base" in summaries:
        b = summaries["base"]
        for name, s in summaries.items():
            print(f"{name:7} passed {s['passed_mean']:5}/{s['total']} ({s['passed_mean'] - b['passed_mean']:+.1f})  "
                  f"prompt tokens {s['prompt_tokens_per_question']:6} ({s['prompt_tokens_per_question'] / b['prompt_tokens_per_question'] - 1:+.0%})  "
                  f"median {s['median_seconds']} s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
