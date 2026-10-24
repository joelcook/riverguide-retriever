"""The 75 abbreviations as real questions, through RiverGuide's own loop: Qwen writes the searches.

    cd ../riverguide && .venv/bin/python ../riverguide-retriever/loop_abbreviations.py --search keyword
    cd ../riverguide && .venv/bin/python ../riverguide-retriever/loop_abbreviations.py --search embedding \
        --ollama http://127.0.0.1:11435 --embed-model riverguide-retriever --name finetuned

Same prompt.md, tools, and loop as evals/search_ab.py. Each abbreviation is asked as "I'm interested
in {query} at IRSC. What do you offer?". Graded two ways: found (a right page came back from any
search the model ran) and linked (the answer links a right page). Also records every search query
the model wrote, to see whether it spells acronyms out.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

RIVERGUIDE = Path("/home/irsc/Projects/riverguide")
sys.path.insert(0, str(RIVERGUIDE))

import yaml  # noqa: E402

from evals import run  # noqa: E402
from evals.search_ab import EmbeddingSearch, answer, search_tool  # noqa: E402
from riverguide.index import default_index  # noqa: E402

OUT = Path(__file__).resolve().parent / "runs"
LINK = re.compile(r"\]\((https?://[^)\s]+)\)")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--search", choices=["keyword", "embedding"], required=True)
    parser.add_argument("--model", default="qwen3.8:27b")
    parser.add_argument("--embed-model", default="qwen3-embedding:0.6b")
    parser.add_argument("--ollama", default="http://127.0.0.1:11434")
    parser.add_argument("--name", default=None)
    args = parser.parse_args()
    name = args.name or args.search

    cases = yaml.safe_load((RIVERGUIDE / "evals/abbreviations.yaml").read_text())
    system = run.v2_system()
    dispatch = system.dispatch
    if args.search == "embedding":
        engine = EmbeddingSearch(default_index(), args.embed_model, args.ollama)
        system.dispatch = lambda n, a: search_tool(engine, **a) if n == "search" else dispatch(n, a)
    inner = system.dispatch
    trace: list[tuple[str, dict, dict]] = []

    def recording(n: str, a: dict) -> dict:
        out = inner(n, a)
        trace.append((n, a, out))
        return out

    system.dispatch = recording
    rows = []
    for case in cases:
        trace.clear()
        question = f"I'm interested in {case['query']} at IRSC. What do you offer?"
        t = time.perf_counter()
        try:
            result = answer(system, question, args.model, args.ollama, max_steps=8)
        except Exception as exc:  # noqa: BLE001
            result = {"response": "", "tool_calls": [], "error": str(exc)}
        frags = case["url_contains"]
        seen = [r["url"] for n, _, out in trace if n == "search" for r in (out or {}).get("results", [])]
        linked = LINK.findall(result["response"])
        row = {
            "query": case["query"],
            "found": any(f in u for f in frags for u in seen),
            "linked": any(f in u for f in frags for u in linked),
            "searches": [a.get("query") for n, a, _ in trace if n == "search"],
            "other_tools": [n for n, _, _ in trace if n != "search"],
            "links": linked,
            "error": result.get("error"),
            "seconds": round(time.perf_counter() - t, 1),
        }
        rows.append(row)
        print(f"{'FOUND' if row['found'] else 'miss '} {'LINK' if row['linked'] else '    '} {case['query']:28} {row['searches']}", flush=True)

    summary = {"name": name, "search": args.search, "embed_model": args.embed_model if args.search == "embedding" else None,
               "found": sum(r["found"] for r in rows), "linked": sum(r["linked"] for r in rows), "n": len(rows),
               "no_search": sum(not r["searches"] for r in rows)}
    OUT.mkdir(exist_ok=True)
    (OUT / f"loop-{name}.json").write_text(json.dumps({"summary": summary, "rows": rows}, indent=2))
    print(json.dumps(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
