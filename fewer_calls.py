"""Can RiverGuide answer with fewer model calls? Search first, then let Qwen answer.

    cd ../riverguide && .venv/bin/python ../riverguide-retriever/fewer_calls.py --prefetch none
    cd ../riverguide && .venv/bin/python ../riverguide-retriever/fewer_calls.py --prefetch keyword
    cd ../riverguide && .venv/bin/python ../riverguide-retriever/fewer_calls.py --prefetch full \
        --ollama-embed http://127.0.0.1:11435     # serve.py --model models/full

The golden set through evals/search_ab.py's loop, except that before Qwen's first step the
search tool has already run on the student's question, exactly as if Qwen had called
search(query=<question>), and its result is in the conversation. Qwen can answer from it or
search again. --prefetch none is today's loop, for the comparison. Records golden grades, model
calls per question, tokens, and seconds. Results: runs/fewer-<prefetch>.json.

--system legacy runs RiverGuide 1 instead (the LLM wiki: its prompt, wiki, and database tools from
../irsc-knowledge-base) on the same questions with the same model, graded by the same answer-text
checks, for a fair wiki-versus-pages comparison: runs/fewer-legacy.json.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path
from typing import Any

RIVERGUIDE = Path("/home/irsc/Projects/riverguide")
sys.path.insert(0, str(RIVERGUIDE))
ROOT = Path(__file__).resolve().parent

import yaml  # noqa: E402

from evals import run  # noqa: E402
from evals.search_ab import ollama, search_tool  # noqa: E402
from riverguide.index import default_index  # noqa: E402
from trained_search import TrainedSearch, wait_for_canary  # noqa: E402


def answer(system: run.System, question: str, model: str, base: str, prefetch, max_steps: int = 8) -> dict:
    """search_ab.answer, plus an optional search already done on the question."""
    tools = [{"type": "function", "function": {"name": t["name"], "description": t["description"], "parameters": t["input_schema"]}}
             for t in system.tools]
    messages: list[dict[str, Any]] = [{"role": "system", "content": system.prompt}, {"role": "user", "content": question}]
    calls: list[dict] = []
    if prefetch is not None:
        args = {"query": question}
        messages.append({"role": "assistant", "content": "", "tool_calls": [
            {"id": "prefetch", "type": "function", "function": {"name": "search", "arguments": json.dumps(args)}}]})
        messages.append({"role": "tool", "tool_call_id": "prefetch", "content": json.dumps(prefetch(question), default=str, ensure_ascii=False)})
        calls.append({"name": "search", "input": args, "prefetched": True})
    texts: list[str] = []
    model_calls = 0
    tokens = {"prompt": 0, "completion": 0}
    for _ in range(max_steps):
        model_calls += 1
        reply = ollama(base, "/v1/chat/completions", {"model": model, "messages": messages, "tools": tools, "max_tokens": 8000})
        tokens["prompt"] += (reply.get("usage") or {}).get("prompt_tokens", 0)
        tokens["completion"] += (reply.get("usage") or {}).get("completion_tokens", 0)
        choice = reply["choices"][0]
        message = choice["message"]
        if (message.get("content") or "").strip():
            texts.append(message["content"].strip())
        if not message.get("tool_calls"):
            error = "answer did not finish (finish reason 'length')" if choice.get("finish_reason") == "length" else None
            return {"response": "\n\n".join(texts), "tool_calls": calls, "model_calls": model_calls, "tokens": tokens, "error": error, "notes": []}
        messages.append({"role": "assistant", "content": message.get("content") or "", "tool_calls": message["tool_calls"]})
        for call in message["tool_calls"]:
            name = call["function"]["name"]
            try:
                args = json.loads(call["function"].get("arguments") or "{}")
            except json.JSONDecodeError:
                args = {}
            calls.append({"name": name, "input": args})
            output = system.dispatch(name, args)
            messages.append({"role": "tool", "tool_call_id": call.get("id", name), "content": json.dumps(output, default=str, ensure_ascii=False)})
    return {"response": "\n\n".join(texts), "tool_calls": calls, "model_calls": model_calls, "tokens": tokens, "error": "tool loop cap reached", "notes": []}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prefetch", choices=["none", "keyword", "full"], required=True)
    parser.add_argument("--system", choices=["v2", "legacy"], default="v2")
    parser.add_argument("--model", default="qwen3.8:27b")
    parser.add_argument("--ollama", default=os.environ.get("PROD_BASE", "http://127.0.0.1:11434"))
    parser.add_argument("--ollama-embed", default="http://127.0.0.1:11435", help="serve.py, for --prefetch full")
    args = parser.parse_args()

    system = run.legacy_system(RIVERGUIDE.parent / "irsc-knowledge-base") if args.system == "legacy" else run.v2_system()
    if args.system == "legacy" and args.prefetch != "none":
        parser.error("RiverGuide 1 runs as it is: --prefetch none")
    dispatch = system.dispatch
    prefetch = None
    if args.prefetch == "keyword":
        prefetch = lambda q: dispatch("search", {"query": q})  # noqa: E731  production's search tool
    elif args.prefetch == "full":
        engine = TrainedSearch(default_index(), "riverguide-retriever-full", args.ollama_embed)
        prefetch = lambda q: search_tool(engine, query=q)  # noqa: E731
        system.dispatch = lambda n, a: search_tool(engine, **a) if n == "search" else dispatch(n, a)

    items = yaml.safe_load((RIVERGUIDE / "evals/golden.yaml").read_text(encoding="utf-8"))  # the same questions for both systems
    rows = []
    for item in items:
        wait_for_canary()
        t = time.perf_counter()
        try:
            result = answer(system, item["question"], args.model, args.ollama, prefetch)
        except Exception as exc:  # noqa: BLE001
            result = {"response": "", "tool_calls": [], "model_calls": 0, "tokens": {"prompt": 0, "completion": 0}, "error": f"error: {exc}", "notes": []}
        reasons = system.grade(item, result)
        answer_reasons = run.text_reasons(item, result["response"])
        if result["error"]:
            reasons.insert(0, result["error"])
        rows.append({"id": item["id"], "passed": not reasons, "answer_passed": not answer_reasons, "reasons": reasons,
                     "model_calls": result["model_calls"], "tool_calls": len(result["tool_calls"]), "tokens": result["tokens"],
                     "seconds": round(time.perf_counter() - t, 1)})
        print(f"{'PASS' if not reasons else 'FAIL'} {item['id']} ({result['model_calls']} model calls)", flush=True)

    summary = {"system": args.system, "prefetch": args.prefetch, "passed": sum(r["passed"] for r in rows), "answer_passed": sum(r["answer_passed"] for r in rows),
               "total": len(rows), "model_calls_per_question": round(statistics.mean(r["model_calls"] for r in rows), 2),
               "one_call_answers": sum(r["model_calls"] == 1 for r in rows),
               "median_seconds": statistics.median(r["seconds"] for r in rows),
               "tokens_per_question": round(statistics.mean(r["tokens"]["prompt"] + r["tokens"]["completion"] for r in rows)),
               "prompt_tokens_per_question": round(statistics.mean(r["tokens"]["prompt"] for r in rows)),
               "completion_tokens_per_question": round(statistics.mean(r["tokens"]["completion"] for r in rows))}
    name = "legacy" if args.system == "legacy" else args.prefetch
    (ROOT / "runs" / f"fewer-{name}.json").write_text(json.dumps({"summary": summary, "rows": rows}, indent=2))
    print(json.dumps(summary), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
