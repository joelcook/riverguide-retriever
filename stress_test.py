"""How much can production take? RiverGuide's real path, questions arriving at random, as students do.

    cd ../riverguide && .venv/bin/python ../riverguide-retriever/stress_test.py [--rates 2 4 6 8 10 12] [--soak-minutes 60]

Each question goes the way the Worker sends it (web/src/chat.ts): the retriever searches it first,
through the model gateway's /v1/search, then Qwen answers through the gateway's chat route with the
three tools, up to 8 steps, everything authenticated with the gateway's token. When the gateway is
full it answers 503; like the Worker's AI SDK, a call waits the retry hint and tries twice more, and
if it is still full the question counts as overflow (in production, Claude would answer it).

Arrivals are random (a Poisson process), not fixed groups of students:
  ramp    --rates questions a minute, --minutes each, letting the queue drain between rates
  burst   --burst questions spread over one minute (a class asking together)
  soak    the highest ramp rate with no overflow and p90 under 20 s, for --soak-minutes
Golden-set questions (graded, so quality under load shows too), never students'. Arrivals pause
around RiverGuide's :47 canary. Results: runs/stress-<date>.json and a table.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import random
import statistics
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fewer_calls import RIVERGUIDE, ROOT, own_record  # noqa: E402  (also puts riverguide on sys.path)

import yaml  # noqa: E402

from evals import run  # noqa: E402

MODEL = "qwen3.8:27b"
FOUND = ("Excerpts from official College pages, all re-checked against the live site on the "
         "'checked' date. Quote dates, amounts, and contacts exactly as written.")


class Busy(Exception):
    """The gateway was still full after the retries: in production, Claude answers this one."""


def gateway_env() -> tuple[str, str]:
    env = dict(line.split("=", 1) for line in (Path.home() / ".config/riverguide/model-gateway.env").read_text().splitlines()
               if "=" in line and not line.startswith("#"))
    return f"http://127.0.0.1:{env.get('RIVERGUIDE_MODEL_PORT', '8790')}", env["RIVERGUIDE_MODEL_TOKEN"]


class RiverGuide:
    """RiverGuide's path through the gateway, the Worker's way."""

    def __init__(self, base: str, token: str):
        self.base, self.token = base, token
        self.system = run.v2_system()
        self.tools = [{"type": "function", "function": {"name": t["name"], "description": t["description"], "parameters": t["input_schema"]}}
                      for t in self.system.tools]
        self.local = threading.Lock()  # read_page and search_sections share one in-memory SQLite index

    def post(self, path: str, body: dict, timeout: float = 320) -> dict:
        req = urllib.request.Request(f"{self.base}{path}", json.dumps(body).encode(),
                                     {"content-type": "application/json", "authorization": f"Bearer {self.token}"})
        return json.load(urllib.request.urlopen(req, timeout=timeout))

    def search(self, query: str, limit: int | None = None) -> dict:
        try:
            found = self.post("/v1/search", {"query": query, "limit": max(1, min(int(limit or 8), 10))}, timeout=4)
            return {"checked": found.get("checked"), "results": found.get("results", []), "note": FOUND}
        except Exception:  # noqa: BLE001  the Worker falls back to keyword search
            with self.local:
                return self.system.dispatch("search", {"query": query, "limit": limit or 8})

    def chat(self, messages: list, stats: dict) -> dict:
        body = {"model": MODEL, "messages": messages, "tools": self.tools, "max_tokens": 8000}
        for attempt in range(3):  # the AI SDK's default: two retries, honoring the retry hint
            try:
                return self.post("/v1/chat/completions", body)
            except urllib.error.HTTPError as e:
                if e.code not in (429, 503):
                    raise
                stats["busy_replies"] += 1
                if attempt == 2:
                    raise Busy() from e
                time.sleep(float(e.headers.get("retry-after") or 2))
        raise AssertionError("unreachable")

    def answer(self, question: str) -> dict:
        stats = {"model_calls": 0, "busy_replies": 0, "prompt_tokens": 0}
        messages = [{"role": "system", "content": self.system.prompt}, {"role": "user", "content": question}]
        calls = []
        if not own_record(question):
            args = {"query": question}
            messages.append({"role": "assistant", "content": "", "tool_calls": [
                {"id": "prefetch", "type": "function", "function": {"name": "search", "arguments": json.dumps(args)}}]})
            messages.append({"role": "tool", "tool_call_id": "prefetch", "content": json.dumps(self.search(question), ensure_ascii=False)})
            calls.append({"name": "search", "input": args, "prefetched": True})
        texts = []
        for _ in range(8):
            stats["model_calls"] += 1
            reply = self.chat(messages, stats)
            stats["prompt_tokens"] += (reply.get("usage") or {}).get("prompt_tokens", 0)
            message = reply["choices"][0]["message"]
            if (message.get("content") or "").strip():
                texts.append(message["content"].strip())
            if not message.get("tool_calls"):
                return {"response": "\n\n".join(texts), "tool_calls": calls, "error": None, "notes": [], **stats}
            messages.append({"role": "assistant", "content": message.get("content") or "", "tool_calls": message["tool_calls"]})
            for call in message["tool_calls"]:
                name = call["function"]["name"]
                try:
                    args = json.loads(call["function"].get("arguments") or "{}")
                except json.JSONDecodeError:
                    args = {}
                calls.append({"name": name, "input": args})
                if name == "search":
                    output = self.search(args.get("query", ""), args.get("limit"))
                else:
                    with self.local:
                        output = self.system.dispatch(name, args)
                messages.append({"role": "tool", "tool_call_id": call.get("id", name), "content": json.dumps(output, default=str, ensure_ascii=False)})
        return {"response": "\n\n".join(texts), "tool_calls": calls, "error": "tool loop cap reached", "notes": [], **stats}


def canary_pause() -> float:
    """Hold new arrivals while RiverGuide's hourly :47 canary runs; returns the seconds held."""
    held = 0.0
    while 46 <= time.localtime().tm_min < 49:
        time.sleep(5)
        held += 5
    return held


def run_phase(path: RiverGuide, items: list[dict], name: str, arrivals: list[float], log: list[dict]) -> dict:
    """Start one question at each arrival offset (seconds from now); wait for all to finish."""
    rows, lock, threads = [], threading.Lock(), []
    in_flight = [0]
    peak = [0]
    start = time.time()

    def one(item: dict, arrived: float) -> None:
        with lock:
            in_flight[0] += 1
            peak[0] = max(peak[0], in_flight[0])
        t = time.time()
        try:
            result = path.answer(item["question"])
            reasons = path.system.grade(item, result)
            row = {"passed": not reasons and not result["error"], "overflow": False, "error": result["error"],
                   "model_calls": result["model_calls"], "busy_replies": result["busy_replies"], "prompt_tokens": result["prompt_tokens"]}
        except Busy:
            row = {"passed": False, "overflow": True, "error": None, "model_calls": 0, "busy_replies": 3, "prompt_tokens": 0}
        except Exception as exc:  # noqa: BLE001
            row = {"passed": False, "overflow": False, "error": f"{type(exc).__name__}: {str(exc)[:120]}", "model_calls": 0,
                   "busy_replies": 0, "prompt_tokens": 0}
        row.update({"id": item["id"], "arrived": round(arrived, 1), "seconds": round(time.time() - t, 2)})
        with lock:
            in_flight[0] -= 1
            rows.append(row)

    held = 0.0
    for offset in arrivals:
        delay = start + held + offset - time.time()
        if delay > 0:
            time.sleep(delay)
        held += canary_pause()
        th = threading.Thread(target=one, args=(random.choice(items), offset), daemon=True)
        th.start()
        threads.append(th)
    for th in threads:
        th.join()
    wall = time.time() - start - held
    done = [r for r in rows if not r["overflow"] and not r["error"]]
    secs = sorted(r["seconds"] for r in done)
    summary = {
        "phase": name, "arrivals": len(rows), "minutes": round(wall / 60, 1), "answered": len(done),
        "overflow": sum(r["overflow"] for r in rows), "errors": sum(bool(r["error"]) for r in rows),
        "overflow_share": round(sum(r["overflow"] for r in rows) / max(1, len(rows)), 3),
        "passed_share": round(sum(r["passed"] for r in done) / max(1, len(done)), 3),
        "median_s": round(statistics.median(secs), 1) if secs else None,
        "p90_s": round(secs[int(0.9 * (len(secs) - 1))], 1) if secs else None, "max_s": secs[-1] if secs else None,
        "answers_per_minute": round(len(done) / max(1e-9, wall / 60), 2), "peak_in_flight": peak[0],
        "busy_replies": sum(r["busy_replies"] for r in rows),
        "model_calls_per_question": round(statistics.mean(r["model_calls"] for r in done), 2) if done else None,
        "canary_hold_s": held,
    }
    log.append({"summary": summary, "rows": rows})
    print(json.dumps(summary), flush=True)
    return summary


def poisson(rate_per_min: float, minutes: float, rng: random.Random) -> list[float]:
    t, out = 0.0, []
    while True:
        t += rng.expovariate(rate_per_min / 60)
        if t >= minutes * 60:
            return out
        out.append(t)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rates", type=float, nargs="+", default=[2, 4, 6, 8, 10, 12])
    parser.add_argument("--minutes", type=float, default=6)
    parser.add_argument("--burst", type=int, default=30)
    parser.add_argument("--soak-minutes", type=float, default=60)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    base, token = gateway_env()
    path = RiverGuide(base, token)
    items = yaml.safe_load((RIVERGUIDE / "evals/golden.yaml").read_text(encoding="utf-8"))
    rng = random.Random(args.seed)
    random.seed(args.seed)
    log: list[dict] = []
    ramp = []
    for rate in args.rates:
        s = run_phase(path, items, f"ramp {rate:g}/min", poisson(rate, args.minutes, rng), log)
        ramp.append((rate, s))
        if s["overflow_share"] > 0.5:  # far past what it can take: higher rates only say so louder
            print(f"stopping the ramp at {rate:g}/min: over half overflowed", flush=True)
            break
    run_phase(path, items, f"burst {args.burst} in 1 min", sorted(rng.uniform(0, 60) for _ in range(args.burst)), log)
    good = [rate for rate, s in ramp if s["overflow"] == 0 and s["errors"] == 0 and (s["p90_s"] or 99) <= 20]
    if args.soak_minutes > 0 and good:
        run_phase(path, items, f"soak {max(good):g}/min", poisson(max(good), args.soak_minutes, rng), log)
    out = ROOT / "runs" / f"stress-{dt.date.today()}.json"
    out.write_text(json.dumps({"gateway": base, "phases": log}, indent=1))
    print(f"\n{'phase':22} {'arrived':>7} {'answered':>8} {'overflow':>8} {'errors':>6} {'ans/min':>7} {'median':>6} {'p90':>6} {'max':>6} {'passed':>6} {'peak':>4}")
    for p in log:
        s = p["summary"]
        print(f"{s['phase']:22} {s['arrivals']:>7} {s['answered']:>8} {s['overflow']:>8} {s['errors']:>6} {s['answers_per_minute']:>7} "
              f"{s['median_s']!s:>6} {s['p90_s']!s:>6} {s['max_s']!s:>6} {s['passed_share']:>6.0%} {s['peak_in_flight']:>4}")
    print(f"wrote {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
