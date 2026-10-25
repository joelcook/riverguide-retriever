"""Students' kind of first message for each training chunk, written by Qwen or by Claude, for the v2 retriever.

    .venv/bin/python gen_student.py prep [--batches 16]      # data/student/bNN.ids: the same chunks for every writer
    .venv/bin/python gen_student.py qwen [--workers 4]       # data/queries.student.qwen.jsonl, from production's Qwen
    .venv/bin/python gen_student.py show b01 [--start 0]     # 25 passages of a batch, as the Claude writer reads them
    .venv/bin/python gen_student.py save b01 < lines.jsonl   # the Claude writer's lines, checked and kept
    .venv/bin/python gen_student.py status [sonnet]          # chunks written and still to write, per batch
    .venv/bin/python gen_student.py merge sonnet             # data/queries.student.sonnet.jsonl
    .venv/bin/python gen_student.py ready                    # exit 0 once both writers covered 97% of the chunks

v1's queries (gen_queries.py) search for a passage: a question, a 2-5 word search, shorthand, a
typo. Students' real first messages are different: conversational, about their situation, without
the College's terms. The retriever finds the right page for 96% of v1-style questions on held-out
pages but 88% of real ones. Both writers get the same prompt and the same passages (cut to 900
characters at a word), so the v2 comparison changes only the writer. Held-out (val) pages are left
out: they are never trained on.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import urllib.request
from concurrent.futures import ThreadPoolExecutor

from build_data import is_val_page
from common import DATA, load_docs

OLLAMA = os.environ.get("PROD_BASE", "http://127.0.0.1:11440")  # production's engine (lab/production.env)
MODEL = "qwen3.8:27b"
STUDENT = DATA / "student"
CHARS = 900
SHOW = 25

PROMPT = """You write training data for the search engine behind RiverGuide, the AI advisor for Indian River State College (IRSC) in Florida. The search engine has to find the College web page passage that answers what a student or parent types as their first message.

Real first messages are conversational. They describe the student's situation or goal instead of naming the office, form, or policy, often in lowercase with little punctuation, and sometimes ask two things at once. Made-up examples of the style (don't reuse them): "hi i need help my financial aid says i have a hold", "how do i sign up for spring classes im coming back after a few years", "my son wants to take college classes in high school how does that work".

Read the passage below and write two first messages that this passage answers:
- chat: the way a real student or parent would type it, 8 to 30 words, in their own words rather than the page's title or official terms.
- indirect: 3 to 10 words asking for what the passage gives, without its title or key terms.
Each must be answered by this passage specifically, not just by the College in general. Don't add facts the passage doesn't support.
- useful: false if the passage is navigation, a list of links, boilerplate, or has nothing a student or parent would ask about. Leave chat and indirect empty then.

Page: {title}
URL: {url}
Section: {section}
Passage:
{text}"""

SCHEMA = {
    "type": "object",
    "properties": {"useful": {"type": "boolean"}, "chat": {"type": "string"}, "indirect": {"type": "string"}},
    "required": ["useful", "chat", "indirect"],
}

CLAUDE_HEAD = """Write two first messages for each passage below, following these instructions exactly (they are
the same ones the other writer gets, one passage at a time):

{instructions}

Save one JSON line per passage, with its id:
  {{"id": 123, "useful": true, "chat": "...", "indirect": "..."}}
  {{"id": 124, "useful": false, "chat": "", "indirect": ""}}
"""


def cut(text: str) -> str:
    if len(text) <= CHARS:
        return text
    head = text[:CHARS]
    return (head[: head.rfind(" ")] if " " in head else head) + " ..."


def prompt_for(doc: dict) -> str:
    return PROMPT.format(title=doc["title"], url=doc["url"], section=doc["section"] or "(top of page)", text=cut(doc["text"]))


def chunk_ids(docs: list[dict]) -> list[int]:
    """The chunks v1 found worth a query (useful), minus held-out pages."""
    useful = [json.loads(line)["id"] for line in open(DATA / "queries.raw.jsonl", encoding="utf-8") if json.loads(line).get("useful")]
    return [i for i in useful if not is_val_page(docs[i]["url"])]


def batch_ids(name: str) -> list[int]:
    return [int(x) for x in (STUDENT / f"{name}.ids").read_text().split()]


def batches() -> list[str]:
    return sorted(p.stem for p in STUDENT.glob("b*.ids"))


def written(writer: str, name: str | None = None) -> dict[int, dict]:
    """The writer's lines so far, last one per chunk wins."""
    files = [STUDENT / writer / f"{name}.jsonl"] if name else sorted((STUDENT / writer).glob("b*.jsonl"))
    if writer == "qwen":
        files = [DATA / "queries.student.qwen.jsonl"]
    out = {}
    for f in files:
        if f.exists():
            for line in open(f, encoding="utf-8"):
                r = json.loads(line)
                out[r["id"]] = r
    return out


def valid(r: dict) -> str | None:
    if not isinstance(r.get("id"), int) or not isinstance(r.get("useful"), bool):
        return "needs an integer id and true/false useful"
    if r["useful"]:
        for f, lo, hi in (("chat", 8, 200), ("indirect", 2, 120)):
            if not isinstance(r.get(f), str) or not lo <= len(r[f].strip()) <= hi:
                return f"{f} must be {lo}-{hi} characters"
    return None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("cmd", choices=["prep", "qwen", "show", "save", "status", "merge", "ready"])
    parser.add_argument("arg", nargs="?")
    parser.add_argument("--batches", type=int, default=16)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--start", type=int, default=0)
    args = parser.parse_args()
    docs = load_docs()

    if args.cmd == "prep":
        ids = chunk_ids(docs)
        STUDENT.mkdir(parents=True, exist_ok=True)
        for k in range(args.batches):
            (STUDENT / f"b{k + 1:02d}.ids").write_text(" ".join(map(str, ids[k :: args.batches])) + "\n")
        print(f"{len(ids)} chunks in {args.batches} batches")

    elif args.cmd == "show":
        ids = batch_ids(args.arg)
        part = ids[args.start : args.start + SHOW]
        if args.start == 0:
            print(CLAUDE_HEAD.format(instructions=PROMPT.split("\n\nPage:")[0]))
        for i in part:
            d = docs[i]
            print(f"=== id {i}\nPage: {d['title']}\nURL: {d['url']}\nSection: {d['section'] or '(top of page)'}\nPassage:\n{cut(d['text'])}\n")
        end = args.start + len(part)
        print(f"--- passages {args.start + 1}-{end} of {len(ids)} in {args.arg}."
              + (f" Next: show {args.arg} --start {end}" if end < len(ids) else " That was the last part."))

    elif args.cmd == "save":
        ids, saved, errors = set(batch_ids(args.arg)), [], []
        for n, line in enumerate(sys.stdin, 1):
            if not line.strip():
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                errors.append(f"line {n}: not JSON")
                continue
            problem = valid(r) or (None if r.get("id") in ids else f"id {r.get('id')} is not in {args.arg}")
            if problem:
                errors.append(f"line {n}: {problem}")
            else:
                saved.append({"id": r["id"], "useful": r["useful"], "chat": r.get("chat", "").strip(),
                              "indirect": r.get("indirect", "").strip()})
        (STUDENT / "sonnet").mkdir(exist_ok=True)
        with open(STUDENT / "sonnet" / f"{args.arg}.jsonl", "a", encoding="utf-8") as fh:
            for r in saved:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        done = written("sonnet", args.arg)
        print(f"saved {len(saved)}; {len(ids) - len(set(done) & ids)} of {len(ids)} in {args.arg} still to write")
        for e in errors[:10]:
            print("  not saved:", e)

    elif args.cmd == "status":
        writer = args.arg or "sonnet"
        have = written(writer)
        for b in batches():
            ids = batch_ids(b)
            print(f"{b}: {sum(i in have for i in ids)}/{len(ids)}")
        total = sum(len(batch_ids(b)) for b in batches())
        print(f"{writer}: {sum(1 for b in batches() for i in batch_ids(b) if i in have)}/{total}, "
              f"useful {sum(1 for r in have.values() if r.get('useful'))}")

    elif args.cmd == "ready":
        ids = {i for b in batches() for i in batch_ids(b)}
        return 0 if all(len(set(written(w)) & ids) >= 0.97 * len(ids) for w in ("qwen", "sonnet")) else 1

    elif args.cmd == "merge":
        have = written(args.arg)
        ids = [i for b in batches() for i in batch_ids(b)]
        with open(DATA / f"queries.student.{args.arg}.jsonl", "w", encoding="utf-8") as fh:
            for i in sorted(ids):
                if i in have:
                    fh.write(json.dumps({**have[i], "writer": args.arg}, ensure_ascii=False) + "\n")
        print(f"{sum(i in have for i in ids)}/{len(ids)} chunks to data/queries.student.{args.arg}.jsonl")

    elif args.cmd == "qwen":
        ids = [i for b in batches() for i in batch_ids(b)]
        out, lock = DATA / "queries.student.qwen.jsonl", threading.Lock()
        have = written("qwen")
        todo = [i for i in ids if i not in have]
        print(f"{len(todo)} of {len(ids)} chunks to write", flush=True)

        def one(i: int) -> None:
            body = {"model": MODEL, "messages": [{"role": "user", "content": prompt_for(docs[i])}], "format": SCHEMA,
                    "think": False, "stream": False, "options": {"temperature": 0.8, "top_p": 0.95}}
            req = urllib.request.Request(f"{OLLAMA}/api/chat", json.dumps(body).encode(), {"content-type": "application/json"})
            for _ in range(2):
                try:
                    r = json.loads(json.load(urllib.request.urlopen(req, timeout=300))["message"]["content"])
                    break
                except (ValueError, KeyError, OSError):
                    r = None
            if r is None or valid({"id": i, **r}):
                return
            with lock, open(out, "a", encoding="utf-8") as fh:
                fh.write(json.dumps({"id": i, "useful": r["useful"], "chat": r["chat"].strip(), "indirect": r["indirect"].strip(),
                                     "writer": "qwen"}, ensure_ascii=False) + "\n")

        with ThreadPoolExecutor(args.workers) as pool:
            for n, _ in enumerate(pool.map(one, todo), 1):
                if n % 500 == 0:
                    print(f"{n}/{len(todo)}", flush=True)
        print(f"qwen: {len(written('qwen'))}/{len(ids)} chunks", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
