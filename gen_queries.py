"""Write student search queries for every chunk with the 27B on omarchy's Ollama.

    ../riverguide/.venv/bin/python gen_queries.py [--limit N] [--workers 2]

Appends to data/queries.raw.jsonl and skips chunks already done, so it can be
stopped and restarted. Program, service, and admissions pages go first; the
1,700 course pages go last. Uses Ollama's default context so it shares the
loaded model with production instead of forcing a reload.
"""
from __future__ import annotations

import argparse
import json
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

from common import DATA, load_docs

OLLAMA = "http://127.0.0.1:11434"
MODEL = "qwen3.8:27b"
OUT = DATA / "queries.raw.jsonl"

PROMPT = """You write training data for the search engine behind RiverGuide, the AI advisor for Indian River State College (IRSC) in Florida. Students and parents type questions, and the search engine has to find the College web page passage that answers them.

Read the passage below and write searches that this passage answers. Write the way real students and parents type: casual, often lowercase, sometimes with abbreviations, nicknames, or typos. Don't copy the page title word for word. Each query must be answered by this passage specifically, not just by the College in general.

Fields:
- useful: false if the passage is navigation, a list of links, boilerplate, or has nothing a student or parent would search for. Leave the other fields empty then.
- question: a full question a student or parent would ask.
- search: what they would type into a search box, 2 to 5 words.
- shorthand: the most casual way to ask. Use the abbreviation, acronym, nickname, or slang students actually use for this program, office, form, or topic (like "fafsa", "DE classes", "comp 1", "the portal"). If there is none, the shortest 1 to 3 word version.
- typo: a short search with one or two realistic typos or misspellings.

Page: {title}
URL: {url}
Section: {section}
Passage:
{text}"""

SCHEMA = {
    "type": "object",
    "properties": {
        "useful": {"type": "boolean"},
        "question": {"type": "string"},
        "search": {"type": "string"},
        "shorthand": {"type": "string"},
        "typo": {"type": "string"},
    },
    "required": ["useful", "question", "search", "shorthand", "typo"],
}


def targets(docs: list[dict]) -> list[dict]:
    """Chunks worth writing queries for, most important pages first."""
    keep = [
        d for d in docs
        if d["repeats"] <= 2 and d["chunk_chars"] >= 120 and len(d["text"]) >= 200
    ]
    return sorted(keep, key=lambda d: ("/catalog/courses/" in d["url"], d["id"]))


def generate(doc: dict) -> dict:
    body = {
        "model": MODEL,
        "messages": [{"role": "user", "content": PROMPT.format(
            title=doc["title"], url=doc["url"], section=doc["section"] or "(top of page)", text=doc["text"])}],
        "format": SCHEMA,
        "think": False,
        "stream": False,
        "keep_alive": "10m",
        "options": {"temperature": 0.8, "top_p": 0.95},
    }
    request = urllib.request.Request(f"{OLLAMA}/api/chat", json.dumps(body).encode(), {"content-type": "application/json"})
    with urllib.request.urlopen(request, timeout=600) as response:
        reply = json.load(response)
    out = json.loads(reply["message"]["content"])
    return {"id": doc["id"], **out, "eval_tokens": reply.get("eval_count"), "seconds": reply.get("total_duration", 0) / 1e9}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()

    docs = load_docs()
    todo = targets(docs)
    done = set()
    if OUT.exists():
        done = {json.loads(line)["id"] for line in open(OUT, encoding="utf-8")}
    todo = [d for d in todo if d["id"] not in done]
    if args.limit:
        todo = todo[: args.limit]
    print(f"{len(docs)} chunks, {len(done)} done, {len(todo)} to go", flush=True)

    lock = threading.Lock()
    started, count, failures = time.time(), 0, 0

    def work(doc: dict) -> None:
        nonlocal count, failures
        for attempt in range(3):
            try:
                row = generate(doc)
                break
            except Exception as exc:  # noqa: BLE001 - retry anything, log, move on
                if attempt == 2:
                    with lock:
                        failures += 1
                        print(f"failed {doc['id']}: {exc}", flush=True)
                    return
                time.sleep(5)
        with lock:
            with open(OUT, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(row) + "\n")
            count += 1
            if count % 50 == 0 or count == len(todo):
                rate = count / (time.time() - started)
                print(f"{count}/{len(todo)}  {rate:.2f}/s  eta {(len(todo) - count) / rate / 60:.0f} min  failures {failures}", flush=True)

    with ThreadPoolExecutor(args.workers) as pool:
        list(pool.map(work, todo))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
