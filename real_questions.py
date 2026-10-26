"""Grade keyword search, the stock embedder, and the retriever on students' real first questions.

    .venv/bin/python real_questions.py --org https://<org>.crm.dynamics.com --bot <River Guide bot id>
        [--systems keyword=keyword stock=Qwen/Qwen3-Embedding-0.6B production=models/production] [--deep production] [--label NAME]

--systems names the three lists the judge sees (keyword=keyword is keyword search; MODEL+rerankN
has Qwen3-Reranker-0.6B re-sort MODEL's top N pages before the top 5 are judged, and
MODEL+rerankN:RERANKER a trained one, e.g. models/production+rerank20:models/reranker-v1). --deep NAME sorts
that system's misses: production's Qwen reads the top 20 pages of every system together and says
whether any of them answers the question. If one does, search missed it; if none does, the website
may not answer it (a list for the College's web team). Only counts, ranks, topics, and page URLs
are kept.

Questions are the first typed message of each River Guide (Copilot Studio) session in Dataverse,
which keeps 30 days; the az login on omarchy reads them. Each system's top 5 pages on today's
crawl go to production's Qwen, blind and shuffled as A/B/C, which judges whether each list
answers the question and which serves the student best.

Privacy, as irsc-knowledge-base/scripts/mine_copilot_sessions.py: messages are redacted in memory
(emails, phones, SSNs, long IDs) before search or judging, nothing raw is written, and output is
aggregate counts plus only the questions asked at least --min-show times. Output goes to logs/,
which is not committed.
"""
from __future__ import annotations

import argparse
import datetime as dt
import gc
import hashlib
import json
import os
import random
import re
import subprocess
import time
import urllib.parse
import urllib.request
from collections import Counter
from math import comb
from pathlib import Path

import numpy as np
import torch
import yaml

from common import QUERY_INSTRUCT, RIVERGUIDE, ROOT, build_docs
from refresh import git_head, load_model
from riverguide.corpus import load_pages, read_checked
from riverguide.index import SearchIndex

STARTERS = Path("/home/irsc/Projects/irsc-knowledge-base/prompts/copilot_studio_conversation_start.yaml")
OLLAMA = os.environ.get("PROD_BASE", "http://127.0.0.1:11434")  # production's engine (lab/production.env)
JUDGE = "qwen3.8:27b"
SYSTEMS = {"keyword": None, "stock": "Qwen/Qwen3-Embedding-0.6B", "production": str(ROOT / "models" / "production")}
RERANK: dict[str, tuple[int, str]] = {}  # system -> (how many of its top pages, which reranker) (MODEL+rerankN[:RERANKER])
TOP = 5
DEEP = 20
TOPICS = ["admissions and applying", "registration and classes", "financial aid and scholarships",
          "tuition, fees, and payment", "programs and degrees", "courses and class schedules",
          "records, transcripts, and grades", "accounts, email, and technology", "advising and support services",
          "campus, contacts, and hours", "other"]
EXCERPT = 500

REDACT = [
    (re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.I), "<email>"),
    (re.compile(r"(?<!\d)\d{3}-\d{2}-\d{4}(?!\d)"), "<ssn>"),
    (re.compile(r"(?<!\d)(?:\+?1[\s.-]?)?(?:\(?\d{3}\)?[\s.-]?)\d{3}[\s.-]?\d{4}(?!\d)"), "<phone>"),
    (re.compile(r"(?<![\w])[A-Z]?\d{6,}(?!\d)", re.I), "<id>"),
]

PROMPT = """You grade search results for Indian River State College's student help assistant.
A student typed this as the first message of a chat:

<message>{question}</message>

Three search engines each returned their top 5 College web pages, as lists A, B and C.

{lists}

Decide:
- kind: "question" if the College's website could help with it (information, or how to do something).
  "other" for greetings, tests, gibberish, or only asking for a human.
- A, B, C: true if at least one page in that list gives the student what they need (the answer itself,
  or the official page for doing the task). Judge each list on its own.
- best: the list that serves the student best: "A", "B", "C", or "tie".

Reply with JSON only: {{"kind": "question", "A": true, "B": false, "C": true, "best": "A"}}"""


def redact(text: str) -> str:
    for pattern, token in REDACT:
        text = pattern.sub(token, text)
    return " ".join(text.split())


def first_questions(org: str, bot: str, days: int) -> list[str]:
    token = subprocess.run(["az", "account", "get-access-token", "--resource", org, "--query", "accessToken", "-o", "tsv"],
                           capture_output=True, text=True, check=True).stdout.strip()
    since = (dt.datetime.now(dt.timezone.utc) - dt.timedelta(days=days)).strftime("%Y-%m-%dT%H:%M:%SZ")
    query = urllib.parse.quote(f"_bot_conversationtranscriptid_value eq {bot} and createdon ge {since}")
    url = f"{org}/api/data/v9.2/conversationtranscripts?$select=content&$filter={query}"
    out = []
    while url:
        request = urllib.request.Request(url, headers={"Authorization": f"Bearer {token}", "Prefer": "odata.maxpagesize=100"})
        page = json.load(urllib.request.urlopen(request, timeout=120))
        for row in page["value"]:
            acts = json.loads(row["content"] or "{}").get("activities", [])
            typed = [a for a in acts if a.get("type") == "message" and (a.get("from") or {}).get("role") == 1
                     and (a.get("text") or "").strip()]
            if typed:
                out.append(redact(min(typed, key=lambda a: a.get("timestamp", 0))["text"]))
        url = page.get("@odata.nextLink")
    return out


DEEP_PROMPT = """You check search results for Indian River State College's student help assistant.
A student typed this as the first message of a chat:

<message>{question}</message>

Below are College web pages, numbered. Which of them give the student what they need (the answer
itself, or the official page for doing the task)? List every one that does, or none.

{pages}

Also name the topic of the message, one of: {topics}.

Reply with JSON only: {{"answers": [3, 12], "topic": "financial aid and scholarships"}}"""


def deep_check(question: str, pages: list[dict]) -> dict | None:
    """Which of the pooled pages answer the question (their urls), and its topic."""
    body = "\n".join(f"  {i}. {r['title']}" + (f" > {r['section']}" if r.get("section") else "") +
                     f" ({r['url']})\n     {' '.join(r['excerpt'].split())[:300]}" for i, r in enumerate(pages, 1))
    for _ in range(2):
        while 44 <= time.localtime().tm_min < 50:
            time.sleep(15)
        request = {"model": JUDGE, "stream": False, "think": False, "format": "json", "options": {"temperature": 0},
                   "messages": [{"role": "user", "content": DEEP_PROMPT.format(question=question, pages=body, topics="; ".join(TOPICS))}]}
        try:
            reply = json.load(urllib.request.urlopen(urllib.request.Request(
                f"{OLLAMA}/api/chat", json.dumps(request).encode(), {"content-type": "application/json"}), timeout=300))
            v = json.loads(reply["message"]["content"])
            urls = [pages[n - 1]["url"] for n in v.get("answers", []) if isinstance(n, int) and 1 <= n <= len(pages)]
            return {"urls": urls, "topic": v.get("topic") if v.get("topic") in TOPICS else "other"}
        except (KeyError, ValueError, urllib.error.URLError):
            continue
    return None


def starters() -> set[str]:
    replies = yaml.safe_load(STARTERS.read_text())["beginDialog"]["actions"][0]["activity"]["quickReplies"]
    return {key(r["text"]) for r in replies}


def key(text: str) -> str:
    return " ".join(re.findall(r"[a-z0-9<>']+", text.lower()))


def dense_rankings(name: str, path: str, index: SearchIndex, docs: list[dict], questions: list[str], depth: int = TOP) -> list[list[int]]:
    """Top pages as chunk ids (each page's best chunk), for each question."""
    model = load_model(path)
    # Keyed by the model's own directory, not the system's name: models/production moves to each new
    # model that passes its gate, and a system named after it mustn't reuse the last model's embeddings.
    tag = Path(path).resolve().name if Path(path).exists() else name
    cache = ROOT / "runs" / f"emb-{tag}-{git_head(RIVERGUIDE)}.npy"
    if cache.exists():
        emb = np.load(cache)
    else:
        emb = model.encode([d["text"] for d in docs], batch_size=32, normalize_embeddings=True,
                           convert_to_numpy=True).astype(np.float32)
        np.save(cache, emb)
    q = model.encode([QUERY_INSTRUCT + x for x in questions], batch_size=64, normalize_embeddings=True,
                     convert_to_numpy=True).astype(np.float32)
    del model
    gc.collect()  # the model sits in a reference cycle: without this its 1.9 GB stays on the GPU
    torch.cuda.empty_cache()
    doc_page = np.array([d["page"] for d in docs])
    out = []
    for scores in q @ emb.T:
        picked, seen = [], set()
        for cid in np.argsort(-scores):
            if doc_page[cid] not in seen:
                seen.add(doc_page[cid])
                picked.append(int(cid))
                if len(picked) == depth:
                    break
        out.append(picked)
    return out


def rerank(questions: list[str], ids: list[list[int]], docs: list[dict], n: int, reranker: str) -> list[list[int]]:
    """Each question's first n pages (their best chunks) re-sorted by the reranker, the rest after."""
    from rerank import Judge  # loads only when a system asks for it

    judge = Judge(device=os.environ.get("RERANK_DEVICE", "cpu"), model=reranker)  # the GPU is production's (rerank.py)
    pairs = [(q, docs[c]["text"]) for q, r in zip(questions, ids) for c in r[:n]]
    scores = judge.score(pairs, batch=int(os.environ.get("JUDGE_BATCH", 16)))
    del judge
    gc.collect()
    torch.cuda.empty_cache()
    out, k = [], 0
    for r in ids:
        head = r[:n]
        order = sorted(range(len(head)), key=lambda j: -scores[k + j])
        out.append([head[j] for j in order] + r[n:])
        k += len(head)
    return out


def show(results: list[dict]) -> str:
    lines = []
    for i, r in enumerate(results, 1):
        head = r["title"] + (f" > {r['section']}" if r.get("section") else "")
        lines.append(f"  {i}. {head} ({r['url']})\n     {' '.join(r['excerpt'].split())[:EXCERPT]}")
    return "\n".join(lines) or "  (no results)"


def judge(question: str, lists: dict[str, list[dict]]) -> dict | None:
    names = sorted(lists)
    random.Random(hashlib.sha256(question.encode()).digest()).shuffle(names)
    labels = dict(zip("ABC", names))
    body = "\n\n".join(f"List {label}:\n{show(lists[name])}" for label, name in labels.items())
    for _ in range(2):
        while 44 <= time.localtime().tm_min < 50:  # RiverGuide's :47 canary gets Qwen to itself
            time.sleep(15)
        request = {"model": JUDGE, "stream": False, "think": False, "format": "json", "options": {"temperature": 0},
                   "messages": [{"role": "user", "content": PROMPT.format(question=question, lists=body)}]}
        try:
            reply = json.load(urllib.request.urlopen(urllib.request.Request(
                f"{OLLAMA}/api/chat", json.dumps(request).encode(), {"content-type": "application/json"}), timeout=300))
            verdict = json.loads(reply["message"]["content"])
            best = verdict.get("best", "tie")
            return {"kind": verdict["kind"], **{labels[l]: bool(verdict[l]) for l in "ABC"},
                    "best": labels.get(best, "tie")}
        except (KeyError, ValueError, urllib.error.URLError):
            continue
    return None


def mcnemar(a: int, b: int) -> float:
    n, k = a + b, min(a, b)
    return 1.0 if n == 0 else min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--org", required=True)
    parser.add_argument("--bot", required=True)
    parser.add_argument("--days", type=int, default=30)
    parser.add_argument("--min-show", type=int, default=3)
    parser.add_argument("--systems", nargs=3, metavar="NAME=MODEL", help="the three lists to judge (MODEL keyword is keyword search)")
    parser.add_argument("--deep", metavar="NAME", help="sort this system's misses: search missed, or no page answers")
    parser.add_argument("--label", help="output logs/real-questions-DATE-LABEL.json")
    args = parser.parse_args()
    global SYSTEMS
    if args.systems:
        SYSTEMS = {}
        for n, m in (x.split("=", 1) for x in args.systems):
            m, _, spec = m.partition("+rerank")
            if spec:
                depth, _, reranker = spec.partition(":")
                RERANK[n] = (int(depth), str(ROOT / reranker) if reranker else "Qwen/Qwen3-Reranker-0.6B")
            SYSTEMS[n] = None if m == "keyword" else str(ROOT / m) if (ROOT / m).exists() else m
    assert not args.deep or args.deep in SYSTEMS, "--deep must name one of --systems"
    assert args.min_show >= 2, "never show questions asked once"

    asked = first_questions(args.org, args.bot, args.days)
    counts = Counter(key(q) for q in asked)
    text = {}
    for q in asked:
        text.setdefault(key(q), q)
    unique = [k for k in counts if k]
    print(f"{len(asked)} sessions with a typed first message, {len(unique)} distinct", flush=True)

    corpus = RIVERGUIDE / "corpus"
    index = SearchIndex(load_pages(corpus), read_checked(corpus))
    docs = build_docs(index)
    questions = [text[k] for k in unique]
    depth = DEEP if args.deep else TOP
    lists, pools = {}, {}  # pools: each system's top `depth` pages, for --deep
    for name, path in SYSTEMS.items():
        if path and name in RERANK:
            n, reranker = RERANK[name]
            ids = dense_rankings(name, path, index, docs, questions, max(depth, n))
            ranked = [[index._result(c) for c in r] for r in rerank(questions, ids, docs, n, reranker)]
        elif path:
            ranked = [[index._result(c) for c in r] for r in dense_rankings(name, path, index, docs, questions, depth)]
        else:
            ranked = [index.search(q, limit=depth) for q in questions]
        lists[name], pools[name] = [r[:TOP] for r in ranked], ranked
    print("searched; judging", flush=True)

    verdicts = {}
    for i, k in enumerate(unique):
        verdicts[k] = judge(text[k], {name: lists[name][i] for name in SYSTEMS})
        if (i + 1) % 50 == 0:
            print(f"judged {i + 1}/{len(unique)}", flush=True)

    start = starters()
    graded = [k for k in unique if verdicts[k] and verdicts[k]["kind"] == "question"]
    summary = {"sessions": len(asked), "distinct": len(unique), "judge_failed": sum(v is None for v in verdicts.values()),
               "not_a_question": sum(1 for v in verdicts.values() if v and v["kind"] != "question"), "groups": {}}
    for group, members in (("typed", [k for k in graded if k not in start]), ("starter clicks", [k for k in graded if k in start])):
        sessions = sum(counts[k] for k in members)
        g = {"distinct": len(members), "sessions": sessions}
        for name in SYSTEMS:
            g[f"{name} answered"] = sum(verdicts[k][name] for k in members)
            g[f"{name} answered (sessions)"] = sum(counts[k] for k in members if verdicts[k][name])
        g["best"] = dict(Counter(verdicts[k]["best"] for k in members))
        names = list(SYSTEMS)
        for a, b in ((names[0], names[1]), (names[0], names[2]), (names[1], names[2])):
            only_a = sum(verdicts[k][a] and not verdicts[k][b] for k in members)
            only_b = sum(verdicts[k][b] and not verdicts[k][a] for k in members)
            g[f"{a} vs {b}"] = {f"{a} only": only_a, f"{b} only": only_b, "p": round(mcnemar(only_a, only_b), 4)}
        summary["groups"][group] = g
    summary["frequent"] = [{"question": text[k], "sessions": counts[k], **{n: verdicts[k][n] for n in SYSTEMS},
                            "best": verdicts[k]["best"]}
                           for k in sorted(graded, key=counts.__getitem__, reverse=True) if counts[k] >= args.min_show]
    if args.deep:
        typed = [k for k in graded if k not in start]
        missed = [k for k in typed if not verdicts[k][args.deep]]
        rows = []
        for k in missed:
            i = unique.index(k)
            pool, seen = [], set()
            for rank in range(depth):  # interleave the systems' lists, so no system's pages come first
                for name in SYSTEMS:
                    r = pools[name][i][rank] if rank < len(pools[name][i]) else None
                    if r and r["url"] not in seen:
                        seen.add(r["url"])
                        pool.append(r)
            v = deep_check(text[k], pool)
            if v is None:
                continue
            own = [r["url"] for r in pools[args.deep][i]]
            ranks = [own.index(u) + 1 for u in v["urls"] if u in own]
            rows.append({"answerable": bool(v["urls"]), "topic": v["topic"], "rank": min(ranks) if ranks else None,
                         "first_url": v["urls"][0] if v["urls"] else None, "sessions": counts[k]})
        found = [r for r in rows if r["answerable"]]
        summary["deep"] = {
            "system": args.deep, "missed": len(missed), "checked": len(rows),
            "search_missed": len(found), "no_page_answers": len(rows) - len(found),
            "search_missed_at": dict(Counter("6-10" if r["rank"] and r["rank"] <= 10 else "11-20" if r["rank"] else "not in its top 20"
                                             for r in found)),
            "search_missed_topics": dict(Counter(r["topic"] for r in found).most_common()),
            "no_page_topics": dict(Counter(r["topic"] for r in rows if not r["answerable"]).most_common()),
            "pages_it_should_have_found": dict(Counter(r["first_url"] for r in found).most_common(40)),
        }
        print(json.dumps(summary["deep"], indent=2), flush=True)
    out = ROOT / "logs" / (f"real-questions-{dt.date.today()}" + (f"-{args.label}" if args.label else "") + ".json")
    out.write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: v for k, v in summary.items() if k != "frequent"}, indent=2))
    for f in summary["frequent"]:
        print(f"{f['sessions']:3d}x {''.join(n[0].upper() if f[n] else '.' for n in SYSTEMS)} "
              f"best={f['best']:7} {f['question'][:90]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
