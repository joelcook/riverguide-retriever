"""Turn generated queries into training rows: leakage filters, a held-out page split, a judge, hard negatives.

    .venv/bin/python build_data.py --mode strict   # the honest one: no test abbreviation in any training query
    .venv/bin/python build_data.py --mode full     # production candidate: keeps them (abbreviation score is then contaminated)
    .venv/bin/python build_data.py --mode full --name v2-sonnet --extra data/queries.student.sonnet.jsonl
                                                   # full plus students' kind of first message (gen_student.py)

Both modes drop training queries that copy a golden question. Pages whose URL hashes to 0 mod 20
are held out: their queries are data/val.jsonl and never trained on.

Qwen3-Reranker-0.6B judges every pair (rerank.py). A generated query is kept only when its own
chunk answers it (p >= 0.1 for training, 0.02 for program-name queries, 0.5 for val). Each kept query gets up to two hard
negatives: the stock embedder's closest chunk from another page and keyword search's best chunk
from another page, each only when the judge says it does not answer the query (p < 0.2). The
College repeats itself across irsc.edu and the catalog, so a close chunk is often another right
answer; the judge is what keeps those out.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re

import numpy as np
import torch
import yaml
from sentence_transformers import SentenceTransformer

from common import DATA, QUERY_INSTRUCT, RIVERGUIDE, ROOT, load_docs, norm, snapshot_index
from rerank import Judge
from riverguide.index import _WEIGHTS, fts_query

CASE_SENSITIVE = {"IT", "AS", "HIM"}  # also English words in lowercase
LOWER_TECH_IT = re.compile(
    r"\bit (support|degree|degrees|certificate|certificates|certification|program|programs|class|classes|"
    r"course|courses|career|careers|job|jobs|major|technician|tech|help desk|specialist|training|school)\b"
)
LOWER_AS = re.compile(r"\bas (degree|nursing|business|program|in)\b")
FIELDS = ("question", "search", "shorthand", "typo")
STUDENT_FIELDS = ("chat", "indirect")  # gen_student.py
POS_MIN, VAL_POS_MIN, NEG_MAX = 0.1, 0.5, 0.2
PROGRAM_POS_MIN = 0.02  # "how do i become a police officer" is only loosely answered by a program page's top chunk
CANDIDATES = 8
CACHE = DATA / "judge.jsonl"


def is_val_page(url: str) -> bool:
    return int(hashlib.md5(url.encode()).hexdigest(), 16) % 20 == 0


def abbreviation_filter():
    tests = [c["query"] for c in yaml.safe_load((RIVERGUIDE / "evals/abbreviations.yaml").read_text())]
    phrases = [re.compile(rf"(?<![a-z0-9]){re.escape(norm(t))}(?:s|es)?(?![a-z0-9])") for t in tests if t not in CASE_SENSITIVE]
    upper = [re.compile(rf"(?<![A-Za-z0-9]){t}s?(?![A-Za-z0-9])") for t in CASE_SENSITIVE]

    def leaks(query: str) -> bool:
        n = norm(query)
        low = query.lower()
        return (any(p.search(n) for p in phrases) or any(p.search(query) for p in upper)
                or bool(LOWER_TECH_IT.search(low)) or bool(LOWER_AS.search(low)))
    return leaks


def golden_filter():
    questions = [g["question"] for g in yaml.safe_load((RIVERGUIDE / "evals/golden.yaml").read_text())]
    sets = [set(norm(q).split()) for q in questions]
    exact = {norm(q) for q in questions}

    def leaks(query: str) -> bool:
        n = norm(query)
        if n in exact:
            return True
        words = set(n.split())
        return len(words) >= 3 and any(len(words & s) / len(words | s) >= 0.8 for s in sets)
    return leaks


def load_rows(extra: list[str] = ()) -> list[dict]:
    rows = []
    for path in extra:
        for line in open(path, encoding="utf-8"):
            r = json.loads(line)
            if r.get("useful"):
                rows += [{"q": r[f].strip(), "doc": r["id"], "kind": f} for f in STUDENT_FIELDS if r.get(f, "").strip()]
    for line in open(DATA / "queries.raw.jsonl", encoding="utf-8"):
        r = json.loads(line)
        if r.get("useful"):
            rows += [{"q": r[f].strip(), "doc": r["id"], "kind": f} for f in FIELDS if r.get(f, "").strip()]
    program = DATA / "queries.program.jsonl"
    if program.exists():
        for line in open(program, encoding="utf-8"):
            r = json.loads(line)
            rows += [{"q": q.strip(), "doc": r["id"], "kind": "program"} for q in r.get("queries", []) if q.strip()]
    seen, out = set(), []
    for r in rows:
        key = (norm(r["q"]), r["doc"])
        if 2 <= len(r["q"]) <= 200 and key not in seen:
            seen.add(key)
            out.append(r)
    return out


def bm25_chunks(index, query: str, depth: int = 40) -> list[int]:
    match = fts_query(query)
    if match is None:
        return []
    with index._lock:
        rows = index._db.execute(
            "SELECT chunk_id FROM chunks WHERE chunks MATCH ? "
            f"ORDER BY bm25(chunks, {', '.join(map(str, _WEIGHTS))}) LIMIT {depth}", (match,)).fetchall()
    return [r[0] for r in rows]


class CachedJudge:
    """The judge, remembering every (query, chunk) it has scored across runs and modes."""

    def __init__(self, docs: list[dict]):
        self.docs = docs
        self.judge: Judge | None = None
        self.cache: dict[str, float] = {}
        if CACHE.exists():
            for line in open(CACHE, encoding="utf-8"):
                k, p = json.loads(line)
                self.cache[k] = p

    def scores(self, pairs: list[tuple[str, int]]) -> list[float]:
        keys = [f"{d}|{q}" for q, d in pairs]
        todo = list(dict.fromkeys(k for k in keys if k not in self.cache))
        if todo:
            self.judge = self.judge or Judge()
            split = [k.split("|", 1) for k in todo]
            got = self.judge.score([(q, self.docs[int(d)]["text"]) for d, q in split], batch=int(os.environ.get("JUDGE_BATCH", 32)))
            with open(CACHE, "a", encoding="utf-8") as fh:
                for k, p in zip(todo, got):
                    self.cache[k] = p
                    fh.write(json.dumps([k, p]) + "\n")
        return [self.cache[k] for k in keys]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("strict", "full"), required=True)
    parser.add_argument("--name", help="output name (train.NAME.jsonl); default the mode")
    parser.add_argument("--extra", action="append", default=[], help="more queries, as gen_student.py writes them")
    args = parser.parse_args()
    name = args.name or args.mode
    random.seed(0)

    docs = load_docs()
    index = snapshot_index()
    rows = load_rows(args.extra)
    golden_leak, abbr_leak = golden_filter(), abbreviation_filter()
    stats: dict = {"mode": args.mode, "name": name, "extra": args.extra, "generated": len(rows), "dropped_golden": 0, "dropped_abbreviation": 0}
    kept = []
    for r in rows:
        if golden_leak(r["q"]):
            stats["dropped_golden"] += 1
        elif args.mode == "strict" and abbr_leak(r["q"]):
            stats["dropped_abbreviation"] += 1
        else:
            kept.append(r)

    judge = CachedJudge(docs)
    for r, p in zip(kept, judge.scores([(r["q"], r["doc"]) for r in kept])):
        r["p"] = p
    def answered(r: dict) -> bool:
        return r["p"] >= (PROGRAM_POS_MIN if r["kind"] == "program" else POS_MIN)

    train = [r for r in kept if not is_val_page(docs[r["doc"]]["url"]) and answered(r)]
    stats["dropped_unanswered"] = sum(1 for r in kept if not is_val_page(docs[r["doc"]]["url"]) and not answered(r))
    if args.mode == "strict" and name == "strict":  # one val set, from the strict rows, shared by every model
        val_rows = [r for r in kept if is_val_page(docs[r["doc"]]["url"]) and r["p"] >= VAL_POS_MIN]
        random.shuffle(val_rows)
        with open(DATA / "val.jsonl", "w", encoding="utf-8") as fh:
            for r in val_rows[:1500]:
                fh.write(json.dumps({"query": r["q"], "url": docs[r["doc"]]["url"], "kind": r["kind"]}) + "\n")
        stats["val"] = min(1500, len(val_rows))

    stock = SentenceTransformer("Qwen/Qwen3-Embedding-0.6B", device="cuda", model_kwargs={"dtype": torch.bfloat16})
    stock.max_seq_length = 512
    doc_emb = np.load(ROOT / "runs/emb-stock.npy")
    q_emb = stock.encode([QUERY_INSTRUCT + r["q"] for r in train], batch_size=128, normalize_embeddings=True,
                         convert_to_numpy=True, show_progress_bar=True).astype(np.float32)
    del stock
    torch.cuda.empty_cache()
    np.save(DATA / f"qemb-stock-{name}.npy", q_emb)
    page = np.array([d["page"] for d in docs])
    text_key = [hashlib.md5(d["text"].split("\n\n", 1)[-1].encode()).hexdigest() for d in docs]

    # Candidate negatives: other pages, other text, in the stock model's and keyword search's order.
    dense_c, keyword_c = [], []
    for i, r in enumerate(train):
        pos = r["doc"]

        def other(c: int) -> bool:
            return page[c] != page[pos] and text_key[c] != text_key[pos]

        sims = doc_emb @ q_emb[i]
        dense_c.append([int(c) for c in np.argsort(-sims)[:60] if other(int(c))][:CANDIDATES])
        keyword_c.append([c for c in bm25_chunks(index, r["q"]) if other(c)][:CANDIDATES])

    # Judge the first three of each, then the rest only where nothing passed.
    negs: list[list[int]] = [[] for _ in train]
    for lo, hi in ((0, 3), (3, CANDIDATES)):
        for cands in (dense_c, keyword_c):
            want = [i for i in range(len(train)) if not any(n in cands[i] for n in negs[i])]
            pairs = [(train[i]["q"], c) for i in want for c in cands[i][lo:hi] if c not in negs[i]]
            score = dict(zip(pairs, judge.scores(pairs)))
            for i in want:
                pick = next((c for c in cands[i][lo:hi] if c not in negs[i] and score.get((train[i]["q"], c), 1) < NEG_MAX), None)
                if pick is not None:
                    negs[i].append(pick)

    path = DATA / f"train.{name}.jsonl"
    with open(path, "w", encoding="utf-8") as fh:
        for i, r in enumerate(train):
            fh.write(json.dumps({"q": r["q"], "doc": r["doc"], "page": int(page[r["doc"]]), "kind": r["kind"],
                                 "p": round(r["p"], 4), "negs": negs[i], "row": i}) + "\n")
    stats["train"] = len(train)
    stats["kinds"] = {k: sum(r["kind"] == k for r in train) for k in (*FIELDS, "program", *STUDENT_FIELDS)}
    stats["negatives"] = {n: sum(len(x) == n for x in negs) for n in (0, 1, 2)}
    print(json.dumps(stats, indent=2))
    (DATA / f"stats.{name}.json").write_text(json.dumps(stats, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
