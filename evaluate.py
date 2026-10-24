"""Grade keyword search, a dense model, and their fusion on RiverGuide's retrieval tests.

    .venv/bin/python evaluate.py --model Qwen/Qwen3-Embedding-0.6B --name stock
    .venv/bin/python evaluate.py --model models/strict --name strict

Tests, all graded on pages (a page's score is its best chunk's):
- abbreviations: evals/abbreviations.yaml, right page in the top 5 (as evals/search_compare.py)
- golden: the golden questions that name pages to cite (must_cite_url), one in the top 5
- val: generated queries for held-out pages, never trained on (data/val.jsonl), top 5 and MRR@10
Hybrid is weighted reciprocal-rank fusion of the two page rankings; its keyword weight is
picked on val only, then applied to the frozen tests.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import torch
import yaml
from sentence_transformers import SentenceTransformer

from common import DATA, QUERY_INSTRUCT, RIVERGUIDE, ROOT, load_docs, snapshot_index
from riverguide.corpus import normalize_url

RUNS = ROOT / "runs"
DEPTH = 50
RRF_K = 60


def load_tests() -> dict[str, list[dict]]:
    abbreviations = yaml.safe_load((RIVERGUIDE / "evals/abbreviations.yaml").read_text())
    golden = yaml.safe_load((RIVERGUIDE / "evals/golden.yaml").read_text())
    tests = {
        "abbreviations": [{"query": c["query"], "fragments": c["url_contains"]} for c in abbreviations],
        "golden": [
            {"query": g["question"], "urls": [normalize_url(u) for u in g["must_cite_url"]]}
            for g in golden if g.get("must_cite_url")
        ],
    }
    val = DATA / "val.jsonl"
    if val.exists():
        tests["val"] = [json.loads(line) for line in open(val, encoding="utf-8")]
    return tests


def hit(case: dict, urls: list[str], pages_url: list[str]) -> bool:
    if "fragments" in case:
        return any(f in u for f in case["fragments"] for u in urls)
    if "urls" in case:
        return any(normalize_url(u) in case["urls"] for u in urls)
    return case["url"] in urls


class Dense:
    def __init__(self, model: str, name: str, docs: list[dict], n_pages: int):
        self.model = SentenceTransformer(model, device="cuda", model_kwargs={"torch_dtype": torch.bfloat16})
        self.model.max_seq_length = 512
        self.doc_page = np.array([d["page"] for d in docs])
        self.n_pages = n_pages
        cache = RUNS / f"emb-{name}.npy"
        if cache.exists():
            self.docs = np.load(cache)
        else:
            self.docs = self.model.encode([d["text"] for d in docs], batch_size=32, normalize_embeddings=True,
                                          convert_to_numpy=True, show_progress_bar=True).astype(np.float32)
            np.save(cache, self.docs)

    def rank(self, queries: list[str]) -> list[list[int]]:
        q = self.model.encode([QUERY_INSTRUCT + x for x in queries], batch_size=64, normalize_embeddings=True,
                              convert_to_numpy=True).astype(np.float32)
        out = []
        for scores in q @ self.docs.T:
            best = np.full(self.n_pages, -np.inf, dtype=np.float32)
            np.maximum.at(best, self.doc_page, scores)
            out.append(list(np.argsort(-best)[:DEPTH]))
        return out


def fuse(dense: list[int], keyword: list[int], weight: float) -> list[int]:
    score: dict[int, float] = {}
    for rank, p in enumerate(dense):
        score[p] = score.get(p, 0) + 1 / (RRF_K + rank)
    for rank, p in enumerate(keyword):
        score[p] = score.get(p, 0) + weight / (RRF_K + rank)
    return sorted(score, key=score.__getitem__, reverse=True)


def grade(cases: list[dict], rankings: list[list[int]], urls: list[str]) -> dict:
    hits, rr, misses = 0, 0.0, []
    for case, ranking in zip(cases, rankings):
        top = [urls[p] for p in ranking[:10]]
        ok = hit(case, top[:5], urls)
        hits += ok
        if not ok:
            misses.append(case["query"])
        for i in range(10):
            if hit(case, top[i : i + 1], urls):
                rr += 1 / (i + 1)
                break
    return {"top5": hits, "n": len(cases), "mrr10": round(rr / max(1, len(cases)), 4), "misses": misses}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--name", required=True)
    args = parser.parse_args()
    RUNS.mkdir(exist_ok=True)

    index = snapshot_index()
    docs = load_docs()
    urls = [p.url for p in index.pages]
    page_of = {normalize_url(u): i for i, u in enumerate(urls)}
    tests = load_tests()
    dense = Dense(args.model, args.name, docs, len(urls))

    result: dict = {"model": args.model, "name": args.name}
    rankings: dict[str, dict[str, list[list[int]]]] = {}
    for test, cases in tests.items():
        queries = [c["query"] for c in cases]
        keyword = [[page_of[normalize_url(r["url"])] for r in index.search(q, limit=DEPTH)] for q in queries]
        rankings[test] = {"keyword": keyword, "dense": dense.rank(queries)}

    # Pick the keyword weight on val (generated, held-out pages), never on the frozen tests.
    weight = 0.5
    if "val" in tests:
        sweep = {}
        for w in (0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0):
            fused = [fuse(d, k, w) for d, k in zip(rankings["val"]["dense"], rankings["val"]["keyword"])]
            sweep[w] = grade(tests["val"], fused, urls)["mrr10"]
        weight = max(sweep, key=sweep.__getitem__)
        result["hybrid_sweep_val_mrr10"] = sweep
    result["hybrid_keyword_weight"] = weight

    for test, cases in tests.items():
        r = rankings[test]
        r["hybrid"] = [fuse(d, k, weight) for d, k in zip(r["dense"], r["keyword"])]
        result[test] = {system: grade(cases, r[system], urls) for system in ("keyword", "dense", "hybrid")}
        if test == "val":
            for system in result[test].values():
                system["misses"] = system["misses"][:40]

    # Real first questions from RiverGuide 1 (questions only), for reading, not grading.
    examples = (RIVERGUIDE / "web/app/examples.ts").read_text()
    top = re.search(r"TOP_QUESTIONS = \[(.*?)\];", examples, re.S)
    if top:
        questions = re.findall(r'"([^"]+)"', top.group(1))
        dense_top = dense.rank(questions)
        lines = [f"# Top 3 pages for students' most-asked questions ({args.name})\n"]
        for q, d in zip(questions, dense_top):
            k = [page_of[normalize_url(x["url"])] for x in index.search(q, limit=DEPTH)]
            lines.append(f"## {q}\n")
            for label, ranking in (("keyword", k), ("dense", d), ("hybrid", fuse(d, k, weight))):
                lines.append(f"- {label}: " + " | ".join(index.pages[p].title for p in ranking[:3]))
            lines.append("")
        (RUNS / f"{args.name}-top-questions.md").write_text("\n".join(lines))

    (RUNS / f"{args.name}.json").write_text(json.dumps(result, indent=2))
    for test in tests:
        row = "  ".join(
            f"{s} {result[test][s]['top5']}/{result[test][s]['n']} (mrr {result[test][s]['mrr10']})"
            for s in ("keyword", "dense", "hybrid")
        )
        print(f"{test:14} {row}")
    print(f"hybrid keyword weight {weight}")
    for s in ("keyword", "dense", "hybrid"):
        print(f"abbreviation misses, {s}: {', '.join(result['abbreviations'][s]['misses']) or 'none'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
