"""Questions and candidate pages for training RiverGuide's reranker, for Claude Opus 5.5 to grade.

    .venv/bin/python rerank_data.py prep [--n 1200] [--batches 12]     # PREP_DEVICE=cpu off the GPU
        data/rerank/questions.jsonl and data/rerank/bNN.ids; riverguide-lab grades show/save/status
        then serve them to the graders and keep the grades in data/rerank/opus/bNN.jsonl

The questions are Claude Sonnet 5.5's made-up first messages from the v2 experiment
(data/queries.student.sonnet.jsonl): students' style, never students' words, and unseen by
v2-qwen, which trained on Qwen's messages instead, so v2's ranking of them is honest. (v2's own
training questions would almost always put the right page first, and teach a reranker nothing
about near-misses.) Each gets v2's top 20 pages on the training snapshot (each page's best chunk),
leaving out held-out pages so `val` stays clean. 60% are picked where v2 doesn't rank the page the
message was written for first, and the rest at random, so the reranker also learns when to leave
v2's order alone.
"""
from __future__ import annotations

import argparse
import json
import os
import random

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from build_data import is_val_page
from common import DATA, QUERY_INSTRUCT, ROOT, load_docs

OUT = DATA / "rerank"
DEPTH = 20


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("cmd", choices=["prep"])
    parser.add_argument("--n", type=int, default=1200)
    parser.add_argument("--batches", type=int, default=12)
    parser.add_argument("--hard", type=float, default=0.6)
    args = parser.parse_args()
    random.seed(0)

    docs = load_docs()
    emb = np.load(ROOT / "runs/emb-v2-qwen.npy")  # evaluate.py's embeddings of these docs, in this order
    assert len(emb) == len(docs), "runs/emb-v2-qwen.npy doesn't match data/docs.jsonl"
    page = np.array([d["page"] for d in docs])
    held_out = np.array([is_val_page(d["url"]) for d in docs])

    rows = [json.loads(line) for line in open(DATA / "queries.student.sonnet.jsonl", encoding="utf-8")]
    items = [(r[k], k, r["id"]) for r in rows if r.get("useful") for k in ("chat", "indirect") if r.get(k)]
    # One embedding model fits on the GPU beside production (the retriever service runs the same one);
    # on the CPU, 9,000 messages take minutes, longer when another job shares the cores.
    device = os.environ.get("PREP_DEVICE", "cuda")
    model = SentenceTransformer(str(ROOT / "models" / "v2-qwen"), device=device,
                                model_kwargs={"dtype": torch.bfloat16} if device == "cuda" else {})
    q = model.encode([QUERY_INSTRUCT + t for t, _, _ in items], batch_size=64, normalize_embeddings=True,
                     convert_to_numpy=True, show_progress_bar=False).astype(np.float32)

    ranked = []
    for (text, kind, source), scores in zip(items, q @ emb.T):
        picked, seen = [], set()
        for c in np.argsort(-scores):
            if held_out[c] or page[c] in seen:
                continue
            seen.add(page[c])
            picked.append(int(c))
            if len(picked) == DEPTH:
                break
        pages = [page[c] for c in picked]
        rank = pages.index(page[source]) + 1 if page[source] in pages else DEPTH + 1
        ranked.append({"text": text, "kind": kind, "source": source, "source_rank": rank, "candidates": picked})

    hard = [r for r in ranked if r["source_rank"] > 1]
    easy = [r for r in ranked if r["source_rank"] == 1]
    random.shuffle(hard)
    random.shuffle(easy)
    n_hard = min(len(hard), round(args.n * args.hard))
    chosen = hard[:n_hard] + easy[: args.n - n_hard]
    random.shuffle(chosen)

    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "questions.jsonl", "w", encoding="utf-8") as fh:
        for qid, r in enumerate(chosen, 1):
            fh.write(json.dumps({"qid": qid, **r}, ensure_ascii=False) + "\n")
    for b in range(args.batches):
        (OUT / f"b{b + 1:02d}.ids").write_text(" ".join(str(i) for i in range(b + 1, len(chosen) + 1, args.batches)) + "\n")
    at = [r["source_rank"] for r in ranked]
    print(json.dumps({"messages": len(items), "v2_first": at.count(1), "v2_2_to_5": sum(2 <= x <= 5 for x in at),
                      "v2_6_to_20": sum(6 <= x <= 20 for x in at), "v2_past_20": at.count(DEPTH + 1),
                      "chosen": len(chosen), "hard": n_hard, "batches": args.batches}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
