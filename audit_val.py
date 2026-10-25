"""Audit val: per-kind scores, near-duplicate leakage from training pages, paired comparison."""
import json, sys
from collections import defaultdict
from math import comb
sys.path.insert(0, ".")
import numpy as np, torch
from sentence_transformers import SentenceTransformer
from common import DATA, QUERY_INSTRUCT, ROOT, load_docs, norm

docs = load_docs()
doc_page = np.array([d["page"] for d in docs])
n_pages = int(doc_page.max()) + 1
url_of = {}
for d in docs:
    url_of[d["page"]] = d["url"]
page_of_url = {u: p for p, u in url_of.items()}
val = [json.loads(l) for l in open(DATA / "val.jsonl")]
train = [json.loads(l) for l in open(DATA / "train.full.jsonl")]
train_pages = {r["page"] for r in train}
val_pages = {page_of_url[c["url"]] for c in val}
print(f"val pages {len(val_pages)}, train pages {len(train_pages)}, overlap {len(val_pages & train_pages)}")

# Near-twin: a val page's chunk nearly identical (stock cosine) to some train page's chunk.
E = np.load(ROOT / "runs/emb-stock.npy")
train_mask = np.isin(doc_page, list(train_pages))
T = E[train_mask]
twin = {}
for p in val_pages:
    idx = np.where(doc_page == p)[0]
    twin[p] = float((E[idx] @ T.T).max())
sims = np.array(list(twin.values()))
print("val page max-cos to any train chunk: " + ", ".join(f">={t}: {(sims >= t).sum()}" for t in (0.99, 0.97, 0.95, 0.9)))

# Exact query-string overlap with training queries.
train_q = {norm(r["q"]) for r in train}
dup = [norm(c["query"]) in train_q for c in val]
print(f"val queries whose text also appears as a training query: {sum(dup)}/{len(val)}")

def ranks(model_path):
    m = SentenceTransformer(model_path, device="cuda", model_kwargs={"dtype": torch.bfloat16})
    name = {"Qwen/Qwen3-Embedding-0.6B": "stock"}.get(model_path, model_path.split("/")[-1])
    D = np.load(ROOT / f"runs/emb-{name}.npy")
    q = m.encode([QUERY_INSTRUCT + c["query"] for c in val], batch_size=64, normalize_embeddings=True).astype(np.float32)
    out = []
    for c, s in zip(val, q @ D.T):
        best = np.full(n_pages, -np.inf, dtype=np.float32)
        np.maximum.at(best, doc_page, s)
        order = np.argsort(-best)
        out.append(int(np.where(order == page_of_url[c["url"]])[0][0]) + 1)
    del m; torch.cuda.empty_cache()
    return np.array(out)

R = {n: ranks(p) for n, p in (("stock", "Qwen/Qwen3-Embedding-0.6B"), ("strict", "models/strict"), ("full", "models/full"))}

def row(mask, label):
    n = int(mask.sum())
    cells = "  ".join(f"{k} {int((r[mask] <= 5).sum()):4d} ({(r[mask] <= 5).mean():.0%})" for k, r in R.items())
    print(f"{label:34} n={n:4d}  {cells}")

print("\nright page in top 5")
row(np.ones(len(val), bool), "all val")
kinds = np.array([c["kind"] for c in val])
for k in ("question", "search", "shorthand", "typo", "program"):
    row(kinds == k, f"  kind={k}")
vt = np.array([twin[page_of_url[c["url"]]] for c in val])
row(vt >= 0.95, "  page has near-twin in train (>=.95)")
row(vt < 0.95, "  no near-twin (<.95)")
row(vt < 0.90, "  no near-twin (<.90)")
row(~np.array(dup), "  query text not in train")

def mcnemar(a, b):
    x, y = int((a & ~b).sum()), int((~a & b).sum())
    n, k = x + y, min(x, y)
    return x, y, min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n) if n else 1.0
for a, b in (("strict", "stock"), ("full", "stock"), ("full", "strict")):
    x, y, p = mcnemar(R[a] <= 5, R[b] <= 5)
    print(f"{a} vs {b}: {a} only {x}, {b} only {y}, p={p:.2g}")
lose = [(c["query"], c["kind"], int(R['stock'][i]), int(R['full'][i])) for i, c in enumerate(val) if R['stock'][i] <= 5 < R['full'][i]]
print(f"\nfull loses vs stock ({len(lose)}):")
for q in lose[:15]: print("  ", q)
