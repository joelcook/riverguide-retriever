"""Show a model's top pages for some queries: .venv/bin/python inspect_queries.py NAME MODEL "q1" "q2" ..."""
import sys

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from common import QUERY_INSTRUCT, ROOT, load_docs

name, model_path, queries = sys.argv[1], sys.argv[2], sys.argv[3:]
docs = load_docs()
emb = np.load(ROOT / f"runs/emb-{name}.npy")
m = SentenceTransformer(model_path, device="cuda", model_kwargs={"dtype": torch.bfloat16})
q = m.encode([QUERY_INSTRUCT + x for x in queries], normalize_embeddings=True)
page = np.array([d["page"] for d in docs])
url = {d["page"]: d["url"] for d in docs}
for text, qv in zip(queries, q):
    s = emb @ qv
    best = {}
    for i in np.argsort(-s)[:200]:
        best.setdefault(int(page[i]), float(s[i]))
    print(f"\n{text}")
    for p, sc in list(best.items())[:5]:
        print(f"  {sc:.3f} {url[p].replace('https://', '')[:110]}")
