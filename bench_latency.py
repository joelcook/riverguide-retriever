"""Time one query through the retriever: embed it, score every chunk, take the top pages.

    .venv/bin/python bench_latency.py MODEL
"""
import statistics
import sys
import time

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from common import QUERY_INSTRUCT

model_path = sys.argv[1]
queries = ["RN", "how do I drop a class", "when is the fall 2026 drop deadline", "rad tech program requirements", "pharm tech"]
docs = np.random.default_rng(0).standard_normal((8484, 1024)).astype(np.float32)
for device, dtype in (("cuda", torch.bfloat16), ("cpu", torch.float32)):
    torch.set_num_threads(8)
    m = SentenceTransformer(model_path, device=device, model_kwargs={"dtype": dtype})
    for q in queries:
        m.encode(QUERY_INSTRUCT + q)
    times = []
    for _ in range(10):
        for q in queries:
            t = time.perf_counter()
            v = m.encode(QUERY_INSTRUCT + q, normalize_embeddings=True)
            top = np.argsort(-(docs @ v.astype(np.float32)))[:50]
            times.append((time.perf_counter() - t) * 1000)
    print(f"{device:4} {str(dtype):15} median {statistics.median(times):.1f} ms  p90 {sorted(times)[int(0.9 * len(times))]:.1f} ms  (8 CPU threads)")
    del m
