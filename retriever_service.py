"""RiverGuide's retriever as a service: models/production, over the nightly index.

    .venv/bin/python retriever_service.py           # 127.0.0.1:8791 (riverguide-retriever.service)

models/production links to the model that last passed RiverGuide's golden gate (gate_retriever.sh
moves it). POST /search {"query": "...", "limit": 8} returns {"checked", "results"} shaped exactly
like RiverGuide's search tool (title, url, updated, section, excerpt; the best chunk per page, at
most 10 pages). The Worker reaches it through the model gateway's /v1/search and falls back to
keyword search when it can't. On the GPU in bfloat16 (RETRIEVER_DEVICE=cuda, 15 ms a search) or
the CPU. It serves index/ as refresh.py last swapped it in, and picks up a new one by itself.
Logs outcomes and timing, never the query (RiverGuide's rule 7).

With a reranker (--rerank, by default models/production-rerank when that link exists), the model's
top 20 pages (their best chunks) are re-sorted by it before the first `limit` are returned: a
fine-tuned Qwen3-Reranker-0.6B (train_reranker.py), scored as rerank.Judge scores, in this process
so it shares the GPU context (about 1.3 GB more). --plain-port also serves the same index without
the reranker, from the same models: gate_retriever.sh --rerank compares the two that way, because
two retriever processes beside Qwen don't fit on the card.
"""
from __future__ import annotations

import argparse
import json
import os
import threading
import time

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")  # less fragmentation in a tight GPU
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from common import QUERY_INSTRUCT, ROOT

INDEX = ROOT / "index"
RERANK_DEPTH = 20  # pages the reranker re-sorts, as it was trained and tested (real_questions.py +rerank20)
RERANK_BATCH = int(os.environ.get("RERANK_BATCH", 5))
MAX_RESULTS = 10
MAX_QUERY_CHARS = 2000


class Index:
    """index/ as one consistent snapshot, reloaded when refresh.py swaps in a new one."""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.stamp = None
        self.load()

    def load(self) -> None:
        meta = json.loads((INDEX / "meta.json").read_text())
        if meta["built"] == self.stamp:
            return
        emb = np.load(INDEX / "emb.npy")
        docs = [json.loads(line) for line in open(INDEX / "docs.jsonl", encoding="utf-8")]
        pages = np.array([d["page"] for d in docs])
        texts = [d["text"] for d in docs]  # what the reranker reads, as it trained on
        results = [json.loads(line) for line in open(INDEX / "results.jsonl", encoding="utf-8")]
        assert len(emb) == len(pages) == len(results), "index/ is inconsistent"
        with self.lock:
            self.meta, self.emb, self.pages, self.texts, self.results, self.stamp = meta, emb, pages, texts, results, meta["built"]
        print(json.dumps({"event": "index", "built": meta["built"], "checked": meta["corpus_checked"], "chunks": len(emb)}), flush=True)

    def search(self, q: np.ndarray, limit: int, rerank=None) -> dict:
        """Each page's best chunk, best first; with rerank(chunk texts) -> scores, the top RERANK_DEPTH re-sorted."""
        with self.lock:
            meta, emb, pages, texts, results = self.meta, self.emb, self.pages, self.texts, self.results
        scores = emb @ q
        picked, seen = [], set()
        for cid in np.argsort(-scores)[:400]:
            if pages[cid] in seen:
                continue
            seen.add(pages[cid])
            picked.append(int(cid))
            if len(picked) == (RERANK_DEPTH if rerank else limit):
                break
        s = rerank([texts[c] for c in picked]) if rerank else None
        if s:
            picked = [picked[j] for j in sorted(range(len(picked)), key=lambda j: -s[j])]
        return {"checked": meta["corpus_checked"], "results": [results[c] for c in picked[:limit]]}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=str(ROOT / "models" / "production"))
    parser.add_argument("--port", type=int, default=8791)
    parser.add_argument("--threads", type=int, default=8)
    # cuda: about 1.2 GB beside production's Qwen, in bfloat16 as refresh.py builds the index.
    parser.add_argument("--device", default=os.environ.get("RETRIEVER_DEVICE", "cpu"), choices=["cpu", "cuda"])
    production_rerank = ROOT / "models" / "production-rerank"
    parser.add_argument("--rerank", default=str(production_rerank) if production_rerank.exists() else None,
                        help="a trained reranker; 'none' for none")
    parser.add_argument("--plain-port", type=int, help="also serve without the reranker on this port")
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    if args.device == "cuda":
        model = SentenceTransformer(args.model, device="cuda", model_kwargs={"dtype": torch.bfloat16})
    else:
        model = SentenceTransformer(args.model, device="cpu")
    model.max_seq_length = 256
    encode_lock = threading.Lock()
    index = Index()
    reranker = rerank_lock = None
    if args.rerank and args.rerank != "none":
        from rerank import Judge

        reranker, rerank_lock = Judge(device=args.device, model=args.rerank), threading.Lock()

    def watch() -> None:  # refresh.py swaps index/ whole; reload within a minute
        while True:
            time.sleep(60)
            try:
                index.load()
            except Exception as exc:  # noqa: BLE001  keep serving the old index
                print(json.dumps({"event": "index_error", "error": str(exc)[:200]}), flush=True)

    threading.Thread(target=watch, daemon=True).start()

    class Handler(BaseHTTPRequestHandler):
        reranks = reranker is not None

        def _send(self, status: int, body: dict) -> None:
            data = json.dumps(body).encode()
            self.send_response(status)
            self.send_header("content-type", "application/json")
            self.send_header("content-length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self) -> None:  # noqa: N802
            if self.path != "/health":
                return self._send(404, {"error": "Not found."})
            self._send(200, {"built": index.meta["built"], "checked": index.meta["corpus_checked"], "chunks": len(index.results)})

        def do_POST(self) -> None:  # noqa: N802
            started = time.perf_counter()
            if self.path != "/search":
                return self._send(404, {"error": "Not found."})
            try:
                body = json.loads(self.rfile.read(min(int(self.headers.get("content-length") or 0), 16_000)))
                query = str(body["query"]).strip()[:MAX_QUERY_CHARS]
                limit = max(1, min(int(body.get("limit") or 8), MAX_RESULTS))
            except (ValueError, KeyError, TypeError):
                return self._send(400, {"error": "Send {\"query\": \"...\", \"limit\": 8}."})
            if not query:
                return self._send(400, {"error": "Empty query."})
            with encode_lock:
                q = model.encode(QUERY_INSTRUCT + query, normalize_embeddings=True).astype(np.float32)

            def rerank(texts: list[str]) -> list[float] | None:
                # Five at a time: beside Qwen the card has under 2 GB to spare, and twenty at once
                # outgrew it. Out of memory anyway, the search keeps the model's own order.
                with rerank_lock:
                    try:
                        return reranker.score([(query, t) for t in texts], batch=RERANK_BATCH, wait=False)
                    except torch.OutOfMemoryError:
                        torch.cuda.empty_cache()
                        print(json.dumps({"event": "rerank_out_of_memory"}), flush=True)
                        return None

            out = index.search(q, limit, rerank if self.reranks else None)
            self._send(200, out)
            print(json.dumps({"event": "search", "ms": round((time.perf_counter() - started) * 1000, 1),
                              "results": len(out["results"]), "reranked": self.reranks}), flush=True)

        def log_message(self, *a) -> None:
            pass

    class Plain(Handler):
        reranks = False

    print(f"retriever on 127.0.0.1:{args.port} ({args.model}, {args.device}, reranker {args.rerank if reranker else 'none'})", flush=True)
    if args.plain_port:
        print(f"without the reranker on 127.0.0.1:{args.plain_port}", flush=True)
        threading.Thread(target=ThreadingHTTPServer(("127.0.0.1", args.plain_port), Plain).serve_forever, daemon=True).start()
    ThreadingHTTPServer(("127.0.0.1", args.port), Handler).serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
