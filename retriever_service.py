"""RiverGuide's retriever as a service: models/production, over the nightly index.

    .venv/bin/python retriever_service.py           # 127.0.0.1:8791 (riverguide-retriever.service)

models/production links to the model that last passed RiverGuide's golden gate (gate_retriever.sh
moves it). POST /search {"query": "...", "limit": 8} returns {"checked", "results"} shaped exactly
like RiverGuide's search tool (title, url, updated, section, excerpt; the best chunk per page, at
most 10 pages). The Worker reaches it through the model gateway's /v1/search and falls back to
keyword search when it can't. On the GPU in bfloat16 (RETRIEVER_DEVICE=cuda, 15 ms a search) or
the CPU. It serves index/ as refresh.py last swapped it in, and picks up a new one by itself.
Logs outcomes and timing, never the query (RiverGuide's rule 7).
"""
from __future__ import annotations

import argparse
import json
import os
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from common import QUERY_INSTRUCT, ROOT

INDEX = ROOT / "index"
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
        pages = np.array([json.loads(line)["page"] for line in open(INDEX / "docs.jsonl", encoding="utf-8")])
        results = [json.loads(line) for line in open(INDEX / "results.jsonl", encoding="utf-8")]
        assert len(emb) == len(pages) == len(results), "index/ is inconsistent"
        with self.lock:
            self.meta, self.emb, self.pages, self.results, self.stamp = meta, emb, pages, results, meta["built"]
        print(json.dumps({"event": "index", "built": meta["built"], "checked": meta["corpus_checked"], "chunks": len(emb)}), flush=True)

    def search(self, q: np.ndarray, limit: int) -> dict:
        with self.lock:
            meta, emb, pages, results = self.meta, self.emb, self.pages, self.results
        scores = emb @ q
        picked, seen = [], set()
        for cid in np.argsort(-scores)[:400]:
            if pages[cid] in seen:
                continue
            seen.add(pages[cid])
            picked.append(results[cid])
            if len(picked) == limit:
                break
        return {"checked": meta["corpus_checked"], "results": picked}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=str(ROOT / "models" / "production"))
    parser.add_argument("--port", type=int, default=8791)
    parser.add_argument("--threads", type=int, default=8)
    # cuda: about 1.2 GB beside production's Qwen, in bfloat16 as refresh.py builds the index.
    parser.add_argument("--device", default=os.environ.get("RETRIEVER_DEVICE", "cpu"), choices=["cpu", "cuda"])
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    if args.device == "cuda":
        model = SentenceTransformer(args.model, device="cuda", model_kwargs={"dtype": torch.bfloat16})
    else:
        model = SentenceTransformer(args.model, device="cpu")
    model.max_seq_length = 256
    encode_lock = threading.Lock()
    index = Index()

    def watch() -> None:  # refresh.py swaps index/ whole; reload within a minute
        while True:
            time.sleep(60)
            try:
                index.load()
            except Exception as exc:  # noqa: BLE001  keep serving the old index
                print(json.dumps({"event": "index_error", "error": str(exc)[:200]}), flush=True)

    threading.Thread(target=watch, daemon=True).start()

    class Handler(BaseHTTPRequestHandler):
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
            out = index.search(q, limit)
            self._send(200, out)
            print(json.dumps({"event": "search", "ms": round((time.perf_counter() - started) * 1000, 1),
                              "results": len(out["results"])}), flush=True)

        def log_message(self, *a) -> None:
            pass

    print(f"retriever on 127.0.0.1:{args.port} ({args.model}, {args.device})", flush=True)
    ThreadingHTTPServer(("127.0.0.1", args.port), Handler).serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
