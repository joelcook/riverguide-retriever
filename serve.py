"""Serve a trained retriever behind Ollama's API, so riverguide's evals can use it unchanged.

    .venv/bin/python serve.py --model models/strict --port 11435
    # then, in ../riverguide:
    .venv/bin/python -m evals.search_ab --search embedding --ollama http://127.0.0.1:11435 --embed-model riverguide-retriever

POST /api/embed is answered by the model (any "model" name); every other request is passed
through to Ollama on 11434, so the answering model is still production's Qwen.
"""
from __future__ import annotations

import argparse
import json
import threading
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import torch
from sentence_transformers import SentenceTransformer

UPSTREAM = "http://127.0.0.1:11434"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--port", type=int, default=11435)
    args = parser.parse_args()
    model = SentenceTransformer(args.model, device="cuda", model_kwargs={"dtype": torch.bfloat16})
    model.max_seq_length = 512
    lock = threading.Lock()

    class Handler(BaseHTTPRequestHandler):
        def _body(self) -> bytes:
            return self.rfile.read(int(self.headers.get("content-length") or 0))

        def _send(self, status: int, body: bytes, ctype: str = "application/json") -> None:
            self.send_response(status)
            self.send_header("content-type", ctype)
            self.send_header("content-length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _proxy(self, method: str) -> None:
            body = self._body() if method == "POST" else None
            request = urllib.request.Request(UPSTREAM + self.path, body, method=method,
                                             headers={"content-type": self.headers.get("content-type", "application/json")})
            try:
                with urllib.request.urlopen(request, timeout=900) as response:
                    self._send(response.status, response.read(), response.headers.get("content-type", "application/json"))
            except urllib.error.HTTPError as exc:
                self._send(exc.code, exc.read())

        def do_GET(self) -> None:  # noqa: N802
            self._proxy("GET")

        def do_POST(self) -> None:  # noqa: N802
            if self.path != "/api/embed":
                return self._proxy("POST")
            request = json.loads(self._body())
            texts = request["input"] if isinstance(request["input"], list) else [request["input"]]
            with lock:
                vectors = model.encode(texts, batch_size=16, normalize_embeddings=True).tolist()
            self._send(200, json.dumps({"model": request.get("model"), "embeddings": vectors}).encode())

        def log_message(self, *a) -> None:
            pass

    print(f"serving {args.model} on 127.0.0.1:{args.port}", flush=True)
    ThreadingHTTPServer(("127.0.0.1", args.port), Handler).serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
