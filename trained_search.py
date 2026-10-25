"""RiverGuide's search tool answered by a trained retriever, for the production-loop tests.

evals/search_ab.py's EmbeddingSearch embeds chunks as "title\\nsection\\ntext". The retriever was
trained on "title > section\\n\\nexcerpt" (common.build_docs), so this embeds them that way, for
every model under test. Also: wait out RiverGuide's :47 canary, which shares Qwen with these tests.
"""
from __future__ import annotations

import hashlib
import time
from array import array
from pathlib import Path

from common import build_docs
from evals.search_ab import CACHE_DIR, EmbeddingSearch
from riverguide.index import SearchIndex


class TrainedSearch(EmbeddingSearch):
    def __init__(self, index: SearchIndex, model: str, base: str):
        self.index, self.model, self.base = index, model, base
        texts = [d["text"] for d in build_docs(index)]
        key = hashlib.sha256((model + "\0" + "\0".join(texts)).encode()).hexdigest()[:16]
        cache = CACHE_DIR / f"{model.replace(':', '-').replace('/', '-')}-trained-{key}.f32"
        flat = array("f")
        if cache.exists():
            flat.frombytes(cache.read_bytes())
        else:
            started = time.time()
            for i in range(0, len(texts), 64):
                for vector in self._embed(texts[i : i + 64]):
                    flat.extend(vector)
            CACHE_DIR.mkdir(parents=True, exist_ok=True)
            cache.write_bytes(flat.tobytes())
            print(f"embedded {len(texts)} chunks with {model} in {time.time() - started:.0f}s", flush=True)
        dim = len(flat) // len(texts)
        self.vectors = [flat[i * dim : (i + 1) * dim] for i in range(len(texts))]


MAINTENANCE = Path(__file__).resolve().parent / "lab" / ".maintenance"


def wait_for_canary() -> None:
    """Hold the next question from :44 to :50 so the :47 canary has Qwen to itself (not needed
    while the lab has RiverGuide offline: the canary is paused then)."""
    while 44 <= time.localtime().tm_min < 50 and not MAINTENANCE.exists():
        time.sleep(15)


def canary_safe(answer):
    def wrapped(*args, **kwargs):
        wait_for_canary()
        return answer(*args, **kwargs)
    return wrapped
