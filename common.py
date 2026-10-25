"""What the retriever searches: RiverGuide's own chunks, built from corpus/ by riverguide.index.

A doc is one search chunk shown the way search shows it: the page title, the
section, and the excerpt (the chunk plus the short chunks after it), so the
model embeds what the answering model would read.
"""
from __future__ import annotations

import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RIVERGUIDE = Path("/home/irsc/Projects/riverguide")
sys.path.insert(0, str(RIVERGUIDE))

from functools import lru_cache  # noqa: E402

from riverguide.corpus import load_pages, read_checked  # noqa: E402
from riverguide.index import EXCERPT_CHARS, SearchIndex  # noqa: E402

DATA = ROOT / "data"
# Everything is pinned to the corpus the queries were written from (riverguide 5d125e7, the
# 2026-10-22 crawl): the nightly crawl renumbers chunks. Rebuild it with
#   git -C ../riverguide archive 5d125e7 corpus | tar -x -C data/corpus-2026-10-22
SNAPSHOT = DATA / "corpus-2026-10-22" / "corpus"
DOC_CHARS = 1800
QUERY_INSTRUCT = (
    "Instruct: Given a college student's search query, retrieve the College web page that answers it\nQuery: "
)


@lru_cache(maxsize=1)
def snapshot_index() -> SearchIndex:
    return SearchIndex(load_pages(SNAPSHOT), read_checked(SNAPSHOT))


def build_docs(index: SearchIndex | None = None) -> list[dict]:
    index = index or snapshot_index()
    chunks = index._chunks
    repeats = Counter(hashlib.md5(c.text.encode()).hexdigest() for c in chunks)
    docs = []
    for cid, chunk in enumerate(chunks):
        page = index.pages[chunk.page]
        excerpt = chunk.text
        nxt = cid + 1
        while len(excerpt) < EXCERPT_CHARS // 2 and nxt < len(chunks) and chunks[nxt].page == chunk.page:
            excerpt = f"{excerpt}\n\n{chunks[nxt].text}"
            nxt += 1
        head = page.title if not chunk.section else f"{page.title} > {chunk.section}"
        docs.append({
            "id": cid,
            "page": chunk.page,
            "url": page.url,
            "title": page.title,
            "section": chunk.section,
            "chunk_chars": len(chunk.text),
            "repeats": repeats[hashlib.md5(chunk.text.encode()).hexdigest()],
            "text": f"{head}\n\n{excerpt}"[:DOC_CHARS],
        })
    return docs


def load_docs() -> list[dict]:
    path = DATA / "docs.jsonl"
    if not path.exists():
        DATA.mkdir(exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            for d in build_docs():
                fh.write(json.dumps(d) + "\n")
    return [json.loads(line) for line in open(path, encoding="utf-8")]


def norm(text: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", text.lower()))
