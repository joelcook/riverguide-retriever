"""Package a trained model for release: bf16 weights, a model card, zipped.

    .venv/bin/python package.py strict full

Writes dist/riverguide-retriever-<name>.zip for each. The card carries the query prefix,
which the model needs at search time.
"""
from __future__ import annotations

import json
import shutil
import sys

import torch
from sentence_transformers import SentenceTransformer

from common import QUERY_INSTRUCT, ROOT

CARD = """# riverguide-retriever-{name}

Qwen3-Embedding-0.6B fine-tuned to find Indian River State College web pages for students'
questions, abbreviations, and nicknames (RN, rad tech, BSN, plumber). One forward pass per query,
no generated tokens. Built in https://github.com/joelcook/riverguide-retriever.

{variant}

## Use

```python
from sentence_transformers import SentenceTransformer
model = SentenceTransformer("riverguide-retriever-{name}")
INSTRUCT = {instruct!r}
q = model.encode(INSTRUCT + "rad tech", normalize_embeddings=True)
d = model.encode(["<page title> > <section>\\n\\n<chunk text>"], normalize_embeddings=True)
scores = d @ q
```

Queries need the instruction prefix; chunks are embedded as plain text. A page's score is its
best chunk's.

## Results (right page in the top 5)

{results}
"""

VARIANTS = {
    "strict": "**strict**: no training query contains any of the 75 test abbreviations, so its abbreviation "
              "score measures abbreviations it never saw in training.",
    "full": "**full**: trained on every generated query, including ones with the test abbreviations. Its "
            "abbreviation score is contaminated; use it in production, report strict.",
}


def results_table(name: str) -> str:
    rows = ["| | keyword (BM25) | stock 0.6B | this model | hybrid |", "|---|---|---|---|---|"]
    stock = json.loads((ROOT / "runs/stock.json").read_text())
    mine = json.loads((ROOT / f"runs/{name}.json").read_text())
    for test in ("abbreviations", "golden", "val"):
        if test in mine:
            k, s, m, h = stock[test]["keyword"], stock[test]["dense"], mine[test]["dense"], mine[test]["hybrid"]
            rows.append(f"| {test} | {k['top5']}/{k['n']} | {s['top5']}/{s['n']} | {m['top5']}/{m['n']} | {h['top5']}/{h['n']} |")
    return "\n".join(rows)


def main() -> int:
    for name in sys.argv[1:]:
        out = ROOT / "dist" / f"riverguide-retriever-{name}"
        shutil.rmtree(out, ignore_errors=True)
        model = SentenceTransformer(str(ROOT / "models" / name), device="cpu", model_kwargs={"dtype": torch.bfloat16})
        model.save(str(out))
        (out / "README.md").write_text(CARD.format(name=name, variant=VARIANTS.get(name, ""), instruct=QUERY_INSTRUCT,
                                                   results=results_table(name)))
        archive = shutil.make_archive(str(out), "zip", root_dir=out.parent, base_dir=out.name)
        print(archive)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
