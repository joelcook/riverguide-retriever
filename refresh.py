"""Keep the retriever current with RiverGuide's corpus, and say when it needs retraining.

    .venv/bin/python refresh.py            # nightly: re-embed tonight's crawl with models/production
    .venv/bin/python refresh.py --check    # Sundays: also test it on that crawl; alert on a drop

Facts live in the pages, so a changed fee or deadline needs only a re-embed, never a retrain.
The chunks of ../riverguide/corpus, shown as common.build_docs shows them, go to index/:
docs.jsonl, emb.npy, results.jsonl (each chunk as RiverGuide's search tool returns it, for
retriever_service.py) and meta.json (corpus date and commit, model, pages new since training).
It is built in index.new/ and swapped in whole.

The check runs the frozen tests on the new crawl: the 75 abbreviations, the golden questions
that name pages to cite, and the val queries whose pages still exist. Retraining is due when
the College's vocabulary moves (new programs, new names), which shows up here as a drop: a
test under its floor opens riverguide:retriever-drift through ops-alert, and the next passing
check clears it. Every check is appended to logs/drift.jsonl.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import shutil
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from common import DATA, QUERY_INSTRUCT, RIVERGUIDE, ROOT, build_docs
from evaluate import DEPTH, grade, load_tests
from riverguide.corpus import load_pages, read_checked
from riverguide.index import SearchIndex

INDEX = ROOT / "index"
DRIFT_LOG = ROOT / "logs" / "drift.jsonl"
ALERT_KEY = "riverguide:retriever-drift"
# Set by the first check from its own scores, a little under them (MARGIN), and kept: a later
# check below a floor means the College's pages have moved away from what the model learned.
FLOORS_FILE = ROOT / "drift_floors.json"
MARGIN = {"abbreviations": 3, "golden": 1, "val": 0.02}


def wait_for_canary() -> None:
    """Stay off the GPU from :44 to :56, while RiverGuide's :47 canary loads Qwen."""
    while 44 <= time.localtime().tm_min < 56:
        time.sleep(30)


def load_model(path: str) -> SentenceTransformer:
    if torch.cuda.is_available():
        try:
            model = SentenceTransformer(path, device="cuda", model_kwargs={"dtype": torch.bfloat16})
            model.max_seq_length = 512
            return model
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
    model = SentenceTransformer(path, device="cpu")
    model.max_seq_length = 512
    return model


def encode(model: SentenceTransformer, texts: list[str], batch_size: int) -> np.ndarray:
    """On the GPU beside Qwen (kept loaded) and the live retriever: 1.6 GiB at batch 32, about 1 GiB
    to spare. If the GPU still runs out part way, all of it again on the CPU, in one precision."""
    try:
        return model.encode(texts, batch_size=batch_size, normalize_embeddings=True, convert_to_numpy=True)
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        model.to("cpu", dtype=torch.float32)
        return model.encode(texts, batch_size=batch_size, normalize_embeddings=True, convert_to_numpy=True)


def git_head(repo: Path) -> str:
    return subprocess.run(["git", "-C", str(repo), "rev-parse", "--short", "HEAD"],
                          capture_output=True, text=True).stdout.strip()


def refresh(model: SentenceTransformer, model_path: str) -> tuple[SearchIndex, list[dict], np.ndarray, dict]:
    corpus = RIVERGUIDE / "corpus"
    index = SearchIndex(load_pages(corpus), read_checked(corpus))
    docs = build_docs(index)
    emb = encode(model, [d["text"] for d in docs], 32).astype(np.float32)
    trained = DATA / "docs.jsonl"
    trained_urls = {json.loads(line)["url"] for line in open(trained, encoding="utf-8")} if trained.exists() else None
    meta = {
        "built": dt.datetime.now().astimezone().isoformat(timespec="seconds"),
        "model": model_path,
        "corpus_checked": index.checked,
        "corpus_commit": git_head(RIVERGUIDE),
        "pages": len(index.pages),
        "chunks": len(docs),
        "pages_new_since_training": None if trained_urls is None else len({p.url for p in index.pages} - trained_urls),
    }
    new, old = INDEX.with_name("index.new"), INDEX.with_name("index.old")
    shutil.rmtree(new, ignore_errors=True)
    new.mkdir()
    with open(new / "docs.jsonl", "w", encoding="utf-8") as fh:
        for d in docs:
            fh.write(json.dumps(d) + "\n")
    np.save(new / "emb.npy", emb)
    with open(new / "results.jsonl", "w", encoding="utf-8") as fh:
        for cid in range(len(docs)):
            fh.write(json.dumps(index._result(cid)) + "\n")
    (new / "meta.json").write_text(json.dumps(meta, indent=2))
    shutil.rmtree(old, ignore_errors=True)
    if INDEX.exists():
        INDEX.rename(old)
    new.rename(INDEX)
    shutil.rmtree(old, ignore_errors=True)
    return index, docs, emb, meta


def check(model: SentenceTransformer, index: SearchIndex, docs: list[dict], emb: np.ndarray) -> dict:
    urls = [p.url for p in index.pages]
    doc_page = np.array([d["page"] for d in docs])
    tests = load_tests()
    present = set(urls)
    tests["val"] = [c for c in tests.get("val", []) if c["url"] in present]
    scores = {}
    for name, cases in tests.items():
        q = encode(model, [QUERY_INSTRUCT + c["query"] for c in cases], 64).astype(np.float32)
        rankings = []
        for s in q @ emb.T:
            best = np.full(len(urls), -np.inf, dtype=np.float32)
            np.maximum.at(best, doc_page, s)
            rankings.append(list(np.argsort(-best)[:DEPTH]))
        result = grade(cases, rankings, urls)
        scores[name] = {"top5": result["top5"], "n": result["n"], "misses": result["misses"][:20]}
    return scores


def val_rate(scores: dict) -> float:
    return scores["val"]["top5"] / max(1, scores["val"]["n"])


def floors(scores: dict) -> dict:
    if not FLOORS_FILE.exists():
        FLOORS_FILE.write_text(json.dumps({
            "set": dt.date.today().isoformat(),
            "abbreviations": scores["abbreviations"]["top5"] - MARGIN["abbreviations"],
            "golden": scores["golden"]["top5"] - MARGIN["golden"],
            "val": round(val_rate(scores) - MARGIN["val"], 4),
        }, indent=2))
        print(f"floors set from this check: {FLOORS_FILE.read_text()}", flush=True)
    return json.loads(FLOORS_FILE.read_text())


def problems(scores: dict, floor: dict) -> list[str]:
    out = []
    for test in ("abbreviations", "golden"):
        if scores[test]["top5"] < floor[test]:
            out.append(f"{test} {scores[test]['top5']}/{scores[test]['n']} (floor {floor[test]})")
    if val_rate(scores) < floor["val"]:
        out.append(f"val {val_rate(scores):.1%} of {scores['val']['n']} (floor {floor['val']:.0%})")
    return out


def alert(action: str, message: str = "") -> None:
    ops = shutil.which("ops-alert") or str(Path.home() / ".local/bin/ops-alert")
    if not Path(ops).exists():
        print(f"ops-alert not found; would {action} {ALERT_KEY}: {message}")
        return
    argv = [ops, action, "riverguide"] + ([message] if action == "open" else []) + ["--key", ALERT_KEY]
    subprocess.run(argv, check=False, timeout=30)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=str((ROOT / "models" / "production").resolve()))
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    wait_for_canary()
    model = load_model(args.model)
    index, docs, emb, meta = refresh(model, args.model)
    print(f"{meta['built']} re-embedded corpus {meta['corpus_checked']} ({meta['corpus_commit']}): "
          f"{meta['chunks']} chunks, {meta['pages_new_since_training']} pages new since training", flush=True)
    if not args.check:
        return 0

    scores = check(model, index, docs, emb)
    found = problems(scores, floors(scores))
    last = json.loads(DRIFT_LOG.read_text().splitlines()[-1]) if DRIFT_LOG.exists() and DRIFT_LOG.read_text().strip() else {}
    record = {**meta, "scores": scores, "problems": found}
    DRIFT_LOG.parent.mkdir(exist_ok=True)
    with open(DRIFT_LOG, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(record) + "\n")
    print("  ".join(f"{t} {s['top5']}/{s['n']}" for t, s in scores.items()), flush=True)
    if found:
        alert("open", f"retriever below its floor on corpus {meta['corpus_checked']}: {'; '.join(found)}. "
                      "Retrain on the new crawl (riverguide-retriever README, Reproduce).")
    elif last.get("problems"):
        alert("clear")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
