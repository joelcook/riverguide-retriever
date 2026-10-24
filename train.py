"""Fine-tune Qwen3-Embedding-0.6B into RiverGuide's retriever.

    .venv/bin/python train.py --mode strict --name strict

Contrastive (InfoNCE) training: each query against its chunk, its two mined hard negatives, and
every other chunk in the batch. A batch candidate is masked out for a query when it is another
chunk of the right page, the same text, or when the stock model already scores it above the right
chunk (likely another right answer, the GISTEmbed idea, with the stock model's embeddings
precomputed). Validation is page-level top 5 and MRR@10 on held-out pages; the best step is kept.

omarchy shares this GPU with RiverGuide's Qwen. From :44 to :56 each hour the run moves to the
CPU so the :47 canary can load Qwen, then asks Ollama to unload it and moves back.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import random
import time
import urllib.request

import numpy as np
import torch
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer

from common import DATA, QUERY_INSTRUCT, ROOT, load_docs

BASE = "Qwen/Qwen3-Embedding-0.6B"
OLLAMA = "http://127.0.0.1:11434"


def log(name: str, msg: str) -> None:
    line = f"{dt.datetime.now():%H:%M:%S} {msg}"
    print(line, flush=True)
    with open(ROOT / "logs" / f"train-{name}.log", "a") as fh:
        fh.write(line + "\n")


def ollama_loaded() -> list[str]:
    try:
        with urllib.request.urlopen(f"{OLLAMA}/api/ps", timeout=10) as r:
            return [m["name"] for m in json.load(r).get("models", [])]
    except OSError:
        return []


def unload_ollama() -> None:
    for name in ollama_loaded():
        body = json.dumps({"model": name, "keep_alive": 0}).encode()
        try:
            urllib.request.urlopen(urllib.request.Request(f"{OLLAMA}/api/generate", body, {"content-type": "application/json"}), timeout=120).read()
        except OSError:
            pass
    for _ in range(60):
        if not ollama_loaded():
            return
        time.sleep(2)


def move_optimizer(optimizer: torch.optim.Optimizer, device: str) -> None:
    for state in optimizer.state.values():
        for k, v in state.items():
            if torch.is_tensor(v):
                state[k] = v.to(device)


def gpu_window(name: str, model, optimizer) -> None:
    if not 44 <= dt.datetime.now().minute < 56:
        return
    log(name, "moving to CPU for the :47 canary")
    model.to("cpu")
    move_optimizer(optimizer, "cpu")
    torch.cuda.empty_cache()
    while 44 <= dt.datetime.now().minute < 56:
        time.sleep(15)
    unload_ollama()
    model.to("cuda")
    move_optimizer(optimizer, "cuda")
    log(name, "back on the GPU")


def encode(model, texts: list[str], micro: int) -> torch.Tensor:
    """Embed with gradients, in length-sorted micro-batches to cut padding."""
    order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
    out = [None] * len(texts)
    for s in range(0, len(order), micro):
        idx = order[s : s + micro]
        features = model.preprocess([texts[i] for i in idx])
        features = {k: (v.to("cuda") if torch.is_tensor(v) else v) for k, v in features.items()}
        emb = F.normalize(model(features)["sentence_embedding"].float(), dim=-1)
        for j, i in enumerate(idx):
            out[i] = emb[j]
    return torch.stack(out)


@torch.no_grad()
def validate(model, docs, val, page_ids) -> dict:
    model.eval()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        d = model.encode([x["text"] for x in docs], batch_size=64, normalize_embeddings=True, convert_to_tensor=True).float()
        q = model.encode([QUERY_INSTRUCT + v["query"] for v in val], batch_size=128, normalize_embeddings=True, convert_to_tensor=True).float()
    model.train()
    n_pages = int(page_ids.max()) + 1
    url_page = {x["url"]: x["page"] for x in docs}
    hits, rr = 0, 0.0
    scores = q @ d.T
    for i, v in enumerate(val):
        best = torch.full((n_pages,), -1e9, device="cuda").scatter_reduce(0, page_ids, scores[i], reduce="amax")
        top = torch.topk(best, 10).indices.tolist()
        target = url_page[v["url"]]
        if target in top[:5]:
            hits += 1
        if target in top:
            rr += 1 / (top.index(target) + 1)
    return {"top5": hits / len(val), "mrr10": rr / len(val)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default="strict")
    parser.add_argument("--name", required=True)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--tau", type=float, default=0.05)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--micro", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    (ROOT / "logs").mkdir(exist_ok=True)
    log(args.name, f"args {vars(args)}")

    docs = load_docs()
    rows = [json.loads(line) for line in open(DATA / f"train.{args.mode}.jsonl", encoding="utf-8")]
    val = [json.loads(line) for line in open(DATA / "val.jsonl", encoding="utf-8")][:800]
    guide_docs = torch.tensor(np.load(ROOT / "runs/emb-stock.npy"), device="cuda")
    guide_q = torch.tensor(np.load(DATA / f"qemb-stock-{args.mode}.npy"), device="cuda")
    page_ids = torch.tensor([d["page"] for d in docs], device="cuda")
    text_key = [hashlib.md5(d["text"].split("\n\n", 1)[-1].encode()).hexdigest() for d in docs]

    model = SentenceTransformer(BASE, device="cuda", model_kwargs={"dtype": torch.float32})
    model.max_seq_length = 512
    model[0].auto_model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.train()

    steps_per_epoch = len(rows) // args.batch
    total = int(steps_per_epoch * args.epochs)
    warmup = max(1, int(0.05 * total))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01, fused=True)
    schedule = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda s: min(1.0, (s + 1) / warmup) * max(0.0, (total - s) / max(1, total - warmup)))

    best = validate(model, docs, val, page_ids)
    log(args.name, f"step 0 val {best}  ({len(rows)} rows, {total} steps)")
    best_score, best_step = best["mrr10"], 0
    out_dir = ROOT / "models" / args.name
    model.save(str(out_dir))

    order: list[int] = []
    step, started = 0, time.time()
    while step < total:
        if not order:
            order = list(range(len(rows)))
            random.shuffle(order)
        batch = [rows[order.pop()] for _ in range(min(args.batch, len(order)))]
        gpu_window(args.name, model, optimizer)

        cand = list(dict.fromkeys([r["doc"] for r in batch] + [n for r in batch for n in r["negs"]]))
        col = {c: j for j, c in enumerate(cand)}
        cand_t = torch.tensor(cand, device="cuda")
        pos = torch.tensor([col[r["doc"]] for r in batch], device="cuda")

        # Mask candidates that are probably also right for a query.
        with torch.no_grad():
            g = guide_q[torch.tensor([r["row"] for r in batch], device="cuda")] @ guide_docs[cand_t].T
            g_pos = g.gather(1, pos[:, None])
            same_page = page_ids[cand_t][None, :] == page_ids[cand_t[pos]][:, None]
            keys = [text_key[c] for c in cand]
            same_text = torch.tensor([[keys[j] == text_key[r["doc"]] for j in range(len(cand))] for r in batch], device="cuda")
            mask = (same_page | same_text | (g > g_pos))
            mask[torch.arange(len(batch)), pos] = False

        with torch.autocast("cuda", dtype=torch.bfloat16):
            q = encode(model, [QUERY_INSTRUCT + r["q"] for r in batch], args.micro * 2)
            d = encode(model, [docs[c]["text"] for c in cand], args.micro)
        logits = (q @ d.T) / args.tau
        logits = logits.masked_fill(mask, float("-inf"))
        loss = F.cross_entropy(logits, pos)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        schedule.step()
        step += 1

        if step % 10 == 0:
            rate = step / (time.time() - started)
            log(args.name, f"step {step}/{total} loss {loss.item():.4f} lr {schedule.get_last_lr()[0]:.2e} "
                           f"masked/q {mask.sum().item() / len(batch):.1f} eta {(total - step) / rate / 60:.0f} min")
        if step % args.eval_every == 0 or step == total:
            v = validate(model, docs, val, page_ids)
            log(args.name, f"step {step} val {v}")
            if v["mrr10"] > best_score:
                best_score, best_step = v["mrr10"], step
                model.save(str(out_dir))
                log(args.name, f"saved best (mrr10 {best_score:.4f}) to {out_dir}")

    log(args.name, f"done: best step {best_step}, val mrr10 {best_score:.4f}, {(time.time() - started) / 60:.0f} min")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
