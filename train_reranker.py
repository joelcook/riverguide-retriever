"""Fine-tune Qwen3-Reranker-0.6B on grades of v2's top 20 pages (rerank_data.py): production's Qwen's for
all 1,200 items, Claude Opus 5.5's for the 125 it graded before it used up the plan (--grades).

    .venv/bin/python train_reranker.py --name reranker-v1      # a night job: about 10 GB of GPU
        models/reranker-v1 (the best step on held-out questions), logs/train-reranker-v1.log,
        runs/reranker-v1.json (the held-out scores, stock against trained)

Each question's 20 candidates are scored together, as at search time. Two losses:
  - list: softmax over the 20 scores against Opus's grades (2 full, 1 a quarter, 0 none), so
    the pages Opus would put first come first; a question Opus graded all 0 sits this one out.
  - point: binary cross-entropy of P(yes) against the grade (2 -> 1, 1 -> 0.5, 0 -> 0), so the
    score stays a probability: build_data.py's judge reads it as one.
A tenth of the questions (by id) are held out; the step with the best held-out NDCG@5 is kept.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import random
import time

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from common import DATA, ROOT, load_docs
from rerank import INSTRUCT, MODEL, PREFIX, SUFFIX

GAIN = {0: 0.0, 1: 0.25, 2: 1.0}  # list targets
POINT = {0: 0.0, 1: 0.5, 2: 1.0}  # point targets


def log(name: str, msg: str) -> None:
    line = f"{dt.datetime.now():%H:%M:%S} {msg}"
    print(line, flush=True)
    with open(ROOT / "logs" / f"train-{name}.log", "a") as fh:
        fh.write(line + "\n")


def grades_by(grader: str) -> dict[int, list[int]]:
    out = {}
    for path in sorted((DATA / "rerank" / grader).glob("b*.jsonl")):
        for line in open(path, encoding="utf-8"):
            r = json.loads(line)
            out[r["q"]] = r["g"]  # the last save of an item wins
    return out


def load(graders: list[str], name: str) -> list[dict]:
    """Each item's grades from the graders in order, a later grader's replacing an earlier one's."""
    questions = {json.loads(l)["qid"]: json.loads(l) for l in open(DATA / "rerank/questions.jsonl", encoding="utf-8")}
    by = {g: grades_by(g) for g in graders}
    if "qwen" in by and "opus" in by:  # how far Qwen agrees with Opus where both graded
        both = [q for q in by["opus"] if q in by["qwen"]]
        pairs = [(a, b) for q in both for a, b in zip(by["opus"][q], by["qwen"][q])]
        if pairs:
            log(name, f"Qwen vs Opus on {len(both)} items: same grade {sum(a == b for a, b in pairs) / len(pairs):.1%} of passages, "
                      f"same 2s {sum((a == 2) == (b == 2) for a, b in pairs) / len(pairs):.1%}")
    graded: dict[int, list[int]] = {}
    for g in graders:
        graded.update(by[g])
    return [{**questions[q], "grades": g} for q, g in graded.items() if q in questions and len(g) == len(questions[q]["candidates"])]


class Scorer:
    """P(yes) logits for (query, passage) pairs in the model card's format, as rerank.Judge scores them."""

    def __init__(self, path: str, max_length: int = 640, train: bool = False):
        self.tok = AutoTokenizer.from_pretrained(path, padding_side="left")
        # Training keeps fp32 weights (AdamW's state too) under bf16 autocast, as train.py does.
        dtype = torch.float32 if train else torch.bfloat16
        self.model = AutoModelForCausalLM.from_pretrained(path, dtype=dtype).cuda()
        self.yes, self.no = self.tok.convert_tokens_to_ids("yes"), self.tok.convert_tokens_to_ids("no")
        self.prefix = self.tok.encode(PREFIX, add_special_tokens=False)
        self.suffix = self.tok.encode(SUFFIX, add_special_tokens=False)
        self.room = max_length - len(self.prefix) - len(self.suffix)

    def logits(self, query: str, passages: list[str]) -> torch.Tensor:
        texts = [f"<Instruct>: {INSTRUCT}\n<Query>: {query}\n<Document>: {p}" for p in passages]
        enc = self.tok(texts, add_special_tokens=False, truncation=True, max_length=self.room)
        ids = [self.prefix + x + self.suffix for x in enc["input_ids"]]
        enc = self.tok.pad({"input_ids": ids}, padding=True, return_tensors="pt").to("cuda")
        with torch.autocast("cuda", dtype=torch.bfloat16):
            last = self.model(**enc, logits_to_keep=1).logits[:, -1, :]
        return last[:, self.yes].float() - last[:, self.no].float()  # log-odds of "yes"


def ndcg5(scores: list[float], grades: list[int]) -> float:
    order = sorted(range(len(scores)), key=lambda j: -scores[j])
    dcg = sum((2 ** grades[j] - 1) / math.log2(i + 2) for i, j in enumerate(order[:5]))
    ideal = sum((2 ** g - 1) / math.log2(i + 2) for i, g in enumerate(sorted(grades, reverse=True)[:5]))
    return dcg / ideal if ideal else 1.0


@torch.no_grad()
def evaluate(scorer: Scorer, items: list[dict], docs: list[dict]) -> dict:
    scorer.model.eval()
    n, top1, keep = 0.0, 0, 0
    for it in items:
        s = scorer.logits(it["text"], [docs[c]["text"] for c in it["candidates"]]).tolist()
        n += ndcg5(s, it["grades"])
        best = max(range(len(s)), key=lambda j: s[j])
        top1 += it["grades"][best] == 2
        keep += it["grades"][0] == 2  # v2's own first page, for comparison
    scorer.model.train()
    k = max(1, len(items))
    return {"ndcg5": round(n / k, 4), "first_is_2": round(top1 / k, 4), "v2_first_is_2": round(keep / k, 4)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--epochs", type=float, default=2.0)
    parser.add_argument("--point", type=float, default=0.5, help="weight of the point loss")
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--grades", default="qwen,opus", help="graders under data/rerank/, later ones replacing earlier")
    args = parser.parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    (ROOT / "logs").mkdir(exist_ok=True)

    docs = load_docs()
    items = load(args.grades.split(","), args.name)
    held = [it for it in items if it["qid"] % 10 == 0]
    train = [it for it in items if it["qid"] % 10 != 0]
    log(args.name, f"args {vars(args)}; {len(train)} training questions, {len(held)} held out")

    scorer = Scorer(MODEL, train=True)
    stock = evaluate(scorer, held, docs)
    log(args.name, f"stock {stock}")
    scorer.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    scorer.model.train()
    optimizer = torch.optim.AdamW(scorer.model.parameters(), lr=args.lr, weight_decay=0.01)
    total = int(len(train) * args.epochs)
    warmup = max(1, int(0.05 * total))
    schedule = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda s: min(1.0, (s + 1) / warmup) * max(0.0, (total - s) / max(1, total - warmup)))

    out = ROOT / "models" / args.name
    best, best_step = stock["ndcg5"], 0
    order: list[dict] = []
    step, started = 0, time.time()
    while step < total:
        if not order:
            order = train[:]
            random.shuffle(order)
        it = order.pop()
        g = torch.tensor(it["grades"], device="cuda")
        s = scorer.logits(it["text"], [docs[c]["text"] for c in it["candidates"]])
        point = F.binary_cross_entropy_with_logits(s, torch.tensor([POINT[x] for x in it["grades"]], device="cuda"))
        gain = torch.tensor([GAIN[x] for x in it["grades"]], device="cuda")
        loss = args.point * point
        if gain.sum() > 0:
            loss = loss + -(gain / gain.sum() * F.log_softmax(s, dim=0)).sum()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(scorer.model.parameters(), 1.0)
        optimizer.step()
        schedule.step()
        step += 1
        if step % 50 == 0:
            rate = step / (time.time() - started)
            log(args.name, f"step {step}/{total} loss {loss.item():.4f} eta {(total - step) / rate / 60:.0f} min")
        if step % args.eval_every == 0 or step == total:
            v = evaluate(scorer, held, docs)
            log(args.name, f"step {step} held-out {v}")
            if v["ndcg5"] > best:
                best, best_step = v["ndcg5"], step
                scorer.model.save_pretrained(str(out))
                scorer.tok.save_pretrained(str(out))
                log(args.name, f"saved best (ndcg5 {best:.4f}) to {out}")
    result = {"name": args.name, "held_out": len(held), "trained_on": len(train), "stock": stock,
              "best_step": best_step, "best_ndcg5": best}
    if best_step:
        del scorer, optimizer
        torch.cuda.empty_cache()
        result["trained"] = evaluate(Scorer(str(out)), held, docs)
    (ROOT / "runs" / f"{args.name}.json").write_text(json.dumps(result, indent=2))
    log(args.name, f"done: {json.dumps(result)}; {(time.time() - started) / 60:.0f} min")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
