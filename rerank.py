"""Qwen3-Reranker-0.6B as a relevance judge: P(the passage answers the query).

Used by build_data.py to drop generated queries their own chunk doesn't answer and mined
negatives that are really other right answers. Prompt format from the model card.
"""
from __future__ import annotations

import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "Qwen/Qwen3-Reranker-0.6B"
INSTRUCT = "Given a college student's search query, judge whether this College web page passage answers it"
PREFIX = ('<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the '
          'Instruct provided. Note that the answer can only be "yes" or "no".<|im_end|>\n<|im_start|>user\n')
SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"


class Judge:
    def __init__(self, max_length: int = 640):
        self.tok = AutoTokenizer.from_pretrained(MODEL, padding_side="left")
        self.model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).cuda().eval()
        self.yes = self.tok.convert_tokens_to_ids("yes")
        self.no = self.tok.convert_tokens_to_ids("no")
        self.prefix = self.tok.encode(PREFIX, add_special_tokens=False)
        self.suffix = self.tok.encode(SUFFIX, add_special_tokens=False)
        self.max_length = max_length

    @torch.no_grad()
    def score(self, pairs: list[tuple[str, str]], batch: int = 16) -> list[float]:
        order = sorted(range(len(pairs)), key=lambda i: len(pairs[i][1]))
        out = [0.0] * len(pairs)
        room = self.max_length - len(self.prefix) - len(self.suffix)
        for s in range(0, len(order), batch):
            idx = order[s : s + batch]
            texts = [f"<Instruct>: {INSTRUCT}\n<Query>: {pairs[i][0]}\n<Document>: {pairs[i][1]}" for i in idx]
            enc = self.tok(texts, add_special_tokens=False, truncation=True, max_length=room)
            ids = [self.prefix + x + self.suffix for x in enc["input_ids"]]
            enc = self.tok.pad({"input_ids": ids}, padding=True, return_tensors="pt").to("cuda")
            for attempt in range(20):  # the canary may load Qwen mid-run; wait for the GPU, don't die
                try:
                    logits = self.model(**enc, logits_to_keep=1).logits[:, -1, :]
                    break
                except torch.OutOfMemoryError:
                    torch.cuda.empty_cache()
                    time.sleep(30)
            else:
                raise RuntimeError("GPU stayed full for 10 minutes")
            p = torch.softmax(torch.stack([logits[:, self.no], logits[:, self.yes]], dim=1).float(), dim=1)[:, 1]
            for j, i in enumerate(idx):
                out[i] = p[j].item()
        return out
