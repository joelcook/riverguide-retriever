"""Could RiverGuide answer common questions ahead of time, matched by meaning? How often, how safely.

    .venv/bin/python repeat_questions.py --org https://<org>.crm.dynamics.com --bot <River Guide bot id>

The answer cache matches a question word for word (normalized), and most real first questions are
worded differently (551 distinct of 634 in 30 days). Suppose RiverGuide answered the most common
questions each night and served one when a student's question means the same thing: what share of
questions would that cover, and how often would the match give the wrong answer?

Real first questions of River Guide (Copilot Studio) sessions, redacted in memory as
real_questions.py does, minus the conversation-starter buttons (the Worker's prewarm already caches
those). Embedded with the retriever's model (models/production). The first --train-days build the library:
questions grouped greedily (each joins the most similar earlier group's lead at cosine >= t), the
--library largest groups each answered once. The later days test it: a question is covered when the
nearest library lead is at cosine >= t. Qwen judges every covered pair: would one answer serve both?
Output is aggregate counts only, to logs/ (not committed); no question text is printed or written.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import subprocess
import time
import urllib.parse
import urllib.request

import numpy as np

from common import QUERY_INSTRUCT, ROOT
from real_questions import JUDGE, OLLAMA, key, redact, starters
from refresh import load_model

THRESHOLDS = [0.80, 0.85, 0.90, 0.93, 0.96]
PROMPT = """Two students typed these as the first message to Indian River State College's help assistant.

<first>{a}</first>
<second>{b}</second>

Would one answer, written for the first, also fully and correctly answer the second? Answer false if
they differ in anything that changes the answer: the term or year, the program or course, the campus,
the kind of student (new, returning, dual enrollment, international), or what they want done.

Reply with JSON only: {{"same": true}}"""


def first_questions(org: str, bot: str, days: int) -> list[tuple[float, str]]:
    """(time, redacted text) of each session's first typed message, oldest first."""
    token = subprocess.run(["az", "account", "get-access-token", "--resource", org, "--query", "accessToken", "-o", "tsv"],
                           capture_output=True, text=True, check=True).stdout.strip()
    since = (dt.datetime.now(dt.timezone.utc) - dt.timedelta(days=days)).strftime("%Y-%m-%dT%H:%M:%SZ")
    query = urllib.parse.quote(f"_bot_conversationtranscriptid_value eq {bot} and createdon ge {since}")
    url = f"{org}/api/data/v9.2/conversationtranscripts?$select=content&$filter={query}"
    out = []
    while url:
        request = urllib.request.Request(url, headers={"Authorization": f"Bearer {token}", "Prefer": "odata.maxpagesize=100"})
        page = json.load(urllib.request.urlopen(request, timeout=120))
        for row in page["value"]:
            acts = json.loads(row["content"] or "{}").get("activities", [])
            typed = [a for a in acts if a.get("type") == "message" and (a.get("from") or {}).get("role") == 1
                     and (a.get("text") or "").strip() and a.get("timestamp")]
            if typed:
                first = min(typed, key=lambda a: float(a["timestamp"]))
                out.append((float(first["timestamp"]), redact(first["text"])))
        url = page.get("@odata.nextLink")
    return sorted(out)


def same(a: str, b: str) -> bool | None:
    for _ in range(2):
        while 44 <= time.localtime().tm_min < 50:  # RiverGuide's :47 canary gets Qwen to itself
            time.sleep(15)
        request = {"model": JUDGE, "stream": False, "think": False, "format": "json", "options": {"temperature": 0},
                   "messages": [{"role": "user", "content": PROMPT.format(a=a, b=b)}]}
        try:
            reply = json.load(urllib.request.urlopen(urllib.request.Request(
                f"{OLLAMA}/api/chat", json.dumps(request).encode(), {"content-type": "application/json"}), timeout=300))
            return bool(json.loads(reply["message"]["content"])["same"])
        except (KeyError, ValueError, urllib.error.URLError):
            continue
    return None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--org", required=True)
    parser.add_argument("--bot", required=True)
    parser.add_argument("--days", type=int, default=30)
    parser.add_argument("--train-days", type=int, default=20)
    parser.add_argument("--library", type=int, default=50)
    args = parser.parse_args()

    skip = starters()
    asked = [(t, q) for t, q in first_questions(args.org, args.bot, args.days) if key(q) and key(q) not in skip]
    cut = asked[0][0] + args.train_days * 86400
    train = [q for t, q in asked if t < cut]
    test = [q for t, q in asked if t >= cut]
    model = load_model(str(ROOT / "models" / "production"))
    emb = lambda qs: model.encode([QUERY_INSTRUCT + q for q in qs], batch_size=64, normalize_embeddings=True,  # noqa: E731
                                  convert_to_numpy=True).astype(np.float32)
    e_train, e_test = emb(train), emb(test)
    exact = {key(q) for q in train}
    out = {"sessions": len(asked), "train_questions": len(train), "test_questions": len(test),
           "train_days": args.train_days, "library_size": args.library,
           "test_exact_repeat_of_train": round(sum(key(q) in exact for q in test) / max(1, len(test)), 3), "thresholds": {}}
    judged: dict[tuple[int, int], bool | None] = {}
    for t in THRESHOLDS:
        leads, sizes = [], []  # greedy groups over the training days
        for i in range(len(train)):
            if leads:
                sims = e_train[leads] @ e_train[i]
                j = int(np.argmax(sims))
                if sims[j] >= t:
                    sizes[j] += 1
                    continue
            leads.append(i)
            sizes.append(1)
        library = [leads[j] for j in np.argsort(sizes)[::-1][: args.library]]
        lib_sizes = sorted(sizes, reverse=True)[: args.library]
        sims = e_test @ e_train[library].T
        best = sims.argmax(axis=1)
        covered = [(k, library[int(best[k])]) for k in range(len(test)) if sims[k, best[k]] >= t]
        for pair in covered:
            if pair not in judged:
                judged[pair] = same(train[pair[1]], test[pair[0]])
        verdicts = [judged[p] for p in covered if judged[p] is not None]
        right = sum(verdicts)
        out["thresholds"][str(t)] = {
            "groups": len(leads), "library_covers_train_share": round(sum(lib_sizes) / max(1, len(train)), 3),
            "test_covered": len(covered), "test_covered_share": round(len(covered) / max(1, len(test)), 3),
            "judged": len(verdicts), "judged_same": right,
            "precision": round(right / len(verdicts), 3) if verdicts else None,
            "wrong_answers_per_100_questions": round(100 * (len(verdicts) - right) / max(1, len(test)), 2),
        }
        print(json.dumps({"threshold": t, **out["thresholds"][str(t)]}), flush=True)
    path = ROOT / "logs" / f"repeat-questions-{dt.date.today()}.json"
    path.write_text(json.dumps(out, indent=2))
    print(json.dumps({k: v for k, v in out.items() if k != "thresholds"}), flush=True)
    print(f"wrote {path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
