#!/usr/bin/env bash
# Build, train, and score one v2 retriever: v1's training data (full) plus students' kind of first
# message (gen_student.py) written by WRITER, qwen or sonnet. Both v2 models use only the chunks
# both writers covered, so they differ only in the writer.
#   lab/enqueue.sh 37-v2-qwen night ./v2_train.sh qwen     (logs/v2-WRITER-*.log)
# Queue it with "# minutes: 100" and "# ready: .venv/bin/python gen_student.py ready" (lab/lab.sh),
# so it starts only once both writers are done and the window has time for it (about 85 minutes).
set -euo pipefail
cd "$(dirname "$0")"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
W=$1 NAME=v2-$1 PY=.venv/bin/python
say() { echo "$(date +%H:%M:%S) $*"; }
$PY gen_student.py merge sonnet > /dev/null 2>&1 || true
$PY - <<'PYEOF' || exit 75
import json
from gen_student import DATA, batch_ids, batches

ids = {i for b in batches() for i in batch_ids(b)}
files = {w: DATA / f"queries.student.{w}.jsonl" for w in ("qwen", "sonnet")}
have = {w: {json.loads(l)["id"]: json.loads(l) for l in open(f, encoding="utf-8")} for w, f in files.items() if f.exists()}
short = {w: len(set(have.get(w, {})) & ids) for w in files if len(set(have.get(w, {})) & ids) < 0.97 * len(ids)}
if short:
    print(f"waiting for the writers: {short} of {len(ids)}")
    raise SystemExit(1)
both = sorted(set(have["qwen"]) & set(have["sonnet"]) & ids)
for w, h in have.items():
    with open(DATA / f"queries.student.{w}.both.jsonl", "w", encoding="utf-8") as fh:
        for i in both:
            fh.write(json.dumps(h[i], ensure_ascii=False) + "\n")
print(f"{len(both)} chunks written by both; useful: " + ", ".join(f"{w} {sum(h[i]['useful'] for i in both)}" for w, h in have.items()))
PYEOF
say "build $NAME"
$PY build_data.py --mode full --name "$NAME" --extra "data/queries.student.$W.both.jsonl" > "logs/$NAME-build.log" 2>&1
grep -v "Batches\|Loading weights\|Warning" "logs/$NAME-build.log" | tail -30 || true
say "train $NAME"
$PY train.py --mode "$NAME" --name "$NAME" > "logs/$NAME-train.log" 2>&1
grep "saved best\|step" "logs/$NAME-train.log" | tail -3 || true
say "evaluate $NAME"
rm -f "runs/emb-$NAME.npy"
$PY evaluate.py --model "models/$NAME" --name "$NAME" > "logs/$NAME-eval.log" 2>&1
grep -v "Batches\|Loading weights\|Warning\|it/s\]" "logs/$NAME-eval.log" | tail -12 || true
say "done $NAME"
