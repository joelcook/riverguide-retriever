#!/usr/bin/env bash
# Wait for query generation, then build data, train, and evaluate both models.
#   ./pipeline.sh   (logs/pipeline.log)
set -euo pipefail
cd "$(dirname "$0")"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=.venv/bin/python
say() { echo "$(date +%H:%M:%S) $*"; }

while ps -eo args | grep -q "[.]venv/bin/python gen_"; do sleep 60; done
say "generation done: $(wc -l < data/queries.raw.jsonl) chunks, $(wc -l < data/queries.program.jsonl) program pages"

unload() {
  curl -s localhost:11434/api/generate -d '{"model":"qwen3.8:27b","keep_alive":0}' > /dev/null || true
  for _ in $(seq 30); do [ -z "$(curl -s localhost:11434/api/ps | $PY -c 'import json,sys;print(json.load(sys.stdin)["models"] or "")')" ] && return; sleep 2; done
}

# Stay off the GPU while the :47 canary loads Qwen.
avoid_canary() { while m=$((10#$(date +%M))); [ "$m" -ge 44 ] && [ "$m" -lt 56 ]; do sleep 30; done; }

avoid_canary; unload
say "build strict"; $PY build_data.py --mode strict 2>&1 | grep -v "Batches\|Loading weights\|Warning"
avoid_canary
say "build full";   $PY build_data.py --mode full   2>&1 | grep -v "Batches\|Loading weights\|Warning"
say "evaluate stock on the new val set"
$PY evaluate.py --model Qwen/Qwen3-Embedding-0.6B --name stock 2>&1 | grep -v "Batches\|Loading weights\|Warning\|it/s\]"

for mode in strict full; do
  avoid_canary; unload
  say "train $mode"
  $PY train.py --mode "$mode" --name "$mode" 2>&1 | grep -v "Batches\|Loading weights\|Warning\|Writing model shards"
  say "evaluate $mode"
  rm -f "runs/emb-$mode.npy"
  $PY evaluate.py --model "models/$mode" --name "$mode" 2>&1 | grep -v "Batches\|Loading weights\|Warning\|it/s\]"
done
say "pipeline done"
