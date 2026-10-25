#!/usr/bin/env bash
# Production-loop tests for each search: the 75 abbreviations as questions, then the golden set.
# Qwen (production's model, omarchy's Ollama) writes every search. Embedding runs serve the model
# with serve.py and embed chunks the way it was trained (trained_search.py).
#   ./loop_tests.sh   (logs/loop.log)
set -uo pipefail
cd "$(dirname "$0")"
ROOT=$PWD
RG=/home/irsc/Projects/riverguide
say() { echo "$(date +%H:%M:%S) $*"; }

SERVER=
serve() {  # serve NAME MODEL
  .venv/bin/python serve.py --model "$2" --port 11435 > "logs/serve-$1.log" 2>&1 &
  SERVER=$!
  for _ in $(seq 120); do
    curl -sf localhost:11435/api/embed -d '{"input": "x"}' > /dev/null && return 0
    sleep 2
  done
  say "server for $1 did not start"; return 1
}
stop() { [ -n "$SERVER" ] && kill "$SERVER" 2>/dev/null && wait "$SERVER" 2>/dev/null; SERVER=; }
trap stop EXIT

tests() {  # tests NAME [search args...]
  local name=$1; shift
  say "loop $name"
  (cd "$RG" && .venv/bin/python "$ROOT/loop_abbreviations.py" --name "$name" "$@")
  say "golden $name"
  (cd "$RG" && .venv/bin/python "$ROOT/golden_ab.py" --name "$name" "$@")
}

tests keyword --search keyword
for spec in stock:Qwen/Qwen3-Embedding-0.6B full:models/full strict:models/strict; do
  name=${spec%%:*}
  serve "$name" "${spec#*:}" || continue
  tests "$name" --search embedding --ollama http://127.0.0.1:11435 --embed-model "riverguide-retriever-$name"
  stop
done
say "loop tests done"
