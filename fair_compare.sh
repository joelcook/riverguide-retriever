#!/usr/bin/env bash
# Lab job (share): RiverGuide 1's LLM wiki against RiverGuide 2.0 with the retriever first, on the
# same model and questions, once golden.yaml carries the 2026-10-24 check updates (exit 75 until
# then). Results: runs/fewer-legacy.json, runs/fewer-full.json (tokens, model calls, seconds).
set -uo pipefail
cd "$(dirname "$0")"
ROOT=$PWD RG=/home/irsc/Projects/riverguide
grep -q "passwordreset@irsc.edu" "$RG/evals/golden.yaml" || { echo "waiting for the updated golden set on riverguide main"; exit 75; }
(cd "$RG" && .venv/bin/python "$ROOT/fewer_calls.py" --system legacy --prefetch none | tail -1)
.venv/bin/python serve.py --model models/full --port 11435 > logs/serve-fair.log 2>&1 &
SERVER=$!
trap 'kill $SERVER 2>/dev/null' EXIT
for _ in $(seq 120); do curl -sf localhost:11435/api/embed -d '{"input": "x"}' > /dev/null && break; sleep 2; done
(cd "$RG" && .venv/bin/python "$ROOT/fewer_calls.py" --prefetch full | tail -1)
