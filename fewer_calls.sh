#!/usr/bin/env bash
# Lab job: fewer_calls.py's three variants back to back (today's loop, keyword search first, the
# retriever first), so the comparison shares one evening.
set -uo pipefail
cd "$(dirname "$0")"
ROOT=$PWD RG=/home/irsc/Projects/riverguide
(cd "$RG" && .venv/bin/python "$ROOT/fewer_calls.py" --system legacy --prefetch none | tail -1)   # RiverGuide 1, the LLM wiki
for p in none keyword; do (cd "$RG" && .venv/bin/python "$ROOT/fewer_calls.py" --prefetch "$p" | tail -1); done
.venv/bin/python serve.py --model models/full --port 11435 > logs/serve-fewer.log 2>&1 &
SERVER=$!
trap 'kill $SERVER 2>/dev/null' EXIT
for _ in $(seq 120); do curl -sf localhost:11435/api/embed -d '{"input": "x"}' > /dev/null && break; sleep 2; done
(cd "$RG" && .venv/bin/python "$ROOT/fewer_calls.py" --prefetch full | tail -1)
