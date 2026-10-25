#!/usr/bin/env bash
# Lab job (night): does Ollama answer more students at once with 4 slots than with today's 1?
# Production's Ollama (one slot, 127.0.0.1:11434) under the load test, then a second Ollama run
# as irsc on 127.0.0.1:11436 with OLLAMA_NUM_PARALLEL=4 and the same model files, same test.
# The system service's setting needs sudo; this measures the change without it.
set -uo pipefail
cd "$(dirname "$0")"
ROOT=$PWD RG=/home/irsc/Projects/riverguide
unload() { curl -s "$1/api/chat" -d '{"model": "qwen3.8:27b", "messages": [], "keep_alive": 0}' > /dev/null; }

(cd "$RG" && .venv/bin/python "$ROOT/load_test.py" --name ollama-p1 --base http://127.0.0.1:11434 --levels 1 4 8 16)
unload http://127.0.0.1:11434
OLLAMA_HOST=127.0.0.1:11436 OLLAMA_MODELS=/var/lib/ollama OLLAMA_NUM_PARALLEL=4 OLLAMA_MAX_LOADED_MODELS=1 \
  OLLAMA_NOPRUNE=1 OLLAMA_NO_CLOUD=1 HOME=$HOME ollama serve > logs/ollama-p4.log 2>&1 &
SERVER=$!
trap 'unload http://127.0.0.1:11436; kill $SERVER 2>/dev/null; wait $SERVER 2>/dev/null' EXIT
for _ in $(seq 60); do curl -sf localhost:11436/api/version > /dev/null && break; sleep 2; done
(cd "$RG" && .venv/bin/python "$ROOT/load_test.py" --name ollama-p4 --base http://127.0.0.1:11436 --levels 1 4 8 16)
grep -m1 -iE "offload|layers.*gpu|memory" logs/ollama-p4.log || true
