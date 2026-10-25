#!/usr/bin/env bash
# Lab job (night): which engine should serve production's Qwen3.8 27B on the 4090?
#   ./bakeoff.sh [ENGINE...]      (default: all of ollama-p1 ollama-p4 llamacpp vllm sglang)
# Each engine serves the same model under the same name (qwen3.8:27b) with production Ollama's
# sampling (temperature 1, top-k 20, top-p 0.95), then gets: the golden set twice through
# RiverGuide's loop (quality), the load test at 1, 4, 8 and 16 students (speed), fundraising's
# own extraction call (one engine serves both apps), and its GPU memory. An engine that won't
# start tries its fallback settings, then is recorded as failed and loses. Results:
# runs/bake-<engine>.json (+ golden-bake-*, load-bake-*). Once every engine has a result,
# bakeoff_verdict.py picks the survivor and promote_engine.sh makes it production. Out of
# time first: exit 75, and the lab's next window picks up where this left off.
set -uo pipefail
cd "$(dirname "$0")"
ROOT=$PWD RG=/home/irsc/Projects/riverguide FUND=/home/irsc/Projects/fundraising-prospect-discovery E=$HOME/engines
NAME=qwen3.8:27b
GGUF=/var/lib/ollama/blobs/sha256-f5f1dd8920d417aac2718b0bda3403da274301efdd6760b4f0f4b864ff2ad57d
CKPT=$E/models/qwen3.8-27b-w4a16 TEMPLATE=$E/models/qwen3.8-27b-template/chat_template.jinja
SAMPLING='{"temperature": 1.0, "top_p": 0.95, "top_k": 20, "min_p": 0.0}'
ENGINES=("${@:-ollama-p1 ollama-p4 llamacpp vllm sglang}"); ENGINES=(${ENGINES[@]})
say() { echo "$(date +%T) $*"; }
PID= BASE= CMD=
# Wait for install_engines.sh: every engine READY or FAILED, the shared checkpoint too.
for need in llamacpp vllm sglang models/qwen3.8-27b-w4a16; do
  [ -f "$E/$need/READY" ] || [ -f "$E/$need/FAILED" ] || { say "waiting for the install of $need"; exit 75; }
done

# The command for ENGINE at fallback ATTEMPT (1 = best settings); empty when out of attempts.
engine_cmd() {
  local port=11434
  case "$1" in
    ollama-p4) port=11436 ;; llamacpp) port=11437 ;; vllm) port=11438 ;; sglang) port=11439 ;;
  esac
  BASE=http://127.0.0.1:${port:-11434}
  local vllm="$E/vllm/.venv/bin/vllm serve $CKPT --served-model-name $NAME --host 127.0.0.1 --port $port \
    --gpu-memory-utilization 0.95 --max-num-seqs 16 --enable-prefix-caching --reasoning-parser qwen3 \
    --enable-auto-tool-choice --tool-call-parser qwen3_coder --override-generation-config '$SAMPLING'"
  local sglang="$E/sglang/.venv/bin/python -m sglang.launch_server --model-path $CKPT --served-model-name $NAME \
    --host 127.0.0.1 --port $port --max-running-requests 16 --reasoning-parser qwen3 --tool-call-parser qwen3_coder"
  local llama="$E/llamacpp/src/build/bin/llama-server -m $GGUF --alias $NAME --host 127.0.0.1 --port $port \
    -ngl 999 -fa on -np 4 --jinja --chat-template-file $TEMPLATE --cache-reuse 256 --temp 1.0 --top-k 20 --top-p 0.95 --min-p 0"
  case "$1:$2" in
    ollama-p1:1) CMD="(system Ollama, 1 slot)" ;;
    ollama-p4:1) CMD="env OLLAMA_HOST=127.0.0.1:$port OLLAMA_MODELS=/var/lib/ollama OLLAMA_NUM_PARALLEL=4 OLLAMA_MAX_LOADED_MODELS=1 OLLAMA_NOPRUNE=1 OLLAMA_NO_CLOUD=1 ollama serve" ;;
    llamacpp:1) CMD="$llama -c 65536" ;;
    llamacpp:2) CMD="$llama -c 32768" ;;
    vllm:1) CMD="$vllm --max-model-len 32768 --kv-cache-dtype fp8 --speculative-config '{\"method\": \"mtp\", \"num_speculative_tokens\": 3}'" ;;
    vllm:2) CMD="$vllm --max-model-len 32768 --kv-cache-dtype fp8" ;;
    vllm:3) CMD="$vllm --max-model-len 16384 --kv-cache-dtype fp8 --enforce-eager" ;;
    sglang:1) CMD="$sglang --context-length 32768 --mem-fraction-static 0.88 --kv-cache-dtype fp8_e5m2" ;;
    sglang:2) CMD="$sglang --context-length 16384 --mem-fraction-static 0.85" ;;
    *) CMD= ;;
  esac
}

start() {  # start ENGINE ATTEMPT: true once it answers on $BASE
  engine_cmd "$1" "$2"; [ -n "$CMD" ] || return 2
  if [ "$1" = ollama-p1 ]; then
    curl -s "$BASE/api/chat" -d "{\"model\": \"$NAME\", \"messages\": []}" > /dev/null; PID=; return 0
  fi
  [[ $1 == ollama-* ]] || [ -f "$E/$1/READY" ] || { echo "not installed: $(cat "$E/$1/FAILED" 2>/dev/null)" > "logs/bake-$1.log"; return 2; }
  bash -c "exec $CMD" > "logs/bake-$1.log" 2>&1 &
  PID=$!
  for _ in $(seq 180); do                                    # up to 15 minutes: vLLM compiles on first start
    kill -0 "$PID" 2>/dev/null || return 1
    curl -sf "$BASE/v1/models" > /dev/null && return 0
    sleep 5
  done
  return 1
}
stop() {
  if [ -n "$PID" ]; then kill "$PID" 2>/dev/null; wait "$PID" 2>/dev/null; PID=; fi
  curl -s localhost:11434/api/chat -d "{\"model\": \"$NAME\", \"messages\": [], \"keep_alive\": 0}" > /dev/null
  sleep 5
}
trap stop EXIT

fundraising_ok() {  # fundraising's own extraction call against $BASE
  local api=openai; [[ $1 == ollama-* ]] && api=ollama
  (cd "$FUND" && LLM_API=$api OLLAMA_URL=$BASE OLLAMA_MODEL=$NAME timeout 600 .venv/bin/python manage.py shell -c "
from prospect.extraction import request_extraction
r = request_extraction('Extract documented philanthropic facts. Return only the requested JSON schema.',
    'In 2024 Jane Example gave \$50,000 to endow a nursing scholarship at Indian River State College.')
print('FUND_OK', len(r['prospects']))" 2>&1) | grep -q FUND_OK
}

for engine in "${ENGINES[@]}"; do
  [ -f "runs/bake-$engine.json" ] && { say "$engine: already measured"; continue; }
  if [ -n "${LAB_DEADLINE:-}" ] && (( $(date +%s) + 3000 > LAB_DEADLINE )); then
    say "not enough time left for $engine; the next window resumes"; exit 75
  fi
  say "$engine: starting"
  attempt=1 started=false
  while true; do
    start "$engine" "$attempt"; code=$?
    [ "$code" -eq 0 ] && { started=true; break; }
    stop
    [ "$code" -eq 2 ] && break
    say "$engine: settings $attempt didn't start; trying the next"; attempt=$((attempt + 1))
  done
  if ! $started; then
    python3 -c "import json,sys; json.dump({'engine': sys.argv[1], 'started': False, 'why': open(sys.argv[2]).read()[-600:]}, open(sys.argv[3], 'w'), indent=2)" \
      "$engine" "logs/bake-$engine.log" "runs/bake-$engine.json"
    say "$engine: did not start"; continue
  fi
  # Engines that load lazily (Ollama) hold no weights until the first question: ask one, then measure.
  curl -s "$BASE/v1/chat/completions" -H "content-type: application/json" \
    -d "{\"model\": \"$NAME\", \"messages\": [{\"role\": \"user\", \"content\": \"hi\"}], \"max_tokens\": 1}" > /dev/null
  memory=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
  say "$engine: up with settings $attempt (${memory} MiB); golden x2"
  for r in 1 2; do
    (cd "$RG" && .venv/bin/python "$ROOT/golden_ab.py" --name "bake-$engine-r$r" --search keyword --model "$NAME" --ollama "$BASE" | grep -E "passed, answers")
  done
  say "$engine: load test"
  (cd "$RG" && .venv/bin/python "$ROOT/load_test.py" --name "bake-$engine" --base "$BASE" --model "$NAME" --levels 1 4 8 16)
  fund=false; fundraising_ok "$engine" && fund=true
  say "$engine: fundraising call ok: $fund"
  python3 -c "import json,sys; json.dump({'engine': sys.argv[1], 'started': True, 'settings': int(sys.argv[2]), 'command': sys.argv[3],
    'memory_mib': int(sys.argv[4]), 'fundraising_ok': sys.argv[5] == 'true'}, open(sys.argv[6], 'w'), indent=2)" \
    "$engine" "$attempt" "$CMD" "$memory" "$fund" "runs/bake-$engine.json"
  stop
done

python3 bakeoff_verdict.py "${ENGINES[@]}" || exit 1
winner=$(python3 -c "import json; print(json.load(open('runs/bakeoff.json'))['winner'])")
./promote_engine.sh "$winner"
