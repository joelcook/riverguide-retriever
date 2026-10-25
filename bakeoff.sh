#!/usr/bin/env bash
# Lab job (night): which engine should serve production's Qwen3.8 27B on the 4090?
#   ./bakeoff.sh [ENGINE...]      (default: Ollama 1 and 4 slots and tuned; llama.cpp, vLLM and SGLang, each with MTP and with DFlash2)
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
NAME=qwen3.8:27b SERVED=qwen3.8:27b   # SERVED: the name the engine under test answers to
GGUF=/var/lib/ollama/blobs/sha256-f5f1dd8920d417aac2718b0bda3403da274301efdd6760b4f0f4b864ff2ad57d
CKPT=$E/models/qwen3.8-27b-w4a16 TEMPLATE=$E/models/qwen3.8-27b-template/chat_template.jinja
SAMPLING='{"temperature": 1.0, "top_p": 0.95, "top_k": 20, "min_p": 0.0}'
ENGINES=("${@:-ollama-p1 ollama-p4 ollama-max llamacpp-mtp llamacpp-dflash vllm-dflash vllm-mtp sglang-dflash sglang-mtp}"); ENGINES=(${ENGINES[@]})
say() { echo "$(date +%T) $*"; }
PID= BASE= CMD=
# Wait for install_engines.sh: every engine READY or FAILED, the shared checkpoint too.
for need in llamacpp vllm sglang models/qwen3.8-27b-w4a16; do
  [ -f "$E/$need/READY" ] || [ -f "$E/$need/FAILED" ] || { say "waiting for the install of $need"; exit 75; }
done
[ -f "$E/sglang/READY" ] && { python3 engine_patches/sglang_mtp_shared_embed.py "$E"/sglang/.venv/lib/python3.*/site-packages > /dev/null || say "the SGLang patch didn't apply"; }

# The command for ENGINE at fallback ATTEMPT (1 = best settings); empty when out of attempts.
# Every engine runs at its limit, with everything it offers: Qwen3.8's own multi-token prediction
# (MTP) and the DFlash2 drafter where it supports them (each its own variant, so neither hides the
# other), flash attention, an 8-bit KV cache, prefix caching. A variant whose speedup won't start
# is recorded as failed; it never falls back to plain decoding (that would run it handicapped).
# Every setting gives each conversation 32,768 tokens, as production Ollama does (RiverGuide's
# longer tool loops reach 25K). A fallback gives up parallel slots or CUDA graphs, never context:
# a smaller context fails long questions, and that's a handicap, not a setting. (llama.cpp's -c is
# shared by its -np slots, so 4 slots of 32K is -c 131072. Not --kv-unified: in one shared pool
# every decode step scans the pool's whole used range, and MTP decoded at 36 tokens/s per
# stream against 95 with separate slots. A smaller ubatch, -ub 256, frees the memory instead.)
# vLLM runs text-only (--language-model-only), as RiverGuide is: the vision tower costs 0.8 GB,
# and each drafter allocates its own 16-bit output head (2.37 GiB for Qwen's 248K vocabulary).
# SGLang: its MTP draft builds its own embedding and output head (4.7 GiB) before swapping in the
# target's; engine_patches/sglang_mtp_shared_embed.py builds them empty instead (its switch,
# RIVERGUIDE_MTP_SHARED_EMBED, is on for SGLang). Qwen3.8's linear-attention layers keep a 147 MB
# state per cached slot, 3 slots a request with SGLang's lazy prefix buffer and skip-decode-lock
# option; --enable-linear-replayssm-spec drops the per-draft-token copies of that state. What's
# left on 24 GB is one conversation at a time (a 41K-token pool). SGLang starts even when its pool
# can't hold --context-length, so start() checks the pool. Its DFlash2 fits at most 28.6K tokens
# (bf16 state, 4 draft tokens; ReplaySSM refuses DFlash on this model) and accepted 0% of drafts.
# (SGLang samples by the checkpoint's generation_config.json: production's temperature 1, top-k 20, top-p 0.95.)
engine_cmd() {
  local port=11434
  case "${1%%-*}" in
    ollama) [ "$1" = ollama-p1 ] || port=11436 ;; llamacpp) port=11437 ;; vllm) port=11438 ;; sglang) port=11439 ;;
  esac
  BASE=http://127.0.0.1:$port
  SERVED=$NAME; [ "${1%%-*}" = sglang ] && SERVED=qwen3.8-27b   # SGLang reads a colon as "model:adapter"
  local dfl_gguf=$E/models/qwen3.8-27b-dflash2-gguf/Qwen3.8-27B-DFlash2-Q4_K_M.gguf
  local dfl_w4=$E/models/qwen3.8-27b-dflash2-w4a16
  local ollama_env="OLLAMA_HOST=127.0.0.1:$port OLLAMA_MODELS=/var/lib/ollama OLLAMA_NUM_PARALLEL=4 OLLAMA_MAX_LOADED_MODELS=1 OLLAMA_NOPRUNE=1 OLLAMA_NO_CLOUD=1"
  local llama="$E/llamacpp/src/build/bin/llama-server -m $GGUF --alias $NAME --host 127.0.0.1 --port $port \
    -ngl 999 -fa on -ctk q8_0 -ctv q8_0 --jinja --chat-template-file $TEMPLATE --cache-reuse 256 \
    --temp 1.0 --top-k 20 --top-p 0.95 --min-p 0"
  local llama_mtp="-ctkd q8_0 -ctvd q8_0 --spec-type draft-mtp --spec-draft-n-max 3 --spec-draft-p-min 0.75"
  local llama_dfl="-md $dfl_gguf -ngld 999 -ctkd q8_0 -ctvd q8_0 --spec-type draft-dflash --spec-draft-n-max 7"
  # Their kernel JITs (FlashInfer, sgl-kernel) need nvcc and ninja; cron's PATH has neither.
  local vllm="env PATH=$E/vllm/.venv/bin:/opt/cuda/bin:$PATH CUDA_HOME=/opt/cuda $E/vllm/.venv/bin/vllm serve $CKPT --served-model-name $NAME --host 127.0.0.1 --port $port \
    --gpu-memory-utilization 0.95 --max-model-len 32768 --language-model-only --enable-prefix-caching --reasoning-parser qwen3 \
    --enable-auto-tool-choice --tool-call-parser qwen3_coder --override-generation-config '$SAMPLING' --kv-cache-dtype fp8"
  local vllm_mtp="--speculative-config '{\"method\": \"mtp\", \"num_speculative_tokens\": 3}'"
  local vllm_dfl="--speculative-config '{\"method\": \"dflash\", \"model\": \"$dfl_w4\", \"num_speculative_tokens\": 7}'"
  local sglang="env PATH=$E/sglang/.venv/bin:/opt/cuda/bin:$PATH CUDA_HOME=/opt/cuda RIVERGUIDE_MTP_SHARED_EMBED=1 SGLANG_OPT_MAMBA_SKIP_DECODE_LOCK=1 $E/sglang/.venv/bin/python -m sglang.launch_server --model-path $CKPT --served-model-name $SERVED \
    --host 127.0.0.1 --port $port --context-length 32768 --kv-cache-dtype fp8_e5m2 --reasoning-parser qwen3 --tool-call-parser qwen3_coder"
  local sglang_one="--max-running-requests 1 --max-mamba-cache-size 3 --mamba-radix-cache-strategy extra_buffer_lazy"
  local sglang_mtp="--enable-linear-replayssm-spec --speculative-algorithm NEXTN --speculative-num-steps 3 --speculative-eagle-topk 1 --speculative-num-draft-tokens 4"
  local sglang_dfl="--speculative-algorithm DFLASH --speculative-draft-model-path $dfl_w4 --speculative-num-draft-tokens 8"
  case "$1:$2" in
    ollama-p1:1) CMD="(system Ollama, 1 slot)" ;;
    ollama-p4:1) CMD="env $ollama_env ollama serve" ;;
    ollama-max:1) CMD="env $ollama_env OLLAMA_FLASH_ATTENTION=1 OLLAMA_KV_CACHE_TYPE=q8_0 ollama serve" ;;
    llamacpp-mtp:1) CMD="$llama -np 4 -c 131072 $llama_mtp" ;;
    llamacpp-mtp:2) CMD="$llama -np 4 -c 131072 -ub 256 $llama_mtp" ;;
    llamacpp-mtp:3) CMD="$llama -np 3 -c 98304 $llama_mtp" ;;
    llamacpp-mtp:4) CMD="$llama -np 2 -c 65536 $llama_mtp" ;;
    llamacpp-dflash:1) CMD="$llama -np 4 -c 131072 $llama_dfl" ;;
    llamacpp-dflash:2) CMD="$llama -np 4 -c 131072 -ub 256 $llama_dfl" ;;
    llamacpp-dflash:3) CMD="$llama -np 3 -c 98304 -ub 256 $llama_dfl" ;;
    llamacpp-dflash:4) CMD="$llama -np 2 -c 65536 -ub 256 $llama_dfl" ;;
    llamacpp-dflash:5) CMD="$llama -np 1 -c 32768 -ub 256 $llama_dfl" ;;
    vllm-dflash:1) CMD="$vllm --max-num-seqs 16 $vllm_dfl" ;;
    vllm-dflash:2) CMD="$vllm --max-num-seqs 4 $vllm_dfl" ;;
    vllm-dflash:3) CMD="$vllm --max-num-seqs 4 --enforce-eager $vllm_dfl" ;;
    vllm-dflash:4) CMD="$vllm --max-num-seqs 2 --enforce-eager --gpu-memory-utilization 0.97 $vllm_dfl" ;;
    vllm-mtp:1) CMD="$vllm --max-num-seqs 16 $vllm_mtp" ;;
    vllm-mtp:2) CMD="$vllm --max-num-seqs 4 $vllm_mtp" ;;
    vllm-mtp:3) CMD="$vllm --max-num-seqs 4 --enforce-eager $vllm_mtp" ;;
    vllm-mtp:4) CMD="$vllm --max-num-seqs 2 --enforce-eager --gpu-memory-utilization 0.97 $vllm_mtp" ;;
    sglang-dflash:1) CMD="$sglang --mem-fraction-static 0.95 $sglang_one --cuda-graph-max-bs-decode 1 --disable-prefill-cuda-graph --chunked-prefill-size 2048 --mamba-ssm-dtype bfloat16 $sglang_dfl" ;;
    sglang-dflash:2) CMD="$sglang --mem-fraction-static 0.95 $sglang_one --cuda-graph-max-bs-decode 1 --disable-prefill-cuda-graph --chunked-prefill-size 2048 --mamba-ssm-dtype bfloat16 ${sglang_dfl% 8} 4" ;;
    sglang-mtp:1) CMD="$sglang --mem-fraction-static 0.91 $sglang_one --cuda-graph-max-bs-decode 1 --disable-prefill-cuda-graph $sglang_mtp" ;;
    sglang-mtp:2) CMD="$sglang --mem-fraction-static 0.905 $sglang_one --cuda-graph-max-bs-decode 1 --disable-prefill-cuda-graph $sglang_mtp" ;;
    sglang-mtp:3) CMD="$sglang --mem-fraction-static 0.91 $sglang_one --disable-decode-cuda-graph --disable-prefill-cuda-graph $sglang_mtp" ;;
    *) CMD= ;;
  esac
}

start() {  # start ENGINE ATTEMPT: true once it answers on $BASE
  engine_cmd "$1" "$2"; [ -n "$CMD" ] || return 2
  if [ "$1" = ollama-p1 ]; then
    curl -s "$BASE/api/chat" -d "{\"model\": \"$NAME\", \"messages\": []}" > /dev/null; PID=; return 0
  fi
  ln -sfn "bake-$1-s$2.log" "logs/bake-$1.log"              # one log per attempt; this name is the latest
  [[ $1 == ollama-* ]] || [ -f "$E/${1%%-*}/READY" ] || { echo "not installed: $(cat "$E/${1%%-*}/FAILED" 2>/dev/null)" > "logs/bake-$1.log"; return 2; }
  bash -c "exec $CMD" > "logs/bake-$1-s$2.log" 2>&1 &
  PID=$!
  for _ in $(seq 180); do                                    # up to 15 minutes: vLLM compiles on first start
    kill -0 "$PID" 2>/dev/null || return 1
    curl -sf "$BASE/v1/models" > /dev/null && { pool_fits "$1" "$2"; return; }
    sleep 5
  done
  return 1
}
pool_fits() {  # SGLang serves even when its token pool is smaller than --context-length
  [[ $1 == sglang-* ]] || return 0
  local pool; pool=$(grep -o -m1 "max_total_num_tokens=[0-9]*" "logs/bake-$1-s$2.log" | cut -d= -f2)
  [ "${pool:-0}" -ge 32768 ] && return 0
  kill "$PID" 2>/dev/null; wait "$PID" 2>/dev/null; PID=; sleep 3   # first, or SGLang writes over the note
  echo "bakeoff: a ${pool:-unknown}-token pool can't hold a 32,768-token conversation" >> "logs/bake-$1-s$2.log"; return 1
}
stop() {
  if [ -n "$PID" ]; then kill "$PID" 2>/dev/null; wait "$PID" 2>/dev/null; PID=; fi
  curl -s localhost:11434/api/chat -d "{\"model\": \"$NAME\", \"messages\": [], \"keep_alive\": 0}" > /dev/null
  sleep 5
}
trap stop EXIT

fundraising_ok() {  # fundraising's own extraction call against $BASE
  local api=openai; [[ $1 == ollama-* ]] && api=ollama
  (cd "$FUND" && LLM_API=$api OLLAMA_URL=$BASE OLLAMA_MODEL=$SERVED timeout 600 .venv/bin/python manage.py shell -c "
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
  attempt=1 started=false; rm -f logs/bake-"$engine"-s*.log   # an earlier run's attempts aren't this one's
  while true; do
    if (( attempt > 1 )) && [ -n "${LAB_DEADLINE:-}" ] && (( $(date +%s) + 2400 > LAB_DEADLINE )); then
      say "not enough time left for $engine settings $attempt; the next window starts it over"; exit 75
    fi
    start "$engine" "$attempt"; code=$?
    [ "$code" -eq 0 ] && { started=true; break; }
    stop
    [ "$code" -eq 2 ] && break
    say "$engine: settings $attempt didn't start; trying the next"; attempt=$((attempt + 1))
  done
  if ! $started; then
    python3 -c "import glob,json,sys; json.dump({'engine': sys.argv[1], 'started': False, 'why': {f: open(f).read()[-400:] for f in sorted(glob.glob(f'logs/bake-{sys.argv[1]}-s*.log'))}}, open(sys.argv[2], 'w'), indent=2)" \
      "$engine" "runs/bake-$engine.json"
    say "$engine: did not start"; continue
  fi
  # Engines that load lazily (Ollama) hold no weights until the first question: ask one, then measure.
  curl -s "$BASE/v1/chat/completions" -H "content-type: application/json" \
    -d "{\"model\": \"$SERVED\", \"messages\": [{\"role\": \"user\", \"content\": \"hi\"}], \"max_tokens\": 1}" > /dev/null
  memory=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
  say "$engine: up with settings $attempt (${memory} MiB); golden x2"
  for r in 1 2; do
    (cd "$RG" && .venv/bin/python "$ROOT/golden_ab.py" --name "bake-$engine-r$r" --search keyword --model "$SERVED" --ollama "$BASE" | grep -E "passed, answers")
  done
  say "$engine: load test"
  (cd "$RG" && .venv/bin/python "$ROOT/load_test.py" --name "bake-$engine" --base "$BASE" --model "$SERVED" --levels 1 4 8 16)
  fund=false; fundraising_ok "$engine" && fund=true
  say "$engine: fundraising call ok: $fund"
  python3 -c "import json,sys; json.dump({'engine': sys.argv[1], 'started': True, 'settings': int(sys.argv[2]), 'command': sys.argv[3],
    'memory_mib': int(sys.argv[4]), 'fundraising_ok': sys.argv[5] == 'true'}, open(sys.argv[6], 'w'), indent=2)" \
    "$engine" "$attempt" "$CMD" "$memory" "$fund" "runs/bake-$engine.json"
  stop
done

missing=$(for e in "${ENGINES[@]}"; do [ -f "runs/bake-$e.json" ] || echo -n "$e "; done)
[ -z "$missing" ] || { say "not measured yet: $missing; the next window resumes"; exit 75; }
python3 bakeoff_verdict.py "${ENGINES[@]}" || exit 1
winner=$(python3 -c "import json; print(json.load(open('runs/bakeoff.json'))['winner'])")
./promote_engine.sh "$winner"
