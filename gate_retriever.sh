#!/usr/bin/env bash
# Lab job (share): RiverGuide's golden gate for a new retriever model (riverguide AGENTS.md rule 6),
# and the switch if it passes. Production's own Worker (riverguide main, in a worktree) runs in
# wrangler dev on :8787 with production's settings (RETRIEVER=omarchy, SEARCH_FIRST=on, from
# wrangler.jsonc) and today's corpus in a local D1, through a gateway of its own on :8792 to
# production's Qwen. Its searches go to production's retriever (:8791) or to the candidate's (:8793,
# with its own index of the same crawl, served from a worktree of this repo). A committed plan
# declares the objective, minimum gain, baseline and balanced rounds before any measurement:
#   ./gate_retriever.sh --plan plans/EXPERIMENT.json CANDIDATE
#   ./gate_retriever.sh --switch CANDIDATE
# riverguide-lab gate (rust/src/gate.rs) requires a demonstrated end-to-end benefit while holding
# quality, semantic checks, completion and latency guardrails. A tolerance-only pass cannot
# authorize a switch. Production's Qwen judges the semantic checks once each round's answers are in.
# Results go to runs/retriever-gates/ and runs/gate-retriever-CANDIDATE.json.
# --switch replays the measured evidence and checks that the setup still matches the model
# production still serves: models/production points at the candidate, production's index is
# rebuilt with it while the retriever is stopped (the Worker searches by keyword meanwhile), the
# retriever restarts and is checked. Experimental models and measurements are kept;
# any failure puts the old model back. Golden artifacts (Qwen's answers) go to riverguide's
# evals/runs/ (the private repo).
#   ./gate_retriever.sh --plan plans/EXPERIMENT.json --rerank reranker-v1
#   ./gate_retriever.sh --switch --rerank reranker-v1      # then, by hand: models/production-rerank
#   ./gate_retriever.sh --disable-rerank                   # retain retriever-only; keep weights
# With --rerank, one retriever process holds production's model and the reranker on the GPU and
# answers on production's port without the reranker and on :8793 with it (two retriever processes
# beside Qwen don't fit on the card); the systemd retriever steps aside meanwhile and comes back after.
set -uo pipefail
cd "$(dirname "$0")"
export XDG_RUNTIME_DIR="${XDG_RUNTIME_DIR:-/run/user/$(id -u)}"
SWITCH=false DISABLE_RERANK=false RERANKER= PLAN=
while [[ ${1:-} == --* ]]; do
  case $1 in
    --switch) SWITCH=true ;;
    --disable-rerank) DISABLE_RERANK=true ;;
    --plan) PLAN=${2:?--plan takes a committed JSON plan}; shift ;;
    --rerank) RERANKER=${2:?--rerank takes a reranker under models/}; shift ;;
    *) echo "unknown option $1" >&2; exit 2 ;;
  esac
  shift
done
ROOT=$PWD RG=/home/irsc/Projects/riverguide
PY=$RG/.venv/bin/python RL=$HOME/.local/bin/riverguide-lab GW_BIN=$HOME/.local/bin/riverguide-model-gateway
OPS=$(command -v ops-alert || echo "$HOME/.local/bin/ops-alert")
WTR=$HOME/Projects/riverguide-retriever-gate WTW=$HOME/Projects/riverguide-gate
say() { echo "$(date +%T) $*"; }
if $DISABLE_RERANK; then
  if $SWITCH || [ -n "$RERANKER$PLAN" ] || [ "$#" -ne 0 ]; then say "--disable-rerank takes no other options"; exit 2; fi
  if [ ! -L models/production-rerank ]; then
    [ ! -e models/production-rerank ] || { say "production-rerank is not an activation symlink; refusing to remove it"; exit 1; }
    say "production is already retriever-only"
    exit 0
  fi
  previous=$(readlink models/production-rerank)
  unlink models/production-rerank || exit 1
  systemctl --user restart riverguide-retriever || { ln -s "$previous" models/production-rerank; systemctl --user restart riverguide-retriever; exit 1; }
  for _ in $(seq 30); do
    if curl -sf -m 3 localhost:8791/health > /dev/null; then
      say "production is retriever-only; experimental reranker weights and results retained"
      exit 0
    fi
    sleep 2
  done
  say "retriever did not become healthy; restoring the previous link"
  ln -s "$previous" models/production-rerank
  systemctl --user restart riverguide-retriever
  exit 1
fi
if [ -n "$RERANKER" ]; then CAND=rerank-$RERANKER; else CAND=${1:?candidate under models/}; shift; fi
[ "$#" -eq 0 ] || { say "unexpected arguments; model cleanup is separate from promotion"; exit 2; }
[[ "$CAND" =~ ^[a-zA-Z0-9][a-zA-Z0-9._-]*$ ]] || { say "invalid candidate name"; exit 2; }
CAND_PID= GW_PID= DEV_PID= RESTART_RETRIEVER=
cleanup() {
  for p in $GW_PID $DEV_PID $CAND_PID; do kill "$p" 2>/dev/null; done
  [ -n "$CAND_PID" ] && wait "$CAND_PID" 2>/dev/null   # its port must be free for production's retriever
  [ -n "$RESTART_RETRIEVER" ] && systemctl --user start riverguide-retriever
  true
}

[ -L models/production ] || { say "models/production (the link to what production serves) is missing"; exit 1; }
current=$(readlink -f models/production)
if [ -n "$RERANKER" ]; then
  [ -f "models/$RERANKER/config.json" ] || { say "no reranker at models/$RERANKER"; exit 1; }
  [ "$(readlink -f models/production-rerank 2>/dev/null)" = "$(readlink -f "models/$RERANKER")" ] && { say "$RERANKER already reranks production"; exit 0; }
else
  [ -f "models/$CAND/config.json" ] || { say "no model at models/$CAND"; exit 1; }
  [ "$(readlink -f "models/$CAND")" = "$current" ] && { say "$CAND already serves production"; exit 0; }
fi

if $SWITCH && [ -n "$RERANKER" ]; then  # the reranker joins production's retriever; its index stays as it is
  result=runs/gate-retriever-$CAND.json
  "$RL" gate check-switch "$result" || exit 1
  [ "$("$RL" json "$result" candidate)" = "$CAND" ] || { say "result is for another candidate"; exit 1; }
  [ "$("$RL" json "$result" production)" = "$(basename "$current")" ] || { say "$CAND was gated over $("$RL" json "$result" production), but production serves $(basename "$current"): gate it again"; exit 1; }
  summary=$("$RL" json "$result" summary)
  ln -sfn "$RERANKER" models/production-rerank.new && mv -T models/production-rerank.new models/production-rerank
  systemctl --user restart riverguide-retriever
  up=false
  for _ in $(seq 60); do curl -sf -m 10 localhost:8791/search -d '{"query": "financial aid", "limit": 1}' > /dev/null && { up=true; break; }; sleep 2; done
  if ! $up || ! journalctl --user -u riverguide-retriever --since "-3min" --no-pager -o cat | grep -q "reranker .*production-rerank"; then
    say "the switch failed; production's retriever goes back to no reranker"
    rm -f models/production-rerank
    systemctl --user restart riverguide-retriever
    "$OPS" open riverguide "Retriever: $RERANKER passed its gate ($summary), but turning it on failed; production searches without it." --key riverguide:retriever-gate
    exit 1
  fi
  say "production's retriever now reranks with $RERANKER"
  "$OPS" open riverguide "Retriever: the $RERANKER reranker now sorts RiverGuide's search ($summary). Rollback: rm models/production-rerank and restart riverguide-retriever. Details: riverguide-retriever/runs/gate-retriever-$CAND.json" --key riverguide:retriever-gate
  exit 0
fi

# The switch: the retriever stops (the Worker searches by keyword for these two minutes), the link
# moves, production's index is rebuilt with the new model, the retriever starts and must answer.
switch() {  # switch MODEL_DIR
  systemctl --user stop riverguide-retriever
  ln -sfn "$(basename "$1")" models/production.new && mv -T models/production.new models/production
  .venv/bin/python refresh.py --model "$(readlink -f models/production)" 2>&1 | grep -v "Batches\|Loading weights\|Warning"
  systemctl --user start riverguide-retriever
  for _ in $(seq 60); do curl -sf -m 5 localhost:8791/search -d '{"query": "financial aid", "limit": 1}' > /dev/null && return 0; sleep 2; done
  return 1
}
if $SWITCH; then
  result=runs/gate-retriever-$CAND.json
  "$RL" gate check-switch "$result" || exit 1
  [ "$("$RL" json "$result" candidate)" = "$CAND" ] || { say "result is for another candidate"; exit 1; }
  [ "$("$RL" json "$result" production)" = "$(basename "$current")" ] || { say "$CAND was gated against $("$RL" json "$result" production), but production serves $(basename "$current"): gate it again"; exit 1; }
  summary=$("$RL" json "$result" summary)
  if ! switch "models/$CAND" || [ "$("$RL" json index/meta.json model)" != "$(readlink -f "models/$CAND")" ]; then
    say "the switch failed; putting $(basename "$current") back"
    switch "$current"
    "$OPS" open riverguide "Retriever: $CAND passed its gate ($summary), but the switch failed; production is back on $(basename "$current")." --key riverguide:retriever-gate
    exit 1
  fi
  say "production now searches with $CAND; previous models retained"
  "$OPS" open riverguide "Retriever: $CAND now serves RiverGuide ($summary). Previous model retained: $current. Details: riverguide-retriever/runs/gate-retriever-$CAND.json" --key riverguide:retriever-gate
  exit 0
fi
[ -n "$PLAN" ] || { say "a committed --plan is required before measuring (docs/retriever-promotions.md)"; exit 2; }
# Use the inherited GPU-lane lock when cron launched us; acquire it for a manual run.
if [ "$(readlink "/proc/$$/fd/9" 2>/dev/null)" != "$ROOT/lab/.lock-gpu" ]; then exec 9> lab/.lock-gpu; fi
flock -n 9 || { say "another GPU lab job is active"; exit 75; }
# The semantic judge is production's Qwen (evals.run --judge-ollama), so no API key is needed.
JUDGE_BASE=$(. lab/production.env && echo "$BASE")
curl -sf -m 10 "$JUDGE_BASE/api/version" > /dev/null || { say "the judge, production's Qwen at $JUDGE_BASE, doesn't answer"; exit 1; }
manifest=$("$RL" gate prepare "$PLAN" --candidate "$CAND") || exit 1
rounds=$("$RL" json "$manifest" plan.rounds) || exit 1
experiment=$(basename "$(dirname "$manifest")")
trap cleanup EXIT
say "gate: $CAND against production's $(basename "$current")"
set -a; . ~/.config/riverguide/model-gateway.env; set +a   # the token, production's engine

if [ -n "$RERANKER" ]; then
  # One process, production's model and index, both arms: :8791 without the reranker (production's
  # port, so live searches keep working meanwhile), :8793 with it.
  systemctl --user stop riverguide-retriever
  RESTART_RETRIEVER=1
  (RETRIEVER_DEVICE=cuda exec .venv/bin/python retriever_service.py --rerank "models/$RERANKER" --port 8793 --plain-port 8791) > logs/gate-retriever.log 2>&1 &
  CAND_PID=$!
else
  # The candidate's retriever: this repo at HEAD in a worktree (so its index/ is its own), the same
  # venv and models, an index of the same crawl production's was built from.
  if [ ! -d "$WTR" ]; then git worktree add -q --detach "$WTR" HEAD; else git -C "$WTR" checkout -q --detach "$(git rev-parse HEAD)"; fi
  ln -sfn "$ROOT/.venv" "$WTR/.venv"; ln -sfn "$ROOT/models" "$WTR/models"
  (cd "$WTR" && .venv/bin/python refresh.py --model "$ROOT/models/$CAND" 2>&1 | grep -v "Batches\|Loading weights\|Warning") || { say "the candidate's index didn't build"; exit 1; }
  prod_commit=$("$RL" json index/meta.json corpus_commit) cand_commit=$("$RL" json "$WTR/index/meta.json" corpus_commit)
  [ "$prod_commit" = "$cand_commit" ] || { say "production's index is corpus $prod_commit, the candidate's $cand_commit: not the same crawl"; exit 1; }
  (cd "$WTR" && RETRIEVER_DEVICE=cuda exec .venv/bin/python retriever_service.py --model "$ROOT/models/$CAND" --port 8793) > logs/gate-retriever.log 2>&1 &
  CAND_PID=$!
fi
for _ in $(seq 90); do curl -sf -m 10 localhost:8793/search -d '{"query": "financial aid", "limit": 1}' > /dev/null && break; sleep 2; done
curl -sf -m 10 localhost:8793/search -d '{"query": "financial aid", "limit": 1}' > /dev/null || { say "the candidate's retriever didn't come up"; exit 1; }
curl -sf -m 10 localhost:8791/search -d '{"query": "financial aid", "limit": 1}' > /dev/null || { say "production's retriever doesn't answer"; exit 1; }

# Production's Worker, as deployed: riverguide main in a worktree, today's corpus in a local D1.
main=$(git -C "$RG" rev-parse main)
if [ ! -d "$WTW" ]; then git -C "$RG" worktree add -q --detach "$WTW" "$main"; else git -C "$WTW" checkout -q --detach "$main"; fi
grep -v '^OLLAMA_URL=\|^OLLAMA_API_KEY=' "$RG/web/.dev.vars" > "$WTW/web/.dev.vars"
printf 'OLLAMA_URL=http://127.0.0.1:8792/v1\nOLLAMA_API_KEY=%s\n' "$RIVERGUIDE_MODEL_TOKEN" >> "$WTW/web/.dev.vars"
chmod 600 "$WTW/web/.dev.vars"
mkdir -p "$WTW/web/.build"
(cd "$WTW" && $PY -m riverguide.export_d1 > web/.build/d1.sql) || { say "the local D1 didn't export"; exit 1; }
(cd "$WTW/web" && mise exec -- bun install --frozen-lockfile > /dev/null 2>&1 && mise exec -- bun run build > /dev/null \
  && mise exec -- bunx wrangler d1 execute riverguide --local --yes --file .build/d1.sql > /dev/null) || { say "the Worker didn't build"; exit 1; }
(cd "$WTW/web" && exec mise exec -- bunx wrangler dev --ip 127.0.0.1 --port 8787) > logs/gate-dev.log 2>&1 &
DEV_PID=$!
for _ in $(seq 90); do curl -sf localhost:8787/api/health > /dev/null && break; sleep 2; done
curl -sf localhost:8787/api/health > /dev/null || { say "wrangler dev didn't come up"; exit 1; }

gateway() {  # gateway PORT: the gate's own gateway on :8792, its searches to the retriever on PORT
  [ -n "$GW_PID" ] && { kill "$GW_PID" 2>/dev/null; wait "$GW_PID" 2>/dev/null; }
  RIVERGUIDE_MODEL_PORT=8792 RETRIEVER_URL=http://127.0.0.1:$1 "$GW_BIN" >> logs/gate-gateway.log 2>&1 &
  GW_PID=$!
  for _ in $(seq 30); do curl -s -o /dev/null localhost:8792/ && return 0; sleep 1; done
  return 1
}
for round in $(seq "$rounds"); do
  arms=(production candidate)
  (( round % 2 == 0 )) && arms=(candidate production)
  for arm in "${arms[@]}"; do
    port=8791; [ $arm = candidate ] && port=8793
    gateway $port || { say "the gate's gateway didn't start"; exit 1; }
    say "round $round: $arm ($( [ $arm = candidate ] && echo "$CAND" || basename "$current"))"
    # Main's harness, checked clean by prepare. It judges after the round's answers are all in,
    # so judging is never timed and never shares the engine with an answer being timed.
    out=$(cd "$RG" && $PY -m evals.run --tier full --system api --endpoint http://127.0.0.1:8787 --judge-ollama "$JUDGE_BASE" 2>&1)
    echo "$out" | grep -E "passed|^  [a-z]" | tail -12
    art=$(sed -n 's/^artifact: //p' <<< "$out" | tail -1)
    [ -n "$art" ] && [ -f "$RG/$art" ] || { say "no artifact for $arm round $round"; exit 1; }
    dest=$RG/evals/runs/lab-gate-retriever-$experiment-$arm-r$round.json
    cp "$RG/$art" "$dest"
    "$RL" gate add "$manifest" --arm "$arm" --round "$round" --artifact "$dest" || exit 1
  done
done
kill "$GW_PID" "$DEV_PID" "$CAND_PID" 2>/dev/null; wait "$CAND_PID" 2>/dev/null; GW_PID= DEV_PID= CAND_PID=
if [ -n "$RESTART_RETRIEVER" ]; then
  systemctl --user start riverguide-retriever || exit 1
  RESTART_RETRIEVER=
fi

result=runs/gate-retriever-$CAND.json
"$RL" gate evaluate "$manifest" > "$result.new" || { say "experiment invalid; no promotion result written"; exit 1; }
mv "$result.new" "$result"
cp "$result" "$(dirname "$manifest")/verdict.json"
pass=$("$RL" json "$result" passed) summary=$("$RL" json "$result" summary)
say "$CAND: $summary; passed the gate: $pass"
if $pass; then
  "$OPS" open riverguide "Retriever gate: $CAND passed ($summary). Production still serves $(basename "$current"); to switch: ./gate_retriever.sh --switch ${RERANKER:+--rerank }${RERANKER:-$CAND}. Details: riverguide-retriever/runs/gate-retriever-$CAND.json" --key riverguide:retriever-gate
else
  "$OPS" open riverguide "Retriever gate: $CAND didn't pass ($summary). Production stays on $(basename "$current"). Details: riverguide-retriever/runs/gate-retriever-$CAND.json" --key riverguide:retriever-gate
fi
