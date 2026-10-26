#!/usr/bin/env bash
# Lab job (share): RiverGuide's golden gate for a new retriever model (riverguide AGENTS.md rule 6),
# and the switch if it passes. Production's own Worker (riverguide main, in a worktree) runs in
# wrangler dev on :8787 with production's settings (RETRIEVER=omarchy, SEARCH_FIRST=on, from
# wrangler.jsonc) and today's corpus in a local D1, through a gateway of its own on :8792 to
# production's Qwen. Its searches go to production's retriever (:8791) or to the candidate's (:8793,
# with its own index of the same crawl, served from a worktree of this repo), alternating, two
# rounds each, the same evening:
#   lab/enqueue.sh 46c-gate-retriever-v2-qwen share ./gate_retriever.sh v2-qwen     # measure only
#   ./gate_retriever.sh --switch v2-qwen full v2-sonnet                           # then, by hand
# It passes with a mean at least production's minus 1 of the golden set (the bake-off's bar: run-to-
# run noise) and every question answered; the result goes to runs/gate-retriever-CANDIDATE.json and
# an ops-alert, and production isn't touched. --switch acts on a passed result against the model
# production still serves: models/production points at the candidate, production's index is
# rebuilt with it while the retriever is stopped (the Worker searches by keyword meanwhile), the
# retriever restarts and is checked, and the models it beat (the arguments after it) are removed;
# any failure puts the old model back. Golden artifacts (Qwen's answers) go to riverguide's
# evals/runs/ (the private repo).
set -uo pipefail
cd "$(dirname "$0")"
export XDG_RUNTIME_DIR="${XDG_RUNTIME_DIR:-/run/user/$(id -u)}"
SWITCH=false; [ "${1:-}" = --switch ] && { SWITCH=true; shift; }
ROOT=$PWD CAND=${1:?candidate under models/} RG=/home/irsc/Projects/riverguide
shift; BEATEN=("$@")
PY=$RG/.venv/bin/python RL=$HOME/.local/bin/riverguide-lab GW_BIN=$HOME/.local/bin/riverguide-model-gateway
OPS=$(command -v ops-alert || echo "$HOME/.local/bin/ops-alert")
WTR=$HOME/Projects/riverguide-retriever-gate WTW=$HOME/Projects/riverguide-gate
say() { echo "$(date +%T) $*"; }
CAND_PID= GW_PID= DEV_PID=
cleanup() {
  for p in $GW_PID $DEV_PID $CAND_PID; do kill "$p" 2>/dev/null; done
  pkill -f "wrangler dev --ip 127.0.0.1 --port 8787" 2>/dev/null; true
}
trap cleanup EXIT

[ -f "models/$CAND/config.json" ] || { say "no model at models/$CAND"; exit 1; }
[ -L models/production ] || { say "models/production (the link to what production serves) is missing"; exit 1; }
current=$(readlink -f models/production)
[ "$(readlink -f "models/$CAND")" = "$current" ] && { say "$CAND already serves production"; exit 0; }

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
  [ "$("$RL" json "$result" passed 2>/dev/null)" = true ] || { say "no passed gate for $CAND in $result"; exit 1; }
  [ "$("$RL" json "$result" production)" = "$(basename "$current")" ] || { say "$CAND was gated against $("$RL" json "$result" production), but production serves $(basename "$current"): gate it again"; exit 1; }
  summary=$("$RL" json "$result" summary)
  if ! switch "models/$CAND" || [ "$("$RL" json index/meta.json model)" != "$(readlink -f "models/$CAND")" ]; then
    say "the switch failed; putting $(basename "$current") back"
    switch "$current"
    "$OPS" open riverguide "Retriever: $CAND passed its gate ($summary), but the switch failed; production is back on $(basename "$current")." --key riverguide:retriever-gate
    exit 1
  fi
  removed=()
  for m in "${BEATEN[@]}"; do
    [ -n "$m" ] && [ -d "models/$m" ] && [ "$(readlink -f "models/$m")" != "$(readlink -f models/production)" ] && { rm -rf "models/${m:?}"; removed+=("$m"); }
  done
  say "production now searches with $CAND; removed: ${removed[*]:-nothing}"
  "$OPS" open riverguide "Retriever: $CAND now serves RiverGuide ($summary). Removed: ${removed[*]:-nothing}. Rollback: hf download joeltcook/riverguide-retriever-full --local-dir models/full, then the switch in gate_retriever.sh with it. Details: riverguide-retriever/runs/gate-retriever-$CAND.json" --key riverguide:retriever-gate
  exit 0
fi
say "gate: $CAND against production's $(basename "$current")"
set -a; . ~/.config/riverguide/model-gateway.env; set +a   # the token, production's engine

# The candidate's retriever: this repo at HEAD in a worktree (so its index/ is its own), the same
# venv and models, an index of the same crawl production's was built from.
if [ ! -d "$WTR" ]; then git worktree add -q --detach "$WTR" HEAD; else git -C "$WTR" checkout -q --detach "$(git rev-parse HEAD)"; fi
ln -sfn "$ROOT/.venv" "$WTR/.venv"; ln -sfn "$ROOT/models" "$WTR/models"
(cd "$WTR" && .venv/bin/python refresh.py --model "$ROOT/models/$CAND" 2>&1 | grep -v "Batches\|Loading weights\|Warning") || { say "the candidate's index didn't build"; exit 1; }
prod_commit=$("$RL" json index/meta.json corpus_commit) cand_commit=$("$RL" json "$WTR/index/meta.json" corpus_commit)
[ "$prod_commit" = "$cand_commit" ] || { say "production's index is corpus $prod_commit, the candidate's $cand_commit: not the same crawl"; exit 1; }
(cd "$WTR" && RETRIEVER_DEVICE=cuda exec .venv/bin/python retriever_service.py --model "$ROOT/models/$CAND" --port 8793) > logs/gate-retriever.log 2>&1 &
CAND_PID=$!
for _ in $(seq 60); do curl -sf -m 5 localhost:8793/search -d '{"query": "financial aid", "limit": 1}' > /dev/null && break; sleep 2; done
curl -sf -m 5 localhost:8793/search -d '{"query": "financial aid", "limit": 1}' > /dev/null || { say "the candidate's retriever didn't come up"; exit 1; }

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
declare -A sum=([production]=0 [candidate]=0)
runs=()
complete=true
for round in 1 2; do
  for arm in production candidate; do
    port=8791; [ $arm = candidate ] && port=8793
    gateway $port || { say "the gate's gateway didn't start"; exit 1; }
    say "round $round: $arm ($( [ $arm = candidate ] && echo "$CAND" || basename "$current"))"
    out=$(cd "$WTW" && $PY -m evals.run --tier full --system api --endpoint http://127.0.0.1:8787 2>&1)
    echo "$out" | grep -E "passed|^  [a-z]" | tail -12
    art=$(sed -n 's/^artifact: //p' <<< "$out" | tail -1)
    [ -n "$art" ] && [ -f "$WTW/$art" ] || { say "no artifact for $arm round $round"; exit 1; }
    dest=$RG/evals/runs/lab-gate-retriever-$CAND-$arm-r$round.json
    cp "$WTW/$art" "$dest"
    passed=$("$RL" json "$dest" summary.passed) total=$("$RL" json "$dest" summary.total)
    (( passed >= 0 && total > 0 )) || { say "unreadable result in $dest"; exit 1; }
    [ "$total" = "$("$RL" json "$RG/evals/runs/lab-gate-retriever-$CAND-production-r1.json" summary.total)" ] || complete=false
    sum[$arm]=$(( ${sum[$arm]} + passed )); runs+=("{\"arm\": \"$arm\", \"round\": $round, \"passed\": $passed, \"total\": $total}")
  done
done
kill "$GW_PID" "$DEV_PID" "$CAND_PID" 2>/dev/null; GW_PID= DEV_PID= CAND_PID=
pkill -f "wrangler dev --ip 127.0.0.1 --port 8787" 2>/dev/null

# Means over the two rounds, in halves (x2) to stay in integers: pass if candidate >= production - 1.
pass=false; (( sum[candidate] >= sum[production] - 2 )) && $complete && pass=true
summary="golden $(( sum[candidate] / 2 )).$(( sum[candidate] % 2 * 5 )) vs $(( sum[production] / 2 )).$(( sum[production] % 2 * 5 )) of $total passed (means over 2 rounds, production's Worker)"
printf '{"candidate": "%s", "production": "%s", "passed": %s, "summary": "%s", "runs": [%s]}\n' \
  "$CAND" "$(basename "$current")" "$pass" "$summary" "$(IFS=,; echo "${runs[*]}")" > "runs/gate-retriever-$CAND.json"
say "$CAND: $summary; passed the gate: $pass"
if $pass; then
  "$OPS" open riverguide "Retriever gate: $CAND passed ($summary). Production still serves $(basename "$current"); to switch: ./gate_retriever.sh --switch $CAND. Details: riverguide-retriever/runs/gate-retriever-$CAND.json" --key riverguide:retriever-gate
else
  "$OPS" open riverguide "Retriever gate: $CAND didn't pass ($summary). Production stays on $(basename "$current"). Details: riverguide-retriever/runs/gate-retriever-$CAND.json" --key riverguide:retriever-gate
fi
