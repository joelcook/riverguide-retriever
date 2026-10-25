#!/usr/bin/env bash
# Golden gate for riverguide's retriever-first branch (AGENTS.md rule 6), all on omarchy:
# the branch's gateway on :8792 (beside production's on :8790), the branch's Worker in wrangler dev
# on :8787 with today's corpus in a local D1, and evals.run --system api against it, once with
# RETRIEVER=omarchy and once with it off, the same evening. Log: logs/gate-retriever-first.log
set -uo pipefail
WT=/home/irsc/Projects/riverguide-retriever-first PY=/home/irsc/Projects/riverguide/.venv/bin/python
cd "$WT"
say() { echo "$(date +%T) $*"; }
GW_PID= DEV_PID=
cleanup() { [ -n "$DEV_PID" ] && kill "$DEV_PID" 2>/dev/null; [ -n "$GW_PID" ] && kill "$GW_PID" 2>/dev/null; }
trap cleanup EXIT

set -a; . ~/.config/riverguide/model-gateway.env; set +a
RIVERGUIDE_MODEL_PORT=8792 ~/.local/share/mise/installs/bun/1.4.2/bin/bun deploy/omarchy/model-gateway.ts > /tmp/gate-gateway.log 2>&1 &
GW_PID=$!
# The branch's Worker talks to the branch's gateway; secrets stay in the worktree's .dev.vars (gitignored).
grep -v '^OLLAMA_URL=\|^OLLAMA_API_KEY=' /home/irsc/Projects/riverguide/web/.dev.vars > web/.dev.vars
printf 'OLLAMA_URL=http://127.0.0.1:8792/v1\nOLLAMA_API_KEY=%s\n' "$RIVERGUIDE_MODEL_TOKEN" >> web/.dev.vars
chmod 600 web/.dev.vars

say "local D1 from corpus $(cat corpus/CHECKED)"
mkdir -p web/.build && $PY -m riverguide.export_d1 > web/.build/d1.sql
(cd web && mise exec -- bun run build > /dev/null && mise exec -- bunx wrangler d1 execute riverguide --local --yes --file .build/d1.sql > /dev/null) || { say "local D1 load failed"; exit 1; }

for round in $(seq ${GATE_ROUNDS:-2}); do for mode in ${GATE_MODES:-off:off omarchy:off omarchy:on}; do retr=${mode%%:*} first=${mode#*:}
  say "round $round: RETRIEVER=$retr SEARCH_FIRST=$first"
  (cd web && exec mise exec -- bunx wrangler dev --ip 127.0.0.1 --port 8787 --var "RETRIEVER:$retr" --var "SEARCH_FIRST:$first" > /tmp/gate-dev.log 2>&1) &
  DEV_PID=$!
  for _ in $(seq 60); do curl -sf localhost:8787/api/health > /dev/null && break; sleep 2; done
  curl -s localhost:8787/api/health; echo
  $PY -m evals.run --tier full --system api --endpoint http://127.0.0.1:8787 2>&1 | grep -E "passed|^  [a-z]|artifact|Error|error" | tail -12
  kill "$DEV_PID" 2>/dev/null; wait "$DEV_PID" 2>/dev/null; DEV_PID=
  pkill -f "wrangler dev --ip 127.0.0.1 --port 8787" 2>/dev/null; sleep 3
done
done
grep -c "\"route\":\"search\"" /tmp/gate-gateway.log | xargs echo "gateway search calls:"
say "gate done"
