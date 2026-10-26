#!/usr/bin/env bash
# Make the bake-off's survivor production for RiverGuide (and fundraising's config, for when that
# shelved app comes back), then remove the engines it fairly beat. riverguide-lab's verdict lists
# them (runs/bakeoff.json "remove_paths"): an engine whose run wasn't fair stays installed for a
# re-run, and so do the model files it reads (rust/src/verdict.rs).
#   ./promote_engine.sh ENGINE        (inside the lab's maintenance window, from bakeoff.sh)
# Any survivor but production's own Ollama (ollama-p1) runs as the user service riverguide-engine
# on 127.0.0.1:11440, with the exact settings that won. RiverGuide's gateway and fundraising's
# worker are pointed at it and checked; only then are the losing engines' installs and files
# removed. If a check fails, everything goes back and an ops-alert says why.
# Ollama stays installed whatever wins: the lab tries new models in it first (model_ab.sh).
set -uo pipefail
# Cron has no XDG_RUNTIME_DIR, and systemctl --user needs it (deploy/omarchy/nightly.sh does the same).
export XDG_RUNTIME_DIR="${XDG_RUNTIME_DIR:-/run/user/$(id -u)}"
cd "$(dirname "$0")"
ROOT=$PWD WIN=${1:?engine} E=$HOME/engines
GW=$HOME/.config/riverguide/model-gateway.env
FUND=$HOME/Projects/fundraising-prospect-discovery
UNIT=$HOME/.config/systemd/user/riverguide-engine.service
OPS=$(command -v ops-alert || echo "$HOME/.local/bin/ops-alert")
RL=$HOME/.local/bin/riverguide-lab
say() { echo "$(date +%T) [promote] $*"; }
. lab/production.env
setenv() { if grep -q "^$2=" "$1"; then sed -i "s|^$2=.*|$2=$3|" "$1"; else echo "$2=$3" >> "$1"; fi; }

remove_losers() {
  for p in $("$RL" json runs/bakeoff.json remove_paths); do rm -rf "${E:?}/$p"; say "removed $p"; done
}
outcome() {  # what the alert says was removed and kept
  local gone kept
  gone=$("$RL" json runs/bakeoff.json remove | xargs) kept=$("$RL" json runs/bakeoff.json kept_families | xargs)
  echo "Removed: ${gone:-nothing}.${kept:+ Kept for a re-run (not a fair run yet): $kept.}"
}

if [ "$WIN" = "$ENGINE" ]; then
  remove_losers
  "$OPS" open riverguide "Engine bake-off: $WIN stays. $("$RL" json runs/bakeoff.json reason). $(outcome) Details: riverguide-retriever/runs/bakeoff.json" --key riverguide:bakeoff
  exit 0
fi

# The survivor's winning command, moved to the production port.
if [ "$WIN" = ollama-p1 ]; then
  BASE_NEW=http://127.0.0.1:11434 API=ollama
else
  BASE_NEW=http://127.0.0.1:11440 API=openai; [[ $WIN == ollama-* ]] && API=ollama
  cmd=$("$RL" json "runs/bake-$WIN.json" command)
  cmd=$(sed -E 's/--port [0-9]+/--port 11440/; s/OLLAMA_HOST=127\.0\.0\.1:[0-9]+/OLLAMA_HOST=127.0.0.1:11440/' <<< "$cmd")
  if [ "${WIN%%-*}" = llamacpp ]; then   # its own copy of the GGUF, so Ollama can drop models freely
    gguf=$(grep -oE '/var/lib/ollama/blobs/sha256-[0-9a-f]+' <<< "$cmd")
    cp -n "$gguf" "$E/models/qwen3.8-27b.gguf" && cmd=${cmd//$gguf/$E/models/qwen3.8-27b.gguf}
  fi
  # Ollama unloads a model after 5 idle minutes and the next student waits for it (3.5 s from the
  # page cache, longer from disk); the card is Qwen's, so keep it loaded.
  [[ $WIN == ollama-* ]] && cmd=${cmd/ ollama serve/ OLLAMA_KEEP_ALIVE=-1 ollama serve}
  printf '#!/usr/bin/env bash\n# Production engine (%s), written by riverguide-retriever/promote_engine.sh on %s.\nexec %s\n' \
    "$WIN" "$(date +%F)" "$cmd" > "$E/serve-production.sh"
  chmod +x "$E/serve-production.sh"
  cat > "$UNIT" <<UNITEOF
[Unit]
Description=RiverGuide engine: $WIN serving qwen3.8:27b on 127.0.0.1:11440 (lab bake-off winner, $(date +%F))
After=network-online.target

[Service]
ExecStart=$E/serve-production.sh
Restart=always
RestartSec=10

[Install]
WantedBy=default.target
UNITEOF
fi

cp "$GW" "$GW.lab-bak"; cp "$FUND/.env" "$FUND/.env.lab-bak"; cp lab/production.env lab/production.env.lab-bak
rollback() {
  say "rolling back: $1"
  systemctl --user disable --now riverguide-engine 2>/dev/null
  [ "$ENGINE" != ollama-p1 ] && systemctl --user enable --now riverguide-engine 2>/dev/null   # an earlier winner
  mv "$GW.lab-bak" "$GW"; mv "$FUND/.env.lab-bak" "$FUND/.env"; mv lab/production.env.lab-bak lab/production.env
  systemctl --user try-restart river-worker   # only if it isn't shelved
  "$OPS" open riverguide "Engine bake-off picked $WIN, but switching to it failed ($1); production is back on $ENGINE. Details: riverguide-retriever/runs/bakeoff.json" --key riverguide:bakeoff
  exit 1
}

systemctl --user daemon-reload
if [ "$WIN" = ollama-p1 ]; then
  systemctl --user disable --now riverguide-engine 2>/dev/null
else
  systemctl --user enable --now riverguide-engine
  for _ in $(seq 180); do curl -sf "$BASE_NEW/v1/models" > /dev/null && break; sleep 5; done
  curl -sf "$BASE_NEW/v1/models" > /dev/null || rollback "riverguide-engine did not come up"
fi
[ "$BASE_NEW" = http://127.0.0.1:11434 ] && sed -i '/^OLLAMA_URL=/d' "$GW" || setenv "$GW" OLLAMA_URL "$BASE_NEW"
# How many questions the gateway lets through at once: twice what the engine answers together
# (Ollama and llama.cpp run 4 slots), so a short burst waits a few seconds in the engine's queue
# instead of a student seeing "busy" (approved 2026-10-25). An engine that sizes its own batch
# (vLLM or SGLang with no cap on running requests) gets 16.
slots=$("$RL" json runs/bakeoff.json "engines.$WIN.checks.concurrency" 2>/dev/null)
case "$slots" in ''|auto|*[!0-9]*) cap=16 ;; *) cap=$((2 * slots)); (( cap > 16 )) && cap=16 ;; esac
setenv "$GW" RIVERGUIDE_MODEL_MAX_IN_FLIGHT "$cap"
setenv "$FUND/.env" LLM_API "'$API'"
setenv "$FUND/.env" OLLAMA_URL "'$BASE_NEW'"
# Fundraising's worker may be paused for the window: wake it, then restart it on the new engine.
[ -f lab/.maintenance-paused ] && xargs -r kill -CONT < lab/.maintenance-paused 2>/dev/null; rm -f lab/.maintenance-paused
systemctl --user try-restart river-worker   # only if it isn't shelved

# Both apps on the new engine before anything is removed.
(cd "$FUND" && timeout 600 .venv/bin/python manage.py shell -c "
from prospect.extraction import request_extraction
r = request_extraction('Extract documented philanthropic facts. Return only the requested JSON schema.',
    'In 2024 Jane Example gave \$50,000 to endow a nursing scholarship at Indian River State College.')
print('FUND_OK', len(r['prospects']))" 2>&1) | grep -q FUND_OK || say "fundraising's extraction call failed on $WIN (app shelved; check before un-shelving)"
(cd /home/irsc/Projects/riverguide && .venv/bin/python "$ROOT/load_test.py" --name promote-check --base "$BASE_NEW" --levels 1 --per-level 3)
unfinished=$("$RL" json runs/load-promote-check.json levels.0.unfinished) passed=$("$RL" json runs/load-promote-check.json levels.0.passed)
[ "$unfinished" = 0 ] && [ "${passed%.*}" -ge 2 ] 2>/dev/null || rollback "RiverGuide's check questions failed"
rm -f runs/load-promote-check.json

printf '# The engine serving production Qwen, kept by promote_engine.sh. BASE is what the lab'"'"'s tools call.\nENGINE=%s\nBASE=%s\nSERVICE=%s\n' \
  "$WIN" "$BASE_NEW" "$([ "$WIN" = ollama-p1 ] || echo riverguide-engine)" > lab/production.env
rm -f "$GW.lab-bak" "$FUND/.env.lab-bak" lab/production.env.lab-bak
remove_losers
say "production is now $WIN at $BASE_NEW"
"$OPS" open riverguide "Engine bake-off: $WIN now serves RiverGuide and fundraising ($("$RL" json runs/bakeoff.json reason)). $(outcome) Details: riverguide-retriever/runs/bakeoff.json" --key riverguide:bakeoff
