#!/usr/bin/env bash
# Take RiverGuide offline for a lab job, and bring it back.
#   lab/maintenance.sh start NAME DEADLINE   stop the model gateway (students see RiverGuide's
#                                            "unavailable" message), pause fundraising-prospect-
#                                            discovery's worker (its queue waits; approved
#                                            2026-10-24), unload Ollama's model, pause the hourly
#                                            canary, and say so through ops-alert
#   lab/maintenance.sh end                   undo all of it, warm production's model, clear the alert
# The lab's own Ollama calls use /api/chat, so /api/generate in Ollama's log is fundraising's.
# lab.sh calls these around every night job, and closes a window a dead job left open.
set -uo pipefail
# Cron has no XDG_RUNTIME_DIR, and systemctl --user needs it (deploy/omarchy/nightly.sh does the same).
export XDG_RUNTIME_DIR="${XDG_RUNTIME_DIR:-/run/user/$(id -u)}"
LAB="$(cd "$(dirname "$0")" && pwd)"
OPS=$(command -v ops-alert || echo "$HOME/.local/bin/ops-alert")
CANARY='deploy/omarchy/canary.py'"'"
say() { echo "$(date +%F' '%T) [maintenance] $*"; }

case "${1:?start or end}" in
  start)
    echo "$2 $3" > "$LAB/.maintenance"
    crontab -l | sed "s|^\([0-9].*${CANARY}.*\)$|#lab-paused# \1|" | crontab -
    systemctl --user stop riverguide-model
    . "$LAB/production.env"; [ -n "${SERVICE:-}" ] && systemctl --user stop "$SERVICE"   # a non-Ollama production engine
    for pid in $(pgrep -f "fundraising-prospect-discovery/manage.py worker"); do
      [ "$(readlink "/proc/$pid/cwd")" = "$HOME/Projects/fundraising-prospect-discovery" ] || continue
      kill -STOP "$pid" && echo "$pid" >> "$LAB/.maintenance-paused"
    done
    for m in $(curl -s localhost:11434/api/ps | python3 -c 'import json,sys; print(" ".join(x["name"] for x in json.load(sys.stdin)["models"]))'); do
      curl -s localhost:11434/api/chat -d "{\"model\": \"$m\", \"messages\": [], \"keep_alive\": 0}" > /dev/null
    done
    "$OPS" open riverguide "RiverGuide is offline for lab job $2 until $(date -d "@$3" +%H:%M) at the latest; it comes back on its own." --key riverguide:maintenance
    say "offline for $2"
    ;;
  end)
    crontab -l | sed 's|^#lab-paused# ||' | crontab -
    . "$LAB/production.env"; [ -n "${SERVICE:-}" ] && systemctl --user start "$SERVICE"
    [ -f "$LAB/.maintenance-paused" ] && xargs -r kill -CONT < "$LAB/.maintenance-paused" 2>/dev/null
    rm -f "$LAB/.maintenance-paused"
    systemctl --user start riverguide-model
    # Warm production's model when the gateway is in front of Ollama (its default upstream).
    upstream=$(sed -n 's/^OLLAMA_URL=//p' ~/.config/riverguide/model-gateway.env)
    model=$(sed -n 's/^RIVERGUIDE_MODEL=//p' ~/.config/riverguide/model-gateway.env)
    if [ -z "$upstream" ] || [[ $upstream == *:11434* ]]; then
      curl -s localhost:11434/api/chat -d "{\"model\": \"${model:-qwen3.8:27b}\", \"messages\": []}" > /dev/null
    fi
    "$OPS" clear riverguide --key riverguide:maintenance
    rm -f "$LAB/.maintenance"
    say "back online"
    ;;
esac
