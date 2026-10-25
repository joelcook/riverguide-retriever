#!/usr/bin/env bash
# The lab: RiverGuide's experiments, run without anyone starting them. Cron calls it every 10
# minutes for each lane; each lane runs its queued jobs one at a time, in name order, back to back.
#   lab/lab.sh gpu    share and night jobs (the GPU)
#   lab/lab.sh cpu    cpu jobs (installs, downloads)
# A job is an executable in lab/queue/ whose second line is "# class: cpu|share|night":
#   cpu    any time
#   share  uses production's Qwen alongside students: not 04:15-05:30 (crawl) or :44-:50 (canary)
#   night  takes RiverGuide offline (lab/maintenance.sh, around the job): starts 00:30-03:00 and
#          must end by 04:10, or in a window listed in lab/offline.conf ("START END", local ISO)
# A night job gets LAB_DEADLINE (epoch seconds) and should stop before it. Exit 75 means "not
# ready, keep me queued"; any other failure moves the job to lab/done/ and opens an ops-alert.
# Optional lines after the class: "# minutes: N" starts a night job only with N minutes left in its
# window, and "# ready: COMMAND" (run from the repo) must succeed before the job starts, so a job
# that isn't ready never takes RiverGuide offline just to say so.
set -uo pipefail
# Cron has no XDG_RUNTIME_DIR, and systemctl --user needs it (deploy/omarchy/nightly.sh does the same).
export XDG_RUNTIME_DIR="${XDG_RUNTIME_DIR:-/run/user/$(id -u)}"
LAB="$(cd "$(dirname "$0")" && pwd)"
LANE=${1:?lane: gpu or cpu}
mkdir -p "$LAB/queue" "$LAB/done" "$LAB/logs"
say() { echo "$(date +%F' '%T) [$LANE] $*" >> "$LAB/logs/lab.log"; }
OPS=$(command -v ops-alert || echo "$HOME/.local/bin/ops-alert")
exec 9> "$LAB/.lock-$LANE"
flock -n 9 || exit 0
if [ "$LANE" = gpu ]; then
  # A maintenance window left open by a job that died: close it.
  [ -e "$LAB/.maintenance" ] && { say "closing a stale maintenance window"; "$LAB/maintenance.sh" end; }
  pgrep -f "^bash (\S*/)?(loop_tests|gate_retriever_first)\.sh$|^\S*python\S* real_questions.py --org" > /dev/null && exit 0   # hand-started experiments (anchored: a watcher whose command line merely mentions them must not count)
fi

# Jobs call production's engine at PROD_BASE (lab/production.env, kept by promote_engine.sh).
PROD_BASE=$(sed -n 's/^BASE=//p' "$LAB/production.env" 2>/dev/null); export PROD_BASE=${PROD_BASE:-http://127.0.0.1:11434}
# Qwen split onto the CPU (it loaded while something else held GPU memory) answers at CPU speed.
# Between jobs, unload it so the next question loads it whole on the GPU.
heal_offload() {
  curl -s localhost:11434/api/ps | python3 -c '
import json, sys
for m in json.load(sys.stdin)["models"]:
    if m.get("size_vram", 0) < m.get("size", 0):
        print(m["name"])' | while read -r m; do
    curl -s localhost:11434/api/chat -d "{\"model\": \"$m\", \"messages\": [], \"keep_alive\": 0}" > /dev/null
    say "unloaded $m: it was partly on the CPU"
  done
}
# Load production's Qwen before a share job loads its own models, so Qwen gets the GPU first.
warm_production() {
  [[ $PROD_BASE == *:11434* ]] || return 0
  curl -s localhost:11434/api/chat -d '{"model": "qwen3.8:27b", "messages": []}' > /dev/null
}
[ "$LANE" = gpu ] && [ ! -e "$LAB/.maintenance" ] && heal_offload
clock() { now=$(date +%s); hm=$((10#$(date +%H) * 60 + 10#$(date +%M))); }
clock
night_deadline() {  # prints the deadline if a night job may start now
  if (( hm >= 30 && hm < 180 )); then date -d "today 04:10" +%s; return 0; fi
  [ -f "$LAB/offline.conf" ] || return 1
  while read -r start end; do
    [[ -z ${start:-} || $start == \#* ]] && continue
    s=$(date -d "$start" +%s) e=$(date -d "$end" +%s)
    (( now >= s && now < e - 3600 )) && { echo "$e"; return 0; }
  done < "$LAB/offline.conf"
  return 1
}
fundraising_idle() {  # its worker isn't running, or its Ollama calls (/api/generate) stopped 10+ minutes ago
  # (only while it runs: ollama rm also calls /api/generate, to unload the model it deletes)
  pgrep -f "fundraising-prospect-discovery/manage.py worker" > /dev/null || return 0
  ! journalctl -u ollama --since "10 min ago" --no-pager -o cat 2>/dev/null | grep -q 'POST .*"/api/generate"'
}
may_start() {
  case "$1" in
    cpu) [ "$LANE" = cpu ] ;;
    share) [ "$LANE" = gpu ] && (( (hm < 255 || hm >= 330) && (hm % 60 < 44 || hm % 60 >= 50) )) ;;
    night) [ "$LANE" = gpu ] && night_deadline > /dev/null && fundraising_idle ;;
    *) return 1 ;;
  esac
}

# Run jobs back to back until none may start: no waiting for the next cron tick between them.
# Consecutive night jobs share one maintenance window, so RiverGuide isn't brought back up and
# taken down again in between. A job that says "not ready" (exit 75) waits for the next tick.
offline=0
close_window() { if [ "$offline" = 1 ]; then "$LAB/maintenance.sh" end >> "$LAB/logs/lab.log" 2>&1; offline=0; fi; }
trap close_window EXIT
declare -A deferred=()
while true; do
  clock
  ran=0
  for job in "$LAB"/queue/*; do
    [ -x "$job" ] || continue
    name=$(basename "$job")
    [ -n "${deferred[$name]:-}" ] && continue
    class=$(sed -n 's/^# class: *//p' "$job" | head -1)
    may_start "$class" || continue
    minutes=$(sed -n 's/^# minutes: *//p' "$job" | head -1)
    if [ "$class" = night ] && [ -n "$minutes" ] && (( $(night_deadline) - $(date +%s) < minutes * 60 )); then deferred[$name]=1; continue; fi
    ready=$(sed -n 's/^# ready: *//p' "$job" | head -1)
    if [ -n "$ready" ] && ! ( cd "$LAB/.." && bash -c "$ready" ) > /dev/null 2>&1; then deferred[$name]=1; continue; fi
    stamp=$(date +%F-%H%M)
    log="$LAB/done/$stamp-$name.log"
    say "start $name ($class)"
    if [ "$class" = night ]; then
      export LAB_DEADLINE=$(night_deadline)
      if [ "$offline" = 0 ]; then
        "$LAB/maintenance.sh" start "$name" "$LAB_DEADLINE" >> "$LAB/logs/lab.log" 2>&1
        offline=1
      fi
      ( cd "$LAB/.." && "$job" ) > "$log" 2>&1
      code=$?
    else
      close_window
      [ "$class" = share ] && warm_production
      ( cd "$LAB/.." && "$job" ) > "$log" 2>&1
      code=$?
      [ "$class" = share ] && heal_offload
    fi
    say "end $name: exit $code"
    if [ "$code" -eq 75 ]; then rm -f "$log"; deferred[$name]=1; continue; fi   # not ready: stays queued
    mv "$job" "$LAB/done/$stamp-$name"
    [ "$code" -ne 0 ] && "$OPS" open riverguide "lab job $name failed (exit $code); log: riverguide-retriever/lab/done/$stamp-$name.log" --key "riverguide:lab-$name"
    ran=1
    break                                                     # rescan: the queue and the clock have moved
  done
  [ "$ran" = 1 ] || break
done
