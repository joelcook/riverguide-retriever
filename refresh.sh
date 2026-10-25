#!/usr/bin/env bash
# Cron entry for refresh.py: wait for RiverGuide's nightly crawl to finish (its lock), then
# re-embed; on Sundays also run the drift check.
#   ./refresh.sh [--check]   (logs/refresh.log)
set -euo pipefail
cd "$(dirname "$0")"
flock -w 7200 /home/irsc/Projects/riverguide/logs/nightly.lock true
args=()
[ "$(date +%u)" = 7 ] && args+=(--check)
.venv/bin/python refresh.py "${args[@]}" "$@" 2>&1 | grep -v "Batches\|Loading weights\|Warning"
