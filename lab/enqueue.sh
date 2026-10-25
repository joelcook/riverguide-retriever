#!/usr/bin/env bash
# Queue a job for the lab:  lab/enqueue.sh NAME CLASS COMMAND...
#   lab/enqueue.sh 50-model-ab-glimmer night ./model_ab.sh muse-glimmer:30b
# NAME orders the queue (lower runs first). The command runs from the repo root.
set -euo pipefail
LAB="$(cd "$(dirname "$0")" && pwd)"
name=$1 class=$2; shift 2
[[ $class =~ ^(cpu|share|night)$ ]] || { echo "class must be cpu, share, or night" >&2; exit 2; }
mkdir -p "$LAB/queue"
job="$LAB/queue/$name"
[ -e "$job" ] && { echo "already queued: $name" >&2; exit 0; }
printf '#!/usr/bin/env bash\n# class: %s\nset -uo pipefail\n%s\n' "$class" "$(printf '%q ' "$@")" > "$job.tmp"
chmod +x "$job.tmp" && mv "$job.tmp" "$job"
echo "queued $name ($class)"
