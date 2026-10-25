#!/usr/bin/env bash
# Twice a day: has Qwen 4 reached Ollama's library? The first time it has, pull the biggest tag
# that fits the 4090, queue model_ab.sh against production's model for the lab's next night
# window, and say so through ops-alert. Once it has acted it stays quiet (logs/qwen4.seen).
#   ./watch_qwen4.sh   (logs/watch-qwen4.log)
set -uo pipefail
cd "$(dirname "$0")"
exec >> logs/watch-qwen4.log 2>&1
say() { echo "$(date +%F' '%T) $*"; }
SEEN=logs/qwen4.seen
[ -f "$SEEN" ] && exit 0
OPS=$(command -v ops-alert || echo "$HOME/.local/bin/ops-alert")

pick=$(python3 pick_model.py qwen4 qwen4.0)
[ -z "$pick" ] && { say "no qwen4 on Ollama yet"; exit 0; }
if [[ $pick == NOFIT* ]]; then
  say "$pick"
  "$OPS" open riverguide "Qwen 4 is on Ollama, but no tag fits omarchy's 24 GB: ${pick#NOFIT }" --key riverguide:qwen4
  echo "$pick" > "$SEEN"; exit 0
fi
tag=${pick% *}; gb=$(python3 -c "print(f'{${pick#* }/1e9:.1f}')")
say "found $tag ($gb GB); pulling"
ollama pull "$tag" > /dev/null || { say "pull of $tag failed; will try again next run"; exit 1; }
lab/enqueue.sh "40-model-ab-${tag//[:.]/-}" night ./model_ab.sh "$tag"
when="tonight from 00:30"
"$OPS" open riverguide "Qwen 4 is on Ollama: pulled $tag ($gb GB). Golden A/B against qwen3.8:27b runs $when; results land here and in riverguide-retriever/logs/model-ab.log." --key riverguide:qwen4
echo "$tag $when" > "$SEEN"
say "scheduled the A/B for $when"
