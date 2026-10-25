#!/usr/bin/env bash
# Answering-model A/B on RiverGuide's golden set: production's Qwen3.8 27B against a challenger,
# same prompt, tools, keyword search and grading (evals/search_ab.py via golden_ab.py). Rounds
# alternate so both models see the same hour. Ollama holds one model at a time, so run it when
# students are asleep; production's model is reloaded at the end.
#   ./model_ab.sh muse-glimmer:30b   (logs/model-ab.log)
set -uo pipefail
cd "$(dirname "$0")"
ROOT=$PWD
CHALLENGER=${1:?challenger model tag}
PROD=qwen3.8:27b
exec >> logs/model-ab.log 2>&1
say() { echo "$(date +%F' '%T) $*"; }

ollama show "$CHALLENGER" | grep -qw tools || { say "$CHALLENGER has no tool calling; not tested"; exit 1; }
for round in 1 2; do
  for model in "$PROD" "$CHALLENGER"; do
    say "golden $model round $round"
    (cd /home/irsc/Projects/riverguide && .venv/bin/python "$ROOT/golden_ab.py" --name "model-${model//:/-}-r$round" \
        --search keyword --model "$model" | grep -E "passed, answers|^  [a-z]")
  done
done
curl -s localhost:11434/api/chat -d "{\"model\": \"$CHALLENGER\", \"messages\": [], \"keep_alive\": 0}" > /dev/null
curl -s localhost:11434/api/chat -d "{\"model\": \"$PROD\", \"messages\": []}" > /dev/null
say "done; $PROD loaded again"
summary=$(python3 ab_summary.py "$PROD" "$CHALLENGER")
result=$(head -1 <<< "$summary"); winner=$(sed -n 's/^WINNER //p' <<< "$summary")
say "result: $result; winner $winner"
OPS=$(command -v ops-alert || echo "$HOME/.local/bin/ops-alert")
# Only the best model survives. A losing challenger goes now; production's model goes only after
# production has been switched to the winner, or RiverGuide would have nothing to run.
if [ "$winner" = "$PROD" ]; then
  ollama rm "$CHALLENGER" > /dev/null && say "removed $CHALLENGER"
  "$OPS" open riverguide "Model A/B: $PROD stays. $result. Removed $CHALLENGER." --key "riverguide:model-ab-${CHALLENGER//[:.]/-}"
else
  "$OPS" open riverguide "Model A/B: $CHALLENGER beat $PROD. $result. Swap production (MODEL_ID in web/wrangler.jsonc, deploy), then ollama rm $PROD." \
    --key "riverguide:model-ab-${CHALLENGER//[:.]/-}"
fi
