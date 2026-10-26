#!/usr/bin/env bash
# Lab job (share): the answers students tap still work, and still come from the cache, through
# production's Worker. riverguide's deploy/omarchy/prewarm.py asks every question the page offers
# (starter chips, most-asked, typeahead) twice in a row: the first pass must answer every one with a
# finished answer (and fills the cache until the next crawl), the second must find every one in the
# cache (back in under 2 s). Run after a retriever switch (gate_retriever.sh --switch):
#   lab/enqueue.sh 46e-check-popular share ./check_popular.sh
# Result: runs/check-popular-DATE.json and an ops-alert either way (riverguide:popular-answers).
set -uo pipefail
cd "$(dirname "$0")"
ROOT=$PWD RG=/home/irsc/Projects/riverguide RL=$HOME/.local/bin/riverguide-lab
OPS=$(command -v ops-alert || echo "$HOME/.local/bin/ops-alert")
say() { echo "$(date +%T) $*"; }
out=runs/check-popular-$(date +%F-%H%M).json
model=$(basename "$(readlink -f models/production)")
for pass in 1 2; do
  (cd "$RG" && .venv/bin/python deploy/omarchy/prewarm.py) > "/tmp/popular-$pass.json" 2> "/tmp/popular-$pass.err"
  say "pass $pass: $(cat "/tmp/popular-$pass.json")"
  sed 's/^/  /' "/tmp/popular-$pass.err" | head -20
done
asked=$("$RL" json /tmp/popular-1.json questions) failed1=$("$RL" json /tmp/popular-1.json failed)
cached2=$("$RL" json /tmp/popular-2.json already_cached) failed2=$("$RL" json /tmp/popular-2.json failed)
ok=false
[ -n "$asked" ] && [ "$failed1" = 0 ] && [ "$failed2" = 0 ] && [ "$cached2" = "$asked" ] && ok=true
printf '{"model": "%s", "questions": %s, "answered": %s, "cached_on_second_ask": %s, "ok": %s}\n' \
  "$model" "${asked:-0}" "$(( ${asked:-0} - ${failed1:-0} ))" "${cached2:-0}" "$ok" > "$out"
summary="${cached2:-0} of ${asked:-?} popular questions came back from the cache on the second ask; $(( ${asked:-0} - ${failed1:-0} )) of ${asked:-?} answered on the first (retriever $model)"
say "$summary; ok: $ok"
if $ok; then
  "$OPS" open riverguide "Popular answers OK: $summary." --key riverguide:popular-answers
else
  "$OPS" open riverguide "Popular answers need a look: $summary. Failures: riverguide-retriever/lab/done/ (the check-popular log)." --key riverguide:popular-answers
  exit 1
fi
