#!/usr/bin/env bash
# Put SGLang back for a fair re-run: promote_engine.sh removed it on 2026-10-25 after runs that
# riverguide-lab's fairness gate now rejects (one conversation at a time, a drafter that accepted
# nothing). SGLang 0.5.22 (the latest release), the W4A16 checkpoint, Qwen's chat template, and
# two DFlash2 drafters: syvai's W4A16 conversion (1.3 GB, the one that accepted 0% of drafts) and
# Inco AI's original in bf16 (3.8 GB), so bakeoff.sh can tell a bad conversion from an SGLang bug.
# Each directory gets READY or FAILED, as install_engines.sh writes.
#   lab/enqueue.sh 46-install-sglang cpu ./install_sglang.sh
set -uo pipefail
E=$HOME/engines; say() { echo "$(date +%T) $*"; }
mark() { mkdir -p "$E/$1"; rm -f "$E/$1/READY" "$E/$1/FAILED"; echo "$3" > "$E/$1/$2"; say "$1: $2 $3"; }
uv venv -q --python 3.12 "$E/hf" 2>/dev/null; uv pip install -q --python "$E/hf/bin/python" -U huggingface_hub
get() {  # get DIR REPO [FILE...]
  local dir=$1 repo=$2; shift 2
  [ -f "$E/$dir/READY" ] && return 0
  mkdir -p "$E/$dir"
  if "$E/hf/bin/hf" download "$repo" "$@" --local-dir "$E/$dir" > /dev/null 2>&1; then
    mark "$dir" READY "$repo $(du -sh "$E/$dir" | cut -f1)"
  else
    mark "$dir" FAILED "download of $repo failed"
  fi
}
if [ ! -f "$E/sglang/READY" ]; then
  mkdir -p "$E/sglang"
  # ninja in the venv: FlashInfer's and sgl-kernel's JIT need it on PATH, and cron's PATH has none.
  if uv venv -q --python 3.12 "$E/sglang/.venv" && uv pip install -q --python "$E/sglang/.venv/bin/python" "sglang[all]==0.5.22" ninja; then
    mark sglang READY "$("$E/sglang/.venv/bin/python" -c "import sglang; print(sglang.__version__)" 2>&1 | tail -1)"
  else
    mark sglang FAILED "pip install failed"
  fi
fi
get models/qwen3.8-27b-w4a16 dbirks/Qwen3.8-27B-W4A16-AutoRound
get models/qwen3.8-27b-template Qwen/Qwen3.8-27B chat_template.jinja tokenizer_config.json
get models/qwen3.8-27b-dflash2-w4a16 syvai/Qwen3.8-27B-DFlash2-W4A16
get models/qwen3.8-27b-dflash2 incoai/Qwen3.8-27B-DFlash2
say done
