#!/usr/bin/env bash
# Lab job (cpu): install the engines the bake-off compares with Ollama, each on its own in
# ~/engines/<name>, and the model files they need. An engine that installs writes READY (its
# version); one that doesn't writes FAILED (why) and simply loses the bake-off. Safe to rerun.
#   llamacpp  llama.cpp's llama-server, built for the 4090 (sm_89); serves Ollama's own GGUF
#   vllm      vLLM in a Python 3.12 venv; serves the W4A16 checkpoint below
#   sglang    SGLang in a Python 3.12 venv; same checkpoint
#   models/qwen3.8-27b-w4a16   dbirks/Qwen3.8-27B-W4A16-AutoRound (about 19.5 GB)
#   models/qwen3.8-27b-template  Qwen's own chat template, for llama.cpp's tool calls
set -uo pipefail
E=$HOME/engines
mkdir -p "$E/models"
say() { echo "$(date +%T) $*"; }
mark() { rm -f "$E/$1/READY" "$E/$1/FAILED"; echo "$3" > "$E/$1/$2"; say "$1: $2 $3"; }

# Model files first: vLLM and SGLang share them.
if [ ! -f "$E/models/qwen3.8-27b-w4a16/READY" ]; then
  uv venv -q --python 3.12 "$E/hf" 2>/dev/null; uv pip install -q --python "$E/hf/bin/python" -U huggingface_hub
  mkdir -p "$E/models/qwen3.8-27b-w4a16"
  if "$E/hf/bin/hf" download dbirks/Qwen3.8-27B-W4A16-AutoRound --local-dir "$E/models/qwen3.8-27b-w4a16" > /dev/null; then
    mark models/qwen3.8-27b-w4a16 READY "$(du -sh "$E/models/qwen3.8-27b-w4a16" | cut -f1)"
  else
    mark models/qwen3.8-27b-w4a16 FAILED "download failed"
  fi
fi
if [ ! -f "$E/models/qwen3.8-27b-template/READY" ]; then
  mkdir -p "$E/models/qwen3.8-27b-template"
  if "$E/hf/bin/hf" download Qwen/Qwen3.8-27B chat_template.jinja tokenizer_config.json --local-dir "$E/models/qwen3.8-27b-template" > /dev/null; then
    mark models/qwen3.8-27b-template READY "ok"
  else
    mark models/qwen3.8-27b-template FAILED "download failed"
  fi
fi

for py in vllm sglang; do
  [ -f "$E/$py/READY" ] && continue
  mkdir -p "$E/$py"
  pkg=$py; [ "$py" = sglang ] && pkg="sglang[all]"
  if uv venv -q --python 3.12 "$E/$py/.venv" && uv pip install -q --python "$E/$py/.venv/bin/python" "$pkg"; then
    mark "$py" READY "$("$E/$py/.venv/bin/python" -c "import $py; print($py.__version__)" 2>&1 | tail -1)"
  else
    mark "$py" FAILED "pip install failed"
  fi
done

if [ ! -f "$E/llamacpp/READY" ]; then
  mkdir -p "$E/llamacpp"
  [ -d "$E/llamacpp/src" ] || git clone -q --depth 1 https://github.com/ggml-org/llama.cpp "$E/llamacpp/src"
  uv venv -q --python 3.12 "$E/llamacpp/tools" 2>/dev/null; uv pip install -q --python "$E/llamacpp/tools/bin/python" cmake ninja
  export PATH="$E/llamacpp/tools/bin:/opt/cuda/bin:$PATH" CUDACXX=/opt/cuda/bin/nvcc
  if (cd "$E/llamacpp/src" && cmake -S . -B build -G Ninja -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=89 \
        -DCMAKE_CUDA_FLAGS="-allow-unsupported-compiler" -DLLAMA_CURL=OFF -DCMAKE_BUILD_TYPE=Release > /dev/null \
      && cmake --build build -j 24 --target llama-server > "$E/llamacpp/build.log" 2>&1); then
    mark llamacpp READY "$(cd "$E/llamacpp/src" && git rev-parse --short HEAD)"
  else
    mark llamacpp FAILED "build failed: $(tail -3 "$E/llamacpp/build.log" 2>/dev/null | tr '\n' ' ' | cut -c1-200)"
  fi
fi
ls "$E"/*/READY "$E"/models/*/READY 2>/dev/null
ls "$E"/*/FAILED "$E"/models/*/FAILED 2>/dev/null
exit 0
