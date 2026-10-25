"""Find the biggest tag of a model family on Ollama's library that fits omarchy's 24 GB.

    python3 pick_model.py qwen4 qwen4.0      # prints "qwen4:27b 17712345678", "NOFIT ...", or nothing

Plain size tags only (27b, 30b-a3b): the default quantization, no cloud, mlx or fp16 builds.
Sizes come from the registry manifests.
"""
import json
import re
import sys
import urllib.error
import urllib.request

FITS = 20e9  # Qwen3.8 27B is 17.7 GB and leaves room for context on the 4090
PLAIN = re.compile(r"^\d+(\.\d+)?b(-a\d+(\.\d+)?b)?$")


def get(url: str, accept: str = "") -> bytes | None:
    try:
        req = urllib.request.Request(url, headers={"Accept": accept} if accept else {})
        return urllib.request.urlopen(req, timeout=30).read()
    except urllib.error.HTTPError:
        return None


for family in sys.argv[1:]:
    page = get(f"https://ollama.com/library/{family}/tags")
    if page is None:
        continue
    tags = sorted({t for t in re.findall(rf"{re.escape(family)}:([\w.-]+)", page.decode()) if PLAIN.match(t)})
    sizes = {}
    for tag in tags:
        manifest = get(f"https://registry.ollama.ai/v2/library/{family}/manifests/{tag}",
                       "application/vnd.docker.distribution.manifest.v2+json")
        if manifest:
            layers = json.loads(manifest)["layers"]
            sizes[f"{family}:{tag}"] = sum(layer["size"] for layer in layers)
    fitting = {t: s for t, s in sizes.items() if s <= FITS}
    if fitting:
        best = max(fitting, key=fitting.get)
        print(best, fitting[best])
    elif sizes:
        print("NOFIT", " ".join(f"{t}={s / 1e9:.1f}GB" for t, s in sorted(sizes.items())))
    break
