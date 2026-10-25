#!/usr/bin/env python3
"""Let SGLang's Qwen3.5-family MTP draft skip building its own embedding and lm_head.

SGLang builds the MTP draft with a full bf16 embedding and lm_head (248K x 5120 each, 2.37 GiB
apiece), loads the target's embedding into it, then throws both away for the target's own
tensors (set_embed_and_head). On a 24 GB card that brief 4.7 GiB peak is what stops it starting.
With RIVERGUIDE_MTP_SHARED_EMBED=1 in the environment, both are built on the meta device instead,
so nothing is allocated before the share. Unset, SGLang behaves exactly as shipped.

Idempotent: run it after any SGLang upgrade (bakeoff.sh does). Usage: sglang_mtp_shared_embed.py SITE_PACKAGES
"""
import sys
from pathlib import Path

GATE = '(torch.device("meta") if is_nextn and os.environ.get("RIVERGUIDE_MTP_SHARED_EMBED") else __import__("contextlib").nullcontext())'
models = Path(sys.argv[1]) / "sglang/srt/models"
edits = {
    models / "qwen3_5.py": (
        "        self.embed_tokens = self._build_embed_tokens(config)\n",
        f"        with {GATE}:  # riverguide: MTP draft shares the target's\n"
        "            self.embed_tokens = self._build_embed_tokens(config)\n",
    ),
    models / "qwen3_5_mtp.py": (
        "                self.lm_head = ParallelLMHead(\n",
        "                with (torch.device(\"meta\") if __import__(\"os\").environ.get(\"RIVERGUIDE_MTP_SHARED_EMBED\") else __import__(\"contextlib\").nullcontext()):  # riverguide: shares the target's\n"
        "                  self.lm_head = ParallelLMHead(\n",
    ),
}
for path, (old, new) in edits.items():
    text = path.read_text()
    if "riverguide:" in text:
        print(f"{path.name}: already patched")
        continue
    assert text.count(old) == 1, f"{path.name}: expected one match for {old!r}, found {text.count(old)}"
    path.write_text(text.replace(old, new))
    print(f"{path.name}: patched")
