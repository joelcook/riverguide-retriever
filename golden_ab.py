"""evals/search_ab.py's golden A/B, with the search tool answered by TrainedSearch.

    cd ../riverguide && .venv/bin/python ../riverguide-retriever/golden_ab.py --name full \
        --search embedding --ollama http://127.0.0.1:11435 --embed-model riverguide-retriever-full

Arguments after --name go to search_ab unchanged. The run is written by search_ab to riverguide's evals/runs/ as
usual and is moved to runs/golden-<name>.json, keeping riverguide's tree clean.
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, "/home/irsc/Projects/riverguide")

from evals import run, search_ab  # noqa: E402
from trained_search import TrainedSearch, canary_safe  # noqa: E402


def main() -> int:
    name = sys.argv[sys.argv.index("--name") + 1]
    sys.argv = [a for i, a in enumerate(sys.argv) if a != "--name" and sys.argv[i - 1] != "--name"]
    search_ab.EmbeddingSearch = TrainedSearch
    search_ab.answer = canary_safe(search_ab.answer)
    before = set(run.RUNS_DIR.glob("*.json"))
    code = search_ab.main()
    (new,) = set(run.RUNS_DIR.glob("*.json")) - before
    shutil.move(new, ROOT / "runs" / f"golden-{name}.json")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
