"""Second pass: for each program page, how students refer to the program.

    ../riverguide/.venv/bin/python gen_program.py

The first pass writes four queries per chunk. Program names are where students use letters,
nicknames, and job titles the page never says ("the program for becoming a plumber"), so each
program page's top chunk also gets six of those. Writes data/queries.program.jsonl.
"""
from __future__ import annotations

import json
import re
import urllib.request

from common import DATA, load_docs
from gen_queries import MODEL, OLLAMA

OUT = DATA / "queries.program.jsonl"
PROGRAM_URL = re.compile(
    r"irsc\.edu/(program|programs|study_area|schools|salon|professional-training|integrated-education-training-iet|"
    r"what-adult-education|adult-education)(/|$)|/catalog/(aas-degrees-certificates|career-and-technical-education-programs|"
    r"bachelor-degree-programs|associate-in-arts-degree|programs-of-study|certificate-of-professional-preparation|"
    r"adult-educationl|accelerated-opportunities-for-advancement|farmworkers-career-development-program)"
)

PROMPT = """You write training data for the search engine behind RiverGuide, the AI advisor for Indian River State College (IRSC) in Florida.

Below is the top of a College page about a program. Students, parents, and career changers rarely type a program's official name. They type the letters of the credential, a nickname, the job they want, or a casual description.

Write 6 different searches people would type to find this program, as they really type them: often lowercase, short, informal. Cover, where they apply: the abbreviation or credential letters, the common nickname, the job title the program leads to, a casual description, a "how do I become ..." question, and a question a parent might ask. Use only abbreviations and nicknames people really use for this field. Don't repeat the page title word for word.

Page: {title}
URL: {url}
{text}"""

SCHEMA = {
    "type": "object",
    "properties": {"queries": {"type": "array", "items": {"type": "string"}, "minItems": 6, "maxItems": 6}},
    "required": ["queries"],
}


def main() -> int:
    docs = load_docs()
    first: dict[int, dict] = {}
    for d in docs:
        if PROGRAM_URL.search(d["url"]) and d["page"] not in first and d["chunk_chars"] >= 80:
            first[d["page"]] = d
    done = {json.loads(line)["id"] for line in open(OUT, encoding="utf-8")} if OUT.exists() else set()
    todo = [d for d in first.values() if d["id"] not in done]
    print(f"{len(first)} program pages, {len(todo)} to go", flush=True)
    for n, d in enumerate(todo, 1):
        body = {
            "model": MODEL,
            "messages": [{"role": "user", "content": PROMPT.format(title=d["title"], url=d["url"], text=d["text"])}],
            "format": SCHEMA, "think": False, "stream": False, "keep_alive": "10m",
            "options": {"temperature": 0.8, "top_p": 0.95},
        }
        try:
            request = urllib.request.Request(f"{OLLAMA}/api/chat", json.dumps(body).encode(), {"content-type": "application/json"})
            with urllib.request.urlopen(request, timeout=600) as response:
                queries = json.loads(json.load(response)["message"]["content"])["queries"]
        except Exception as exc:  # noqa: BLE001
            print(f"failed {d['id']}: {exc}", flush=True)
            continue
        with open(OUT, "a", encoding="utf-8") as fh:
            fh.write(json.dumps({"id": d["id"], "url": d["url"], "queries": queries}) + "\n")
        if n % 25 == 0:
            print(f"{n}/{len(todo)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
