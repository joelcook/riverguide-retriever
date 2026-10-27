# riverguide-retriever

A small search model for [RiverGuide](https://github.com/joelcook/riverguide): Qwen3-Embedding-0.6B
fine-tuned to find the Indian River State College page that answers a student's question,
including the abbreviations and nicknames keyword search misses (RN, BSN, rad tech, pharm tech, IT).
It is a "System 1" model in the Rampart Edge / Clef sense: one forward pass per query, a score
per chunk, no generated tokens.

The models are release assets (zips with a model card), not files in the repo.

## Results

Right page in the top 5, page-level. MRR@10 in parentheses.

| Test | Keyword (BM25) | Stock 0.6B | **strict** | **full** |
|---|---|---|---|---|
| Abbreviations (75) | 51 (0.54) | 65 (0.64) | 67 (0.69) | 73 (0.77) |
| Golden questions (21) | 18 (0.67) | 19 (0.78) | 19 (0.82) | 19 (0.87) |
| Held-out pages, `val` (942) | 631 (0.56) | 757 (0.59) | 902 (0.77) | 909 (0.77) |

- **strict** never saw the 75 test abbreviations in training, so 67/75 is the honest
  abbreviation score. **full** trained on queries that contain them, so its 73/75 is contaminated.
  Use full in production and cite strict.
- `val` is the clean comparison for both models: 757 → 909 of 942 held-out pages, all from pages
  never trained on.
- Hybrid with keyword search adds nothing to either fine-tuned model. The fusion weight picked on
  `val` was 0.
- full still misses **esthetician** and **BRT**. strict also misses BSN, EV mechanic, CIT, BSBA, ECE,
  and AAS HVAC. Some misses come from strict grading: "AAS HVAC" returns the HVAC program page, but
  the test only accepts the catalog's AAS page.
- Per-question rankings are in `runs/*-top-questions.md` and the raw scores in `runs/*.json`.

Downloads (release `v1`): `riverguide-retriever-full.zip` and `riverguide-retriever-strict.zip`,
949 MB each, bf16 weights plus a model card with the query prefix.

### In RiverGuide (2026-10-24)

- **Real students' questions:** 523 distinct first questions from the College's River Guide chat
  (Sept–Oct 2026), the top 5 pages judged blind by Qwen3.8-27B. This model answered 88%, the stock
  0.6B 79%, and keyword search (BM25) 59%. Head to head with keyword search, it won 160 and lost
  11. Only aggregate counts were kept (`real_questions.py`).
- **Production loop, golden gate** (RiverGuide's Worker run locally, 54 questions): behind the
  search tool and run on the student's first question before the model's first step,
  43.7 → 45.5 passed, 32% fewer tool calls (111 → 75.5), and a 26% lower median answer time. It
  went live on 2026-10-24 (riverguide `cfea663`).
- Published as [joeltcook/riverguide-retriever-full](https://huggingface.co/joeltcook/riverguide-retriever-full).

## How it was built

Everything runs on omarchy (RTX 4090) and is pinned to RiverGuide's corpus at `5d125e7` (the
2026-10-22 crawl, 2,389 pages, 8,484 chunks). A doc is one of RiverGuide's own search chunks,
shown the way search shows it: page title, section, and the excerpt.

1. **Queries** (`gen_queries.py`, `gen_program.py`). Qwen3.8-27B on omarchy's Ollama wrote four
   student-style queries for each of 5,923 chunks (a full question, a 2–5 word search, the
   shorthand or abbreviation, a misspelling), and six "how people refer to this program" queries
   for each of 377 program pages (credential letters, nicknames, job titles).
2. **Leakage** (`build_data.py`). Training queries that copy a golden question are dropped. In
   `strict` mode, so is every query containing one of the 75 test abbreviations (plurals and
   lowercase "it support"/"as degree" included), so the abbreviation score measures abbreviations
   the model never trained on. Pages whose URL hashes to 0 mod 20 are held out as `val`.
3. **A judge** (`rerank.py`). Qwen3-Reranker-0.6B reads each query with its chunk. Queries their
   own chunk doesn't answer are dropped. Hard negatives (the stock embedder's closest chunk from
   another page, keyword search's best chunk from another page) are kept only when the judge says
   they don't answer the query: the College repeats itself across irsc.edu and the catalog, so a
   close chunk is often another right answer.
4. **Training** (`train.py`). InfoNCE (τ = 0.05) against the right chunk, two hard negatives, and
   the rest of the batch (64 queries). In-batch candidates are masked when they are another chunk
   of the right page, the same text, or scored above the right chunk by the stock model (GISTEmbed's
   idea). One epoch, lr 1e-5, best step by val MRR@10. Training steps off the GPU from :44 to :56
   each hour so RiverGuide's canary can load Qwen.
5. **Evaluation** (`evaluate.py`). Page-level, a page's score is its best chunk's:
   `evals/abbreviations.yaml` (right page in the top 5, as `evals/search_compare.py`), the golden
   questions that name pages to cite, and `val`. Hybrid is reciprocal-rank fusion with keyword
   search, its weight picked on `val` only.

## In production

- **Decision, 2026-10-27:** retain `v2-qwen` retrieval without reranking. The reranker's full
  Worker comparison demonstrated no end-to-end benefit: 97/108 full-check passes versus 98/108,
  median 24.6 s versus 22.6 s, and 149 versus 143 tool calls. `models/reranker-v1` remains an
  experimental candidate; its weights and measurements are retained. The old tolerance-only
  gate is preserved separately, and its promotion verdict is withdrawn. See
  [the review and future promotion contract](docs/retriever-promotions.md).
- `retriever_service.py`: models/production (the model that last passed the golden gate,
  `gate_retriever.sh`) on omarchy's **GPU** in bfloat16, as refresh.py builds the index
  (15 ms a query, 60+ a second, 1.7 GB beside Qwen; on the CPU it was 165 ms and 6-8 a second, so 32
  at once waited past the Worker's 4 s timeout), stopped during lab maintenance windows,
  serving `index/` on 127.0.0.1:8791 as the user service `riverguide-retriever`. RiverGuide's
  Worker reaches it through the model gateway's `/v1/search` and falls back to keyword search when
  it can't answer.
- `refresh.py` / `refresh.sh`: after each nightly crawl, re-embed `../riverguide/corpus` into
  `index/` (swapped in whole; the service picks it up). On Sundays, `--check` reruns the frozen tests
  on the new crawl and opens an ops-alert below the floors the first check set (`drift_floors.json`).
  Facts live in the pages, so a changed fee needs only the re-embed; a drop means the College's
  vocabulary moved, and it's time to retrain.

## The lab

`lab/lab.sh` runs queued experiments, one at a time per lane, from cron every 10 minutes:
`cpu` jobs (installs) any time; `share` jobs alongside production; `night` jobs inside a
maintenance window (`lab/maintenance.sh`: the gateway off, Ollama unloaded, the canary paused,
an ops-alert open), 00:30–04:10 or a window in `lab/offline.conf`. Exit 75 keeps a job queued.

- `model_ab.sh MODEL`: RiverGuide's golden set, production's model against a challenger, two
  rounds each. Only the best survives: a losing challenger is removed. `watch_qwen4.sh` (cron)
  queues one the night Qwen 4 reaches Ollama (`pick_model.py` picks the biggest tag that fits).
- `bakeoff.sh` → `riverguide-lab verdict` → `promote_engine.sh`: Ollama (1 and 4 slots), llama.cpp,
  vLLM, and SGLang serving the same model. Each gets the golden set twice, `load_test.py` at 1, 4,
  8, and 16 students, and its GPU memory. A challenger must hold quality and be 15% faster at 4
  students to win; the winner becomes production (`riverguide-engine` on :11440). Only engines
  that lost a fair run are uninstalled. The fairness gate (`rust/src/fair.rs`) checks that each
  engine's best settings served as many students at once as production, its drafts were accepted,
  a conversation could reach 32K tokens, its server logged no errors, and its prefix cache hit.
  An engine whose run fails a check stays installed for a re-run, and an ops-alert says why.
- `rust/`: `riverguide-lab`, the lab's tools in Rust (`cargo install --locked --path rust --root
  ~/.local`): the verdict and its fairness gate, bakeoff.sh's records, JSON reads for the scripts,
  Ollama's loaded models, and the SGLang patch. The repo's Python moves there piece by piece.
- `gate_retriever.sh --plan FILE [--rerank] CANDIDATE` measures a retriever or reranker against
  production using a committed plan. `retriever_gate.py` freezes that plan before testing and
  requires a meaningful product benefit with quality and reliability held. `--switch` replays
  the evidence and refuses legacy tolerance-only passes or changed setups. No experimental
  weights are deleted. `--disable-rerank` restores retriever-only serving without deleting them.
  Details and plan format: [docs/retriever-promotions.md](docs/retriever-promotions.md).
- `load_test.py`, `fewer_calls.py` (search first, plus RiverGuide 1's LLM wiki on the same model),
  `real_questions.py`, `loop_abbreviations.py`, and `golden_ab.py` are the measurements; results
  land in `runs/`.

## Reproduce

```sh
uv venv --python 3.14 .venv && uv pip install --python .venv/bin/python torch sentence-transformers transformers datasets accelerate pyyaml numpy
git -C ../riverguide archive 5d125e7 corpus | tar -x -C data/corpus-2026-10-22
../riverguide/.venv/bin/python gen_queries.py && ../riverguide/.venv/bin/python gen_program.py   # ~2 h on the 4090
./pipeline.sh                                                                                    # judge, train, evaluate both modes
.venv/bin/python package.py strict full                                                          # dist/*.zip
```

## Trying it inside RiverGuide

`serve.py` answers Ollama's `/api/embed` with a trained model and passes everything else to
Ollama, so RiverGuide's own harnesses run unchanged:

```sh
.venv/bin/python serve.py --model models/full --port 11435
cd ../riverguide
.venv/bin/python -m evals.search_ab --search embedding --ollama http://127.0.0.1:11435 --embed-model riverguide-retriever
.venv/bin/python ../riverguide-retriever/loop_abbreviations.py --search embedding --ollama http://127.0.0.1:11435 --embed-model riverguide-retriever --name full
```

`loop_abbreviations.py` asks the 75 abbreviations as questions and lets Qwen write the searches,
the way production does.
