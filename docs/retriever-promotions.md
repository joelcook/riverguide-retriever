# Retriever and reranker promotion

Production retains the trained retriever. An added stage must demonstrate a predefined,
meaningful end-to-end benefit before promotion. Intermediate ranking scores guide experiments;
they do not authorize production changes. Experimental weights and raw measurements are kept.

## Reranker v1 review, 2026-10-27

**Verdict: no demonstrated end-to-end benefit; retain retriever-only.**

The comparison used production's Worker locally, the same Qwen3.8-27B answer model and prompt,
the same crawl, cache bypassed, and four concurrent questions. Each arm answered the same 54
questions twice. Baseline: `v2-qwen`; candidate: that retriever plus `reranker-v1` over 20 pages.

| Measure | Retriever | With reranker |
|---|---:|---:|
| Full-check passes | 98/108 | 97/108 |
| Text/completion-check passes | 102/108 | 100/108 |
| Pooled median seconds | 22.6 | 24.6 |
| Mean seconds | 26.61 | 28.19 |
| Nearest-rank p95 seconds | 49.3 | 49.0 |
| Tool calls | 143 | 149 |
| Incomplete/error answers | 0 | 1 |

There are 54 distinct questions, not 108 independent questions. The semantic judge was off.
Several failures were literal wording checks or missing required citations. One reranked answer,
the long IT Support Specialist inquiry in round 2, returned a fallback instead of a completed
answer. The report does not establish that reranking caused it. No token or metered serving-cost
data was captured; the zero dollar values are placeholders. This evidence does not prove that
reranking is inherently worse, but it gives no product benefit that justifies the added stage.

The old gate accepted a mean score within one question of production: 48.5/54 versus 49/54.
That tolerance check was followed by a production promotion. Its original result is preserved as
`runs/gate-retriever-rerank-reranker-v1-tolerance-only.json`; the current result records the
withdrawn promotion and keeps the candidate experimental. The four original answers and traces
remain in RiverGuide's private `evals/runs/lab-gate-retriever-rerank-reranker-v1-*.json` files.

These old measurements were not run under the new contract. They cannot be relabeled as a
predeclared experiment by selecting a favorable objective afterward.

## Declare the target before measurement

Create a plan with exactly these fields, then commit it before running the gate:

```json
{
  "candidate": "rerank-candidate-name",
  "production": "v2-qwen",
  "production_reranker": "none",
  "objective": "latency",
  "minimum_gain": 0.15,
  "rounds": 4
}
```

`candidate` is a retriever directory name, or `rerank-` followed by a reranker directory name.
`production` and `production_reranker` must name the active baseline (`none` for no reranker).
The example declares a 15% median-latency reduction. Choose the threshold for the actual
requirement and record it before spending a test run. The gate refuses unstaged, staged-only,
or untracked plan changes: the exact plan must exist in HEAD.

Supported objectives and units:

| Objective | `minimum_gain` | Required measurement |
|---|---|---|
| `quality` | Additional semantically passing questions per paired round; whole number >= 1 | Actual judge verdicts on the golden set's `expected_summary` questions |
| `latency` | Fractional reduction in pooled median latency; >= 0.05 and < 1 | Positive per-answer seconds from the full application |
| `cost` | Fractional reduction in total serving cost; >= 0.05 and < 1 | Positive metered dollar costs and nonempty token usage for every answer |
| `reliability` | Fewer incomplete/error answers per paired round; whole number >= 1 | Completion/error results from the full application |

The current API harness does not report metered serving usage or cost. Its ordinary `usage` and
`cost_usd` fields include judge costs when judging is on, so neither those fields nor their old
zero placeholders qualify. A cost objective requires separate `serving_usage` and
`serving_cost_usd` fields for every answer; until that accounting is added, it cannot qualify.
Tool counts are reported, never substituted for measured cost.
Training, GPU memory and maintenance costs still belong in the experiment's review; this gate
does not measure total ownership cost.

## Requirements that every candidate must hold

- At least four paired rounds, with an even round count. Arm order alternates AB/BA to balance
  which arm runs first. The declared benefit must be positive in every paired round and reach
  the declared magnitude over the whole experiment.
- Full passes, text/completion passes and semantic passes cannot decrease in aggregate.
  Full and text/completion passes cannot decrease within any question category either.
- The candidate must complete every answer without a reported operational error or fallback.
- Pooled median and p95 latency cannot increase by more than 5%. When both costs are measured,
  serving cost cannot increase by more than 5%.
- Every run must be full and non-partial, with the semantic judge on. Every declared
  `expected_summary` case needs an actual semantic verdict when an answer completed; an
  incomplete answer counts as a semantic failure. This covers the semantic cases the golden
  set declares, not a comprehensive human audit of every factual claim.
- Both arms must contain the same complete, unique question IDs and categories, answering
  model and prompt. Measurements must begin after the plan was frozen. Summaries must match
  the underlying results. Artifact hashes are checked again before promotion.

The 5% guardrail is a fixed operational bound in gate version 2, not an estimated confidence
interval. Four rounds provide a stronger consistency check, not proof of statistical
significance. Review fresh representative questions and semantic disagreements before claiming
general superiority; do not tune the golden set to rescue a candidate.

## The judge

Production's Qwen (qwen3.8:27b at lab/production.env's `BASE`) judges each run's rubric questions
through RiverGuide's `evals.run --judge-ollama`, thinking first, once all of the round's answers
are in, so judging is never timed. It gets the same prompt and verdict schema as the Claude judge
and needs no API key. Every run of an experiment must name the same judge (`meta.judge_model`).

It was checked on 2026-10-27 (RiverGuide's `evals/runs/judge-check-2026-10-27.json`): it failed
16 of 18 answers given to the wrong question, and agreed with Grok, a judge from another family,
on 60 of 90 verdicts. It is more lenient than Grok with its own model's answers (45 against 18 of
72 passed). Both arms are answered by that model, so the leniency falls on both. Its verdicts are
noisy, though: with and without thinking it split on 16 of 90. Since the guardrails allow no
decrease, a candidate that is only as good as production will often be blocked by chance. That
errs toward keeping production.

## Run, inspect, then switch

```sh
# Commit plans/my-experiment.json first. Production's Qwen judges; it must be up.
./gate_retriever.sh --plan plans/my-experiment.json --rerank candidate-name
# For a retriever instead: --plan plans/my-experiment.json retriever-name

# Inspect runs/gate-retriever-rerank-candidate-name.json and its recorded experiment directory.
./gate_retriever.sh --switch --rerank candidate-name
```

Preparation creates a unique `runs/retriever-gates/EXPERIMENT/manifest.json` before model or
service changes. It records the committed plan, baseline and candidate identities, corpus,
RiverGuide revision, engine settings, source hashes and question IDs. Raw answer artifacts stay
in RiverGuide's private repo. The manifest stores paths and hashes, and the verdict stores only
aggregate metrics. A changed plan, model, baseline, corpus, engine, source or artifact requires a
new experiment. A legacy `passed: true` is insufficient.

The existing reranker harness supports adding a reranker over a plain retriever. If production
already has a reranker, it refuses the comparison rather than silently substituting a plain
baseline. Build a fair incumbent-versus-challenger harness for that case first.

Rollback to the simpler path is always available:

```sh
./gate_retriever.sh --disable-rerank
```

It removes only the activation symlink and restarts the retriever. Weights and results remain.
If the service does not become healthy, it restores the previous link. Verify the current
service invocation reports `reranker none` after a rollback. No crawl, index rebuild, answer
model change or public Worker deployment is required.

## Validation

```sh
cargo test --release --manifest-path rust/Cargo.toml
bash -n gate_retriever.sh
```

The tests cover real promotion failure modes: ties, quality regressions, fallbacks, tail-latency
regressions, inconsistent rounds, missing or mismatched measurements, post-hoc plan edits,
changed raw artifacts, missing cost telemetry, a judge that changes between runs, and legacy
tolerance passes. They make no model calls and do not change services. The decisions are
`riverguide-lab gate` (rust/src/gate.rs), ported from retriever_gate.py on 2026-10-27 and
checked byte for byte against it on 813 experiments, 432 real answers among them.
