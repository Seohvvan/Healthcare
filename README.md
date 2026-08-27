# trialmatch — Interactive Clinical Trial Recommendation

Multi-agent system for the Healthcare Agentic AI Challenge 2026 (SKKU AIHC Lab).
Matches free-text patient descriptions against ClinicalTrials.gov eligibility
criteria, detects missing information, asks clarifying questions, and recommends
the most suitable trials with evidence. See `design.md` for the full design.

## Architecture (funnel + interactive loop)

```
patient note
 -> Patient Profiler      (structured profile + unknown fields + search queries)
 -> Retrieval             (BM25 [+ optional dense] hybrid, corpus -> ~50 candidates)
 -> Criteria Parser       (per-trial criteria -> atomic criteria, cached)
 -> Matching              (per-criterion MET / NOT_MET / UNKNOWN + evidence)
 -> Question loop         (UNKNOWN -> clarifying questions -> re-assess)
 -> Ranking               (deterministic aggregation, exclusion = hard filter)
 -> Report                (verdict + evidence + recommendation + disclaimer)
```

## Setup

```bash
uv sync            # install deps (add `--extra dense` for embedding retrieval)
uv run pytest      # run tests (no network, no API key required)
```

## Quickstart — interactive demo, end to end

```bash
# 1. Fetch a trial corpus (once; ~2k trials for the sample conditions)
uv run trialmatch download --queries "acute pancreatitis" "hyperthyroidism" \
  "nephrotic syndrome" "bladder cancer" "migraine with aura" \
  --out data/trials.jsonl --max-per-query 400

# 2. Build the search index (once)
uv run trialmatch index --trials data/trials.jsonl --out data/bm25.pkl

# 3. Run on a patient description; answer its questions in the terminal
uv run trialmatch run --provider gemini --patient-id DEMO \
  --note "A 34-year-old woman presents with recurrent episodes of severe \
unilateral throbbing headache preceded by visual scotomata." \
  --trials data/trials.jsonl --index data/bm25.pkl --question-rounds 2
```

The system prints its Top-5 recommendation with per-criterion evidence, asks
about the information it could not resolve (press Enter to answer "unknown"),
re-assesses only those criteria, and updates the ranking.

## Evaluation results

Measured on 2026-08-18~20 (details, method, and caveats: `docs/results.md`):

- **Interactive loop** (masked-field eval, 10 synthetic patients): the system
  re-asked **45%** of deliberately hidden facts, converted **3.55 trials per
  patient** from undetermined to determined after answers, and preserved the
  top-1 recommendation label in **95%** of runs. Judging is exactly
  reproducible (same input → same labels).
- **Single-shot engine** (TREC CT 2021 judged-subset rerank): nDCG@10 0.29,
  three-class label accuracy 0.605 — limited by candidate supply and by the
  deliberately conservative "never guess" aggregation, which the question loop
  is designed to resolve interactively.

### LLM providers

Two providers are supported behind one client interface, so switching one does
not change any agent or pipeline code. Model tiers per provider live in
`trialmatch/config.py`.

| provider | credentials | models |
|---|---|---|
| `anthropic` (default) | `ANTHROPIC_API_KEY` or an `ant auth login` profile | Claude tiering, `ModelConfig` |
| `gemini` (experiment) | `GOOGLE_API_KEY` (or `GEMINI_API_KEY`) | `GEMINI_MODELS` |

```bash
uv run trialmatch run --provider gemini --patient-id S001 \
  --note-file note.txt --trials data/trials.jsonl --index data/bm25.pkl
uv run python -m trialmatch.eval.bench llm --provider gemini ...   # same flag
```

Credentials are always read from the environment by the provider SDK; no key is
ever stored in this repository.

The CLI (`trialmatch ...` and `python -m trialmatch.eval.bench ...`) also loads
`<project>/.env` at startup (existing environment variables win). Supported keys:

| key | effect |
|---|---|
| `GOOGLE_API_KEY` / `GEMINI_API_KEY` | Gemini credentials (read by the SDK) |
| `ANTHROPIC_API_KEY` | Anthropic credentials (read by the SDK) |
| `GEMINI_MODEL` | overrides all three Gemini tiers at once |
| `GEMINI_EXTRACT_MODEL` / `GEMINI_REASON_MODEL` / `GEMINI_REPORT_MODEL` | per-tier overrides (win over `GEMINI_MODEL`) |
| `GEMINI_RPM` | client-side request pacing, requests/minute (default 10 — free-tier friendly); 429 responses back off 30/60/120s before failing |

## Data sources & licenses

- **ClinicalTrials.gov** (API v2, `https://clinicaltrials.gov/api/v2/studies`) —
  public domain data provided by the U.S. National Library of Medicine; cite the
  source when redistributing.
- **TREC Clinical Trials 2021/2022** (topics + qrels) — free for research use
  after registration at trec-cds.org; used for offline evaluation only.
- **Synthetic patients** (`synthetic-patients.json` + generated variants) —
  fully synthetic; no real patient data or PHI is used anywhere in this project.

## Medical disclaimer

All outputs of this system are for research and reference purposes only and do
**not** constitute medical advice. Trial eligibility must be confirmed by
qualified clinicians and the trial's own study team.
