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

LLM calls use the Anthropic API; credentials resolve from `ANTHROPIC_API_KEY`
or an `ant auth login` profile. Model tiers are set in `trialmatch/config.py`.

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
