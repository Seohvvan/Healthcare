# Lessons Learned (do-not-repeat)

<!-- Auto-loaded into every session and subagent via the @import in CLAUDE.md. -->
<!-- One generalizable rule per line. Keep it short; prune duplicates/stale items. -->
<!-- Format: - [area] rule (reason) -->

- [data] TREC CT qrels are judged against the 2021-04 corpus snapshot; never score them
  against a fresh ClinicalTrials.gov crawl (coverage gaps + criteria text drift), even
  though NCT IDs themselves are stable.
- [claims] State snapshot dates and evidence scope for any benchmark claim; external
  benchmarks validate the engine, not competition-data performance.
- [ranking] One score cannot serve two orderings — when a benchmark's gain order differs
  from the product's "never recommend" order, rank by an explicit label grade first.
- [seams] A cross-module protocol encoded as a magic string (e.g. ":demographic:")
  belongs in one module with one seam test; never retype it at both ends.
- [eval] Re-exporting a heavy submodule from a package __init__ voids any "stays
  importable without that layer" justification for duplicated code — check the
  __init__ before duplicating.
- [eval] An A/B metric must vary exactly one factor — give the baseline arm the same
  budget (question rounds, tools, retries) as the treatment arm, or the number measures
  the budget, not the change.
- [eval] Never compute agreement/coverage over a top-k-truncated ranking: rank order is
  correlated with the label being compared, so truncation silently selects the agreeing
  subset.
- [eval] Report every rate with its denominator and its dropped/failed unit count; a run
  that assessed almost nothing must not be able to print a perfect score.
