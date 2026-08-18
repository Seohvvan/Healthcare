"""Agent 6: deterministic aggregation and ranking (design.md §3).

No LLM. Trial-level verdicts and scores are pure code so they are reproducible
and auditable; the LLM only ever produces criterion-level labels.
"""

from __future__ import annotations

from trialmatch.agents.matcher import DEMOGRAPHIC_MARKER
from trialmatch.config import RankingConfig
from trialmatch.schemas import (
    CriterionVerdict,
    EligibilityLabel,
    ParsedCriteria,
    TrialAssessment,
    TrialLabel,
)

# Final ordering is by label grade first (design.md §3): a trial we would never
# recommend must never outrank one we would, whatever its score. The grades are
# consistent with the TREC gains (eligible 2 > excluded 1 > not relevant 0);
# an undetermined trial sits between eligible and excluded because it is still
# a live candidate pending one more answer.
_LABEL_GRADE: dict[TrialLabel | None, int] = {
    TrialLabel.ELIGIBLE: 3,
    None: 2,  # undetermined: open UNKNOWNs, still a question target
    TrialLabel.EXCLUDED: 1,
    TrialLabel.NOT_RELEVANT: 0,
}


def aggregate(
    assessment: TrialAssessment,
    parsed: ParsedCriteria,
    cfg: RankingConfig,
) -> TrialAssessment:
    """Return a copy of `assessment` with `trial_label` and `score` filled in.

    Rules:
      1. any exclusion criterion MET            -> EXCLUDED, `cfg.excluded_score`
      2. any inclusion criterion NOT_MET        -> NOT_RELEVANT, met_fraction - 1.0
      3. otherwise score = met_fraction - cfg.unknown_penalty * unknown_fraction,
         labelled ELIGIBLE only when nothing is left undecided; a trial with
         open unknowns stays `None` (undetermined) and becomes a question target.

    A failed hard demographic check (the matcher's synthetic verdict) is treated
    as rule 2: the patient simply does not fit the trial population.
    """
    by_id = {verdict.criterion_id: verdict for verdict in assessment.verdicts}

    inclusion_labels = [_label_of(by_id, c.criterion_id) for c in parsed.inclusion]
    exclusion_labels = [_label_of(by_id, c.criterion_id) for c in parsed.exclusion]

    total = len(inclusion_labels)
    met_fraction = (
        sum(1 for label in inclusion_labels if label is EligibilityLabel.MET) / total
        if total
        else 0.0
    )
    unknown_fraction = (
        sum(1 for label in inclusion_labels if label is EligibilityLabel.UNKNOWN) / total
        if total
        else 0.0
    )

    if any(label is EligibilityLabel.MET for label in exclusion_labels):
        return assessment.model_copy(
            update={"trial_label": TrialLabel.EXCLUDED, "score": cfg.excluded_score}
        )

    if _has_demographic_failure(assessment) or any(
        label is EligibilityLabel.NOT_MET for label in inclusion_labels
    ):
        return assessment.model_copy(
            update={"trial_label": TrialLabel.NOT_RELEVANT, "score": met_fraction - 1.0}
        )

    score = met_fraction - cfg.unknown_penalty * unknown_fraction
    fully_resolved = all(
        label is EligibilityLabel.MET for label in inclusion_labels
    ) and all(label is EligibilityLabel.NOT_MET for label in exclusion_labels)
    trial_label = TrialLabel.ELIGIBLE if fully_resolved else None
    return assessment.model_copy(update={"trial_label": trial_label, "score": score})


def label_grade(trial_label: TrialLabel | None) -> int:
    """Ranking grade of a trial label: higher is recommended earlier."""
    return _LABEL_GRADE[trial_label]


def rank(assessments: list[TrialAssessment]) -> list[TrialAssessment]:
    """Sort best-first by (label grade, score); ties break on `nct_id`.

    The label grade dominates the score so an EXCLUDED trial never outranks an
    undetermined or eligible one, however good its criterion arithmetic looks.
    Scores stay meaningful *within* a grade and are still reported as computed.
    """
    return sorted(
        assessments, key=lambda a: (-label_grade(a.trial_label), -a.score, a.nct_id)
    )


def _label_of(
    by_id: dict[str, CriterionVerdict], criterion_id: str
) -> EligibilityLabel:
    verdict = by_id.get(criterion_id)
    return verdict.label if verdict is not None else EligibilityLabel.UNKNOWN


def _has_demographic_failure(assessment: TrialAssessment) -> bool:
    return any(
        DEMOGRAPHIC_MARKER in verdict.criterion_id
        and verdict.label is EligibilityLabel.NOT_MET
        for verdict in assessment.verdicts
    )
