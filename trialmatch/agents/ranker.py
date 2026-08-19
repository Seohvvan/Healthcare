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
      3. otherwise, with w = cfg.exclusion_cleared_weight,

             score = (met_fraction + w * cleared_fraction) / (1 + w)
                     - cfg.unknown_penalty * unknown_fraction

         `cleared_fraction` is the share of exclusion criteria the matcher
         confirmed the patient does not hit (verdict NOT_MET). A cleared
         exclusion is positive evidence of eligibility, so it lifts the score of
         a trial that is still undetermined; the `(1 + w)` normalisation keeps
         the ceiling at 1.0, so a fully resolved eligible trial with at least one
         inclusion criterion still scores exactly 1.0.
         The trial is labelled ELIGIBLE only when nothing is left undecided; a
         trial with open unknowns stays `None` (undetermined) and becomes a
         question target.

    A trial with no exclusion criteria has `cleared_fraction` 1.0 — vacuously,
    every exclusion it has is cleared — so the formula stays uniform and needs no
    branch on exclusion presence. This matches the ELIGIBLE label, which an empty
    exclusion list has always satisfied vacuously too. The deliberate
    consequence: a trial whose exclusion section the parser missed entirely looks
    like a trial with nothing left to clear, so ranking quality is coupled to
    parser recall — exactly the coupling the ELIGIBLE label already carries.

    Evidence backstop: a verdict counts in the label-improving direction only
    when it quotes evidence. An inclusion MET or an exclusion NOT_MET with empty
    `evidence` is read as UNKNOWN — it lifts neither fraction and blocks the
    ELIGIBLE label, leaving the trial undetermined and still a question target.
    This is the code-side guarantee behind the matcher prompt's safety rules: an
    inference the model never grounded in a quote cannot buy the top grade or a
    higher score. Verdicts in the harmful or neutral directions (an exclusion
    MET, an inclusion NOT_MET, anything UNKNOWN) carry no such requirement.

    A failed hard demographic check (the matcher's synthetic verdict) is treated
    as rule 2: the patient simply does not fit the trial population.
    """
    by_id = {verdict.criterion_id: verdict for verdict in assessment.verdicts}

    inclusion_labels = [
        _label_of(by_id, c.criterion_id, EligibilityLabel.MET) for c in parsed.inclusion
    ]
    exclusion_labels = [
        _label_of(by_id, c.criterion_id, EligibilityLabel.NOT_MET) for c in parsed.exclusion
    ]

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

    # Rule 3: reward exclusions the matcher confirmed the patient does not hit.
    # Without this term every still-open trial is ordered on its inclusion
    # criteria alone, which leaves the large undetermined mass nearly unordered.
    cleared_fraction = (
        sum(1 for label in exclusion_labels if label is EligibilityLabel.NOT_MET)
        / len(exclusion_labels)
        if exclusion_labels
        else 1.0  # vacuously cleared; see the docstring on parser-recall coupling
    )
    weight = cfg.exclusion_cleared_weight
    score = (met_fraction + weight * cleared_fraction) / (
        1 + weight
    ) - cfg.unknown_penalty * unknown_fraction
    fully_resolved = all(
        label is EligibilityLabel.MET for label in inclusion_labels
    ) and all(label is EligibilityLabel.NOT_MET for label in exclusion_labels)
    trial_label = TrialLabel.ELIGIBLE if fully_resolved else None
    # Recompute the open-question list from the *effective* labels: a verdict
    # the backstop downgraded must become a question target again, or the trial
    # would sit undetermined forever with nothing left to ask about.
    unknown_ids = [
        criterion.criterion_id
        for criterion, label in zip(
            (*parsed.inclusion, *parsed.exclusion),
            (*inclusion_labels, *exclusion_labels),
            strict=True,
        )
        if label is EligibilityLabel.UNKNOWN
    ]
    return assessment.model_copy(
        update={
            "trial_label": trial_label,
            "score": score,
            "unknown_criterion_ids": unknown_ids,
        }
    )


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
    by_id: dict[str, CriterionVerdict],
    criterion_id: str,
    grounded: EligibilityLabel,
) -> EligibilityLabel:
    """Effective label of one criterion, with the evidence backstop applied.

    `grounded` is the label-improving direction for this criterion kind (MET for
    an inclusion criterion, NOT_MET for an exclusion one); claiming it without
    quoted evidence reads as UNKNOWN. A missing verdict is UNKNOWN too.
    """
    verdict = by_id.get(criterion_id)
    if verdict is None:
        return EligibilityLabel.UNKNOWN
    if verdict.label is grounded and not verdict.evidence:
        return EligibilityLabel.UNKNOWN
    return verdict.label


def _has_demographic_failure(assessment: TrialAssessment) -> bool:
    return any(
        DEMOGRAPHIC_MARKER in verdict.criterion_id
        and verdict.label is EligibilityLabel.NOT_MET
        for verdict in assessment.verdicts
    )
