"""Agent 7: render the ranked assessments as a human-readable report.

Pure code, no LLM: the report is a deterministic view of the ranker's output, so
identical runs produce identical text. The output language is Korean because the
report is a team deliverable (design.md); all code and comments stay English.
"""

from __future__ import annotations

from trialmatch.schemas import (
    Criterion,
    EligibilityLabel,
    ParsedCriteria,
    PatientProfile,
    TrialAssessment,
    TrialLabel,
    TrialRecord,
)

DISCLAIMER = (
    "본 결과는 의학적 자문이 아닌 연구·참고용입니다. "
    "실제 임상시험 참여 여부는 반드시 담당 의료진과 상의하여 결정하시고, "
    "최종 참여 자격은 시험 실시기관의 공식 심사를 통해 확인해야 합니다."
)

_TRIAL_LABEL_KO = {
    TrialLabel.ELIGIBLE: "적합 (eligible)",
    TrialLabel.EXCLUDED: "제외 (excluded)",
    TrialLabel.NOT_RELEVANT: "부적합 (not relevant)",
}
_UNDETERMINED_KO = "미정 (추가 확인 필요)"

_CRITERION_LABEL_KO = {
    EligibilityLabel.MET: "충족",
    EligibilityLabel.NOT_MET: "미충족",
    EligibilityLabel.UNKNOWN: "정보 없음",
}

_MAX_EVIDENCE_PER_TRIAL = 4
_MAX_UNKNOWNS_PER_TRIAL = 5


def build_report(
    profile: PatientProfile,
    ranked: list[TrialAssessment],
    trials_by_nct: dict[str, TrialRecord],
    top_k: int = 5,
    parsed_by_nct: dict[str, ParsedCriteria] | None = None,
) -> str:
    """Render a Korean markdown report for the top `top_k` ranked trials.

    `parsed_by_nct` is optional and, when supplied, lets unresolved unknowns be
    printed with their criterion text instead of a bare identifier.
    """
    selected = ranked[: max(top_k, 0)]
    lines: list[str] = [
        "# 임상시험 매칭 리포트",
        "",
        f"- 환자 ID: `{profile.patient_id}`",
        f"- 평가한 시험 수: {len(ranked)}건 (상위 {len(selected)}건 표시)",
    ]
    if profile.unknown_fields:
        lines.append(f"- 환자 기록에서 확인되지 않은 항목: {', '.join(profile.unknown_fields)}")
    lines.append("")

    if not selected:
        lines.extend(["## 추천 결과", "", "조건을 만족하는 후보 임상시험이 없습니다.", ""])
    else:
        lines.append("## 추천 결과")
        lines.append("")
        for rank_index, assessment in enumerate(selected, start=1):
            lines.extend(
                _render_trial(
                    rank_index,
                    assessment,
                    trials_by_nct.get(assessment.nct_id),
                    (parsed_by_nct or {}).get(assessment.nct_id),
                )
            )

    lines.extend(["---", "", f"> **의료 면책 고지**: {DISCLAIMER}", ""])
    return "\n".join(lines)


def _render_trial(
    rank_index: int,
    assessment: TrialAssessment,
    trial: TrialRecord | None,
    parsed: ParsedCriteria | None,
) -> list[str]:
    title = (trial.title if trial is not None else "") or "(제목 정보 없음)"
    label = (
        _TRIAL_LABEL_KO[assessment.trial_label]
        if assessment.trial_label is not None
        else _UNDETERMINED_KO
    )
    lines = [
        f"### {rank_index}. {assessment.nct_id} — {title}",
        "",
        f"- 판정: **{label}**",
        f"- 점수: {assessment.score:.3f}",
    ]

    decided = [v for v in assessment.verdicts if v.label is not EligibilityLabel.UNKNOWN]
    if decided:
        lines.append("- 주요 근거:")
        for verdict in decided[:_MAX_EVIDENCE_PER_TRIAL]:
            marker = _CRITERION_LABEL_KO[verdict.label]
            explanation = verdict.explanation or "(설명 없음)"
            lines.append(f"  - [{marker}] `{verdict.criterion_id}` — {explanation}")
            for quote in verdict.evidence:
                lines.append(f'    - 인용: "{quote}"')
    else:
        lines.append("- 주요 근거: 판정된 기준이 없습니다.")

    unknown_ids = assessment.unknown_criterion_ids
    if unknown_ids:
        lines.append("- 미해결 확인 필요 항목:")
        by_id = _criteria_by_id(parsed)
        for criterion_id in unknown_ids[:_MAX_UNKNOWNS_PER_TRIAL]:
            criterion = by_id.get(criterion_id)
            detail = f" — {criterion.text}" if criterion is not None else ""
            lines.append(f"  - `{criterion_id}`{detail}")
        remaining = len(unknown_ids) - _MAX_UNKNOWNS_PER_TRIAL
        if remaining > 0:
            lines.append(f"  - 그 외 {remaining}건")
    else:
        lines.append("- 미해결 확인 필요 항목: 없음")

    lines.append("")
    return lines


def _criteria_by_id(parsed: ParsedCriteria | None) -> dict[str, Criterion]:
    if parsed is None:
        return {}
    return {c.criterion_id: c for c in parsed.all_criteria}
