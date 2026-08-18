"""Pipeline orchestration: the funnel of design.md §3 wired end to end.

profile -> retrieve -> parse criteria -> assess -> (question loop) -> rank ->
report. Every stage is an agent function from `trialmatch.agents`; this module
only sequences them, keeps the per-trial criteria cache, and records a run log.

No stage decides anything here: aggregation and ranking stay in the (pure code)
ranker, so a run is reproducible from the log alone.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

from trialmatch.agents import (
    aggregate,
    answer_questions,
    assess_trial,
    build_report,
    generate_questions,
    parse_criteria,
    profile_patient,
    rank,
)
from trialmatch.agents.prompts import is_unknown_answer
from trialmatch.config import Settings
from trialmatch.llm import LLMClient
from trialmatch.retrieval import BM25Index, rrf
from trialmatch.schemas import (
    ClarifyingQuestion,
    EligibilityLabel,
    ParsedCriteria,
    PatientProfile,
    QuestionAnswer,
    Recommendation,
    TrialAssessment,
    TrialRecord,
)

__all__ = ["AnswerProvider", "run_pipeline", "save_run_log", "simulator_answer_provider"]

AnswerProvider = Callable[[list[ClarifyingQuestion]], list[QuestionAnswer]]


def run_pipeline(
    *,
    patient_id: str,
    note: str,
    trials: dict[str, TrialRecord],
    index: BM25Index,
    llm: LLMClient,
    settings: Settings,
    top_k: int = 5,
    question_rounds: int = 1,
    answer_provider: AnswerProvider | None = None,
    use_llm_criteria: bool = True,
    parsed_cache: dict[str, ParsedCriteria] | None = None,
) -> tuple[Recommendation, dict]:
    """Run the full funnel for one patient and return `(recommendation, run_log)`.

    `parsed_cache` is mutated in place so the caller can persist the per-trial
    criteria parse across patients (design.md §2 — the main cost lever).
    The run log is JSON-serializable; writing it is the caller's choice
    (`save_run_log`).

    A per-candidate LLM failure is contained: the trial is dropped from the
    ranking and recorded under `run_log["errors"]`, so one bad trial never sinks
    a whole patient run. Whole-run calls (profiling, question generation) still
    propagate.
    """
    profile = profile_patient(llm, patient_id, note, settings.models)

    candidate_ids = _retrieve(profile, index, trials, settings)

    parsed_by_nct, skipped_no_criteria = _parse_candidates(
        llm, candidate_ids, trials, settings, use_llm_criteria, parsed_cache
    )

    errors: list[dict] = []
    assessments: dict[str, TrialAssessment] = {}
    for nct_id, parsed in parsed_by_nct.items():
        try:
            assessment = assess_trial(llm, profile, parsed, trials[nct_id], settings.models)
        except Exception as exc:  # noqa: BLE001 — one trial must not sink the run
            errors.append({"nct_id": nct_id, "error": str(exc)})
            continue
        assessments[nct_id] = aggregate(assessment, parsed, settings.ranking)

    rounds = _run_question_loop(
        llm=llm,
        profile=profile,
        trials=trials,
        parsed_by_nct=parsed_by_nct,
        assessments=assessments,
        settings=settings,
        question_rounds=question_rounds,
        answer_provider=answer_provider,
        errors=errors,
    )

    ranked = rank(list(assessments.values()))[: max(top_k, 0)]
    report_markdown = build_report(profile, ranked, trials, top_k, parsed_by_nct)

    run_log = {
        "patient_id": patient_id,
        "profile": profile.model_dump(mode="json"),
        "candidate_ids": candidate_ids,
        "skipped_no_criteria": skipped_no_criteria,
        "errors": errors,
        "rounds": rounds,
        "ranking": [
            {
                "nct_id": assessment.nct_id,
                "label": assessment.trial_label.value
                if assessment.trial_label is not None
                else None,
                "score": assessment.score,
            }
            for assessment in ranked
        ],
    }

    recommendation = Recommendation(
        patient_id=patient_id, ranked=ranked, report_markdown=report_markdown
    )
    return recommendation, run_log


def simulator_answer_provider(
    llm: LLMClient, ground_truth_text: str, settings: Settings
) -> AnswerProvider:
    """Answer clarifying questions with the Patient Simulator agent.

    Used by the masked-field evaluation flow (design.md §5.3): the pipeline runs
    on the masked note while the simulator answers from the full text only.
    """

    def provide(questions: list[ClarifyingQuestion]) -> list[QuestionAnswer]:
        return answer_questions(llm, ground_truth_text, questions, settings.models)

    return provide


def save_run_log(run_log: dict, settings: Settings) -> Path:
    """Write `run_log` to `settings.log_dir` and return the file path.

    The filename timestamp is UTC so logs from different machines sort together.
    """
    log_dir = Path(settings.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    patient_id = str(run_log.get("patient_id") or "unknown")
    path = log_dir / f"{patient_id}-{datetime.now(UTC).strftime('%Y%m%d-%H%M%S')}.json"
    with path.open("w", encoding="utf-8") as fh:
        json.dump(run_log, fh, ensure_ascii=False, indent=2)
    return path


# --------------------------------------------------------------------------- #
# stages
# --------------------------------------------------------------------------- #


def _retrieve(
    profile: PatientProfile,
    index: BM25Index,
    trials: dict[str, TrialRecord],
    settings: Settings,
) -> list[str]:
    """BM25 over the note plus every generated query, fused with RRF.

    Zero-score hits are dropped before the fusion: on a small corpus BM25 still
    returns every document, and a trial that shares no term with any query would
    otherwise reach the matcher and spend an LLM call on an unrelated disease.
    """
    cfg = settings.retrieval
    queries: list[str] = []
    for query in [profile.raw_text, *profile.search_queries]:
        text = query.strip()
        # Duplicate queries would count twice in the fusion; keep the first.
        if text and text not in queries:
            queries.append(text)

    rankings = [
        [nct_id for nct_id, score in index.search(query, k=cfg.bm25_k) if score > 0]
        for query in queries
    ]
    fused = rrf(rankings, k=cfg.rrf_k, top_n=cfg.candidates_k)
    # An index may outlive the snapshot it was built from: keep only known ids.
    return [nct_id for nct_id, _score in fused if nct_id in trials]


def _parse_candidates(
    llm: LLMClient,
    candidate_ids: list[str],
    trials: dict[str, TrialRecord],
    settings: Settings,
    use_llm_criteria: bool,
    parsed_cache: dict[str, ParsedCriteria] | None,
) -> tuple[dict[str, ParsedCriteria], list[str]]:
    """Parse (or reuse) criteria per candidate; drop trials without any criterion.

    A trial with no parsed criterion cannot be judged: assessing it would return
    an empty verdict list, which the ranker would score as a vacuous ELIGIBLE.
    Such trials are reported instead of recommended.
    """
    parsed_by_nct: dict[str, ParsedCriteria] = {}
    skipped_no_criteria: list[str] = []

    for nct_id in candidate_ids:
        parsed = parsed_cache.get(nct_id) if parsed_cache is not None else None
        if parsed is None:
            parsed = parse_criteria(llm, trials[nct_id], settings.models, use_llm=use_llm_criteria)
            if parsed_cache is not None:
                parsed_cache[nct_id] = parsed
        if not parsed.all_criteria:
            skipped_no_criteria.append(nct_id)
            continue
        parsed_by_nct[nct_id] = parsed

    return parsed_by_nct, skipped_no_criteria


def _run_question_loop(
    *,
    llm: LLMClient,
    profile: PatientProfile,
    trials: dict[str, TrialRecord],
    parsed_by_nct: dict[str, ParsedCriteria],
    assessments: dict[str, TrialAssessment],
    settings: Settings,
    question_rounds: int,
    answer_provider: AnswerProvider | None,
    errors: list[dict],
) -> list[dict]:
    """Ask about undecided criteria and re-assess only the trials they belong to.

    Question targets are trials whose verdict is still undetermined *and* that
    have open unknowns (design.md §3): a trial already settled as excluded or
    not relevant never earns a question, however many unknowns it carries.

    The re-assessment sends only the unknown criteria to the matcher and merges
    the new verdicts into the previous ones, so a criterion already decided in
    round 1 cannot flip in round 2.

    `profile`, `assessments` and `errors` are updated in place. Returns one
    JSON-ready record per round.
    """
    rounds: list[dict] = []
    if question_rounds <= 0 or answer_provider is None:
        return rounds

    for round_index in range(1, question_rounds + 1):
        targets = [
            a
            for a in assessments.values()
            if a.trial_label is None and a.unknown_criterion_ids
        ]
        if not targets:
            break

        questions = generate_questions(
            llm, profile, targets, parsed_by_nct, settings.models
        )
        if not questions:
            break

        answers = answer_provider(questions)
        clarifications = _clarifications(questions, answers)
        profile.findings.extend(clarifications)

        # Nothing new was learned: re-assessing would repeat the same verdicts.
        reassessed: list[str] = []
        for target in targets if clarifications else []:
            parsed = parsed_by_nct[target.nct_id]
            try:
                update = assess_trial(
                    llm,
                    profile,
                    parsed,
                    trials[target.nct_id],
                    settings.models,
                    criterion_ids=target.unknown_criterion_ids,
                )
            except Exception as exc:  # noqa: BLE001 — one trial must not sink the run
                errors.append({"nct_id": target.nct_id, "error": str(exc)})
                continue
            assessments[target.nct_id] = aggregate(
                _merge_verdicts(target, update), parsed, settings.ranking
            )
            reassessed.append(target.nct_id)

        rounds.append(
            {
                "round": round_index,
                "questions": [q.model_dump(mode="json") for q in questions],
                "answers": [a.model_dump(mode="json") for a in answers],
                "reassessed": reassessed,
            }
        )

        if not clarifications:
            break

    return rounds


def _merge_verdicts(
    previous: TrialAssessment, update: TrialAssessment
) -> TrialAssessment:
    """Overwrite only the re-assessed criteria, keeping every earlier verdict.

    Verdicts the update does not mention (a MET decided in round 1, say) are
    carried over untouched; ids only the update knows about — the matcher's
    synthetic demographic verdict — are appended.
    """
    fresh = {verdict.criterion_id: verdict for verdict in update.verdicts}
    verdicts = [fresh.pop(v.criterion_id, v) for v in previous.verdicts]
    verdicts.extend(fresh.values())
    return previous.model_copy(
        update={
            "verdicts": verdicts,
            "unknown_criterion_ids": [
                v.criterion_id for v in verdicts if v.label is EligibilityLabel.UNKNOWN
            ],
        }
    )


def _clarifications(
    questions: list[ClarifyingQuestion], answers: list[QuestionAnswer]
) -> list[str]:
    """Render informative answers as profile findings, in the order asked."""
    by_id = {answer.question_id: answer.answer_text for answer in answers}
    lines: list[str] = []
    for question in questions:
        answer_text = (by_id.get(question.question_id) or "").strip()
        if not answer_text or is_unknown_answer(answer_text):
            continue
        lines.append(f"Clarification — Q: {question.text} A: {answer_text}")
    return lines
