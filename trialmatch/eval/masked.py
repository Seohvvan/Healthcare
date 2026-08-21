"""Masked-field synthetic-patient evaluation of the question loop (design.md §5.3).

The interactive loop is only worth its cost if a question actually recovers a
fact the note is missing. This module measures exactly that:

  1. `generate` expands each one-sentence competition topic into a complete
     synthetic record plus the facts that may be masked, and stores the pair as
     JSON. Masking is deterministic — a fact names the sentence indices that
     carry it, so no LLM is involved at evaluation time.
  2. `run` runs the pipeline twice per fact: a *baseline* over the full note and
     a *masked* run over the note with that fact's sentences removed. Both arms
     get the **same** question budget, the same Patient Simulator answering
     from the full note, and the **same candidate trials** (the masked runs pin
     the baseline's retrieval result), so the only thing that differs between
     them is the removed sentences — the number measures the mask, not the
     budget or retrieval variance.

Every metric is computed in pure code from the two run logs, so a result is
reproducible from the saved logs alone:

  * `recovered`            — did an informative answer contain the masked fact?
  * `flipped_count`        — how many re-assessed trials ended determined?
  * `label_agreement`      — do the masked labels match the full-note baseline?
  * `top1_label_preserved` — does the baseline's best trial still carry the same
    label under masking? (It checks the label, not the rank.)

Each row also carries its denominators (`n_baseline`, `n_masked`, `n_shared`,
`masked_errors`), because a run that assessed almost nothing must not be able to
print a perfect score.

`trialmatch.eval.__init__` does not re-export this module (it pulls in the data,
retrieval and agent layers); use `python -m trialmatch.eval.masked`.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

from pydantic import BaseModel, Field

from trialmatch.agents.prompts import is_unknown_answer
from trialmatch.config import Settings, load_dotenv
from trialmatch.data.store import load_trials
from trialmatch.data.trec import load_topics_json
from trialmatch.llm import SupportsLLM, create_llm
from trialmatch.pipeline import run_pipeline, simulator_answer_provider
from trialmatch.retrieval.bm25 import BM25Index
from trialmatch.schemas import ParsedCriteria, TrialRecord

__all__ = [
    "DEFAULT_CANDIDATES_K",
    "DEFAULT_QUESTION_ROUNDS",
    "MIN_SHARED_TRIALS",
    "GeneratedPatient",
    "MaskedFact",
    "MaskedPatient",
    "fact_metrics",
    "fact_recovered",
    "flipped_count",
    "format_table",
    "full_note",
    "generate_patients",
    "label_agreement",
    "load_patients",
    "main",
    "masked_note",
    "run_masked_eval",
    "save_masked_result",
    "summarize",
    "top1_label_preserved",
    "write_patients",
]

logger = logging.getLogger(__name__)

# Budget defaults. Per patient the run count is 1 baseline (shared by all of the
# patient's facts) + 1 masked run per fact; each of those runs costs 1 profiler
# call + up to `candidates` matcher calls, plus one question-generation and one
# simulator call per question round. The baseline pays the same question budget
# as the masked arms on purpose (that is what makes the comparison controlled),
# so both caps matter.
DEFAULT_CANDIDATES_K = 20
DEFAULT_QUESTION_ROUNDS = 2

# A label agreement computed over a handful of shared trials is noise; rows
# below this are flagged in the table rather than silently averaged in.
MIN_SHARED_TRIALS = 5

# Generation-prompt targets ONLY — they shape `GENERATOR_SYSTEM` and nothing
# else. `_validate_patient` deliberately does not enforce them: a record with 7
# or 15 sentences is still perfectly usable, so rejecting it at load time would
# throw away paid-for generations. Loading enforces the structural minimums
# (>= 2 sentences, >= 1 fact) instead.
MIN_SENTENCES = 8
MAX_SENTENCES = 14
MIN_FACTS = 3
MAX_FACTS = 5

# Structural minimums enforced by `_validate_patient`.
_MIN_STRUCTURAL_SENTENCES = 2


# --------------------------------------------------------------------------- #
# 1. Patient dataset
# --------------------------------------------------------------------------- #


class MaskedFact(BaseModel):
    """One maskable clinical fact and where it lives in the note.

    `sentence_indices` are 0-based positions into the patient's `sentences`;
    removing exactly those sentences produces the masked note. `keywords` are
    lowercase fragments that an informative answer about this fact would
    contain, which is what makes recovery measurable without an LLM judge.
    """

    field: str
    sentence_indices: list[int] = Field(default_factory=list)
    keywords: list[str] = Field(default_factory=list)


class MaskedPatient(BaseModel):
    """A synthetic patient stored as sentences plus its maskable facts."""

    patient_id: str
    condition: str = ""
    sentences: list[str] = Field(default_factory=list)
    facts: list[MaskedFact] = Field(default_factory=list)


class GeneratedPatient(BaseModel):
    """LLM output schema for `generate`.

    `patient_id` is owned by the caller (derived from the topic id), so the
    model never sees it and can never corrupt it.
    """

    condition: str = ""
    sentences: list[str] = Field(default_factory=list)
    facts: list[MaskedFact] = Field(default_factory=list)


def full_note(patient: MaskedPatient) -> str:
    """The complete note: every sentence, in order."""
    return " ".join(sentence.strip() for sentence in patient.sentences if sentence.strip())


def masked_note(patient: MaskedPatient, fact: MaskedFact) -> str:
    """The note with `fact`'s sentences removed; the rest keeps its order."""
    hidden = set(fact.sentence_indices)
    return " ".join(
        sentence.strip()
        for index, sentence in enumerate(patient.sentences)
        if index not in hidden and sentence.strip()
    )


def load_patients(path: Path) -> list[MaskedPatient]:
    """Load `{"patients": [...]}` and validate that every mask is applicable.

    Everything that would make a downstream metric meaningless is rejected here
    with a `ValueError` naming the patient and the field: an inapplicable mask
    (empty or out-of-range `sentence_indices`), a fact whose keywords cannot
    measure recovery (empty, or absent from the sentences the mask removes), a
    leaking mask (a keyword still present in the masked note), and duplicate
    `patient_id` / duplicate `fact.field` — a duplicate would silently overwrite
    the other's run log. A payload that is not a `{"patients": [...]}` object
    raises `TypeError`, as in `bench.py`.
    """
    with Path(path).open(encoding="utf-8") as fh:
        payload = json.load(fh)
    raw_patients = payload.get("patients") if isinstance(payload, Mapping) else None
    if not isinstance(raw_patients, list):
        raise TypeError("patients file must hold a JSON object with a 'patients' list")

    patients = [MaskedPatient.model_validate(raw) for raw in raw_patients]
    seen: set[str] = set()
    for patient in patients:
        if patient.patient_id in seen:
            raise ValueError(f"duplicate patient_id {patient.patient_id!r}")
        seen.add(patient.patient_id)
        _validate_patient(patient)
    return patients


def _fact_problem(patient: MaskedPatient, fact: MaskedFact) -> str | None:
    """Why `fact` cannot be used as a mask on `patient`, or `None` when it can.

    Beyond the index checks this is a deterministic *leakage* check: every
    keyword must appear in the sentences the mask removes and must NOT appear
    anywhere in the resulting masked note. Otherwise `fact_recovered` would
    score a fact the masked note still states — the pipeline would not have had
    to ask anything to "recover" it.
    """
    count = len(patient.sentences)
    if not fact.sentence_indices:
        return "sentence_indices must not be empty"
    for index in fact.sentence_indices:
        if index < 0 or index >= count:
            return f"sentence index {index} is out of range (0..{count - 1})"

    keywords = [keyword.strip() for keyword in fact.keywords if keyword.strip()]
    if not keywords:
        return "keywords must not be empty"

    hidden_text = " ".join(patient.sentences[index] for index in fact.sentence_indices).casefold()
    remaining_text = masked_note(patient, fact).casefold()
    for keyword in keywords:
        lowered = keyword.casefold()
        if lowered not in hidden_text:
            return f"keyword {keyword!r} does not appear in the sentences the mask removes"
        if lowered in remaining_text:
            return f"keyword {keyword!r} still appears in the masked note (the mask leaks)"
    return None


def _validate_patient(patient: MaskedPatient) -> None:
    """Structural minimums plus a per-fact check; raises `ValueError`."""
    if len(patient.sentences) < _MIN_STRUCTURAL_SENTENCES:
        raise ValueError(
            f"patient {patient.patient_id!r} has fewer than "
            f"{_MIN_STRUCTURAL_SENTENCES} sentences"
        )
    if not patient.facts:
        raise ValueError(f"patient {patient.patient_id!r} has no facts")

    seen_fields: set[str] = set()
    for fact in patient.facts:
        if fact.field in seen_fields:
            raise ValueError(
                f"patient {patient.patient_id!r}: duplicate fact field {fact.field!r}"
            )
        seen_fields.add(fact.field)
        problem = _fact_problem(patient, fact)
        if problem is not None:
            raise ValueError(
                f"patient {patient.patient_id!r}, fact {fact.field!r}: {problem}"
            )


def write_patients(patients: Sequence[MaskedPatient], out_path: Path) -> Path:
    """Write the dataset as `{"patients": [...]}` and return the path."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"patients": [patient.model_dump(mode="json") for patient in patients]}
    with out_path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)
    return out_path


# --------------------------------------------------------------------------- #
# 2. Generation
# --------------------------------------------------------------------------- #


# Korean translation of GENERATOR_SYSTEM below, kept for team readability
# (owner-requested exception to the English-comments rule in AGENTS.md):
#
# [역할] 너는 임상시험 매칭 시스템을 평가하기 위한 가상 환자 기록을 작성한다.
# 한 문장짜리 환자 개요(vignette)가 주어진다.
#
# [1. 기록 확장] 그 한 문장을 8~14개의 짧은 문장으로 이루어진, 임상적으로
# 앞뒤가 맞는 완전한 기록으로 확장하라. 원문의 사실은 전부 유지하고, 시험
# 심사자가 필요로 할 구체적 정보를 추가하라: 나이, 성별, 진단명과 확진 경위,
# 증상 빈도·기간, 관련 음성 소견, 동반 질환, 복용 약, 과거 치료, (해당 시)
# 임신·피임 상태. 그럴듯하고 구체적인 값(숫자·날짜·약 이름)을 쓰고 얼버무림은
# 금지. 문장 하나에 사실 하나만 — 하나의 값을 두 문장에 쪼개지 마라.
# (문장 단위로 지워서 실험하므로, 사실이 두 문장에 걸치면 마스킹이 샌다.)
#
# [2. 환자 시점] 이 기록은 환자 역할 AI가 질문에 답할 때 참조하는 문서이므로,
# 환자 본인이 자기 진료를 이야기하듯 써라. 검사수치·영상/병리 소견·점수·병기가
# 필요하면 맨 수치("24시간 소변 단백 5.2g")가 아니라 환자가 들은 말("의사가
# 소변 단백이 매우 높다고 말해줬다")로 적어라.
#
# [3. 가릴 사실 목록] 가릴 수 있는 사실 3~5개를 함께 반환하라. 반드시 환자가
# 경험으로 답할 수 있는 것만(증상 빈도·시점·기간, 복약, 들어서 아는 진단·치료·
# 입원, 임신·피임, 흡연·음주, 알레르기, 일상 기능). 날것의 임상 측정값은 절대
# 가리지 마라 — 환자가 답할 수 없어 아무것도 측정하지 못한다. 사실마다:
#   * field: 스네이크케이스 이름(기록 안 중복 금지)
#   * sentence_indices: 그 사실이 적힌 문장 번호(0부터). 지목 문장을 지우면
#     기록 어디에서도 그 사실을 알 수 없어야 한다.
#   * keywords: 환자 답변에 등장할 소문자 조각 2~5개. 지목한 문장 안에 글자
#     그대로 있어야 하고 다른 문장엔 없어야 한다(어기면 그 사실은 폐기).
#
# 문장들은 번호 없이 순수 문자열로 반환하라.
GENERATOR_SYSTEM = f"""\
You write synthetic patient records for evaluating a clinical trial matching
system. You are given a one-sentence patient vignette.

Expand it into a complete, clinically coherent record of {MIN_SENTENCES}-{MAX_SENTENCES}
short sentences. Keep every fact of the vignette, and add the concrete details a
trial screener would need: age, sex, diagnosis and how it was confirmed, symptom
frequency and duration, relevant negatives, comorbidities, current medications,
prior treatments, and pregnancy or contraception status when applicable. Use
plausible, specific values (numbers, dates, drug names) — never placeholders.
One fact per sentence; do not spread a value across two sentences.

The record is what a simulated patient answers questions from, so write it as
this patient's own account of their care. When a laboratory value, an imaging or
pathology finding, a test score or a stage matters to the story, state it as
something the patient was told, never as a bare measurement: write "Her doctor
told her that her urine protein was very high", not "24-hour urine protein was
5.2 g".

Then list {MIN_FACTS}-{MAX_FACTS} maskable facts. A maskable fact must be
patient-knowable: something the patient can report from lived experience when a
screener asks — symptom frequency, timing or duration, current and past
medications, diagnoses, treatments and hospitalizations they were told about,
pregnancy or contraception status, smoking and alcohol use, allergies, daily
functioning. Never mask a raw clinical measurement (a laboratory value, an
imaging or pathology finding, a test score, a stage): the patient could not
answer a question about it, so masking it measures nothing. Such information may
only be masked through a sentence that states what the patient was told about
it. For each fact return:
  * `field`: a short snake_case name, e.g. attack_frequency; every fact of a
    record must use a different name;
  * `sentence_indices`: the 0-based indices of the sentences that state it, and
    nothing else — after removing them the record must no longer reveal the fact;
  * `keywords`: 2-5 short lowercase fragments that a patient's answer about this
    fact would contain, e.g. "attacks per month", "4 attacks". Every keyword
    must appear verbatim in the sentences you listed and nowhere else in the
    record, otherwise the fact is discarded.

Return sentences as plain strings without numbering."""


def _generator_user(topic_text: str) -> str:
    return f"Patient vignette:\n{topic_text}"


def generate_patients(
    llm: SupportsLLM,
    topics: Sequence[tuple[str, str]],
    settings: Settings,
    *,
    max_topics: int | None = None,
) -> tuple[list[MaskedPatient], list[dict]]:
    """Expand each topic into a `MaskedPatient` — one LLM call per topic.

    Facts the model got wrong (bad indices, unmeasurable or leaking keywords)
    are dropped one by one and reported as warnings, so one sloppy fact does not
    waste the whole generation; a topic that keeps no valid fact — or whose call
    fails outright — is recorded under the returned errors and skipped. Patient
    ids mirror the topic ids with a leading `S` replaced by `M`.

    The returned list holds `{"topic_id", "error"}` entries; a dropped fact adds
    a `"field"` key, which is how the CLI tells a warning from a hard error.
    """
    selected = list(topics) if max_topics is None else list(topics)[:max_topics]
    if max_topics is not None and len(topics) > len(selected):
        dropped = [topic_id for topic_id, _text in list(topics)[max_topics:]]
        logger.warning("--max-topics dropped %d topic(s): %s", len(dropped), ", ".join(dropped))

    patients: list[MaskedPatient] = []
    errors: list[dict] = []
    for topic_id, text in selected:
        patient_id = re.sub(r"^S", "M", topic_id)
        try:
            generated = cast(
                GeneratedPatient,
                llm.structured(
                    model=settings.models.reason_model,
                    system=GENERATOR_SYSTEM,
                    user=_generator_user(text),
                    schema=GeneratedPatient,
                    max_tokens=settings.models.max_tokens_reason,
                ),
            )
            patient = MaskedPatient(
                patient_id=patient_id,
                condition=generated.condition,
                sentences=[s.strip() for s in generated.sentences if s.strip()],
                facts=[],
            )
            patient.facts = _keep_valid_facts(patient, generated.facts, topic_id, errors)
            _validate_patient(patient)
        except Exception as exc:  # noqa: BLE001 — one topic must not sink the batch
            errors.append({"topic_id": topic_id, "error": str(exc)})
            logger.warning("generation failed for topic %s: %s", topic_id, exc)
            continue
        patients.append(patient)

    return patients, errors


def _keep_valid_facts(
    patient: MaskedPatient,
    facts: Sequence[MaskedFact],
    topic_id: str,
    errors: list[dict],
) -> list[MaskedFact]:
    """Facts that pass `_fact_problem`; each drop is reported, none raises."""
    kept: list[MaskedFact] = []
    seen_fields: set[str] = set()
    for fact in facts:
        problem = (
            f"duplicate fact field {fact.field!r}"
            if fact.field in seen_fields
            else _fact_problem(patient, fact)
        )
        if problem is not None:
            errors.append({"topic_id": topic_id, "field": fact.field, "error": problem})
            logger.warning("dropped fact %s of topic %s: %s", fact.field, topic_id, problem)
            continue
        seen_fields.add(fact.field)
        kept.append(fact)
    return kept


# --------------------------------------------------------------------------- #
# 3. Metrics (pure functions over run logs)
# --------------------------------------------------------------------------- #


def _answer_texts(run_log: Mapping) -> list[str]:
    """Every answer text of every round, in the order they were given."""
    texts: list[str] = []
    for round_log in run_log.get("rounds") or []:
        for answer in round_log.get("answers") or []:
            text = str(answer.get("answer_text") or "")
            if text.strip():
                texts.append(text)
    return texts


def fact_recovered(run_log: Mapping, keywords: Sequence[str]) -> bool:
    """True when an informative answer mentioned the masked fact.

    An answer that is the simulator's unknown sentinel carries no information
    and never counts, however the keywords match.
    """
    lowered = [keyword.strip().casefold() for keyword in keywords if keyword.strip()]
    if not lowered:
        return False
    for text in _answer_texts(run_log):
        if is_unknown_answer(text):
            continue
        haystack = text.casefold()
        if any(keyword in haystack for keyword in lowered):
            return True
    return False


def _labels_by_nct(run_log: Mapping) -> dict[str, str | None]:
    """Final label per ranked trial; `None` means the run left it undetermined."""
    labels: dict[str, str | None] = {}
    for row in run_log.get("ranking") or []:
        nct_id = row.get("nct_id")
        if nct_id:
            labels[str(nct_id)] = row.get("label")
    return labels


def flipped_count(run_log: Mapping) -> int:
    """Trials that were re-assessed after an answer and ended up determined.

    A trial only enters a round's `reassessed` list while it is undetermined,
    so a determined final label means the question loop settled it. Trials that
    fell outside the reported ranking cannot be checked and are not counted —
    which is why the runner logs an untruncated ranking.
    """
    labels = _labels_by_nct(run_log)
    reassessed = {
        str(nct_id)
        for round_log in run_log.get("rounds") or []
        for nct_id in round_log.get("reassessed") or []
    }
    return sum(1 for nct_id in reassessed if labels.get(nct_id) is not None)


def label_agreement(baseline_log: Mapping, masked_log: Mapping) -> float | None:
    """Share of shared trials whose masked-run label matches the baseline.

    Only trials ranked by *both* runs can be compared; `None` (undetermined) is
    a label value like any other, so a baseline eligible that stays undetermined
    under masking counts as a disagreement. Returns `None` when the two rankings
    share no trial at all. Read it next to the row's `n_shared`.
    """
    baseline = _labels_by_nct(baseline_log)
    masked = _labels_by_nct(masked_log)
    shared = sorted(set(baseline) & set(masked))
    if not shared:
        return None
    agreed = sum(1 for nct_id in shared if baseline[nct_id] == masked[nct_id])
    return agreed / len(shared)


def top1_label_preserved(baseline_log: Mapping, masked_log: Mapping) -> bool:
    """True when the baseline's best trial still carries the same label.

    This says nothing about *rank*: the trial may have moved anywhere in the
    masked ranking (or nowhere), only its label has to match.
    """
    ranking = baseline_log.get("ranking") or []
    if not ranking:
        return False
    top = ranking[0]
    nct_id = str(top.get("nct_id") or "")
    masked = _labels_by_nct(masked_log)
    return nct_id in masked and masked[nct_id] == top.get("label")


def fact_metrics(
    baseline_log: Mapping, masked_log: Mapping, keywords: Sequence[str]
) -> dict:
    """The per-fact metrics plus the denominators they were computed over."""
    baseline_ids = set(_labels_by_nct(baseline_log))
    masked_ids = set(_labels_by_nct(masked_log))
    return {
        "recovered": fact_recovered(masked_log, keywords),
        "flipped_count": flipped_count(masked_log),
        "label_agreement": label_agreement(baseline_log, masked_log),
        "top1_label_preserved": top1_label_preserved(baseline_log, masked_log),
        "n_baseline": len(baseline_ids),
        "n_masked": len(masked_ids),
        "n_shared": len(baseline_ids & masked_ids),
        "masked_errors": len(masked_log.get("errors") or []),
    }


def summarize(rows: Sequence[Mapping]) -> dict:
    """Aggregate per-fact rows; every rate is reported with its denominator."""
    total = len(rows)
    agreements = [
        float(row["label_agreement"])
        for row in rows
        if row.get("label_agreement") is not None
    ]
    num_recovered = sum(1 for row in rows if row["recovered"])
    num_top1 = sum(1 for row in rows if row["top1_label_preserved"])
    return {
        "num_patients": len({row["patient_id"] for row in rows}),
        "num_facts": total,
        "num_recovered": num_recovered,
        "recovery_rate": (num_recovered / total) if total else 0.0,
        "mean_flipped_count": (
            sum(int(row["flipped_count"]) for row in rows) / total
        )
        if total
        else 0.0,
        "mean_label_agreement": (sum(agreements) / len(agreements)) if agreements else None,
        "num_label_agreement": len(agreements),
        "num_top1_label_preserved": num_top1,
        "top1_label_preserved_rate": (num_top1 / total) if total else 0.0,
        "num_masked_errors": sum(int(row.get("masked_errors") or 0) for row in rows),
    }


# --------------------------------------------------------------------------- #
# 4. Evaluation run
# --------------------------------------------------------------------------- #


def run_masked_eval(
    llm: SupportsLLM,
    patients: Sequence[MaskedPatient],
    trials: dict[str, TrialRecord],
    index: BM25Index,
    settings: Settings,
    *,
    question_rounds: int = DEFAULT_QUESTION_ROUNDS,
    top_k: int | None = None,
    max_patients: int | None = None,
    max_facts: int | None = None,
) -> dict:
    """Baseline + one masked run per fact, with the metrics of design.md §5.3.

    The comparison is controlled: the baseline gets the *same* `question_rounds`
    and the *same* simulator (answering from the full note) as every masked arm,
    and every masked arm judges the baseline's candidate trials (profiling is
    stochastic, so per-arm retrieval would mix retrieval variance into the
    numbers) — the only difference between the two arms is the removed
    sentences.

    `top_k` bounds the logged ranking. It defaults to `retrieval.candidates_k`,
    i.e. every assessed trial, because the ranking is what all the comparison
    metrics read and the ranker sorts by the very label they compare — a
    truncated ranking would silently select the agreeing subset. Pass a smaller
    value only if you accept that bias.

    One `parsed_cache` is shared by every run of every patient: the criteria
    parse is the dominant repeated cost and it depends only on the trial.

    A failed run is recorded under `errors` and skipped — a masked run that
    raises loses its fact, a baseline that raises loses its patient, neither
    sinks the batch. Both caps are logged with what they dropped.
    """
    if top_k is None:
        top_k = settings.retrieval.candidates_k

    selected = list(patients) if max_patients is None else list(patients)[:max_patients]
    if max_patients is not None and len(patients) > len(selected):
        dropped = [patient.patient_id for patient in list(patients)[max_patients:]]
        logger.warning(
            "--max-patients dropped %d patient(s): %s", len(dropped), ", ".join(dropped)
        )

    parsed_cache: dict[str, ParsedCriteria] = {}
    rows: list[dict] = []
    errors: list[dict] = []
    run_logs: dict[str, dict] = {}

    for patient in selected:
        note = full_note(patient)
        # One provider for both arms: the simulator always answers from the full
        # note, so baseline and masked runs differ only in the masked sentences.
        answer_provider = simulator_answer_provider(llm, note, settings)
        try:
            _, baseline_log = run_pipeline(
                patient_id=patient.patient_id,
                note=note,
                trials=trials,
                index=index,
                llm=llm,
                settings=settings,
                top_k=top_k,
                question_rounds=question_rounds,
                answer_provider=answer_provider,
                parsed_cache=parsed_cache,
            )
        except Exception as exc:  # noqa: BLE001 — one patient must not sink the batch
            errors.append(
                {"patient_id": patient.patient_id, "stage": "baseline", "error": str(exc)}
            )
            logger.warning("baseline run failed for %s: %s", patient.patient_id, exc)
            continue

        logs: dict = {"baseline": baseline_log, "facts": {}}
        run_logs[patient.patient_id] = logs

        facts = patient.facts if max_facts is None else patient.facts[:max_facts]
        if max_facts is not None and len(patient.facts) > len(facts):
            dropped_fields = [fact.field for fact in patient.facts[max_facts:]]
            logger.warning(
                "--max-facts dropped %d fact(s) of %s: %s",
                len(dropped_fields),
                patient.patient_id,
                ", ".join(dropped_fields),
            )

        for fact in facts:
            try:
                _, masked_log = run_pipeline(
                    patient_id=f"{patient.patient_id}:{fact.field}",
                    note=masked_note(patient, fact),
                    trials=trials,
                    index=index,
                    llm=llm,
                    settings=settings,
                    top_k=top_k,
                    question_rounds=question_rounds,
                    answer_provider=answer_provider,
                    parsed_cache=parsed_cache,
                    # Pin the baseline's candidates so both arms judge the same
                    # trials: profiling is stochastic, and letting each arm
                    # retrieve for itself would mix retrieval variance into the
                    # masking comparison.
                    candidate_ids=baseline_log["candidate_ids"],
                )
            except Exception as exc:  # noqa: BLE001 — one fact must not sink the batch
                errors.append(
                    {
                        "patient_id": patient.patient_id,
                        "field": fact.field,
                        "stage": "masked",
                        "error": str(exc),
                    }
                )
                logger.warning(
                    "masked run failed for %s / %s: %s", patient.patient_id, fact.field, exc
                )
                continue

            logs["facts"][fact.field] = masked_log
            rows.append(
                {
                    "patient_id": patient.patient_id,
                    "field": fact.field,
                    **fact_metrics(baseline_log, masked_log, fact.keywords),
                }
            )

    return {
        "config": {
            "question_rounds": question_rounds,
            "top_k": top_k,
            "candidates_k": settings.retrieval.candidates_k,
            "max_patients": max_patients,
            "max_facts": max_facts,
        },
        "results": rows,
        "summary": summarize(rows),
        "errors": errors,
        "run_logs": run_logs,
    }


def save_masked_result(result: dict, settings: Settings) -> Path:
    """Write the full result (metrics + run logs) into `settings.log_dir`.

    The filename timestamp is UTC so logs from different machines sort together,
    matching `trialmatch.pipeline.save_run_log`.
    """
    log_dir = Path(settings.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    path = log_dir / f"masked-{datetime.now(UTC).strftime('%Y%m%d-%H%M%S')}.json"
    with path.open("w", encoding="utf-8") as fh:
        json.dump(result, fh, ensure_ascii=False, indent=2)
    return path


# --------------------------------------------------------------------------- #
# 5. CLI
# --------------------------------------------------------------------------- #


def _yes_no(value: bool) -> str:
    return "yes" if value else "no"


def format_table(result: Mapping) -> str:
    """Render the per-fact rows, the thin-comparison warnings and the summary."""
    rows = list(result.get("results") or [])
    summary = result.get("summary") or {}

    patient_width = max([len("patient"), *(len(str(row["patient_id"])) for row in rows)])
    field_width = max([len("field"), *(len(str(row["field"])) for row in rows)])

    header = (
        f"{'patient':<{patient_width}}  {'field':<{field_width}}  "
        f"{'recovered':>9}  {'flipped':>7}  {'agreement':>9}  "
        f"{'n_base':>6}  {'n_mask':>6}  {'n_shared':>8}  {'errors':>6}  "
        f"{'top1_label':>10}"
    )
    rule = (
        f"{'-' * patient_width}  {'-' * field_width}  "
        f"{'-' * 9}  {'-' * 7}  {'-' * 9}  "
        f"{'-' * 6}  {'-' * 6}  {'-' * 8}  {'-' * 6}  {'-' * 10}"
    )
    lines = [header, rule]
    for row in rows:
        agreement = row.get("label_agreement")
        agreement_text = "n/a" if agreement is None else f"{float(agreement):.4f}"
        lines.append(
            f"{row['patient_id']:<{patient_width}}  {row['field']:<{field_width}}  "
            f"{_yes_no(bool(row['recovered'])):>9}  {int(row['flipped_count']):>7}  "
            f"{agreement_text:>9}  "
            f"{int(row.get('n_baseline') or 0):>6}  {int(row.get('n_masked') or 0):>6}  "
            f"{int(row.get('n_shared') or 0):>8}  {int(row.get('masked_errors') or 0):>6}  "
            f"{_yes_no(bool(row['top1_label_preserved'])):>10}"
        )

    thin = [row for row in rows if int(row.get("n_shared") or 0) < MIN_SHARED_TRIALS]
    if thin:
        lines.append("")
    for row in thin:
        lines.append(
            f"warning: {row['patient_id']}/{row['field']} compared only "
            f"{int(row.get('n_shared') or 0)} shared trial(s) "
            f"(< {MIN_SHARED_TRIALS}); its label agreement is not meaningful"
        )

    total = int(summary.get("num_facts", 0) or 0)
    mean_agreement = summary.get("mean_label_agreement")
    lines.append("")
    lines.append(f"{'patients evaluated:':<23}{summary.get('num_patients', 0)}")
    lines.append(f"{'facts evaluated:':<23}{total}")
    lines.append(
        f"{'facts recovered:':<23}{summary.get('num_recovered', 0)} / {total} "
        f"({summary.get('recovery_rate', 0.0):.4f})"
    )
    lines.append(f"{'mean flipped trials:':<23}{summary.get('mean_flipped_count', 0.0):.4f}")
    lines.append(
        f"{'mean label agreement:':<23}"
        + ("n/a" if mean_agreement is None else f"{float(mean_agreement):.4f}")
        + f" (over {summary.get('num_label_agreement', 0)} of {total} facts)"
    )
    lines.append(
        f"{'top-1 label preserved:':<23}"
        f"{summary.get('num_top1_label_preserved', 0)} / {total} "
        f"({summary.get('top1_label_preserved_rate', 0.0):.4f})"
    )
    lines.append(f"{'masked-run errors:':<23}{summary.get('num_masked_errors', 0)}")
    return "\n".join(lines)


def _positive_int(raw: str) -> int:
    """argparse type for a count that must be at least 1 (exits 2 otherwise)."""
    try:
        value = int(raw)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"expected an integer, got {raw!r}") from exc
    if value < 1:
        raise argparse.ArgumentTypeError(f"must be >= 1, got {value}")
    return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="trialmatch-eval-masked",
        description=(
            "Masked-field synthetic-patient evaluation of the clarifying-question "
            "loop: generate complete patients, hide one fact at a time, and score "
            "what the questions recovered."
        ),
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    generate = subparsers.add_parser(
        "generate", help="expand topics into synthetic patients with maskable facts"
    )
    generate.add_argument(
        "--topics", required=True, type=Path, help="topics JSON ({'topics': [...]})"
    )
    generate.add_argument("--out", required=True, type=Path, help="patients JSON to write")
    generate.add_argument(
        "--provider",
        choices=["anthropic", "gemini"],
        default="anthropic",
        help="LLM provider; picks the model preset in config.py (default: anthropic)",
    )
    generate.add_argument("--max-topics", type=int, default=None, help="cap topics generated")

    run = subparsers.add_parser("run", help="run the masked-field evaluation")
    run.add_argument("--patients", required=True, type=Path, help="patients JSON")
    run.add_argument("--trials", required=True, type=Path, help="trials JSONL snapshot")
    run.add_argument("--index", required=True, type=Path, help="BM25 index pickle")
    run.add_argument(
        "--provider",
        choices=["anthropic", "gemini"],
        default="anthropic",
        help="LLM provider; picks the model preset in config.py (default: anthropic)",
    )
    run.add_argument(
        "--candidates",
        type=_positive_int,
        default=DEFAULT_CANDIDATES_K,
        help=(
            "candidates per run, overriding retrieval.candidates_k "
            f"(default: {DEFAULT_CANDIDATES_K})"
        ),
    )
    run.add_argument("--question-rounds", type=_positive_int, default=DEFAULT_QUESTION_ROUNDS)
    run.add_argument(
        "--top-k",
        type=_positive_int,
        default=None,
        help=(
            "trials kept in the logged ranking (default: --candidates, i.e. every "
            "assessed trial). A smaller value biases the comparison metrics."
        ),
    )
    run.add_argument("--max-patients", type=int, default=None, help="cap patients evaluated")
    run.add_argument("--max-facts", type=int, default=None, help="cap facts masked per patient")
    run.add_argument(
        "--save-log", action="store_true", help="write the result JSON into the log dir"
    )

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    load_dotenv()  # credentials/model overrides from <project>/.env, if present
    args = build_parser().parse_args(argv)
    settings = Settings().with_provider(args.provider)

    if args.command == "generate":
        # The real client is built here so importing this module needs no key.
        patients, errors = generate_patients(
            create_llm(settings.models),
            load_topics_json(args.topics),
            settings,
            max_topics=args.max_topics,
        )
        for error in errors:
            field = error.get("field")
            if field:
                print(f"warning: topic {error['topic_id']} fact {field}: {error['error']}")
            else:
                print(f"error: topic {error['topic_id']}: {error['error']}")
        for patient in patients:
            print(
                f"{patient.patient_id}\t{len(patient.sentences)} sentences\t"
                f"{len(patient.facts)} facts\t{patient.condition}"
            )
        if not patients:
            # Writing now would replace a good dataset with an empty one.
            print(f"error: no patient could be generated; {args.out} was left untouched")
            return 1
        out_path = write_patients(patients, args.out)
        print(f"wrote {len(patients)} patients to {out_path}")
        return 0

    # Every comparison metric reads the logged ranking, so by default it covers
    # every assessed trial; truncating it would select the agreeing subset.
    top_k = args.candidates if args.top_k is None else args.top_k
    if top_k < args.candidates:
        print(
            f"warning: --top-k {top_k} is below --candidates {args.candidates}; "
            "the comparison metrics are computed over a truncated ranking"
        )

    settings = settings.model_copy(
        update={
            "retrieval": settings.retrieval.model_copy(
                update={"candidates_k": args.candidates}
            )
        }
    )
    result = run_masked_eval(
        create_llm(settings.models),
        load_patients(args.patients),
        load_trials(args.trials),
        BM25Index.load(args.index),
        settings,
        question_rounds=args.question_rounds,
        top_k=top_k,
        max_patients=args.max_patients,
        max_facts=args.max_facts,
    )
    for error in result["errors"]:
        field = error.get("field")
        target = f"{error['patient_id']}/{field}" if field else error["patient_id"]
        print(f"error: {target} ({error['stage']}): {error['error']}")
    print(format_table(result))
    if args.save_log:
        print(f"result log: {save_masked_result(result, settings)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
