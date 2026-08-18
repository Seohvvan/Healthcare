"""Command-line entry point: snapshot download, index build, and pipeline runs.

Importing this module never touches the network or the Anthropic credentials:
`LLMClient` creates its underlying client lazily, and every subcommand builds
what it needs inside its own handler.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from trialmatch.agents.prompts import SIMULATOR_UNKNOWN_ANSWER
from trialmatch.config import Settings
from trialmatch.data import download_snapshot, load_topics_json, load_trials
from trialmatch.llm import LLMClient
from trialmatch.pipeline import (
    AnswerProvider,
    run_pipeline,
    save_run_log,
    simulator_answer_provider,
)
from trialmatch.retrieval import BM25Index
from trialmatch.schemas import ClarifyingQuestion, QuestionAnswer

__all__ = ["build_parser", "main"]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="trialmatch",
        description="Interactive clinical trial recommendation (Healthcare Agentic AI 2026).",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    download = subparsers.add_parser(
        "download", help="fetch a ClinicalTrials.gov snapshot into a JSONL file"
    )
    download.add_argument(
        "--queries", nargs="+", required=True, help="condition queries, e.g. migraine"
    )
    download.add_argument("--out", required=True, type=Path, help="output JSONL path")
    download.add_argument(
        "--max-per-query", type=int, default=None, help="cap studies fetched per query"
    )
    download.set_defaults(handler=_cmd_download)

    index = subparsers.add_parser("index", help="build a BM25 index over a snapshot")
    index.add_argument("--trials", required=True, type=Path, help="trials JSONL snapshot")
    index.add_argument("--out", required=True, type=Path, help="output index pickle path")
    index.set_defaults(handler=_cmd_index)

    run = subparsers.add_parser("run", help="run the pipeline for one patient note")
    note_group = run.add_mutually_exclusive_group(required=True)
    note_group.add_argument("--note", help="patient description as text")
    note_group.add_argument("--note-file", type=Path, help="file holding the patient description")
    run.add_argument("--patient-id", required=True, help="identifier used in logs and the report")
    run.add_argument("--trials", required=True, type=Path, help="trials JSONL snapshot")
    run.add_argument("--index", required=True, type=Path, help="BM25 index pickle")
    run.add_argument("--top-k", type=int, default=5, help="trials shown in the report")
    run.add_argument(
        "--no-questions", action="store_true", help="skip the clarifying-question loop"
    )
    run.add_argument(
        "--simulate-file",
        type=Path,
        default=None,
        help="ground-truth note answered by the Patient Simulator instead of a user",
    )
    run.add_argument("--save-log", action="store_true", help="write the run log JSON")
    run.set_defaults(handler=_cmd_run)

    topics = subparsers.add_parser("topics", help="list topics of a TREC-style topics JSON")
    topics.add_argument("--json", dest="json_path", required=True, type=Path)
    topics.set_defaults(handler=_cmd_topics)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    handler = args.handler
    return int(handler(args))


# --------------------------------------------------------------------------- #
# commands
# --------------------------------------------------------------------------- #


def _cmd_download(args: argparse.Namespace) -> int:
    try:
        written = download_snapshot(
            list(args.queries), args.out, max_per_query=args.max_per_query
        )
    except (OSError, ValueError) as exc:
        print(f"error: {exc}")
        return 2
    print(f"wrote {written} trials to {args.out}")
    return 0


def _cmd_index(args: argparse.Namespace) -> int:
    try:
        trials = load_trials(args.trials)
        index = BM25Index.build(trials.values())
        index.save(args.out)
    except (OSError, ValueError) as exc:
        print(f"error: {exc}")
        return 2
    print(f"indexed {len(index)} trials into {args.out}")
    return 0


def _cmd_run(args: argparse.Namespace) -> int:
    settings = Settings()

    # Every user-supplied path is read here so a missing or unreadable file ends
    # the command with a one-line message and exit code 2, never a traceback.
    try:
        note = args.note if args.note is not None else args.note_file.read_text(encoding="utf-8")
        note = note.strip()
        if not note:
            print("error: the patient note is empty")
            return 2
        trials = load_trials(args.trials)
        index = BM25Index.load(args.index)
        ground_truth = (
            args.simulate_file.read_text(encoding="utf-8")
            if args.simulate_file is not None
            else None
        )
    except (OSError, ValueError) as exc:
        print(f"error: {exc}")
        return 2

    llm = LLMClient()

    answer_provider: AnswerProvider | None
    if args.no_questions:
        answer_provider = None
    elif ground_truth is not None:
        answer_provider = simulator_answer_provider(llm, ground_truth, settings)
    else:
        answer_provider = _interactive_answer_provider

    recommendation, run_log = run_pipeline(
        patient_id=args.patient_id,
        note=note,
        trials=trials,
        index=index,
        llm=llm,
        settings=settings,
        top_k=args.top_k,
        question_rounds=0 if args.no_questions else 1,
        answer_provider=answer_provider,
    )

    print(recommendation.report_markdown)
    if args.save_log:
        print(f"run log: {save_run_log(run_log, settings)}")
    return 0


def _cmd_topics(args: argparse.Namespace) -> int:
    try:
        topics = load_topics_json(args.json_path)
    except (OSError, ValueError) as exc:
        print(f"error: {exc}")
        return 2
    for topic_id, text in topics:
        print(f"{topic_id}\t{text}")
    return 0


def _interactive_answer_provider(questions: list[ClarifyingQuestion]) -> list[QuestionAnswer]:
    """Ask the questions on stdin; an empty or missing answer counts as unknown."""
    print("\n확인이 필요한 질문이 있습니다 (모르면 그냥 Enter):")
    answers: list[QuestionAnswer] = []
    for question in questions:
        print(f"[{question.question_id}] {question.text}")
        try:
            answer_text = input("> ").strip()
        except EOFError:  # non-interactive stdin: treat the rest as unknown
            answer_text = ""
        answers.append(
            QuestionAnswer(
                question_id=question.question_id,
                answer_text=answer_text or SIMULATOR_UNKNOWN_ANSWER,
            )
        )
    print()
    return answers


if __name__ == "__main__":
    raise SystemExit(main())
