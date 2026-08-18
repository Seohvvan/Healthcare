"""Multi-agent pipeline stages (design.md §3).

Each module exposes one public function and takes an `LLMClient` where it needs
the model, so the whole pipeline can be driven with a fake client in tests.
`ranker` and `reporter` are pure code by design: aggregation rules and the final
report must be deterministic and auditable.
"""

from __future__ import annotations

from trialmatch.agents.criteria_parser import parse_criteria
from trialmatch.agents.matcher import assess_trial
from trialmatch.agents.profiler import profile_patient
from trialmatch.agents.question_gen import generate_questions
from trialmatch.agents.ranker import aggregate, rank
from trialmatch.agents.reporter import build_report
from trialmatch.agents.simulator import answer_questions

__all__ = [
    "aggregate",
    "answer_questions",
    "assess_trial",
    "build_report",
    "generate_questions",
    "parse_criteria",
    "profile_patient",
    "rank",
]
