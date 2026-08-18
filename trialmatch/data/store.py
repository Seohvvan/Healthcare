"""JSONL persistence for `TrialRecord` snapshots.

One JSON object per line so snapshots stream without loading everything into
memory, and so a partially written file stays usable.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from pathlib import Path

from trialmatch.schemas import TrialRecord


def write_jsonl(records: Iterable[TrialRecord], path: Path, append: bool = False) -> int:
    """Write records as JSON lines; returns the number of records written."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if append else "w"
    count = 0
    with path.open(mode, encoding="utf-8") as fh:
        for record in records:
            fh.write(record.model_dump_json())
            fh.write("\n")
            count += 1
    return count


def read_jsonl(path: Path) -> Iterator[TrialRecord]:
    """Yield records from a JSONL snapshot, skipping blank lines."""
    with Path(path).open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            yield TrialRecord.model_validate(json.loads(line))


def load_trials(path: Path) -> dict[str, TrialRecord]:
    """Load a whole snapshot into a dict keyed by `nct_id` (last one wins)."""
    return {record.nct_id: record for record in read_jsonl(path)}
