"""Storage for annotation records, with resume."""

from __future__ import annotations

import os
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path

from pydantic import BaseModel, ValidationError

from crosstrace.autoannotate.schema import Evidence, Label, Pair, Verdict
from crosstrace.autoannotate.validate import Assessment, Status

Key = tuple[str, str, str]


class AnnotationRecord(BaseModel):
    """One line of annotated.jsonl."""

    pair_id: str
    source_url: str
    target_url: str
    label: Label | None = None
    confidence: float | None = None
    evidence: list[Evidence] = []
    explanation: str = ""
    status: Status
    review_reasons: list[str] = []
    model: str
    prompt_version: str
    created_at: str

    @property
    def key(self) -> Key:
        return (self.pair_id, self.model, self.prompt_version)

    @property
    def completed(self) -> bool:
        """False when the model gave no valid verdict."""
        return self.label is not None


def make_record(
    pair: Pair,
    verdict: Verdict | None,
    assessment: Assessment,
    model: str,
    prompt_version: str,
    now: datetime | None = None,
) -> AnnotationRecord:
    """Build a record. Pass the ORIGINAL pair, not the result of prepare()."""
    moment = now or datetime.now(UTC)
    return AnnotationRecord(
        pair_id=pair.pair_id,
        source_url=pair.source_url,
        target_url=pair.target_url,
        label=verdict.label if verdict else None,
        confidence=verdict.confidence if verdict else None,
        evidence=list(verdict.evidence) if verdict else [],
        explanation=verdict.explanation if verdict else "",
        status=assessment.status,
        review_reasons=list(assessment.reasons),
        model=model,
        prompt_version=prompt_version,
        created_at=moment.isoformat(timespec="seconds"),
    )


class AnnotationStore:
    """An append-only JSONL file of annotation records."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.skipped_lines = 0
        self._latest: dict[Key, AnnotationRecord] = {}
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            return
        with self.path.open(encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    record = AnnotationRecord.model_validate_json(line)
                except ValidationError:
                    self.skipped_lines += 1
                    continue
                self._latest[record.key] = record

    def _ends_cleanly(self) -> bool:
        if not self.path.exists() or self.path.stat().st_size == 0:
            return True
        with self.path.open("rb") as f:
            f.seek(-1, os.SEEK_END)
            return f.read(1) == b"\n"

    def is_done(self, pair_id: str, model: str, prompt_version: str) -> bool:
        record = self._latest.get((pair_id, model, prompt_version))
        return record is not None and record.completed

    def pending(
        self, pairs: Iterable[Pair], model: str, prompt_version: str
    ) -> list[Pair]:
        """Return the pairs that still need an annotation. Keep the order."""
        return [
            pair
            for pair in pairs
            if not self.is_done(pair.pair_id, model, prompt_version)
        ]

    def records(self) -> list[AnnotationRecord]:
        """Return the newest record for each key."""
        return list(self._latest.values())

    def append(self, record: AnnotationRecord) -> None:
        """Write one record to disk. Flush and sync before returning."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        prefix = "" if self._ends_cleanly() else "\n"
        with self.path.open("a", encoding="utf-8") as f:
            f.write(prefix + record.model_dump_json() + "\n")
            f.flush()
            os.fsync(f.fileno())
        self._latest[record.key] = record