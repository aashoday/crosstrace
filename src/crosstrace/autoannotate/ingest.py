"""Read pairs from a CSV or JSONL file."""

from __future__ import annotations

import csv
import json
from collections.abc import Iterator
from pathlib import Path

from pydantic import ValidationError

from crosstrace.autoannotate.schema import Pair

FIELDS = frozenset(Pair.model_fields)

Row = tuple[int, object, str | None]


def _csv_rows(path: Path) -> Iterator[Row]:
    with path.open(newline="", encoding="utf-8") as f:
        for number, row in enumerate(csv.DictReader(f), start=2):
            yield number, row, None


def _jsonl_rows(path: Path) -> Iterator[Row]:
    with path.open(encoding="utf-8") as f:
        for number, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                yield number, json.loads(line), None
            except json.JSONDecodeError as exc:
                yield number, None, f"invalid JSON ({exc.msg})"


def read_pairs(path: str | Path) -> tuple[list[Pair], list[str]]:
    """Return the valid pairs and a list of problem messages."""
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".csv":
        rows = _csv_rows(path)
    elif suffix == ".jsonl":
        rows = _jsonl_rows(path)
    else:
        raise ValueError(f"Unsupported file type: {path.suffix}. Use .csv or .jsonl.")

    pairs: list[Pair] = []
    problems: list[str] = []
    seen: set[str] = set()
    for number, row, error in rows:
        if error:
            problems.append(f"line {number}: {error}")
            continue
        if not isinstance(row, dict):
            problems.append(f"line {number}: row is not an object")
            continue
        data = {key: value for key, value in row.items() if key in FIELDS}
        try:
            pair = Pair(**data)
        except ValidationError as exc:
            bad = ", ".join(sorted({str(e["loc"][0]) for e in exc.errors() if e["loc"]}))
            problems.append(f"line {number}: bad fields: {bad}")
            continue
        if pair.pair_id in seen:
            problems.append(f"line {number}: duplicate pair")
            continue
        seen.add(pair.pair_id)
        pairs.append(pair)
    return pairs, problems