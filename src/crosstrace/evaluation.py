"""Run ranking systems on a split and report metrics."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from crosstrace.baselines import bm25, frame_rule
from crosstrace.benchmark import Candidate, Query, Split
from crosstrace.metrics import DEFAULT_KS, evaluate


@dataclass(frozen=True)
class SystemOutput:
    """The run of one system and extra counts for the report."""

    run: dict[str, list[str]]
    notes: dict[str, int]


Builder = Callable[[Sequence[Candidate], Sequence[Query]], SystemOutput]


def _run_bm25(candidates: Sequence[Candidate], queries: Sequence[Query]) -> SystemOutput:
    index = bm25.BM25Index(candidates)
    return SystemOutput(bm25.rank_queries(index, queries), {})


def _run_frame_rule(
    candidates: Sequence[Candidate], queries: Sequence[Query]
) -> SystemOutput:
    rule = frame_rule.FrameRule(candidates)
    return SystemOutput(
        frame_rule.rank_queries(rule, queries),
        {"no_upstream_frame": rule.count_without_upstream_frame(queries)},
    )


SYSTEMS: dict[str, Builder] = {
    "bm25": _run_bm25,
    "frame_rule": _run_frame_rule,
}


def _read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows = []
    with Path(path).open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def load_candidates(path: str | Path) -> list[Candidate]:
    return [
        Candidate(
            id=row["id"],
            repo=row["repo"],
            kind=row["kind"],
            title=row["title"],
            body=row.get("body") or "",
            date=datetime.fromisoformat(row["date"]),
        )
        for row in _read_jsonl(path)
    ]


def load_queries(path: str | Path) -> list[Query]:
    return [
        Query(
            id=row["id"],
            repo=row["repo"],
            title=row["title"],
            body=row.get("body") or "",
            date=datetime.fromisoformat(row["date"]),
            gold=frozenset(row["gold"]),
        )
        for row in _read_jsonl(path)
    ]


def evaluate_split(
    split: Split,
    candidates: Sequence[Candidate],
    system_names: Sequence[str],
    ks: Sequence[int] = DEFAULT_KS,
) -> dict[str, Any]:
    """Run the systems on the test queries of one split."""
    unknown = [name for name in system_names if name not in SYSTEMS]
    if unknown:
        raise ValueError(f"unknown system: {', '.join(unknown)}")
    test = list(split.test)
    qrels = {q.id: q.gold for q in test}
    systems: dict[str, dict[str, float]] = {}
    for name in system_names:
        output = SYSTEMS[name](candidates, test)
        metrics = evaluate(output.run, qrels, ks)
        systems[name] = {**metrics, **{k: float(v) for k, v in output.notes.items()}}
    return {"split": split.name, "n_queries": len(test), "systems": systems}


def format_table(result: Mapping[str, Any]) -> str:
    """Make a text table for one split result."""
    systems = result["systems"]
    columns = [
        key
        for key in next(iter(systems.values()))
        if key not in ("n_queries", "no_upstream_frame")
    ]
    lines = [f"split: {result['split']}   queries: {result['n_queries']}"]
    lines.append(f"{'system':<12}" + "".join(f"{c:>11}" for c in columns))
    for name, metrics in systems.items():
        lines.append(f"{name:<12}" + "".join(f"{metrics[c]:>11.3f}" for c in columns))
        if "no_upstream_frame" in metrics:
            count = int(metrics["no_upstream_frame"])
            lines.append(f"  {name}: {count} queries have no upstream frame")
    return "\n".join(lines)


def save_results(
    results: Sequence[Mapping[str, Any]], path: str | Path, meta: Mapping[str, Any]
) -> None:
    """Write the results and the run settings to a JSON file."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {"meta": dict(meta), "results": list(results)}
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")