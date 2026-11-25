"""Benchmark format, time rule and data splits.

A query is one downstream issue. The gold answers are upstream items.
A candidate is an item that can be a suspect. It must be older than the query.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, replace
from datetime import datetime
from typing import Literal

Kind = Literal["issue", "pr", "release"]


def _check_aware(value: datetime, what: str) -> None:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{what} must have a time zone")


@dataclass(frozen=True)
class Candidate:
    """An item that can be a suspect. Example id: 'numpy/numpy#123'."""

    id: str
    repo: str
    kind: Kind
    title: str
    body: str
    date: datetime

    def __post_init__(self) -> None:
        _check_aware(self.date, f"date of {self.id}")


@dataclass(frozen=True)
class Query:
    """One downstream issue and the IDs of its gold answers."""

    id: str
    repo: str
    title: str
    body: str
    date: datetime
    gold: frozenset[str]

    def __post_init__(self) -> None:
        _check_aware(self.date, f"date of {self.id}")


def is_older(candidate: Candidate, query: Query) -> bool:
    """The time rule. The candidate must be strictly older than the query."""
    return candidate.date < query.date


def eligible(
    query: Query,
    candidates: Iterable[Candidate],
    exclude_same_repo: bool = True,
) -> list[Candidate]:
    """Return the candidates that the query may use. Keep the order."""
    return [
        c
        for c in candidates
        if is_older(c, query) and not (exclude_same_repo and c.repo == query.repo)
    ]


@dataclass(frozen=True)
class CleanReport:
    """What the cleaning step did."""

    queries_in: int
    queries_kept: int
    gold_not_found: int
    gold_ineligible: int


def clean_queries(
    queries: Iterable[Query],
    candidates: Sequence[Candidate],
    exclude_same_repo: bool = True,
) -> tuple[list[Query], CleanReport]:
    """Remove gold answers that break the rules. Remove queries with no gold."""
    by_id = {c.id: c for c in candidates}
    kept: list[Query] = []
    total = 0
    not_found = 0
    ineligible = 0
    for query in queries:
        total += 1
        valid: set[str] = set()
        for gold_id in sorted(query.gold):
            candidate = by_id.get(gold_id)
            if candidate is None:
                not_found += 1
            elif not eligible(query, [candidate], exclude_same_repo):
                ineligible += 1
            else:
                valid.add(gold_id)
        if valid:
            kept.append(replace(query, gold=frozenset(valid)))
    report = CleanReport(
        queries_in=total,
        queries_kept=len(kept),
        gold_not_found=not_found,
        gold_ineligible=ineligible,
    )
    return kept, report


@dataclass(frozen=True)
class Split:
    """A train set and a test set."""

    name: str
    train: tuple[Query, ...]
    test: tuple[Query, ...]


def time_split(queries: Iterable[Query], test_fraction: float = 0.2) -> Split:
    """The newest queries are the test set. The cut is at a date."""
    if not 0.0 < test_fraction < 1.0:
        raise ValueError("test_fraction must be between 0 and 1")
    ordered = sorted(queries, key=lambda q: (q.date, q.id))
    count = len(ordered)
    if count < 2:
        raise ValueError("need at least 2 queries")
    index = max(1, min(count - 1, count - round(count * test_fraction)))
    cutoff = ordered[index].date
    train = tuple(q for q in ordered if q.date < cutoff)
    test = tuple(q for q in ordered if q.date >= cutoff)
    if not train:
        raise ValueError("cannot split: the queries have no earlier date")
    return Split("time", train, test)


def leave_one_project_out(queries: Iterable[Query]) -> dict[str, Split]:
    """One split for each repository. The test set is that repository."""
    items = list(queries)
    repos = sorted({q.repo for q in items})
    if len(repos) < 2:
        raise ValueError("need queries from at least 2 repositories")
    return {
        repo: Split(
            name=f"lopo:{repo}",
            train=tuple(q for q in items if q.repo != repo),
            test=tuple(q for q in items if q.repo == repo),
        )
        for repo in repos
    }