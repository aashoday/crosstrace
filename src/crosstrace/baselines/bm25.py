"""BM25 baseline: rank candidates by keyword match.

The index is built once from the whole candidate pool.
At query time, only candidates that pass the time rule are scored.
"""

from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from collections.abc import Iterable

from crosstrace.benchmark import Candidate, Query, eligible

_TOKEN = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    """Lower case. Keep letters and digits. Split on everything else."""
    return _TOKEN.findall(text.lower())


def _text(title: str, body: str) -> str:
    return f"{title}\n{body}"


class BM25Index:
    """An inverted index over a pool of candidates."""

    def __init__(
        self,
        candidates: Iterable[Candidate],
        k1: float = 1.5,
        b: float = 0.75,
    ) -> None:
        if k1 < 0:
            raise ValueError("k1 must not be negative")
        if not 0.0 <= b <= 1.0:
            raise ValueError("b must be between 0 and 1")
        pool = list(candidates)
        if not pool:
            raise ValueError("the candidate pool is empty")
        ids = [c.id for c in pool]
        if len(set(ids)) != len(ids):
            raise ValueError("candidate IDs must be unique")

        self._pool = pool
        self._k1 = k1
        self._b = b
        self._length: dict[str, int] = {}
        self._postings: dict[str, list[tuple[str, int]]] = {}
        for candidate in pool:
            tokens = tokenize(_text(candidate.title, candidate.body))
            self._length[candidate.id] = len(tokens)
            for term, count in Counter(tokens).items():
                self._postings.setdefault(term, []).append((candidate.id, count))
        total = sum(self._length.values())
        self._avg_length = total / len(pool) or 1.0

    def _idf(self, document_frequency: int) -> float:
        n = len(self._pool)
        return math.log(1 + (n - document_frequency + 0.5) / (document_frequency + 0.5))

    def search(
        self,
        query: Query,
        exclude_same_repo: bool = True,
        top_k: int = 100,
    ) -> list[tuple[str, float]]:
        """Return (candidate ID, score), best first. Only scores above 0."""
        if top_k < 1:
            raise ValueError("top_k must be at least 1")
        allowed = {c.id for c in eligible(query, self._pool, exclude_same_repo)}
        scores: dict[str, float] = defaultdict(float)
        for term in set(tokenize(_text(query.title, query.body))):
            postings = self._postings.get(term)
            if not postings:
                continue
            idf = self._idf(len(postings))
            for candidate_id, count in postings:
                if candidate_id not in allowed:
                    continue
                length = self._length[candidate_id]
                norm = count + self._k1 * (
                    1 - self._b + self._b * length / self._avg_length
                )
                scores[candidate_id] += idf * count * (self._k1 + 1) / norm
        ranked = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
        return ranked[:top_k]


def rank_queries(
    index: BM25Index,
    queries: Iterable[Query],
    exclude_same_repo: bool = True,
    top_k: int = 100,
) -> dict[str, list[str]]:
    """Return the run: query ID to ranked candidate IDs."""
    return {
        query.id: [
            candidate_id
            for candidate_id, _ in index.search(query, exclude_same_repo, top_k)
        ]
        for query in queries
    }