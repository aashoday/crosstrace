"""Ranking metrics: Recall@k, MRR and nDCG."""

from __future__ import annotations

import math
from collections.abc import Collection, Hashable, Mapping, Sequence

DEFAULT_KS = (1, 5, 10, 20)


def _unique(ranked: Sequence[Hashable]) -> list[Hashable]:
    """Keep the first copy of each item. Keep the order."""
    seen: set[Hashable] = set()
    result: list[Hashable] = []
    for item in ranked:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result


def _check(gold: Collection[Hashable], k: int | None = None) -> set[Hashable]:
    gold_set = set(gold)
    if not gold_set:
        raise ValueError("gold must not be empty")
    if k is not None and k < 1:
        raise ValueError("k must be at least 1")
    return gold_set


def recall_at_k(ranked: Sequence[Hashable], gold: Collection[Hashable], k: int) -> float:
    """Share of gold items found in the top k."""
    gold_set = _check(gold, k)
    top = _unique(ranked)[:k]
    hits = sum(1 for item in top if item in gold_set)
    return hits / len(gold_set)


def reciprocal_rank(ranked: Sequence[Hashable], gold: Collection[Hashable]) -> float:
    """1 divided by the rank of the first gold item. 0 if there is none."""
    gold_set = _check(gold)
    for rank, item in enumerate(_unique(ranked), start=1):
        if item in gold_set:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(ranked: Sequence[Hashable], gold: Collection[Hashable], k: int) -> float:
    """Normalized discounted cumulative gain in the top k."""
    gold_set = _check(gold, k)
    top = _unique(ranked)[:k]
    dcg = sum(
        1.0 / math.log2(rank + 1)
        for rank, item in enumerate(top, start=1)
        if item in gold_set
    )
    ideal_hits = min(len(gold_set), k)
    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))
    return dcg / idcg


def evaluate(
    run: Mapping[str, Sequence[Hashable]],
    qrels: Mapping[str, Collection[Hashable]],
    ks: Sequence[int] = DEFAULT_KS,
) -> dict[str, float]:
    """Mean metrics over all queries in qrels.

    run: query ID to ranked list.
    qrels: query ID to the gold set.
    A query that is missing from run scores 0.
    A query that is in run but not in qrels is ignored.
    """
    if not qrels:
        raise ValueError("qrels must not be empty")
    queries = list(qrels)
    n = len(queries)
    result: dict[str, float] = {"n_queries": float(n)}
    for k in ks:
        result[f"recall@{k}"] = (
            sum(recall_at_k(run.get(q, []), qrels[q], k) for q in queries) / n
        )
    result["mrr"] = sum(reciprocal_rank(run.get(q, []), qrels[q]) for q in queries) / n
    for k in ks:
        result[f"ndcg@{k}"] = (
            sum(ndcg_at_k(run.get(q, []), qrels[q], k) for q in queries) / n
        )
    return result