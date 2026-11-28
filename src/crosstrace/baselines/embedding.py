"""Embedding baseline: rank candidates by cosine similarity.

Candidates and queries are encoded with a pretrained model.
Vectors are saved on disk. At query time, only candidates that pass
the time rule are scored.
"""

from __future__ import annotations

import hashlib
import os
import re
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Protocol

import numpy as np

from crosstrace.benchmark import Candidate, Query, eligible

DEFAULT_MODEL = "BAAI/bge-small-en-v1.5"


class Encoder(Protocol):
    """Turns texts into vectors. `name` is part of the cache key."""

    name: str

    def encode(self, texts: Sequence[str]) -> np.ndarray: ...


class SentenceTransformerEncoder:
    """A pretrained sentence-transformers model. It loads on first use."""

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        max_seq_length: int = 256,
        batch_size: int = 32,
        device: str | None = None,
    ) -> None:
        from sentence_transformers import SentenceTransformer

        self.name = f"{model_name}@{max_seq_length}"
        self._batch_size = batch_size
        self._model = SentenceTransformer(model_name, device=device)
        self._model.max_seq_length = max_seq_length

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        vectors = self._model.encode(
            list(texts),
            batch_size=self._batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return np.asarray(vectors, dtype=np.float32)


class EmbeddingCache:
    """Vectors on disk. One file for each encoder name."""

    def __init__(self, directory: str | Path, encoder_name: str) -> None:
        safe = re.sub(r"[^A-Za-z0-9._-]+", "_", encoder_name)
        self.path = Path(directory) / f"{safe}.npz"
        self._vectors: dict[str, np.ndarray] = {}
        if self.path.exists():
            with np.load(self.path) as data:
                self._vectors = {key: data[key] for key in data.files}

    @staticmethod
    def _key(text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def get(self, text: str) -> np.ndarray | None:
        return self._vectors.get(self._key(text))

    def put(self, text: str, vector: np.ndarray) -> None:
        self._vectors[self._key(text)] = np.asarray(vector, dtype=np.float32)

    def save(self) -> None:
        """Write the file. Write to a temporary file first, then replace."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_name(f"{self.path.stem}.tmp.npz")
        np.savez(temporary, **self._vectors)
        os.replace(temporary, self.path)


def _text(title: str, body: str) -> str:
    return f"{title}\n{body}"


def _normalize(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


def embed_texts(
    texts: Sequence[str],
    encoder: Encoder,
    cache: EmbeddingCache | None = None,
    batch_size: int = 64,
) -> np.ndarray:
    """Return one vector for each text. Encode only what the cache lacks."""
    if not texts:
        return np.zeros((0, 0), dtype=np.float32)
    result: list[np.ndarray | None] = [None] * len(texts)
    missing: list[str] = []
    for position, text in enumerate(texts):
        cached = cache.get(text) if cache else None
        if cached is None:
            missing.append(text)
        else:
            result[position] = cached

    unique = list(dict.fromkeys(missing))
    fresh: dict[str, np.ndarray] = {}
    for start in range(0, len(unique), batch_size):
        batch = unique[start : start + batch_size]
        vectors = np.asarray(encoder.encode(batch), dtype=np.float32)
        if vectors.ndim != 2 or vectors.shape[0] != len(batch):
            raise ValueError("the encoder returned the wrong shape")
        for text, vector in zip(batch, vectors, strict=True):
            fresh[text] = vector
            if cache:
                cache.put(text, vector)
    if cache and fresh:
        cache.save()

    for position, text in enumerate(texts):
        if result[position] is None:
            result[position] = fresh[text]
    return np.vstack(result)


class EmbeddingIndex:
    """Candidate vectors and cosine search."""

    def __init__(
        self,
        candidates: Iterable[Candidate],
        encoder: Encoder,
        cache: EmbeddingCache | None = None,
        batch_size: int = 64,
    ) -> None:
        pool = list(candidates)
        if not pool:
            raise ValueError("the candidate pool is empty")
        ids = [c.id for c in pool]
        if len(set(ids)) != len(ids):
            raise ValueError("candidate IDs must be unique")
        self._pool = pool
        self._encoder = encoder
        self._cache = cache
        self._batch_size = batch_size
        self._row = {candidate_id: row for row, candidate_id in enumerate(ids)}
        texts = [_text(c.title, c.body) for c in pool]
        self._matrix = _normalize(
            embed_texts(texts, encoder, cache, batch_size)
        )

    def _embed_queries(self, queries: Sequence[Query]) -> np.ndarray:
        texts = [_text(q.title, q.body) for q in queries]
        vectors = embed_texts(texts, self._encoder, self._cache, self._batch_size)
        return _normalize(vectors)

    def _rank(
        self,
        query: Query,
        vector: np.ndarray,
        exclude_same_repo: bool,
        top_k: int,
    ) -> list[tuple[str, float]]:
        allowed = eligible(query, self._pool, exclude_same_repo)
        if not allowed:
            return []
        rows = [self._row[c.id] for c in allowed]
        scores = self._matrix[rows] @ vector
        ranked = sorted(
            ((c.id, float(s)) for c, s in zip(allowed, scores, strict=True)),
            key=lambda item: (-item[1], item[0]),
        )
        return ranked[:top_k]

    def search(
        self,
        query: Query,
        exclude_same_repo: bool = True,
        top_k: int = 100,
    ) -> list[tuple[str, float]]:
        """Return (candidate ID, cosine score), best first."""
        if top_k < 1:
            raise ValueError("top_k must be at least 1")
        vector = self._embed_queries([query])[0]
        return self._rank(query, vector, exclude_same_repo, top_k)

    def search_many(
        self,
        queries: Sequence[Query],
        exclude_same_repo: bool = True,
        top_k: int = 100,
    ) -> dict[str, list[tuple[str, float]]]:
        """Like search(), but all queries are encoded in one batch."""
        if top_k < 1:
            raise ValueError("top_k must be at least 1")
        if not queries:
            return {}
        vectors = self._embed_queries(queries)
        return {
            query.id: self._rank(query, vector, exclude_same_repo, top_k)
            for query, vector in zip(queries, vectors, strict=True)
        }


def rank_queries(
    index: EmbeddingIndex,
    queries: Iterable[Query],
    exclude_same_repo: bool = True,
    top_k: int = 100,
) -> dict[str, list[str]]:
    """Return the run: query ID to ranked candidate IDs."""
    found = index.search_many(list(queries), exclude_same_repo, top_k)
    return {
        query_id: [candidate_id for candidate_id, _ in items]
        for query_id, items in found.items()
    }