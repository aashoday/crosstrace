from datetime import UTC, datetime

import numpy as np
import pytest

from crosstrace.baselines.embedding import (
    EmbeddingCache,
    EmbeddingIndex,
    embed_texts,
    rank_queries,
)
from crosstrace.benchmark import Candidate, Query
from crosstrace.evaluation import SYSTEMS
from crosstrace.metrics import evaluate

VOCAB = ("apple", "banana", "cherry")


class FakeEncoder:
    """Counts vocabulary words. Records every text it encodes."""

    def __init__(self, name="fake"):
        self.name = name
        self.encoded = []

    def encode(self, texts):
        self.encoded.extend(texts)
        rows = [[float(t.lower().split().count(w)) for w in VOCAB] for t in texts]
        return np.array(rows, dtype=np.float32)


def day(n):
    return datetime(2024, 1, n, tzinfo=UTC)


def cand(id_, title, repo="numpy/numpy", d=1):
    return Candidate(
        id=id_, repo=repo, kind="issue", title=title, body="", date=day(d)
    )


def query(title, repo="scipy/scipy", d=10, id_="q"):
    return Query(
        id=id_, repo=repo, title=title, body="", date=day(d), gold=frozenset()
    )


def test_scores_match_hand_calculation():
    pool = [cand("d1", "apple banana"), cand("d2", "apple"), cand("d3", "cherry")]
    result = EmbeddingIndex(pool, FakeEncoder()).search(query("apple"))
    assert [item[0] for item in result] == ["d2", "d1", "d3"]
    assert result[0][1] == pytest.approx(1.0)
    assert result[1][1] == pytest.approx(0.7071, abs=1e-3)
    assert result[2][1] == pytest.approx(0.0)


def test_new_candidate_is_never_returned():
    pool = [cand("old", "apple", d=2), cand("new", "apple", d=20)]
    result = EmbeddingIndex(pool, FakeEncoder()).search(query("apple", d=10))
    assert [item[0] for item in result] == ["old"]


def test_candidate_with_same_time_is_not_returned():
    pool = [cand("same", "apple", d=10)]
    assert EmbeddingIndex(pool, FakeEncoder()).search(query("apple", d=10)) == []


def test_same_repo_is_excluded_by_default():
    pool = [
        cand("own", "apple", repo="scipy/scipy"),
        cand("other", "apple", repo="numpy/numpy"),
    ]
    index = EmbeddingIndex(pool, FakeEncoder())
    assert [i[0] for i in index.search(query("apple"))] == ["other"]
    both = index.search(query("apple"), exclude_same_repo=False)
    assert {i[0] for i in both} == {"own", "other"}


def test_ties_are_sorted_by_id():
    pool = [cand("b", "apple"), cand("a", "apple"), cand("c", "apple")]
    result = EmbeddingIndex(pool, FakeEncoder()).search(query("apple"))
    assert [item[0] for item in result] == ["a", "b", "c"]


def test_top_k_cuts_the_list():
    pool = [cand(f"d{n}", "apple") for n in range(5)]
    result = EmbeddingIndex(pool, FakeEncoder()).search(query("apple"), top_k=2)
    assert len(result) == 2


def test_cache_avoids_second_encoding(tmp_path):
    pool = [cand("d1", "apple banana"), cand("d2", "cherry")]
    first = FakeEncoder()
    EmbeddingIndex(pool, first, EmbeddingCache(tmp_path, first.name))
    assert len(first.encoded) == 2

    second = FakeEncoder()
    index = EmbeddingIndex(pool, second, EmbeddingCache(tmp_path, second.name))
    assert second.encoded == []
    assert index.search(query("apple"))[0][0] == "d1"
    assert len(second.encoded) == 1


def test_cache_file_depends_on_encoder_name(tmp_path):
    a = EmbeddingCache(tmp_path, "model/a@256")
    b = EmbeddingCache(tmp_path, "model/a@512")
    assert a.path != b.path
    a.put("text", np.array([1.0, 2.0]))
    a.save()
    assert EmbeddingCache(tmp_path, "model/a@256").get("text") is not None
    assert EmbeddingCache(tmp_path, "model/a@512").get("text") is None
    assert not list(tmp_path.glob("*.tmp.npz"))


def test_same_text_is_encoded_once():
    encoder = FakeEncoder()
    vectors = embed_texts(["apple", "apple", "cherry"], encoder)
    assert vectors.shape == (3, 3)
    assert encoder.encoded == ["apple", "cherry"]


def test_wrong_encoder_shape_is_an_error():
    class Broken:
        name = "broken"

        def encode(self, texts):
            return np.zeros((1, 3), dtype=np.float32)

    with pytest.raises(ValueError):
        embed_texts(["a", "b"], Broken())


def test_bad_arguments():
    with pytest.raises(ValueError):
        EmbeddingIndex([], FakeEncoder())
    with pytest.raises(ValueError):
        EmbeddingIndex([cand("x", "a"), cand("x", "b")], FakeEncoder())
    index = EmbeddingIndex([cand("d1", "apple")], FakeEncoder())
    with pytest.raises(ValueError):
        index.search(query("apple"), top_k=0)


def test_rank_queries_works_with_the_metrics():
    pool = [cand("good", "apple banana"), cand("bad", "cherry")]
    q = Query(
        id="q1",
        repo="scipy/scipy",
        title="apple",
        body="banana",
        date=day(10),
        gold=frozenset({"good"}),
    )
    run = rank_queries(EmbeddingIndex(pool, FakeEncoder()), [q])
    assert run["q1"] == ["good", "bad"]
    result = evaluate(run, {"q1": q.gold}, ks=(1,))
    assert result["mrr"] == 1.0
    assert result["recall@1"] == 1.0


def test_system_is_in_the_runner_registry():
    assert "embedding" in SYSTEMS