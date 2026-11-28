from datetime import UTC, datetime

import pytest

from crosstrace.baselines.bm25 import BM25Index, rank_queries, tokenize
from crosstrace.benchmark import Candidate, Query
from crosstrace.metrics import evaluate


def day(n):
    return datetime(2024, 1, n, tzinfo=UTC)


def cand(id_, title, body="", repo="numpy/numpy", d=1):
    return Candidate(
        id=id_, repo=repo, kind="issue", title=title, body=body, date=day(d)
    )


def query(title, body="", repo="scipy/scipy", d=10, id_="q"):
    return Query(
        id=id_, repo=repo, title=title, body=body, date=day(d), gold=frozenset()
    )


def test_tokenize():
    assert tokenize("np.linalg.inv(A) fails_2x") == [
        "np",
        "linalg",
        "inv",
        "a",
        "fails",
        "2x",
    ]
    assert tokenize("") == []


def test_scores_match_hand_calculation():
    pool = [
        cand("d1", "apple banana"),
        cand("d2", "apple"),
        cand("d3", "cherry"),
    ]
    result = BM25Index(pool).search(query("apple"))
    assert [item[0] for item in result] == ["d2", "d1"]
    assert result[0][1] == pytest.approx(0.5296, abs=1e-3)
    assert result[1][1] == pytest.approx(0.3837, abs=1e-3)


def test_rare_term_beats_common_term():
    pool = [
        cand("d1", "rare common"),
        cand("d2", "common"),
        cand("d3", "common"),
        cand("d4", "common other"),
    ]
    result = BM25Index(pool).search(query("rare common"))
    assert result[0][0] == "d1"


def test_repeated_query_terms_count_once():
    pool = [cand("d1", "apple"), cand("d2", "cherry")]
    index = BM25Index(pool)
    once = index.search(query("apple"))
    many = index.search(query("apple apple apple"))
    assert once == many


def test_new_candidate_is_never_returned():
    pool = [cand("old", "apple", d=2), cand("new", "apple apple", d=20)]
    result = BM25Index(pool).search(query("apple", d=10))
    assert [item[0] for item in result] == ["old"]


def test_candidate_with_same_time_is_not_returned():
    pool = [cand("same", "apple", d=10)]
    assert BM25Index(pool).search(query("apple", d=10)) == []


def test_same_repo_is_excluded_by_default():
    pool = [
        cand("own", "apple", repo="scipy/scipy"),
        cand("other", "apple", repo="numpy/numpy"),
    ]
    index = BM25Index(pool)
    assert [i[0] for i in index.search(query("apple"))] == ["other"]
    both = index.search(query("apple"), exclude_same_repo=False)
    assert {i[0] for i in both} == {"own", "other"}


def test_candidates_without_a_match_are_not_returned():
    pool = [cand("d1", "apple"), cand("d2", "cherry")]
    result = BM25Index(pool).search(query("apple"))
    assert [item[0] for item in result] == ["d1"]


def test_query_without_a_match_gives_empty_list():
    pool = [cand("d1", "apple")]
    assert BM25Index(pool).search(query("zzz")) == []


def test_ties_are_sorted_by_id():
    pool = [cand("b", "apple"), cand("a", "apple"), cand("c", "apple")]
    result = BM25Index(pool).search(query("apple"))
    assert [item[0] for item in result] == ["a", "b", "c"]


def test_top_k_cuts_the_list():
    pool = [cand(f"d{n}", "apple") for n in range(5)]
    result = BM25Index(pool).search(query("apple"), top_k=2)
    assert len(result) == 2


def test_body_text_is_used():
    pool = [cand("d1", "title", body="the dtype changed")]
    result = BM25Index(pool).search(query("dtype"))
    assert [item[0] for item in result] == ["d1"]


def test_bad_arguments():
    pool = [cand("d1", "apple")]
    with pytest.raises(ValueError):
        BM25Index([])
    with pytest.raises(ValueError):
        BM25Index([cand("x", "a"), cand("x", "b")])
    with pytest.raises(ValueError):
        BM25Index(pool, k1=-1)
    with pytest.raises(ValueError):
        BM25Index(pool, b=1.5)
    with pytest.raises(ValueError):
        BM25Index(pool).search(query("apple"), top_k=0)


def test_rank_queries_works_with_the_metrics():
    pool = [
        cand("good", "linalg inv singular matrix"),
        cand("bad", "plot colors"),
    ]
    q = Query(
        id="q1",
        repo="scipy/scipy",
        title="inv fails",
        body="singular matrix in linalg",
        date=day(10),
        gold=frozenset({"good"}),
    )
    run = rank_queries(BM25Index(pool), [q])
    assert run == {"q1": ["good"]}
    result = evaluate(run, {"q1": q.gold}, ks=(1,))
    assert result["mrr"] == 1.0
    assert result["recall@1"] == 1.0