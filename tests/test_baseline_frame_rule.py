from datetime import UTC, datetime

import pytest

from crosstrace.baselines.frame_rule import FrameRule, rank_queries
from crosstrace.benchmark import Candidate, Query
from crosstrace.metrics import evaluate


def day(n):
    return datetime(2024, 1, n, tzinfo=UTC)


def cand(id_, repo="numpy/numpy", d=1, kind="issue"):
    return Candidate(id=id_, repo=repo, kind=kind, title="t", body="b", date=day(d))


def traceback(*packages):
    lines = [
        "Traceback (most recent call last):",
        '  File "test.py", line 1, in <module>',
    ]
    for package in packages:
        lines.append(f'  File "/env/site-packages/{package}/mod.py", line 5, in f')
    lines.append("ValueError: bad value")
    return "\n".join(lines)


def query(body, repo="scipy/scipy", d=10, id_="q"):
    return Query(
        id=id_, repo=repo, title="fails", body=body, date=day(d), gold=frozenset()
    )


def test_newest_first():
    pool = [cand("a", d=2), cand("b", d=5), cand("c", d=3)]
    result = FrameRule(pool).search(query(traceback("numpy")))
    assert result == ["b", "c", "a"]


def test_only_repos_from_the_frames():
    pool = [
        cand("np1", repo="numpy/numpy"),
        cand("pd1", repo="pandas-dev/pandas"),
    ]
    result = FrameRule(pool).search(query(traceback("numpy")))
    assert result == ["np1"]


def test_two_packages_are_both_used():
    pool = [
        cand("np1", repo="numpy/numpy", d=2),
        cand("pd1", repo="pandas-dev/pandas", d=4),
    ]
    result = FrameRule(pool).search(query(traceback("numpy", "pandas")))
    assert result == ["pd1", "np1"]


def test_query_without_traceback_gets_empty_list():
    pool = [cand("np1")]
    assert FrameRule(pool).search(query("I get a wrong result.")) == []


def test_frames_without_a_package_get_empty_list():
    text = (
        "Traceback (most recent call last):\n"
        '  File "<stdin>", line 1, in <module>\n'
        "ValueError: bad value"
    )
    assert FrameRule([cand("np1")]).search(query(text)) == []


def test_own_package_alone_is_no_upstream_frame():
    pool = [cand("sp1", repo="scipy/scipy"), cand("np1")]
    rule = FrameRule(pool)
    q = query(traceback("scipy"), repo="scipy/scipy")
    assert rule.search(q) == []
    assert rule.count_without_upstream_frame([q]) == 1


def test_own_package_is_dropped_but_others_stay():
    pool = [cand("sp1", repo="scipy/scipy", d=2), cand("np1", d=3)]
    result = FrameRule(pool).search(query(traceback("scipy", "numpy")))
    assert result == ["np1"]


def test_unknown_package_is_ignored():
    pool = [cand("np1")]
    assert FrameRule(pool).search(query(traceback("requests"))) == []


def test_time_rule():
    pool = [cand("old", d=2), cand("new", d=20), cand("same", d=10)]
    result = FrameRule(pool).search(query(traceback("numpy"), d=10))
    assert result == ["old"]


def test_all_kinds_are_candidates():
    pool = [
        cand("issue1", d=2, kind="issue"),
        cand("pr1", d=3, kind="pr"),
        cand("rel1", d=4, kind="release"),
    ]
    result = FrameRule(pool).search(query(traceback("numpy")))
    assert result == ["rel1", "pr1", "issue1"]


def test_ties_are_sorted_by_id():
    pool = [cand("b", d=3), cand("a", d=3), cand("c", d=3)]
    result = FrameRule(pool).search(query(traceback("numpy")))
    assert result == ["a", "b", "c"]


def test_top_k_cuts_the_list():
    pool = [cand(f"c{n}", d=n) for n in range(1, 6)]
    result = FrameRule(pool).search(query(traceback("numpy")), top_k=2)
    assert result == ["c5", "c4"]


def test_custom_mapping():
    pool = [cand("x1", repo="org/mylib")]
    rule = FrameRule(pool, package_repos={"mylib": "org/mylib"})
    assert rule.search(query(traceback("mylib"))) == ["x1"]
    assert rule.search(query(traceback("numpy"))) == []


def test_count_without_upstream_frame():
    rule = FrameRule([cand("np1")])
    queries = [
        query(traceback("numpy"), id_="q1"),
        query("no traceback", id_="q2"),
        query(traceback("requests"), id_="q3"),
    ]
    assert rule.count_without_upstream_frame(queries) == 2


def test_duplicate_ids_are_an_error():
    with pytest.raises(ValueError):
        FrameRule([cand("x"), cand("x")])


def test_bad_top_k():
    with pytest.raises(ValueError):
        FrameRule([cand("a")]).search(query(traceback("numpy")), top_k=0)


def test_rank_queries_works_with_the_metrics():
    pool = [cand("good", d=5), cand("other", d=3)]
    q = Query(
        id="q1",
        repo="scipy/scipy",
        title="fails",
        body=traceback("numpy"),
        date=day(10),
        gold=frozenset({"other"}),
    )
    run = rank_queries(FrameRule(pool), [q])
    assert run == {"q1": ["good", "other"]}
    result = evaluate(run, {"q1": q.gold}, ks=(1, 2))
    assert result["mrr"] == pytest.approx(0.5)
    assert result["recall@1"] == 0.0
    assert result["recall@2"] == 1.0