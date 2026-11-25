from datetime import UTC, datetime

import pytest

from crosstrace.benchmark import (
    Candidate,
    Query,
    clean_queries,
    eligible,
    is_older,
    leave_one_project_out,
    time_split,
)


def day(n):
    return datetime(2024, 1, n, tzinfo=UTC)


def cand(id_, repo="numpy/numpy", d=1, kind="issue"):
    return Candidate(id=id_, repo=repo, kind=kind, title="t", body="b", date=day(d))


def query(id_, repo="scipy/scipy", d=10, gold=()):
    return Query(
        id=id_, repo=repo, title="t", body="b", date=day(d), gold=frozenset(gold)
    )


def test_naive_date_is_an_error():
    naive = datetime(2024, 1, 1)
    with pytest.raises(ValueError):
        Candidate(id="a", repo="r", kind="issue", title="t", body="b", date=naive)
    with pytest.raises(ValueError):
        Query(id="q", repo="r", title="t", body="b", date=naive, gold=frozenset())


def test_is_older_is_strict():
    q = query("q", d=5)
    assert is_older(cand("a", d=4), q)
    assert not is_older(cand("b", d=5), q)
    assert not is_older(cand("c", d=6), q)


def test_eligible_removes_new_and_same_repo():
    q = query("q", repo="scipy/scipy", d=5)
    pool = [
        cand("old", d=2),
        cand("new", d=7),
        cand("same-time", d=5),
        cand("own", repo="scipy/scipy", d=2),
    ]
    assert [c.id for c in eligible(q, pool)] == ["old"]
    both = eligible(q, pool, exclude_same_repo=False)
    assert [c.id for c in both] == ["old", "own"]


def test_no_candidate_is_newer_than_its_query():
    pool = [cand(str(n), d=n) for n in range(1, 20)]
    for d in (1, 5, 12, 25):
        q = query("q", d=d)
        assert all(c.date < q.date for c in eligible(q, pool))


def test_clean_keeps_valid_gold():
    pool = [cand("a", d=2), cand("b", d=3)]
    kept, report = clean_queries([query("q", d=10, gold={"a", "b"})], pool)
    assert len(kept) == 1
    assert kept[0].gold == frozenset({"a", "b"})
    assert report.queries_in == 1
    assert report.queries_kept == 1
    assert report.gold_not_found == 0
    assert report.gold_ineligible == 0


def test_clean_drops_unknown_gold():
    pool = [cand("a", d=2)]
    kept, report = clean_queries([query("q", gold={"a", "zzz"})], pool)
    assert kept[0].gold == frozenset({"a"})
    assert report.gold_not_found == 1


def test_clean_drops_gold_that_is_too_new():
    pool = [cand("a", d=2), cand("late", d=20)]
    kept, report = clean_queries([query("q", d=10, gold={"a", "late"})], pool)
    assert kept[0].gold == frozenset({"a"})
    assert report.gold_ineligible == 1


def test_clean_drops_gold_from_same_repo():
    pool = [cand("own", repo="scipy/scipy", d=2)]
    kept, report = clean_queries([query("q", repo="scipy/scipy", gold={"own"})], pool)
    assert kept == []
    assert report.gold_ineligible == 1


def test_clean_removes_query_without_gold():
    pool = [cand("a", d=2)]
    kept, report = clean_queries(
        [query("q1", gold={"a"}), query("q2", gold={"zzz"}), query("q3")], pool
    )
    assert [q.id for q in kept] == ["q1"]
    assert report.queries_in == 3
    assert report.queries_kept == 1


def make_queries(count=10):
    return [query(f"q{n}", d=n, gold={"a"}) for n in range(1, count + 1)]


def test_time_split_newest_are_test():
    split = time_split(make_queries(10), test_fraction=0.2)
    assert split.name == "time"
    assert [q.id for q in split.test] == ["q9", "q10"]
    assert len(split.train) == 8
    assert max(q.date for q in split.train) < min(q.date for q in split.test)


def test_time_split_ignores_input_order():
    queries = list(reversed(make_queries(10)))
    split = time_split(queries)
    assert [q.id for q in split.test] == ["q9", "q10"]


def test_time_split_keeps_same_date_together():
    queries = [
        query("a", d=1),
        query("b", d=2),
        query("c", d=3),
        query("d", d=3),
        query("e", d=3),
    ]
    split = time_split(queries, test_fraction=0.4)
    assert [q.id for q in split.train] == ["a", "b"]
    assert [q.id for q in split.test] == ["c", "d", "e"]


def test_time_split_same_date_for_all_is_an_error():
    queries = [query(f"q{n}", d=5) for n in range(4)]
    with pytest.raises(ValueError):
        time_split(queries)


def test_time_split_needs_two_queries():
    with pytest.raises(ValueError):
        time_split([query("only")])


def test_time_split_bad_fraction():
    for bad in (0.0, 1.0, -0.5, 2.0):
        with pytest.raises(ValueError):
            time_split(make_queries(), test_fraction=bad)


def test_leave_one_project_out():
    queries = [
        query("a1", repo="astropy/astropy"),
        query("a2", repo="astropy/astropy"),
        query("s1", repo="scipy/scipy"),
        query("p1", repo="pandas-dev/pandas"),
    ]
    splits = leave_one_project_out(queries)
    assert set(splits) == {"astropy/astropy", "scipy/scipy", "pandas-dev/pandas"}
    for repo, split in splits.items():
        assert split.name == f"lopo:{repo}"
        assert {q.repo for q in split.test} == {repo}
        assert repo not in {q.repo for q in split.train}
        assert len(split.train) + len(split.test) == len(queries)
    test_ids = sorted(q.id for s in splits.values() for q in s.test)
    assert test_ids == sorted(q.id for q in queries)


def test_leave_one_project_out_needs_two_repositories():
    with pytest.raises(ValueError):
        leave_one_project_out([query("a", repo="r"), query("b", repo="r")])