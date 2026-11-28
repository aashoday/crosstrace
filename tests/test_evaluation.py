import json
from datetime import UTC, datetime

import pytest

from crosstrace.benchmark import Candidate, Query, leave_one_project_out, time_split
from crosstrace.evaluation import (
    evaluate_split,
    format_table,
    load_candidates,
    load_queries,
    save_results,
)

TRACEBACK = (
    "Traceback (most recent call last):\n"
    '  File "/env/site-packages/numpy/mod.py", line 5, in f\n'
    "ValueError: bad value"
)


def day(n):
    return datetime(2024, 1, n, tzinfo=UTC)


def make_candidates():
    return [
        Candidate(
            id="np1",
            repo="numpy/numpy",
            kind="issue",
            title="numpy dtype change",
            body="",
            date=day(1),
        )
    ]


def make_queries(count=5, repo="scipy/scipy"):
    return [
        Query(
            id=f"{repo}-q{n}",
            repo=repo,
            title="numpy dtype fails",
            body=TRACEBACK,
            date=day(10 + n),
            gold=frozenset({"np1"}),
        )
        for n in range(count)
    ]


def test_evaluate_split_time():
    split = time_split(make_queries())
    result = evaluate_split(split, make_candidates(), ["bm25", "frame_rule"])
    assert result["split"] == "time"
    assert result["n_queries"] == len(split.test)
    for metrics in result["systems"].values():
        assert metrics["mrr"] == 1.0
        assert metrics["recall@1"] == 1.0
    assert result["systems"]["frame_rule"]["no_upstream_frame"] == 0.0


def test_evaluate_split_lopo():
    queries = make_queries(2, "scipy/scipy") + make_queries(2, "pandas-dev/pandas")
    splits = leave_one_project_out(queries)
    result = evaluate_split(splits["scipy/scipy"], make_candidates(), ["bm25"])
    assert result["split"] == "lopo:scipy/scipy"
    assert result["n_queries"] == 2


def test_unknown_system():
    split = time_split(make_queries())
    with pytest.raises(ValueError):
        evaluate_split(split, make_candidates(), ["nope"])


def test_format_table_has_system_names_and_counts():
    split = time_split(make_queries())
    result = evaluate_split(split, make_candidates(), ["bm25", "frame_rule"])
    text = format_table(result)
    assert "bm25" in text
    assert "frame_rule" in text
    assert "mrr" in text
    assert "no upstream frame" in text


def test_load_files(tmp_path):
    cand_path = tmp_path / "c.jsonl"
    query_path = tmp_path / "q.jsonl"
    cand_path.write_text(
        json.dumps(
            {
                "id": "np1",
                "repo": "numpy/numpy",
                "kind": "issue",
                "title": "t",
                "body": None,
                "date": "2024-01-01T00:00:00Z",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    query_path.write_text(
        json.dumps(
            {
                "id": "q1",
                "repo": "scipy/scipy",
                "title": "t",
                "body": "b",
                "date": "2024-02-01T00:00:00+00:00",
                "gold": ["np1"],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    candidates = load_candidates(cand_path)
    queries = load_queries(query_path)
    assert candidates[0].body == ""
    assert candidates[0].date.tzinfo is not None
    assert queries[0].gold == frozenset({"np1"})


def test_date_without_time_zone_is_an_error(tmp_path):
    path = tmp_path / "c.jsonl"
    row = {
        "id": "a",
        "repo": "r",
        "kind": "issue",
        "title": "t",
        "body": "",
        "date": "2024-01-01T00:00:00",
    }
    path.write_text(json.dumps(row) + "\n", encoding="utf-8")
    with pytest.raises(ValueError):
        load_candidates(path)


def test_save_results(tmp_path):
    split = time_split(make_queries())
    result = evaluate_split(split, make_candidates(), ["bm25"])
    out = tmp_path / "out" / "eval.json"
    save_results([result], out, {"split": "time"})
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["meta"]["split"] == "time"
    assert data["results"][0]["split"] == "time"