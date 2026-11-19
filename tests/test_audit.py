from crosstrace.audit import (
    context,
    load_labels,
    next_unlabeled,
    sample_pairs,
    save_label,
)
from crosstrace.db import connect

ROWS = [
    ("a/x", 1, "Issue", "b/y", 2, "2024-01-01", 0),
    ("a/x", 3, "Issue", "c/z", 4, "2024-01-02", 0),
    ("a/x", 5, "Issue", "b/y", 6, "2024-01-03", 0),
]


def make_conn(tmp_path):
    conn = connect(tmp_path / "test.sqlite")
    conn.executemany("INSERT INTO cross_refs VALUES (?, ?, ?, ?, ?, ?, ?)", ROWS)
    conn.commit()
    return conn


def test_sample_keeps_only_listed_repos(tmp_path):
    pairs = sample_pairs(make_conn(tmp_path), {"a/x", "b/y"})
    assert sorted(pairs) == [("a/x", 1, "b/y", 2), ("a/x", 5, "b/y", 6)]


def test_sample_is_repeatable(tmp_path):
    conn = make_conn(tmp_path)
    repos = {"a/x", "b/y", "c/z"}
    assert sample_pairs(conn, repos, seed=1) == sample_pairs(conn, repos, seed=1)


def test_labels_roundtrip_and_next(tmp_path):
    path = tmp_path / "labels" / "audit.csv"
    pairs = [("a/x", 1, "b/y", 2), ("a/x", 5, "b/y", 6)]

    save_label(pairs[0], "cause", "a note", path=path)
    labels = load_labels(path)
    assert labels == {pairs[0]: "cause"}
    assert next_unlabeled(pairs, labels) == pairs[1]

    save_label(pairs[1], "noise", path=path)
    assert next_unlabeled(pairs, load_labels(path)) is None


def test_context_finds_reference():
    text = "x" * 300 + " see numpy/numpy#123 for details " + "y" * 300
    snippet = context(text, "numpy/numpy", 123, width=40)
    assert "numpy/numpy#123" in snippet
    assert len(snippet) < 120


def test_context_is_none_when_missing():
    assert context("nothing here", "numpy/numpy", 1) is None