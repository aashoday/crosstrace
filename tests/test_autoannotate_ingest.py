import csv
import json

import pytest

from crosstrace.autoannotate.ingest import read_pairs
from crosstrace.autoannotate.schema import Pair

ROW = {
    "source_repo": "scipy/scipy",
    "source_url": "https://github.com/scipy/scipy/issues/1",
    "source_title": "Test fails",
    "source_body": "It fails.",
    "target_repo": "numpy/numpy",
    "target_url": "https://github.com/numpy/numpy/issues/2",
    "target_title": "Change in dtype",
    "target_body": "We changed dtype.",
}


def write_jsonl(path, rows):
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")


def test_read_jsonl_ignores_extra_columns(tmp_path):
    path = tmp_path / "pairs.jsonl"
    write_jsonl(path, [{**ROW, "extra": 1}])
    pairs, problems = read_pairs(path)
    assert len(pairs) == 1
    assert problems == []
    assert pairs[0].source_repo == "scipy/scipy"


def test_read_csv(tmp_path):
    path = tmp_path / "pairs.csv"
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(ROW))
        writer.writeheader()
        writer.writerow(ROW)
    pairs, problems = read_pairs(path)
    assert len(pairs) == 1
    assert problems == []


def test_missing_field_is_reported(tmp_path):
    path = tmp_path / "pairs.jsonl"
    write_jsonl(path, [{**ROW, "target_title": ""}, ROW])
    pairs, problems = read_pairs(path)
    assert len(pairs) == 1
    assert len(problems) == 1
    assert "line 1" in problems[0]
    assert "target_title" in problems[0]


def test_duplicate_is_skipped(tmp_path):
    path = tmp_path / "pairs.jsonl"
    write_jsonl(path, [ROW, ROW])
    pairs, problems = read_pairs(path)
    assert len(pairs) == 1
    assert "duplicate" in problems[0]


def test_invalid_json_line_is_reported(tmp_path):
    path = tmp_path / "pairs.jsonl"
    path.write_text(json.dumps(ROW) + "\n{not json\n", encoding="utf-8")
    pairs, problems = read_pairs(path)
    assert len(pairs) == 1
    assert "line 2" in problems[0]


def test_null_body_becomes_empty(tmp_path):
    path = tmp_path / "pairs.jsonl"
    write_jsonl(path, [{**ROW, "source_body": None}])
    pairs, _ = read_pairs(path)
    assert pairs[0].source_body == ""


def test_unsupported_suffix(tmp_path):
    path = tmp_path / "pairs.txt"
    path.write_text("x", encoding="utf-8")
    with pytest.raises(ValueError):
        read_pairs(path)


def test_pair_id_is_stable_and_depends_on_direction():
    a = Pair(**ROW)
    b = Pair(**ROW)
    swapped = Pair(
        **{
            **ROW,
            "source_url": ROW["target_url"],
            "target_url": ROW["source_url"],
        }
    )
    assert a.pair_id == b.pair_id
    assert a.pair_id != swapped.pair_id