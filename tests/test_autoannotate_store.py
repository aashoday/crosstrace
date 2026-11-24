from datetime import UTC, datetime

from crosstrace.autoannotate.schema import Pair, Verdict
from crosstrace.autoannotate.store import (
    AnnotationRecord,
    AnnotationStore,
    make_record,
)
from crosstrace.autoannotate.validate import assess

MODEL = "test-model"
VERSION = "v1"


def make_pair(n=1):
    return Pair(
        source_repo="scipy/scipy",
        source_url=f"https://github.com/scipy/scipy/issues/{n}",
        source_title="Test fails",
        source_body="It fails.",
        target_repo="numpy/numpy",
        target_url=f"https://github.com/numpy/numpy/issues/{n}",
        target_title="Change in dtype",
        target_body="We changed dtype.",
    )


def good_verdict():
    return Verdict(
        label="related",
        confidence=0.9,
        evidence=[],
        explanation="Same topic.",
    )


def record_for(pair, verdict="default", model=MODEL, version=VERSION):
    if verdict == "default":
        verdict = good_verdict()
    return make_record(pair, verdict, assess(pair, verdict), model, version)


def test_make_record_for_accepted_verdict():
    pair = make_pair()
    verdict = good_verdict()
    now = datetime(2026, 11, 24, 12, 0, tzinfo=UTC)
    record = make_record(pair, verdict, assess(pair, verdict), MODEL, VERSION, now)
    assert record.pair_id == pair.pair_id
    assert record.label == "related"
    assert record.status == "accepted"
    assert record.review_reasons == []
    assert record.model == MODEL
    assert record.prompt_version == VERSION
    assert record.created_at == "2026-11-24T12:00:00+00:00"
    assert record.completed


def test_make_record_without_verdict():
    pair = make_pair()
    record = make_record(pair, None, assess(pair, None), MODEL, VERSION)
    assert record.label is None
    assert record.confidence is None
    assert record.status == "review"
    assert record.review_reasons == ["invalid_output"]
    assert not record.completed


def test_append_and_reload(tmp_path):
    path = tmp_path / "annotated.jsonl"
    pair = make_pair()
    store = AnnotationStore(path)
    store.append(record_for(pair))
    assert store.is_done(pair.pair_id, MODEL, VERSION)

    reloaded = AnnotationStore(path)
    assert reloaded.is_done(pair.pair_id, MODEL, VERSION)
    assert len(reloaded.records()) == 1
    assert reloaded.records()[0].label == "related"


def test_one_json_line_per_record(tmp_path):
    path = tmp_path / "annotated.jsonl"
    store = AnnotationStore(path)
    store.append(record_for(make_pair(1)))
    store.append(record_for(make_pair(2)))
    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    for line in lines:
        AnnotationRecord.model_validate_json(line)


def test_creates_parent_folder(tmp_path):
    path = tmp_path / "out" / "annotated.jsonl"
    AnnotationStore(path).append(record_for(make_pair()))
    assert path.exists()


def test_resume_after_stop(tmp_path):
    path = tmp_path / "annotated.jsonl"
    pairs = [make_pair(n) for n in range(1, 5)]

    first_run = AnnotationStore(path)
    for pair in first_run.pending(pairs, MODEL, VERSION)[:2]:
        first_run.append(record_for(pair))

    second_run = AnnotationStore(path)
    todo = second_run.pending(pairs, MODEL, VERSION)
    assert [p.pair_id for p in todo] == [p.pair_id for p in pairs[2:]]
    for pair in todo:
        second_run.append(record_for(pair))

    final = AnnotationStore(path)
    assert final.pending(pairs, MODEL, VERSION) == []
    assert len(final.records()) == 4
    assert len(path.read_text(encoding="utf-8").splitlines()) == 4


def test_new_prompt_version_annotates_again(tmp_path):
    store = AnnotationStore(tmp_path / "annotated.jsonl")
    pair = make_pair()
    store.append(record_for(pair, version="v1"))
    assert not store.is_done(pair.pair_id, MODEL, "v2")
    assert store.pending([pair], MODEL, "v2") == [pair]


def test_new_model_annotates_again(tmp_path):
    store = AnnotationStore(tmp_path / "annotated.jsonl")
    pair = make_pair()
    store.append(record_for(pair, model="model-a"))
    assert not store.is_done(pair.pair_id, "model-b", VERSION)


def test_failed_record_is_retried_and_replaced(tmp_path):
    path = tmp_path / "annotated.jsonl"
    pair = make_pair()
    store = AnnotationStore(path)
    store.append(record_for(pair, verdict=None))
    assert not store.is_done(pair.pair_id, MODEL, VERSION)
    assert store.pending([pair], MODEL, VERSION) == [pair]

    store.append(record_for(pair))
    reloaded = AnnotationStore(path)
    assert reloaded.is_done(pair.pair_id, MODEL, VERSION)
    assert len(reloaded.records()) == 1
    assert reloaded.records()[0].label == "related"
    assert len(path.read_text(encoding="utf-8").splitlines()) == 2


def test_half_written_last_line_is_skipped(tmp_path):
    path = tmp_path / "annotated.jsonl"
    store = AnnotationStore(path)
    store.append(record_for(make_pair(1)))
    with path.open("a", encoding="utf-8") as f:
        f.write('{"pair_id": "abc", "lab')

    after_crash = AnnotationStore(path)
    assert after_crash.skipped_lines == 1
    assert len(after_crash.records()) == 1

    after_crash.append(record_for(make_pair(2)))
    final = AnnotationStore(path)
    assert final.skipped_lines == 1
    assert len(final.records()) == 2
    assert len(path.read_text(encoding="utf-8").splitlines()) == 3


def test_pending_keeps_the_input_order(tmp_path):
    store = AnnotationStore(tmp_path / "annotated.jsonl")
    pairs = [make_pair(n) for n in (3, 1, 2)]
    store.append(record_for(pairs[1]))
    todo = store.pending(pairs, MODEL, VERSION)
    assert [p.pair_id for p in todo] == [pairs[0].pair_id, pairs[2].pair_id]