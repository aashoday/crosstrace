import pytest

from crosstrace.autoannotate.prompt import MAX_CHARS, prepare
from crosstrace.autoannotate.schema import Evidence, Pair, Verdict
from crosstrace.autoannotate.validate import (
    assess,
    quote_is_supported,
    unsupported_quotes,
)


def make_pair(**over):
    data = {
        "source_repo": "scipy/scipy",
        "source_url": "https://github.com/scipy/scipy/issues/1",
        "source_title": "Test fails with new numpy",
        "source_body": "The test fails since numpy 2.0 changed the dtype.",
        "target_repo": "numpy/numpy",
        "target_url": "https://github.com/numpy/numpy/issues/2",
        "target_title": "Change default dtype",
        "target_body": "We changed the default dtype in this release.",
    }
    data.update(over)
    return Pair(**data)


def make_verdict(label="cause", confidence=0.9, evidence=None, **over):
    if evidence is None:
        evidence = [Evidence(side="source", quote="numpy 2.0 changed the dtype")]
    return Verdict(
        label=label,
        confidence=confidence,
        evidence=evidence,
        explanation=over.get("explanation", "The source blames the change."),
    )


def test_good_verdict_is_accepted():
    result = assess(make_pair(), make_verdict())
    assert result.status == "accepted"
    assert not result.needs_review
    assert result.reasons == ()
    assert result.reason_text == ""


def test_quote_in_title_is_supported():
    evidence = Evidence(side="target", quote="Change default dtype")
    assert quote_is_supported(make_pair(), evidence)


def test_quote_in_body_is_supported():
    evidence = Evidence(side="target", quote="in this release")
    assert quote_is_supported(make_pair(), evidence)


def test_changed_quote_is_rejected():
    evidence = Evidence(side="source", quote="numpy 2.0 altered the dtype")
    assert not quote_is_supported(make_pair(), evidence)


def test_case_change_is_rejected():
    evidence = Evidence(side="source", quote="Numpy 2.0 changed the dtype")
    assert not quote_is_supported(make_pair(), evidence)


def test_quote_on_the_wrong_side_is_rejected():
    evidence = Evidence(side="target", quote="numpy 2.0 changed the dtype")
    assert not quote_is_supported(make_pair(), evidence)


def test_quote_cannot_cross_title_and_body():
    pair = make_pair(source_title="Title end", source_body="body start")
    evidence = Evidence(side="source", quote="Title end body start")
    assert not quote_is_supported(pair, evidence)


def test_unsupported_quotes_returns_only_bad_ones():
    good = Evidence(side="source", quote="numpy 2.0 changed the dtype")
    bad = Evidence(side="source", quote="this text is not there")
    verdict = make_verdict(evidence=[good, bad])
    assert unsupported_quotes(make_pair(), verdict) == [bad]


def test_trigger_invalid_output():
    result = assess(make_pair(), None)
    assert result.needs_review
    assert result.reasons == ("invalid_output",)


def test_trigger_unclear_label():
    verdict = make_verdict(label="unclear", evidence=[])
    result = assess(make_pair(), verdict)
    assert result.reasons == ("unclear_label",)


def test_trigger_low_confidence():
    result = assess(make_pair(), make_verdict(confidence=0.5))
    assert result.reasons == ("low_confidence",)


def test_confidence_at_the_limit_is_accepted():
    result = assess(make_pair(), make_verdict(confidence=0.7))
    assert result.status == "accepted"


def test_limit_is_a_setting():
    verdict = make_verdict(confidence=0.8)
    assert assess(make_pair(), verdict, min_confidence=0.9).needs_review
    assert not assess(make_pair(), verdict, min_confidence=0.5).needs_review


def test_trigger_unsupported_quote():
    good = Evidence(side="source", quote="numpy 2.0 changed the dtype")
    bad = Evidence(side="source", quote="this text is not there")
    result = assess(make_pair(), make_verdict(evidence=[good, bad]))
    assert result.reasons == ("unsupported_quote",)


def test_trigger_cause_without_any_quote():
    result = assess(make_pair(), make_verdict(evidence=[]))
    assert result.reasons == ("cause_without_evidence",)


def test_cause_with_only_bad_quotes_has_two_reasons():
    bad = Evidence(side="source", quote="this text is not there")
    result = assess(make_pair(), make_verdict(evidence=[bad]))
    assert result.reasons == ("unsupported_quote", "cause_without_evidence")


def test_related_without_quote_is_accepted():
    verdict = make_verdict(label="related", evidence=[])
    assert assess(make_pair(), verdict).status == "accepted"


def test_noise_without_quote_is_accepted():
    verdict = make_verdict(label="noise", evidence=[])
    assert assess(make_pair(), verdict).status == "accepted"


def test_reason_text_joins_all_reasons():
    bad = Evidence(side="source", quote="this text is not there")
    result = assess(make_pair(), make_verdict(confidence=0.1, evidence=[bad]))
    assert "confidence" in result.reason_text
    assert "quote" in result.reason_text


def test_quote_from_cut_text_is_rejected():
    body = "a" * MAX_CHARS + " hidden causal sentence"
    pair = prepare(make_pair(source_body=body))
    evidence = Evidence(side="source", quote="hidden causal sentence")
    assert not quote_is_supported(pair, evidence)


def test_bad_limit_is_an_error():
    with pytest.raises(ValueError):
        assess(make_pair(), make_verdict(), min_confidence=1.5)
    with pytest.raises(ValueError):
        assess(make_pair(), make_verdict(), min_confidence=-0.1)