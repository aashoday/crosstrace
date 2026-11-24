"""Evidence validation and review rules for the AutoAnnotator."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from crosstrace.autoannotate.schema import Evidence, Pair, Verdict

DEFAULT_MIN_CONFIDENCE = 0.7

Status = Literal["accepted", "review"]

REASON_TEXT = {
    "invalid_output": "The model output was invalid after all retries.",
    "unclear_label": "The label is unclear.",
    "low_confidence": "The confidence is below the limit.",
    "unsupported_quote": "A quote was not found in the text.",
    "cause_without_evidence": "The label is cause but no valid quote supports it.",
}


def _texts(pair: Pair, side: str) -> tuple[str, str]:
    if side == "source":
        return pair.source_title, pair.source_body
    return pair.target_title, pair.target_body


def quote_is_supported(pair: Pair, evidence: Evidence) -> bool:
    """True if the quote is an exact part of the title or the body."""
    title, body = _texts(pair, evidence.side)
    return evidence.quote in title or evidence.quote in body


def unsupported_quotes(pair: Pair, verdict: Verdict) -> list[Evidence]:
    """Return the quotes that are not found in the prepared text."""
    return [e for e in verdict.evidence if not quote_is_supported(pair, e)]


@dataclass(frozen=True)
class Assessment:
    """The result of the checks for one pair."""

    status: Status
    reasons: tuple[str, ...] = ()

    @property
    def needs_review(self) -> bool:
        return self.status == "review"

    @property
    def reason_text(self) -> str:
        return " ".join(REASON_TEXT[code] for code in self.reasons)


def assess(
    pair: Pair,
    verdict: Verdict | None,
    min_confidence: float = DEFAULT_MIN_CONFIDENCE,
) -> Assessment:
    """Check one verdict. Use None when the model gave no valid verdict."""
    if not 0.0 <= min_confidence <= 1.0:
        raise ValueError("min_confidence must be between 0 and 1")
    if verdict is None:
        return Assessment("review", ("invalid_output",))

    reasons: list[str] = []
    if verdict.label == "unclear":
        reasons.append("unclear_label")
    if verdict.confidence < min_confidence:
        reasons.append("low_confidence")

    bad = unsupported_quotes(pair, verdict)
    if bad:
        reasons.append("unsupported_quote")
    valid_count = len(verdict.evidence) - len(bad)
    if verdict.label == "cause" and valid_count == 0:
        reasons.append("cause_without_evidence")

    if reasons:
        return Assessment("review", tuple(reasons))
    return Assessment("accepted")