"""Data models for the AutoAnnotator."""

from __future__ import annotations

import hashlib
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

LABELS = ("cause", "related", "noise", "unclear")

Label = Literal["cause", "related", "noise", "unclear"]
Side = Literal["source", "target"]


class Pair(BaseModel):
    """One pair of items from two different repositories."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    source_repo: str = Field(min_length=1)
    source_url: str = Field(min_length=1)
    source_title: str = Field(min_length=1)
    source_body: str = ""
    target_repo: str = Field(min_length=1)
    target_url: str = Field(min_length=1)
    target_title: str = Field(min_length=1)
    target_body: str = ""

    @field_validator("source_body", "target_body", mode="before")
    @classmethod
    def _none_to_empty(cls, value: object) -> object:
        return "" if value is None else value

    @property
    def pair_id(self) -> str:
        key = f"{self.source_url}|{self.target_url}"
        return hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]


class Evidence(BaseModel):
    """One quote copied from the title or body of one item."""

    side: Side
    quote: str = Field(min_length=1, max_length=500)


class Verdict(BaseModel):
    """The answer the LLM must return for one pair."""

    label: Label
    confidence: float = Field(ge=0.0, le=1.0)
    evidence: list[Evidence] = Field(max_length=5)
    explanation: str = Field(min_length=1, max_length=800)