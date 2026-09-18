from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from dayframe.cluster.sessions import Cluster

CATEGORIES = (
    "trip",
    "family",
    "meal",
    "celebration",
    "outing",
    "hobby",
    "nature",
    "other",
)

Category = Literal[
    "trip",
    "family",
    "meal",
    "celebration",
    "outing",
    "hobby",
    "nature",
    "other",
]


class _Strict(BaseModel):
    model_config = ConfigDict(extra="ignore")


class MemoryDraft(_Strict):
    cluster_id: str
    title: str
    body: str
    category: Category
    confidence: float = Field(ge=0, le=1)

    @field_validator("cluster_id", "title", "body", mode="before")
    @classmethod
    def _strip(cls, value: object) -> object:
        return value.strip() if isinstance(value, str) else value

    @field_validator("title")
    @classmethod
    def _title(cls, value: str) -> str:
        return value[:70]

    @field_validator("body")
    @classmethod
    def _body(cls, value: str) -> str:
        return value[:300]


class DiscardDraft(_Strict):
    cluster_id: str
    reason: str

    @field_validator("cluster_id", "reason", mode="before")
    @classmethod
    def _strip(cls, value: object) -> object:
        return value.strip() if isinstance(value, str) else value


class DaySubmission(_Strict):
    memories: list[MemoryDraft] = Field(default_factory=list)
    discards: list[DiscardDraft] = Field(default_factory=list)


@dataclass
class LabeledCluster:
    label: str
    seq: int
    cluster: Cluster


@dataclass
class AttachedImage:
    label: str
    asset_uuid: str
    captured_at: str
    place: str | None
    jpeg_bytes: bytes
    media_type: str = "image/jpeg"


@dataclass
class BaselineResult:
    labeled: list[LabeledCluster]
    memories: list[MemoryDraft]
    discarded: list[DiscardDraft]
    skipped_unreviewed: list[str]
    images_sent: int
    input_tokens: int
    output_tokens: int
    cost_usd: float
    wall_seconds: float
    model: str
    prompt_version: str = "baseline_v1"
    raw_response: dict = field(default_factory=dict)
