from __future__ import annotations

import re
import tomllib
from pathlib import Path
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator

from dayframe.paths import config_path

_HHMM = re.compile(r"^(?:[01]\d|2[0-3]):[0-5]\d$")


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ProviderConfig(_Strict):
    name: str = "anthropic"
    model: str = "claude-sonnet-4-5"


class WindowConfig(_Strict):
    mode: Literal["previous_calendar_day"] = "previous_calendar_day"
    basis: Literal["date_added"] = "date_added"
    run_at: str = "07:00"

    @field_validator("run_at")
    @classmethod
    def _hhmm(cls, value: str) -> str:
        if not _HHMM.match(value):
            raise ValueError("run_at must be HH:MM in 24-hour local time")
        return value


class ClusterConfig(_Strict):
    time_gap_minutes: int = Field(default=90, gt=0)
    geo_gap_km: float = Field(default=1.5, gt=0)
    min_gap_for_geo_split_minutes: int = Field(default=20, gt=0)


class BudgetConfig(_Strict):
    max_turns: int = Field(default=12, gt=0)
    max_images: int = Field(default=16, gt=0)
    max_cost_usd: float = Field(default=0.25, gt=0)
    timeout_seconds: int = Field(default=180, gt=0)


class PrivacyConfig(_Strict):
    send_coordinates: bool = False
    strip_exif: bool = True
    max_image_px: int = Field(default=1024, ge=256)


class CalendarConfig(_Strict):
    provider: Literal["google"] = "google"
    name: str = "Dayframe"
    calendar_id: str = ""
    timezone: str = "Asia/Jerusalem"
    approval_mode: Literal["auto", "review"] = "auto"
    min_confidence: float = Field(default=0.7, ge=0, le=1)
    title_prefix: str = "Dayframe: "
    language: Literal["en"] = "en"


class Config(_Strict):
    provider: ProviderConfig = Field(default_factory=ProviderConfig)
    window: WindowConfig = Field(default_factory=WindowConfig)
    cluster: ClusterConfig = Field(default_factory=ClusterConfig)
    budget: BudgetConfig = Field(default_factory=BudgetConfig)
    privacy: PrivacyConfig = Field(default_factory=PrivacyConfig)
    calendar: CalendarConfig = Field(default_factory=CalendarConfig)

    @classmethod
    def load(cls, path: Path | None = None) -> Self:
        target = path or config_path()
        if not target.exists():
            raise FileNotFoundError(
                f"config not found: {target}\n"
                "copy config.example.toml to ~/Dayframe/config.toml and edit"
            )
        data = tomllib.loads(target.read_text(encoding="utf-8"))
        return cls.model_validate(data)
