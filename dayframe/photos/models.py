from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


class Asset(BaseModel):
    """One Photos library asset. Tests use JSON fixtures; production uses osxphotos."""

    model_config = ConfigDict(extra="forbid")

    uuid: str
    date: datetime
    date_added: datetime
    latitude: float | None = None
    longitude: float | None = None
    place_name: str | None = None
    screenshot: bool = False
    uti: str | None = None
    burst: bool = False
    burst_id: str | None = None
    persons: list[str] = Field(default_factory=list)
    score_overall: float | None = None
    score_well_timed: float | None = None
    path: str | None = None
    path_edited: str | None = None
    has_camera_exif: bool = False

    @property
    def coords(self) -> tuple[float, float] | None:
        if self.latitude is None or self.longitude is None:
            return None
        return (self.latitude, self.longitude)
