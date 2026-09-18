from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime

from dayframe.cluster.sessions import Cluster, cluster_assets
from dayframe.config import Config
from dayframe.paths import config_path
from dayframe.photos.models import Asset
from dayframe.photos.reader import PhotosReader
from dayframe.prefilter.rules import Dropped, PrefilterResult, prefilter
from dayframe.window import calendar_day_window, parse_target_date


def load_config_optional() -> Config:
    path = config_path()
    if path.exists():
        return Config.load(path)
    return Config()


def inspect_run_id(target_date: date) -> str:
    return f"run_{target_date.isoformat()}"


@dataclass
class PipelineResult:
    run_id: str
    target_date: date
    start: datetime
    end: datetime
    raw: list[Asset]
    prefilter: PrefilterResult
    clusters: list[Cluster]

    @property
    def kept(self) -> list[Asset]:
        return self.prefilter.kept

    @property
    def dropped(self) -> list[Dropped]:
        return self.prefilter.dropped


def run_photos_window(
    reader: PhotosReader,
    start: datetime,
    end: datetime,
    *,
    cfg: Config | None = None,
    seen: set[str] | None = None,
    run_id: str | None = None,
    target_date: date | None = None,
) -> PipelineResult:
    config = cfg or load_config_optional()
    raw = reader.assets_added_between(start, end)
    filtered = prefilter(raw, seen)
    clusters = cluster_assets(filtered.kept, config.cluster)
    day = target_date or start.date()
    return PipelineResult(
        run_id=run_id or inspect_run_id(day),
        target_date=day,
        start=start,
        end=end,
        raw=raw,
        prefilter=filtered,
        clusters=clusters,
    )


def run_for_date(
    reader: PhotosReader,
    target: str,
    *,
    cfg: Config | None = None,
    seen: set[str] | None = None,
    now: datetime | None = None,
) -> PipelineResult:
    config = cfg or load_config_optional()
    day = parse_target_date(target, now=now)
    start, end = calendar_day_window(day)
    return run_photos_window(
        reader,
        start,
        end,
        cfg=config,
        seen=seen,
        run_id=inspect_run_id(day),
        target_date=day,
    )
