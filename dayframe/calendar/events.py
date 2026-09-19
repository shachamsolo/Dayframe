from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from dayframe.baseline.models import LabeledCluster, MemoryDraft
from dayframe.cluster.sessions import Cluster
from dayframe.config import CalendarConfig
from dayframe.window import ensure_aware

MIN_DURATION = timedelta(minutes=1)


class CalendarError(RuntimeError):
    """Auth or calendar-write failure with a user-facing message."""


@dataclass(frozen=True)
class MemoryEvent:
    event_id: str
    cluster_id: str
    summary: str
    description: str
    location: str | None
    start: datetime
    end: datetime
    timezone: str
    run_id: str
    confidence: float


def event_id_for(cluster_id: str) -> str:
    """SHA-1 hex is valid Google Calendar base32hex (`0-9`, `a-v`)."""
    return hashlib.sha1(f"dayframe:{cluster_id}".encode()).hexdigest()


def localize(value: datetime, timezone: str) -> datetime:
    try:
        zone = ZoneInfo(timezone)
    except ZoneInfoNotFoundError as exc:
        raise CalendarError(f"unknown calendar timezone {timezone!r}") from exc
    return ensure_aware(value).astimezone(zone)


def span(start: datetime, end: datetime) -> tuple[datetime, datetime]:
    if end <= start:
        return start, start + MIN_DURATION
    return start, end


def build_memory_event(
    memory: MemoryDraft,
    cluster: Cluster,
    cfg: CalendarConfig,
    run_id: str,
) -> MemoryEvent:
    start, end = span(localize(cluster.start, cfg.timezone), localize(cluster.end, cfg.timezone))
    prefix = cfg.title_prefix or ""
    return MemoryEvent(
        event_id=event_id_for(cluster.id),
        cluster_id=cluster.id,
        summary=f"{prefix}{memory.title}" if prefix else memory.title,
        description=memory.body,
        location=cluster.place,
        start=start,
        end=end,
        timezone=cfg.timezone,
        run_id=run_id,
        confidence=memory.confidence,
    )


def build_memory_events(
    memories: list[MemoryDraft],
    labeled: list[LabeledCluster],
    cfg: CalendarConfig,
    run_id: str,
) -> list[MemoryEvent]:
    by_label = {item.label: item.cluster for item in labeled}
    events: list[MemoryEvent] = []
    for memory in memories:
        cluster = by_label.get(memory.cluster_id)
        if cluster is None:
            raise CalendarError(f"memory refers to unknown cluster {memory.cluster_id}")
        events.append(build_memory_event(memory, cluster, cfg, run_id))
    return events


def event_body(event: MemoryEvent) -> dict[str, object]:
    body: dict[str, object] = {
        "id": event.event_id,
        "summary": event.summary,
        "description": event.description,
        "start": {
            "dateTime": event.start.isoformat(),
            "timeZone": event.timezone,
        },
        "end": {
            "dateTime": event.end.isoformat(),
            "timeZone": event.timezone,
        },
        "transparency": "transparent",
        "status": "confirmed",
        "extendedProperties": {
            "private": {
                "dayframe_run_id": event.run_id,
                "dayframe_cluster_id": event.cluster_id,
                "dayframe_confidence": f"{event.confidence:.2f}",
            }
        },
    }
    if event.location:
        body["location"] = event.location
    return body
