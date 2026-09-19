from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from google.oauth2.credentials import Credentials
from tenacity.wait import wait_base

from dayframe.baseline.models import LabeledCluster, MemoryDraft
from dayframe.calendar.auth import load_credentials
from dayframe.calendar.events import CalendarError, MemoryEvent, build_memory_events
from dayframe.calendar.google import GoogleWriter, build_api
from dayframe.calendar.ics import write_ics
from dayframe.config import Config
from dayframe.paths import out_dir
from dayframe.store.db import connect, delete_run, fetch_run, init_db

OnPublished = Callable[[str, str], None]


@dataclass
class PublishResult:
    backend: Literal["google", "ics"]
    events: list[MemoryEvent]
    calendar_id: str | None = None
    ics_path: Path | None = None


@dataclass
class UndoResult:
    run_id: str
    google_deleted: int = 0
    ics_deleted: bool = False
    local_deleted: bool = False
    warnings: list[str] = field(default_factory=list)

    @property
    def changed(self) -> bool:
        return self.google_deleted > 0 or self.ics_deleted or self.local_deleted


def ics_path_for(run_id: str) -> Path:
    return out_dir() / f"{run_id}.ics"


def publish_memories(
    memories: list[MemoryDraft],
    labeled: list[LabeledCluster],
    cfg: Config,
    run_id: str,
    *,
    on_published: OnPublished | None = None,
    credentials: Credentials | None | Literal[False] = None,
    api: Any | None = None,
    wait: wait_base | None = None,
) -> PublishResult:
    events = build_memory_events(memories, labeled, cfg.calendar, run_id)
    if not events:
        return PublishResult(backend="ics", events=[])

    creds: Credentials | None
    if credentials is False:
        creds = None
    elif credentials is not None:
        creds = credentials
    elif api is not None:
        creds = None
    else:
        creds = load_credentials()

    if api is not None or creds is not None:
        if api is None:
            if creds is None:
                raise CalendarError("Google credentials missing")
            writer_api = build_api(creds)
        else:
            writer_api = api
        writer = GoogleWriter(writer_api, cfg.calendar, wait=wait)
        try:
            calendar_id = writer.ensure_calendar()
            for event in events:
                event_id = writer.upsert_event(calendar_id, event)
                if on_published is not None:
                    on_published(event.cluster_id, event_id)
        except Exception as exc:
            if isinstance(exc, CalendarError):
                raise
            raise CalendarError(f"Google Calendar write failed: {exc}") from exc
        return PublishResult(backend="google", events=events, calendar_id=calendar_id)

    path = write_ics(events, ics_path_for(run_id))
    if on_published is not None:
        for event in events:
            on_published(event.cluster_id, event.event_id)
    return PublishResult(backend="ics", events=events, ics_path=path)


def undo_run(
    run_id: str,
    cfg: Config,
    *,
    api: Any | None = None,
    credentials: Credentials | None | Literal[False] = None,
    wait: wait_base | None = None,
) -> UndoResult:
    result = UndoResult(run_id=run_id)
    creds: Credentials | None
    if credentials is False:
        creds = None
    elif credentials is not None:
        creds = credentials
    elif api is not None:
        creds = None
    else:
        try:
            creds = load_credentials()
        except CalendarError as exc:
            result.warnings.append(str(exc))
            creds = None

    if api is not None or creds is not None:
        try:
            writer_api = api if api is not None else build_api(creds)  # type: ignore[arg-type]
            writer = GoogleWriter(writer_api, cfg.calendar, wait=wait)
            calendar_id = writer.find_calendar()
            if calendar_id:
                result.google_deleted = writer.delete_run_events(calendar_id, run_id)
        except Exception as exc:
            message = str(exc) if isinstance(exc, CalendarError) else f"Google undo failed: {exc}"
            result.warnings.append(message)

    ics_path = ics_path_for(run_id)
    if ics_path.is_file():
        ics_path.unlink()
        result.ics_deleted = True

    conn = init_db(connect())
    try:
        if fetch_run(conn, run_id) is not None:
            delete_run(conn, run_id)
            conn.commit()
            result.local_deleted = True
    finally:
        conn.close()

    if not result.changed:
        raise CalendarError(f"no run {run_id}")
    return result
