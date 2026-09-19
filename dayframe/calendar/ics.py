from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from dayframe.calendar.events import MemoryEvent


def write_ics(events: list[MemoryEvent], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(tz=UTC).strftime("%Y%m%dT%H%M%SZ")
    lines = [
        "BEGIN:VCALENDAR",
        "VERSION:2.0",
        "PRODID:-//Dayframe//EN",
        "CALSCALE:GREGORIAN",
        "METHOD:PUBLISH",
    ]
    for event in events:
        lines.extend(_vevent(event, stamp=stamp))
    lines.append("END:VCALENDAR")
    path.write_text("\r\n".join(_fold(line) for line in lines) + "\r\n", encoding="utf-8")
    return path


def _vevent(event: MemoryEvent, *, stamp: str) -> list[str]:
    lines = [
        "BEGIN:VEVENT",
        f"UID:{event.event_id}@dayframe.local",
        f"DTSTAMP:{stamp}",
        f"DTSTART;TZID={event.timezone}:{_ics_local(event.start)}",
        f"DTEND;TZID={event.timezone}:{_ics_local(event.end)}",
        f"SUMMARY:{_escape(event.summary)}",
        f"DESCRIPTION:{_escape(event.description)}",
        "TRANSP:TRANSPARENT",
        f"X-DAYFRAME-RUN-ID:{event.run_id}",
        f"X-DAYFRAME-CLUSTER-ID:{event.cluster_id}",
        f"X-DAYFRAME-CONFIDENCE:{event.confidence:.2f}",
    ]
    if event.location:
        lines.append(f"LOCATION:{_escape(event.location)}")
    lines.append("END:VEVENT")
    return lines


def _ics_local(value: datetime) -> str:
    return value.strftime("%Y%m%dT%H%M%S")


def _escape(value: str) -> str:
    return (
        value.replace("\\", "\\\\")
        .replace(";", "\\;")
        .replace(",", "\\,")
        .replace("\r\n", "\\n")
        .replace("\n", "\\n")
    )


def _fold(line: str) -> str:
    if len(line) <= 75:
        return line
    chunks = [line[:75]]
    rest = line[75:]
    while rest:
        chunks.append(" " + rest[:74])
        rest = rest[74:]
    return "\r\n".join(chunks)
