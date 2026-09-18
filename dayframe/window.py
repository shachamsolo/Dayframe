from __future__ import annotations

import re
from datetime import date, datetime, time, timedelta, tzinfo

_SINCE = re.compile(r"^(\d+)([hdm])$")
_UNITS = {
    "h": lambda n: timedelta(hours=n),
    "d": lambda n: timedelta(days=n),
    "m": lambda n: timedelta(minutes=n),
}


def local_tz() -> tzinfo:
    tz = datetime.now().astimezone().tzinfo
    if tz is None:  # pragma: no cover
        raise RuntimeError("system local timezone is missing")
    return tz


def parse_since(value: str) -> timedelta:
    match = _SINCE.match(value.strip().lower())
    if not match:
        raise ValueError("lookback must look like 24h, 7d, or 90m")
    count = int(match.group(1))
    if count < 1:
        raise ValueError("lookback must be at least 1")
    return _UNITS[match.group(2)](count)


def parse_target_date(value: str, *, now: datetime | None = None) -> date:
    current = now or datetime.now(tz=local_tz())
    if value.strip().lower() == "yesterday":
        return current.date() - timedelta(days=1)
    try:
        return date.fromisoformat(value.strip())
    except ValueError as exc:
        raise ValueError("date must be 'yesterday' or YYYY-MM-DD") from exc


def calendar_day_window(day: date, *, tz: tzinfo | None = None) -> tuple[datetime, datetime]:
    zone = tz or local_tz()
    start = datetime.combine(day, time.min, tzinfo=zone)
    return start, start + timedelta(days=1)


def since_window(delta: timedelta, *, now: datetime | None = None) -> tuple[datetime, datetime]:
    end = now or datetime.now(tz=local_tz())
    return end - delta, end


def ensure_aware(value: datetime, *, tz: tzinfo | None = None) -> datetime:
    if value.tzinfo is not None:
        return value
    return value.replace(tzinfo=tz or local_tz())
