from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import pytest

from dayframe.window import calendar_day_window, parse_since, parse_target_date, since_window
from tests.factories import TZ


def test_parse_since() -> None:
    assert parse_since("24h") == timedelta(hours=24)
    assert parse_since("7d") == timedelta(days=7)
    assert parse_since("90m") == timedelta(minutes=90)
    assert parse_since(" 24H ") == timedelta(hours=24)


def test_parse_since_rejects_junk() -> None:
    with pytest.raises(ValueError, match="lookback"):
        parse_since("yesterday")
    with pytest.raises(ValueError, match="at least 1"):
        parse_since("0h")


def test_yesterday_and_iso_date() -> None:
    now = datetime(2026, 9, 18, 7, 0, tzinfo=TZ)
    assert parse_target_date("yesterday", now=now).isoformat() == "2026-09-17"
    assert parse_target_date("2026-09-17").isoformat() == "2026-09-17"
    with pytest.raises(ValueError, match="YYYY-MM-DD"):
        parse_target_date("17/09/2026")


def test_calendar_day_window_jerusalem() -> None:
    start, end = calendar_day_window(parse_target_date("2026-09-17"), tz=TZ)
    assert start == datetime(2026, 9, 17, 0, 0, tzinfo=TZ)
    assert end == datetime(2026, 9, 18, 0, 0, tzinfo=TZ)


def test_since_window() -> None:
    now = datetime(2026, 9, 18, 7, 0, tzinfo=ZoneInfo("UTC"))
    start, end = since_window(timedelta(hours=24), now=now)
    assert end == now
    assert start == datetime(2026, 9, 17, 7, 0, tzinfo=ZoneInfo("UTC"))
