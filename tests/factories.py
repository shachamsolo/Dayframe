from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

from dayframe.photos.models import Asset

TZ = ZoneInfo("Asia/Jerusalem")


def make_asset(
    uuid: str,
    hour: int,
    minute: int = 0,
    **overrides: object,
) -> Asset:
    captured = datetime(2026, 9, 17, hour, minute, tzinfo=TZ)
    added = overrides.pop("date_added", datetime(2026, 9, 17, 23, 0, tzinfo=TZ))
    data: dict[str, object] = {
        "uuid": uuid,
        "date": captured,
        "date_added": added,
        "uti": "public.heic",
        "has_camera_exif": True,
    }
    data.update(overrides)
    return Asset.model_validate(data)
