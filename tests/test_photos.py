from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from zoneinfo import ZoneInfo

from dayframe.photos.reader import FixturePhotosReader, asset_from_photoinfo
from dayframe.window import calendar_day_window

FIXTURE = Path(__file__).parent / "fixtures" / "days" / "beach.json"
TZ = ZoneInfo("Asia/Jerusalem")


def test_fixture_reader_filters_added_window() -> None:
    reader = FixturePhotosReader.from_path(FIXTURE)
    start, end = calendar_day_window(datetime(2026, 9, 17).date(), tz=TZ)
    assets = reader.assets_added_between(start, end)
    assert len(assets) == 18
    assert reader.asset("palmachim-1").place_name == "Palmachim Beach"

    empty = reader.assets_added_between(*calendar_day_window(datetime(2026, 9, 16).date(), tz=TZ))
    assert empty == []


def test_asset_from_photoinfo_maps_fields() -> None:
    tz = TZ
    photo = SimpleNamespace(
        uuid="abc",
        date=datetime(2026, 9, 17, 16, 12, tzinfo=tz),
        date_added=datetime(2026, 9, 17, 20, 0, tzinfo=tz),
        latitude=31.93,
        longitude=34.70,
        place=SimpleNamespace(name="Palmachim Beach"),
        screenshot=False,
        uti="public.heic",
        burst=True,
        burst_key_photo=SimpleNamespace(uuid="burst-key"),
        persons=["Maya", ""],
        score=SimpleNamespace(overall=0.8, well_timed_shot=0.4),
        path="/tmp/a.jpg",
        path_edited=None,
        exif_info=SimpleNamespace(camera_make="Apple", camera_model="iPhone"),
        ismovie=False,
    )
    asset = asset_from_photoinfo(photo)
    assert asset is not None
    assert asset.burst_id == "burst-key"
    assert asset.persons == ["Maya"]
    assert asset.has_camera_exif is True
    assert asset.score_well_timed == 0.4


def test_asset_from_photoinfo_skips_missing_dates() -> None:
    photo = SimpleNamespace(uuid="x", date=None, date_added=None)
    assert asset_from_photoinfo(photo) is None
