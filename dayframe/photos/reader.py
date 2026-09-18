from __future__ import annotations

import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Protocol, runtime_checkable

from pydantic import BaseModel, TypeAdapter

from dayframe.paths import photos_fixture_path, photos_library_path
from dayframe.photos.models import Asset
from dayframe.window import ensure_aware


@runtime_checkable
class PhotosReader(Protocol):
    def assets_added_between(self, start: datetime, end: datetime) -> list[Asset]: ...

    def asset(self, uuid: str) -> Asset: ...


class FixtureFile(BaseModel):
    assets: list[Asset]


_FIXTURE_FILE = TypeAdapter(FixtureFile)


def _in_window(asset: Asset, start: datetime, end: datetime) -> bool:
    added = ensure_aware(asset.date_added)
    return start <= added < end


class FixturePhotosReader:
    """JSON fixture reader so tests never touch a real Photos library."""

    def __init__(self, assets: list[Asset]) -> None:
        self._assets = {asset.uuid: asset for asset in assets}
        if len(self._assets) != len(assets):
            raise ValueError("fixture contains duplicate asset uuids")

    @classmethod
    def from_path(cls, path: Path) -> FixturePhotosReader:
        data = _FIXTURE_FILE.validate_json(path.read_text(encoding="utf-8"))
        return cls(data.assets)

    def assets_added_between(self, start: datetime, end: datetime) -> list[Asset]:
        start_a, end_a = ensure_aware(start), ensure_aware(end)
        found = [asset for asset in self._assets.values() if _in_window(asset, start_a, end_a)]
        found.sort(key=lambda asset: (ensure_aware(asset.date_added), asset.uuid))
        return found

    def asset(self, uuid: str) -> Asset:
        try:
            return self._assets[uuid]
        except KeyError as exc:
            raise KeyError(f"asset not in fixture: {uuid}") from exc


class OsxPhotosReader:
    """Thin wrapper around osxphotos. Do not query Photos.sqlite directly."""

    def __init__(self, library: Path | None = None) -> None:
        self._library = library or photos_library_path()
        self._db = None

    def _photosdb(self):
        if self._db is None:
            try:
                from osxphotos import PhotosDB
            except ImportError as exc:  # pragma: no cover - darwin-only dep
                raise RuntimeError("osxphotos is not installed; pip install -e . on macOS") from exc
            if not self._library.exists():
                raise FileNotFoundError(f"Photos library not found: {self._library}")
            logging.getLogger("osxphotos").setLevel(logging.ERROR)
            try:
                self._db = PhotosDB(dbfile=str(self._library))
            except Exception as exc:
                detail = str(exc).lower()
                if (
                    isinstance(exc, PermissionError)
                    or "error copying" in detail
                    or "permission" in detail
                ):
                    raise PermissionError(
                        f"cannot read Photos library at {self._library}; "
                        "grant Full Disk Access to this terminal "
                        "(System Settings → Privacy & Security → Full Disk Access)"
                    ) from exc
                raise
        return self._db

    def assets_added_between(self, start: datetime, end: datetime) -> list[Asset]:
        from osxphotos import QueryOptions

        start_a, end_a = ensure_aware(start), ensure_aware(end)
        options = QueryOptions(
            added_after=start_a - timedelta(seconds=1),
            added_before=end_a,
            photos=True,
            movies=False,
        )
        photos = self._photosdb().query(options)
        assets: list[Asset] = []
        for photo in photos:
            if getattr(photo, "ismovie", False):
                continue
            asset = asset_from_photoinfo(photo)
            if asset is None:
                continue
            if start_a <= asset.date_added < end_a:
                assets.append(asset)
        assets.sort(key=lambda item: (item.date_added, item.uuid))
        return assets

    def asset(self, uuid: str) -> Asset:
        from osxphotos import QueryOptions

        photos = self._photosdb().query(QueryOptions(uuid=[uuid], photos=True, movies=True))
        if not photos:
            raise KeyError(f"asset not in Photos library: {uuid}")
        converted = asset_from_photoinfo(photos[0])
        if converted is None:
            raise KeyError(f"asset missing capture or added date: {uuid}")
        return converted


def asset_from_photoinfo(photo: object) -> Asset | None:
    date = getattr(photo, "date", None)
    date_added = getattr(photo, "date_added", None)
    if date is None or date_added is None:
        return None
    place_obj = getattr(photo, "place", None)
    place_name = getattr(place_obj, "name", None) if place_obj is not None else None
    score = getattr(photo, "score", None)
    exif = getattr(photo, "exif_info", None)
    burst = bool(getattr(photo, "burst", False))
    burst_id = None
    if burst:
        key = getattr(photo, "burst_key_photo", None)
        burst_id = getattr(key, "uuid", None) if key is not None else getattr(photo, "uuid", None)
    persons = [name for name in (getattr(photo, "persons", None) or []) if name]
    return Asset(
        uuid=photo.uuid,
        date=ensure_aware(date),
        date_added=ensure_aware(date_added),
        latitude=getattr(photo, "latitude", None),
        longitude=getattr(photo, "longitude", None),
        place_name=place_name or None,
        screenshot=bool(getattr(photo, "screenshot", False)),
        uti=getattr(photo, "uti", None) or None,
        burst=burst,
        burst_id=burst_id,
        persons=persons,
        score_overall=getattr(score, "overall", None) if score is not None else None,
        score_well_timed=getattr(score, "well_timed_shot", None) if score is not None else None,
        path=getattr(photo, "path", None),
        path_edited=getattr(photo, "path_edited", None),
        has_camera_exif=bool(
            exif is not None
            and (getattr(exif, "camera_make", None) or getattr(exif, "camera_model", None))
        ),
    )


def get_reader() -> PhotosReader:
    fixture = photos_fixture_path()
    if fixture is not None:
        return FixturePhotosReader.from_path(fixture)
    return OsxPhotosReader()
