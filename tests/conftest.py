from pathlib import Path
from zoneinfo import ZoneInfo

import pytest


@pytest.fixture(autouse=True)
def isolated_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    home = tmp_path / "Dayframe"
    home.mkdir()
    monkeypatch.setenv("DAYFRAME_HOME", str(home))
    monkeypatch.setenv("DAYFRAME_PHOTOS_LIBRARY", str(tmp_path / "missing.photoslibrary"))
    monkeypatch.delenv("DAYFRAME_API_KEY", raising=False)
    monkeypatch.delenv("DAYFRAME_GOOGLE_CLIENT_SECRET", raising=False)
    monkeypatch.delenv("DAYFRAME_PHOTOS_FIXTURE", raising=False)
    monkeypatch.setenv("DAYFRAME_SKIP_DOTENV", "1")
    monkeypatch.setattr(
        "dayframe.doctor.launchd_plist_path",
        lambda: tmp_path / "com.dayframe.daily.plist",
    )
    return home


@pytest.fixture(autouse=True)
def jerusalem_local_tz(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("dayframe.window.local_tz", lambda: ZoneInfo("Asia/Jerusalem"))
