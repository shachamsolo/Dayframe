from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

ENV_HOME = "DAYFRAME_HOME"
ENV_API_KEY = "DAYFRAME_API_KEY"
ENV_GOOGLE_CLIENT_SECRET = "DAYFRAME_GOOGLE_CLIENT_SECRET"
ENV_PHOTOS_LIBRARY = "DAYFRAME_PHOTOS_LIBRARY"
ENV_PHOTOS_FIXTURE = "DAYFRAME_PHOTOS_FIXTURE"
ENV_SKIP_DOTENV = "DAYFRAME_SKIP_DOTENV"
ENV_UNATTENDED = "DAYFRAME_UNATTENDED"

LAUNCHD_LABEL = "com.dayframe.daily"
DEFAULT_PHOTOS_LIBRARY = Path.home() / "Pictures" / "Photos Library.photoslibrary"


def load_env() -> None:
    """Load `.env` files. Existing process env wins (including launchd)."""
    if os.environ.get(ENV_SKIP_DOTENV) == "1":
        _load_one(_home_path() / ".env")
        return
    _load_one(Path.cwd() / ".env")
    _load_one(_home_path() / ".env")


def _home_path() -> Path:
    raw = os.environ.get(ENV_HOME)
    if raw:
        return Path(raw).expanduser()
    return Path.home() / "Dayframe"


def _load_one(path: Path) -> None:
    if path.is_file():
        load_dotenv(path, override=False)


def home() -> Path:
    load_env()
    return _home_path()


def config_path() -> Path:
    return home() / "config.toml"


def db_path() -> Path:
    return home() / "dayframe.db"


def google_token_path() -> Path:
    return home() / "google_token.json"


def traces_dir() -> Path:
    return home() / "traces"


def out_dir() -> Path:
    return home() / "out"


def logs_dir() -> Path:
    return home() / "logs"


def launchd_plist_path() -> Path:
    return Path.home() / "Library" / "LaunchAgents" / f"{LAUNCHD_LABEL}.plist"


def is_unattended() -> bool:
    value = os.environ.get(ENV_UNATTENDED, "").strip().lower()
    return value in {"1", "true", "yes"}


def photos_library_path() -> Path:
    load_env()
    raw = os.environ.get(ENV_PHOTOS_LIBRARY)
    if raw:
        return Path(raw).expanduser()
    return DEFAULT_PHOTOS_LIBRARY


def photos_fixture_path() -> Path | None:
    load_env()
    raw = os.environ.get(ENV_PHOTOS_FIXTURE)
    if not raw or not raw.strip():
        return None
    return Path(raw.strip()).expanduser()


def api_key() -> str | None:
    load_env()
    value = os.environ.get(ENV_API_KEY)
    return value.strip() if value and value.strip() else None


def google_client_secret_path() -> Path | None:
    load_env()
    raw = os.environ.get(ENV_GOOGLE_CLIENT_SECRET)
    if not raw or not raw.strip():
        return None
    return Path(raw.strip()).expanduser()
