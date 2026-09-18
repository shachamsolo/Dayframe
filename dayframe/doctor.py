from __future__ import annotations

import tomllib
from dataclasses import dataclass

from pydantic import ValidationError

from dayframe.config import Config
from dayframe.paths import (
    api_key,
    config_path,
    google_client_secret_path,
    google_token_path,
    launchd_plist_path,
    photos_library_path,
)


@dataclass(frozen=True)
class Check:
    name: str
    ok: bool
    detail: str
    fix: str = ""


def check_config() -> Check:
    path = config_path()
    if not path.exists():
        return Check(
            name="config",
            ok=False,
            detail=f"{path} does not exist",
            fix="cp config.example.toml ~/Dayframe/config.toml",
        )
    try:
        Config.load(path)
    except ValidationError as exc:
        return Check(
            name="config",
            ok=False,
            detail=f"{path} is invalid: {exc.error_count()} error(s)",
            fix="see config.example.toml and fix the failing fields",
        )
    except (OSError, tomllib.TOMLDecodeError) as exc:
        return Check(
            name="config",
            ok=False,
            detail=f"cannot read {path}: {exc}",
            fix="fix the file encoding or TOML syntax",
        )
    return Check(name="config", ok=True, detail=str(path))


def check_api_key() -> Check:
    if api_key() is None:
        return Check(
            name="api_key",
            ok=False,
            detail="DAYFRAME_API_KEY is not set",
            fix="set DAYFRAME_API_KEY in .env or export it (never put it in config.toml)",
        )
    return Check(name="api_key", ok=True, detail="DAYFRAME_API_KEY is set")


def check_photos_library() -> Check:
    library = photos_library_path()
    db = library / "database" / "Photos.sqlite"
    if not library.exists():
        return Check(
            name="photos_library",
            ok=False,
            detail=f"Photos library not found at {library}",
            fix=(
                "grant Full Disk Access to this terminal "
                "(System Settings → Privacy & Security → Full Disk Access)"
            ),
        )
    if not db.exists():
        return Check(
            name="photos_library",
            ok=False,
            detail=f"library exists but {db} is missing",
            fix="confirm this is an Apple Photos library, not an iPhoto or album folder",
        )
    try:
        with db.open("rb") as fh:
            fh.read(16)
    except PermissionError:
        return Check(
            name="photos_library",
            ok=False,
            detail=f"permission denied reading {db}",
            fix=(
                "grant Full Disk Access to this terminal "
                "(System Settings → Privacy & Security → Full Disk Access)"
            ),
        )
    return Check(name="photos_library", ok=True, detail=str(library))


def check_google_calendar() -> Check:
    secret = google_client_secret_path()
    token = google_token_path()
    problems: list[str] = []
    fixes: list[str] = []
    if secret is None:
        problems.append("DAYFRAME_GOOGLE_CLIENT_SECRET is not set")
        fixes.append("export DAYFRAME_GOOGLE_CLIENT_SECRET to your OAuth desktop client JSON")
    elif not secret.exists():
        problems.append(f"client secret not found: {secret}")
        fixes.append("point DAYFRAME_GOOGLE_CLIENT_SECRET at the downloaded client JSON")
    if not token.exists():
        problems.append(f"{token} does not exist")
        fixes.append("run: dayframe auth")
    if problems:
        return Check(
            name="google_calendar",
            ok=False,
            detail="; ".join(problems),
            fix="; ".join(fixes),
        )
    return Check(name="google_calendar", ok=True, detail=f"token at {token}")


def check_launchd() -> Check:
    plist = launchd_plist_path()
    if not plist.exists():
        return Check(
            name="launchd",
            ok=False,
            detail=f"{plist} is not installed",
            fix="run: dayframe install-agent",
        )
    return Check(name="launchd", ok=True, detail=str(plist))


CHECKS = (
    check_config,
    check_api_key,
    check_photos_library,
    check_google_calendar,
    check_launchd,
)


def run_checks() -> list[Check]:
    return [fn() for fn in CHECKS]


def format_report(results: list[Check]) -> str:
    lines = ["Dayframe doctor", ""]
    for check in results:
        status = "PASS" if check.ok else "FAIL"
        lines.append(f"[{status}] {check.name:<16} {check.detail}")
        if not check.ok and check.fix:
            lines.append(f"{'':8}fix: {check.fix}")
        lines.append("")
    n_fail = sum(1 for check in results if not check.ok)
    n_pass = len(results) - n_fail
    lines.append(f"{n_pass} passed, {n_fail} failed")
    return "\n".join(lines)


def failed(results: list[Check]) -> bool:
    return any(not check.ok for check in results)
