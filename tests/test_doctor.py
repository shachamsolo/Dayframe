from pathlib import Path

import pytest

from dayframe.doctor import (
    check_api_key,
    check_config,
    check_google_calendar,
    check_launchd,
    check_photos_library,
    failed,
    format_report,
    run_checks,
)


def test_fresh_home_all_fail(isolated_home: Path) -> None:
    results = run_checks()
    names = [check.name for check in results]
    assert names == [
        "config",
        "api_key",
        "photos_library",
        "google_calendar",
        "launchd",
    ]
    assert failed(results)
    assert all(not check.ok for check in results)
    report = format_report(results)
    assert report.startswith("Dayframe doctor")
    assert "0 passed, 5 failed" in report
    assert "fix:" in report


def test_valid_config_passes(isolated_home: Path) -> None:
    example = Path(__file__).resolve().parents[1] / "config.example.toml"
    (isolated_home / "config.toml").write_text(
        example.read_text(encoding="utf-8"), encoding="utf-8"
    )
    check = check_config()
    assert check.ok
    assert str(isolated_home / "config.toml") in check.detail


def test_invalid_config_fails(isolated_home: Path) -> None:
    (isolated_home / "config.toml").write_text("[window]\nrun_at = 'nope'\n", encoding="utf-8")
    check = check_config()
    assert not check.ok
    assert "invalid" in check.detail


def test_api_key_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    assert not check_api_key().ok
    monkeypatch.setenv("DAYFRAME_API_KEY", "sk-test")
    assert check_api_key().ok


def test_google_calendar_needs_secret_and_token(
    isolated_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    check = check_google_calendar()
    assert not check.ok
    assert "DAYFRAME_GOOGLE_CLIENT_SECRET" in check.detail

    secret = isolated_home / "client.json"
    secret.write_text("{}", encoding="utf-8")
    monkeypatch.setenv("DAYFRAME_GOOGLE_CLIENT_SECRET", str(secret))
    check = check_google_calendar()
    assert not check.ok
    assert "google_token.json" in check.detail

    (isolated_home / "google_token.json").write_text("{}", encoding="utf-8")
    check = check_google_calendar()
    assert check.ok


def test_launchd_missing() -> None:
    check = check_launchd()
    assert not check.ok
    assert "install-agent" in check.fix


def test_launchd_valid_plist_passes() -> None:
    from dayframe.config import Config
    from dayframe.launchd import install_agent

    install_agent(cfg=Config(), load=False)
    check = check_launchd()
    assert check.ok


def test_launchd_invalid_plist_fails(tmp_path: Path) -> None:
    from dayframe.paths import launchd_plist_path

    path = launchd_plist_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("not a plist", encoding="utf-8")
    check = check_launchd()
    assert not check.ok
    assert "invalid" in check.detail


def test_photos_library_missing_when_overridden() -> None:
    check = check_photos_library()
    assert not check.ok
    assert "Photos library not found" in check.detail
