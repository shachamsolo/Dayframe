from pathlib import Path

import pytest

from dayframe.paths import api_key


def test_api_key_from_home_dotenv(isolated_home: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("DAYFRAME_API_KEY", raising=False)
    (isolated_home / ".env").write_text("DAYFRAME_API_KEY=sk-from-home\n", encoding="utf-8")
    assert api_key() == "sk-from-home"


def test_process_env_wins_over_dotenv(isolated_home: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (isolated_home / ".env").write_text("DAYFRAME_API_KEY=from-file\n", encoding="utf-8")
    monkeypatch.setenv("DAYFRAME_API_KEY", "from-env")
    assert api_key() == "from-env"


def test_cwd_dotenv(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("DAYFRAME_SKIP_DOTENV", raising=False)
    monkeypatch.delenv("DAYFRAME_API_KEY", raising=False)
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".env").write_text("DAYFRAME_API_KEY=sk-from-cwd\n", encoding="utf-8")
    assert api_key() == "sk-from-cwd"


def test_empty_dotenv_key_is_missing(isolated_home: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("DAYFRAME_API_KEY", raising=False)
    (isolated_home / ".env").write_text("DAYFRAME_API_KEY=\n", encoding="utf-8")
    assert api_key() is None
