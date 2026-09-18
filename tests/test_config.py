from pathlib import Path

import pytest
from pydantic import ValidationError

from dayframe.config import Config

EXAMPLE = Path(__file__).resolve().parents[1] / "config.example.toml"


def test_example_config_loads() -> None:
    cfg = Config.load(EXAMPLE)
    assert cfg.provider.model == "claude-sonnet-4-5"
    assert cfg.window.basis == "date_added"
    assert cfg.cluster.time_gap_minutes == 90
    assert cfg.privacy.send_coordinates is False
    assert cfg.calendar.min_confidence == 0.7
    assert cfg.calendar.approval_mode == "auto"


def test_missing_config_raises(isolated_home: Path) -> None:
    with pytest.raises(FileNotFoundError, match="config not found"):
        Config.load()


def test_defaults_fill_partial_toml(tmp_path: Path) -> None:
    path = tmp_path / "partial.toml"
    path.write_text("[provider]\nname = 'openai'\n", encoding="utf-8")
    cfg = Config.load(path)
    assert cfg.provider.name == "openai"
    assert cfg.budget.max_turns == 12
    assert cfg.calendar.timezone == "Asia/Jerusalem"


def test_unknown_key_rejected(tmp_path: Path) -> None:
    path = tmp_path / "bad.toml"
    path.write_text("[provider]\nname = 'anthropic'\napi_key = 'nope'\n", encoding="utf-8")
    with pytest.raises(ValidationError):
        Config.load(path)


def test_invalid_run_at(tmp_path: Path) -> None:
    path = tmp_path / "bad.toml"
    path.write_text("[window]\nrun_at = '7am'\n", encoding="utf-8")
    with pytest.raises(ValidationError):
        Config.load(path)


def test_confidence_bounds(tmp_path: Path) -> None:
    path = tmp_path / "bad.toml"
    path.write_text("[calendar]\nmin_confidence = 1.5\n", encoding="utf-8")
    with pytest.raises(ValidationError):
        Config.load(path)
