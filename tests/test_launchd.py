from __future__ import annotations

import plistlib
from pathlib import Path

from dayframe.config import Config, WindowConfig
from dayframe.launchd import (
    agent_program,
    install_agent,
    parse_hhmm,
    plist_is_valid,
    render_plist,
)
from dayframe.paths import ENV_UNATTENDED, LAUNCHD_LABEL, logs_dir


def test_parse_hhmm() -> None:
    assert parse_hhmm("07:00") == (7, 0)
    assert parse_hhmm("08:30") == (8, 30)


def test_render_plist_schedule_and_logs() -> None:
    cfg = Config(window=WindowConfig(run_at="08:30"))
    payload = render_plist(cfg=cfg, program=["/usr/bin/python3", "-m", "dayframe", "run"])
    assert payload["Label"] == LAUNCHD_LABEL
    assert payload["RunAtLoad"] is False
    assert payload["StartCalendarInterval"] == {"Hour": 8, "Minute": 30}
    assert payload["StandardErrorPath"]
    assert payload["EnvironmentVariables"][ENV_UNATTENDED] == "1"
    assert payload["ProgramArguments"][1:4] == ["-m", "dayframe", "run"]


def test_install_agent_writes_plist(isolated_home: Path) -> None:
    result = install_agent(cfg=Config(), load=False)
    assert result.plist_path.is_file()
    assert result.loaded is False
    payload = plistlib.loads(result.plist_path.read_bytes())
    assert payload["Label"] == LAUNCHD_LABEL
    assert payload["RunAtLoad"] is False
    assert payload["StartCalendarInterval"]["Hour"] == 7
    assert Path(payload["StandardOutPath"]).parent == logs_dir()
    assert logs_dir().is_dir()
    assert plist_is_valid(result.plist_path) is None


def test_plist_is_valid_rejects_incomplete_plist(tmp_path: Path) -> None:
    path = tmp_path / "bad.plist"
    path.write_bytes(plistlib.dumps({"Label": LAUNCHD_LABEL, "ProgramArguments": ["x"]}))
    assert plist_is_valid(path) == "missing StartCalendarInterval"


def test_agent_program_uses_python_module() -> None:
    args = agent_program()
    assert args[1:5] == ["-m", "dayframe", "run", "--date"]
