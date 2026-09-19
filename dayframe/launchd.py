from __future__ import annotations

import os
import plistlib
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from dayframe import paths
from dayframe.config import Config
from dayframe.paths import ENV_UNATTENDED, LAUNCHD_LABEL, home, logs_dir


@dataclass(frozen=True)
class InstallResult:
    plist_path: Path
    hour: int
    minute: int
    program: list[str]
    stdout_log: Path
    stderr_log: Path
    loaded: bool
    load_error: str | None = None


def parse_hhmm(value: str) -> tuple[int, int]:
    hour_s, minute_s = value.split(":", 1)
    return int(hour_s), int(minute_s)


def agent_program(executable: Path | None = None) -> list[str]:
    python = str((executable or Path(sys.executable)).resolve())
    return [python, "-m", "dayframe", "run", "--date", "yesterday"]


def launchd_log_paths() -> tuple[Path, Path]:
    directory = logs_dir()
    return directory / "launchd.out.log", directory / "launchd.err.log"


def render_plist(
    *,
    cfg: Config,
    program: list[str] | None = None,
    stdout_log: Path | None = None,
    stderr_log: Path | None = None,
) -> dict[str, object]:
    hour, minute = parse_hhmm(cfg.window.run_at)
    out_log, err_log = launchd_log_paths()
    exe_dir = str(Path(program[0] if program else sys.executable).resolve().parent)
    path = os.environ.get("PATH", "/usr/bin:/bin:/usr/sbin:/sbin")
    if exe_dir not in path.split(":"):
        path = f"{exe_dir}:{path}"
    env = {
        "PATH": path,
        "DAYFRAME_HOME": str(home()),
        ENV_UNATTENDED: "1",
        "LANGCHAIN_TRACING_V2": "false",
        "LANGSMITH_TRACING": "false",
        "PYTHONUNBUFFERED": "1",
    }
    return {
        "Label": LAUNCHD_LABEL,
        "ProgramArguments": program or agent_program(),
        "StartCalendarInterval": {"Hour": hour, "Minute": minute},
        "RunAtLoad": False,
        "StandardOutPath": str(stdout_log or out_log),
        "StandardErrorPath": str(stderr_log or err_log),
        "WorkingDirectory": str(home()),
        "EnvironmentVariables": env,
        "ProcessType": "Background",
    }


def write_plist(payload: dict[str, object], path: Path | None = None) -> Path:
    target = path or paths.launchd_plist_path()
    target.parent.mkdir(parents=True, exist_ok=True)
    logs_dir().mkdir(parents=True, exist_ok=True)
    home().mkdir(parents=True, exist_ok=True)
    target.write_bytes(plistlib.dumps(payload, fmt=plistlib.FMT_XML))
    os.chmod(target, 0o644)
    return target


def bootstrap_agent(plist: Path) -> None:
    uid = os.getuid()
    domain = f"gui/{uid}"
    subprocess.run(
        ["launchctl", "bootout", f"{domain}/{LAUNCHD_LABEL}"],
        check=False,
        capture_output=True,
        text=True,
    )
    result = subprocess.run(
        ["launchctl", "bootstrap", domain, str(plist)],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode == 0:
        return
    fallback = subprocess.run(
        ["launchctl", "load", "-w", str(plist)],
        check=False,
        capture_output=True,
        text=True,
    )
    if fallback.returncode == 0:
        return
    detail = (result.stderr or fallback.stderr or result.stdout or fallback.stdout or "").strip()
    raise RuntimeError(detail or "launchctl bootstrap failed")


def install_agent(
    *,
    cfg: Config | None = None,
    load: bool = True,
    executable: Path | None = None,
) -> InstallResult:
    config = cfg or Config()
    program = agent_program(executable)
    stdout_log, stderr_log = launchd_log_paths()
    payload = render_plist(
        cfg=config, program=program, stdout_log=stdout_log, stderr_log=stderr_log
    )
    path = write_plist(payload)
    hour, minute = parse_hhmm(config.window.run_at)
    loaded = False
    load_error: str | None = None
    if load:
        try:
            bootstrap_agent(path)
            loaded = True
        except FileNotFoundError:
            load_error = "launchctl not found"
        except (RuntimeError, OSError) as exc:
            load_error = str(exc)
    return InstallResult(
        plist_path=path,
        hour=hour,
        minute=minute,
        program=program,
        stdout_log=stdout_log,
        stderr_log=stderr_log,
        loaded=loaded,
        load_error=load_error,
    )


def plist_is_valid(path: Path) -> str | None:
    """Return a problem description, or None if the plist looks installable."""
    try:
        payload = plistlib.loads(path.read_bytes())
    except Exception as exc:
        return f"cannot read plist: {exc}"
    if payload.get("Label") != LAUNCHD_LABEL:
        return f"Label is {payload.get('Label')!r}, expected {LAUNCHD_LABEL!r}"
    if "StartCalendarInterval" not in payload:
        return "missing StartCalendarInterval"
    if not payload.get("StandardErrorPath"):
        return "missing StandardErrorPath"
    args = payload.get("ProgramArguments")
    if not isinstance(args, list) or not args:
        return "missing ProgramArguments"
    return None
