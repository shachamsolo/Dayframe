from __future__ import annotations

import traceback
from datetime import datetime
from pathlib import Path

from dayframe.paths import logs_dir


def log_failure(run_id: str, exc: BaseException) -> Path:
    directory = logs_dir()
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "dayframe.log"
    stamp = datetime.now().astimezone().isoformat(timespec="seconds")
    body = "".join(traceback.format_exception(exc))
    with path.open("a", encoding="utf-8") as fh:
        fh.write(f"{stamp} {run_id} FAILED\n{body}\n")
    return path
