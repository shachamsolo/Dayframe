from __future__ import annotations

from pathlib import Path


def prompts_dir() -> Path:
    here = Path(__file__).resolve()
    repo = here.parents[1] / "prompts"
    if repo.is_dir():
        return repo
    packaged = here.parent / "prompts"
    return packaged


def load_prompt(name: str) -> str:
    path = prompts_dir() / f"{name}.md"
    if not path.is_file():
        raise FileNotFoundError(f"prompt not found: {path}")
    return path.read_text(encoding="utf-8").strip()
