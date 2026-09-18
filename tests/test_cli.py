from pathlib import Path

import pytest
from typer.testing import CliRunner

from dayframe import __version__
from dayframe.cli import app

runner = CliRunner()
BEACH = Path(__file__).parent / "fixtures" / "days" / "beach.json"


def test_help_lists_commands() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    for name in (
        "doctor",
        "auth",
        "run",
        "install-agent",
        "replay",
        "undo",
        "photos",
        "clusters",
        "eval",
    ):
        assert name in result.stdout


def test_version() -> None:
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert __version__ in result.stdout


def test_doctor_exits_nonzero_when_red() -> None:
    result = runner.invoke(app, ["doctor"])
    assert result.exit_code == 1
    assert "Dayframe doctor" in result.stdout
    assert "[FAIL]" in result.stdout
    assert "0 passed, 5 failed" in result.stdout


def test_run_requires_api_key() -> None:
    result = runner.invoke(app, ["run", "--date", "yesterday", "--dry-run"])
    assert result.exit_code == 1
    assert "DAYFRAME_API_KEY" in result.output


def _fake_baseline(photos, cfg, **kwargs):
    from dayframe.baseline.digest import label_clusters
    from dayframe.baseline.models import BaselineResult, DiscardDraft, MemoryDraft

    labeled = label_clusters(photos.clusters)
    return BaselineResult(
        labeled=labeled,
        memories=[
            MemoryDraft(
                cluster_id="c004",
                title="Sunset at Palmachim Beach",
                body="Swimming with Maya and Noa, then ice cream at sunset.",
                category="outing",
                confidence=0.86,
            )
        ],
        discarded=[
            DiscardDraft(cluster_id=item.label, reason="not memorable")
            for item in labeled
            if item.label != "c004"
        ],
        skipped_unreviewed=[],
        images_sent=4,
        input_tokens=1200,
        output_tokens=180,
        cost_usd=0.0123,
        wall_seconds=1.2,
        model="claude-sonnet-4-5",
    )


def test_run_dry_run_prints_and_writes_nothing(
    monkeypatch: pytest.MonkeyPatch, isolated_home: Path
) -> None:
    monkeypatch.setenv("DAYFRAME_API_KEY", "sk-test")
    monkeypatch.setenv("DAYFRAME_PHOTOS_FIXTURE", str(BEACH))
    monkeypatch.setattr("dayframe.cli.run_baseline", _fake_baseline)
    result = runner.invoke(app, ["run", "--date", "2026-09-17", "--dry-run"])
    assert result.exit_code == 0, result.output
    assert "KEEP  [c004] Dayframe: Sunset at Palmachim Beach" in result.stdout
    assert "dry-run: wrote nothing" in result.stdout
    from dayframe.store.db import connect, init_db

    conn = init_db(connect())
    try:
        assert conn.execute("SELECT COUNT(*) AS n FROM assets_seen").fetchone()["n"] == 0
        assert conn.execute("SELECT COUNT(*) AS n FROM memories").fetchone()["n"] == 0
    finally:
        conn.close()


def test_run_persists_without_calendar(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DAYFRAME_API_KEY", "sk-test")
    monkeypatch.setenv("DAYFRAME_PHOTOS_FIXTURE", str(BEACH))
    monkeypatch.setattr("dayframe.cli.run_baseline", _fake_baseline)
    result = runner.invoke(app, ["run", "--date", "2026-09-17", "--no-calendar"])
    assert result.exit_code == 0, result.output
    assert "Wrote run run_2026-09-17" in result.stdout
    assert "M4" not in result.stdout
    from dayframe.store.db import connect, init_db

    conn = init_db(connect())
    try:
        assert conn.execute("SELECT COUNT(*) AS n FROM memories").fetchone()["n"] == 1
        assert conn.execute("SELECT COUNT(*) AS n FROM assets_seen").fetchone()["n"] == 18
        row = conn.execute("SELECT * FROM memories").fetchone()
        assert row["title"] == "Sunset at Palmachim Beach"
        cluster = conn.execute(
            "SELECT decision FROM clusters WHERE id = ?", ("2026-09-17-004",)
        ).fetchone()
        assert cluster["decision"] == "kept"
    finally:
        conn.close()


def test_photos_list_from_fixture(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DAYFRAME_PHOTOS_FIXTURE", str(BEACH))
    result = runner.invoke(app, ["photos", "list", "--date", "2026-09-17"])
    assert result.exit_code == 0, result.output
    assert "18 assets added" in result.stdout
    assert "palmachim-1" in result.stdout
    assert "Palmachim Beach" in result.stdout


def test_photos_list_rejects_bad_since() -> None:
    result = runner.invoke(app, ["photos", "list", "--since", "yesterday"])
    assert result.exit_code == 1
    assert "lookback" in result.output


def test_clusters_show_from_fixture(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DAYFRAME_PHOTOS_FIXTURE", str(BEACH))
    result = runner.invoke(app, ["clusters", "show", "--date", "2026-09-17"])
    assert result.exit_code == 0, result.output
    assert "18 photos across 6 candidate sessions" in result.stdout
    assert "[c004] 16:12–19:40 · 4 photos · Palmachim Beach · Maya, Noa" in result.stdout
    assert "screenshot 2" in result.stdout
    stored = runner.invoke(app, ["clusters", "show", "run_2026-09-17"])
    assert stored.exit_code == 0, stored.output
    assert "6 clusters" in stored.stdout
    assert "Palmachim Beach" in stored.stdout
