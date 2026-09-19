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


def _fake_agent(target, cfg, **kwargs):  # noqa: ANN001
    from datetime import date

    from dayframe.agent.run import AgentResult
    from dayframe.baseline.digest import label_clusters
    from dayframe.baseline.models import DiscardDraft, MemoryDraft
    from dayframe.photos.reader import FixturePhotosReader
    from dayframe.pipeline import run_for_date

    photos = run_for_date(FixturePhotosReader.from_path(BEACH), "2026-09-17")
    labeled = label_clusters(photos.clusters)
    interrupted = cfg.calendar.approval_mode == "review" and not kwargs.get("resume")
    return AgentResult(
        run_id="run_2026-09-17",
        target_date=date(2026, 9, 17),
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
        images_sent=3,
        input_tokens=1200,
        output_tokens=180,
        cost_usd=0.0123,
        wall_seconds=1.2,
        turns=4,
        model="claude-sonnet-4-5",
        raw=photos.raw,
        dropped=photos.dropped,
        clusters=photos.clusters,
        interrupted=interrupted,
        status="pending_review" if interrupted else "ok",
    )


def _fake_baseline(photos, cfg, **kwargs):  # noqa: ANN001
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
    monkeypatch.setattr("dayframe.cli.run_agent", _fake_agent)
    result = runner.invoke(app, ["run", "--date", "2026-09-17", "--dry-run"])
    assert result.exit_code == 0, result.output
    assert "KEEP  [c004] Dayframe: Sunset at Palmachim Beach" in result.stdout
    assert "4 turns" in result.stdout
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
    monkeypatch.setattr("dayframe.cli.run_agent", _fake_agent)
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


def test_run_writes_ics_without_google_auth(
    monkeypatch: pytest.MonkeyPatch, isolated_home: Path
) -> None:
    monkeypatch.setenv("DAYFRAME_API_KEY", "sk-test")
    monkeypatch.setenv("DAYFRAME_PHOTOS_FIXTURE", str(BEACH))
    monkeypatch.setattr("dayframe.cli.run_agent", _fake_agent)
    result = runner.invoke(app, ["run", "--date", "2026-09-17"])
    assert result.exit_code == 0, result.output
    ics = isolated_home / "out" / "run_2026-09-17.ics"
    assert "Google auth unavailable" in result.stdout
    assert str(ics) in result.stdout
    assert ics.is_file()
    text = ics.read_text(encoding="utf-8")
    assert "Dayframe: Sunset at Palmachim Beach" in text
    assert "TRANSP:TRANSPARENT" in text
    from dayframe.store.db import connect, init_db

    conn = init_db(connect())
    try:
        row = conn.execute("SELECT event_id FROM memories").fetchone()
        assert row["event_id"]
    finally:
        conn.close()


def test_undo_removes_ics_and_ledger(monkeypatch: pytest.MonkeyPatch, isolated_home: Path) -> None:
    monkeypatch.setenv("DAYFRAME_API_KEY", "sk-test")
    monkeypatch.setenv("DAYFRAME_PHOTOS_FIXTURE", str(BEACH))
    monkeypatch.setattr("dayframe.cli.run_agent", _fake_agent)
    written = runner.invoke(app, ["run", "--date", "2026-09-17"])
    assert written.exit_code == 0, written.output
    result = runner.invoke(app, ["undo", "run_2026-09-17"])
    assert result.exit_code == 0, result.output
    assert "removed .ics" in result.stdout
    assert "removed the local ledger" in result.stdout
    assert not (isolated_home / "out" / "run_2026-09-17.ics").exists()
    from dayframe.store.db import connect, fetch_run, init_db

    conn = init_db(connect())
    try:
        assert fetch_run(conn, "run_2026-09-17") is None
    finally:
        conn.close()


def test_undo_missing_run() -> None:
    result = runner.invoke(app, ["undo", "run_missing"])
    assert result.exit_code == 1
    assert "no run" in result.output


def test_auth_requires_secret() -> None:
    result = runner.invoke(app, ["auth"])
    assert result.exit_code == 1
    assert "DAYFRAME_GOOGLE_CLIENT_SECRET" in result.output


def test_run_baseline_flag(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DAYFRAME_API_KEY", "sk-test")
    monkeypatch.setenv("DAYFRAME_PHOTOS_FIXTURE", str(BEACH))
    monkeypatch.setattr("dayframe.cli.run_baseline", _fake_baseline)
    result = runner.invoke(app, ["run", "--date", "2026-09-17", "--baseline", "--dry-run"])
    assert result.exit_code == 0, result.output
    assert "1 call" in result.stdout
    assert "KEEP  [c004] Dayframe: Sunset at Palmachim Beach" in result.stdout


def test_replay_missing_trace() -> None:
    result = runner.invoke(app, ["replay", "run_missing"])
    assert result.exit_code == 1
    assert "no trace" in result.output


def test_replay_prints_turns(isolated_home: Path) -> None:
    from dayframe.agent.trace import append_jsonl, trace_path

    path = trace_path("run_2026-09-17")
    append_jsonl(
        path,
        {
            "tool_calls": [{"name": "expand_cluster", "args": {"cluster_id": "c004"}}],
            "tool_results": [{"content": "Palmachim Beach"}],
            "usage": {"input_tokens": 10, "output_tokens": 4},
        },
    )
    result = runner.invoke(app, ["replay", "run_2026-09-17"])
    assert result.exit_code == 0, result.output
    assert "Turn 1" in result.stdout
    assert "expand_cluster" in result.stdout


EXAMPLE = Path(__file__).resolve().parents[1] / "config.example.toml"


def _write_review_config(home: Path) -> None:
    text = EXAMPLE.read_text(encoding="utf-8").replace(
        'approval_mode  = "auto"',
        'approval_mode  = "review"',
    )
    (home / "config.toml").write_text(text, encoding="utf-8")


def test_install_agent_writes_plist(isolated_home: Path) -> None:
    result = runner.invoke(app, ["install-agent", "--no-load"])
    assert result.exit_code == 0, result.output
    assert "Wrote" in result.stdout
    assert "07:00" in result.stdout
    from dayframe.paths import launchd_plist_path

    assert launchd_plist_path().is_file()


def test_run_failure_records_ledger_and_log(
    monkeypatch: pytest.MonkeyPatch, isolated_home: Path
) -> None:
    monkeypatch.setenv("DAYFRAME_API_KEY", "sk-test")
    monkeypatch.setenv("DAYFRAME_PHOTOS_FIXTURE", str(BEACH))

    def boom(*args, **kwargs):  # noqa: ANN001, ANN002
        raise RuntimeError("photos exploded")

    monkeypatch.setattr("dayframe.cli.run_agent", boom)
    result = runner.invoke(app, ["run", "--date", "2026-09-17", "--no-calendar"])
    assert result.exit_code == 1
    assert "photos exploded" in result.output
    log = isolated_home / "logs" / "dayframe.log"
    assert log.is_file()
    assert "photos exploded" in log.read_text(encoding="utf-8")
    from dayframe.store.db import connect, fetch_run, init_db

    conn = init_db(connect())
    try:
        row = fetch_run(conn, "run_2026-09-17")
        assert row is not None
        assert row["status"] == "failed"
        assert "photos exploded" in row["error"]
    finally:
        conn.close()


def test_review_mode_persists_without_calendar(
    monkeypatch: pytest.MonkeyPatch, isolated_home: Path
) -> None:
    _write_review_config(isolated_home)
    monkeypatch.setenv("DAYFRAME_API_KEY", "sk-test")
    monkeypatch.setenv("DAYFRAME_PHOTOS_FIXTURE", str(BEACH))
    monkeypatch.setenv("DAYFRAME_UNATTENDED", "1")
    monkeypatch.setattr("dayframe.cli.run_agent", _fake_agent)
    result = runner.invoke(app, ["run", "--date", "2026-09-17"])
    assert result.exit_code == 0, result.output
    assert "pending_review" in result.stdout
    assert "unattended" in result.stdout
    assert not (isolated_home / "out" / "run_2026-09-17.ics").exists()
    from dayframe.store.db import connect, fetch_run, init_db

    conn = init_db(connect())
    try:
        row = fetch_run(conn, "run_2026-09-17")
        assert row is not None
        assert row["status"] == "pending_review"
        assert conn.execute("SELECT COUNT(*) AS n FROM memories").fetchone()["n"] == 1
    finally:
        conn.close()


def test_review_mode_approve_writes_calendar(
    monkeypatch: pytest.MonkeyPatch, isolated_home: Path
) -> None:
    _write_review_config(isolated_home)
    monkeypatch.setenv("DAYFRAME_API_KEY", "sk-test")
    monkeypatch.setenv("DAYFRAME_PHOTOS_FIXTURE", str(BEACH))
    monkeypatch.setattr("dayframe.cli.run_agent", _fake_agent)
    result = runner.invoke(app, ["run", "--date", "2026-09-17"], input="y\n")
    assert result.exit_code == 0, result.output
    assert "Write 1 memory to the calendar?" in result.output
    ics = isolated_home / "out" / "run_2026-09-17.ics"
    assert ics.is_file()
    from dayframe.store.db import connect, fetch_run, init_db

    conn = init_db(connect())
    try:
        row = fetch_run(conn, "run_2026-09-17")
        assert row is not None
        assert row["status"] == "ok"
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
