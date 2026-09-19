import sqlite3
from pathlib import Path

import pytest

from dayframe.store.db import connect, init_db

EXPECTED_TABLES = {"runs", "assets_seen", "clusters", "memories"}


def test_init_creates_tables(tmp_path: Path) -> None:
    db_file = tmp_path / "dayframe.db"
    conn = init_db(connect(db_file))
    try:
        rows = conn.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table' AND name NOT LIKE 'sqlite_%'"
        ).fetchall()
        names = {row["name"] for row in rows}
        assert EXPECTED_TABLES <= names
        columns = {row[1] for row in conn.execute("PRAGMA table_info(runs)")}
        assert "error" in columns
    finally:
        conn.close()


def test_init_is_idempotent(tmp_path: Path) -> None:
    db_file = tmp_path / "dayframe.db"
    conn = init_db(connect(db_file))
    try:
        init_db(conn)
        count = conn.execute("SELECT COUNT(*) AS n FROM runs").fetchone()["n"]
        assert count == 0
    finally:
        conn.close()


def test_foreign_keys_enforced(tmp_path: Path) -> None:
    conn = init_db(connect(tmp_path / "dayframe.db"))
    try:
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute(
                "INSERT INTO assets_seen (uuid, added_at, run_id) "
                "VALUES ('a', '2026-09-17', 'missing')"
            )
    finally:
        conn.close()


def test_connect_creates_parent_dir(isolated_home: Path) -> None:
    isolated_home.rmdir()
    from dayframe.store.db import connect as connect_default

    conn = connect_default()
    try:
        assert isolated_home.exists()
    finally:
        conn.close()


def test_migrates_error_column_on_existing_db(tmp_path: Path) -> None:
    db_file = tmp_path / "dayframe.db"
    conn = connect(db_file)
    try:
        conn.execute(
            """
            CREATE TABLE runs (
              id TEXT PRIMARY KEY,
              target_date TEXT NOT NULL,
              started_at TEXT NOT NULL,
              finished_at TEXT,
              status TEXT NOT NULL
            )
            """
        )
        conn.commit()
        init_db(conn)
        columns = {row[1] for row in conn.execute("PRAGMA table_info(runs)")}
        assert "error" in columns
    finally:
        conn.close()


def test_record_failed_run(isolated_home: Path) -> None:
    from datetime import datetime

    from dayframe.store.db import fetch_run, record_failed_run

    conn = init_db(connect())
    try:
        record_failed_run(
            conn,
            run_id="run_2026-09-17",
            target_date="2026-09-17",
            started_at=datetime(2026, 9, 18, 7, 0),
            finished_at=datetime(2026, 9, 18, 7, 1),
            error="boom",
            provider="anthropic",
            model="claude-sonnet-4-5",
        )
        row = fetch_run(conn, "run_2026-09-17")
        assert row is not None
        assert row["status"] == "failed"
        assert row["error"] == "boom"
    finally:
        conn.close()
