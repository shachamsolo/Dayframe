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
