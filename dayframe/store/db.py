from __future__ import annotations

import sqlite3
from datetime import datetime
from pathlib import Path

from dayframe.baseline.models import DiscardDraft, LabeledCluster, MemoryDraft
from dayframe.cluster.sessions import Cluster
from dayframe.paths import db_path, home
from dayframe.photos.models import Asset

SCHEMA_PATH = Path(__file__).with_name("schema.sql")

_SCHEMA = SCHEMA_PATH.read_text(encoding="utf-8")


def connect(path: Path | None = None) -> sqlite3.Connection:
    target = path or db_path()
    target.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(target)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db(conn: sqlite3.Connection | None = None) -> sqlite3.Connection:
    own = conn is None
    db = conn or connect()
    try:
        db.executescript(_SCHEMA)
        db.commit()
    except Exception:
        if own:
            db.close()
        raise
    return db


def ensure_home() -> Path:
    path = home()
    path.mkdir(parents=True, exist_ok=True)
    return path


def seen_uuids(conn: sqlite3.Connection, *, exclude_run_id: str | None = None) -> set[str]:
    if exclude_run_id:
        rows = conn.execute(
            "SELECT uuid FROM assets_seen WHERE run_id != ?",
            (exclude_run_id,),
        ).fetchall()
    else:
        rows = conn.execute("SELECT uuid FROM assets_seen").fetchall()
    return {row["uuid"] for row in rows}


def delete_run(conn: sqlite3.Connection, run_id: str) -> None:
    cluster_ids = [
        row["id"]
        for row in conn.execute("SELECT id FROM clusters WHERE run_id = ?", (run_id,)).fetchall()
    ]
    if cluster_ids:
        placeholders = ",".join("?" * len(cluster_ids))
        conn.execute(
            f"DELETE FROM memories WHERE cluster_id IN ({placeholders})",
            cluster_ids,
        )
    conn.execute(
        "DELETE FROM memories WHERE cluster_id IN (SELECT id FROM clusters WHERE run_id = ?)",
        (run_id,),
    )
    conn.execute("DELETE FROM assets_seen WHERE run_id = ?", (run_id,))
    conn.execute("DELETE FROM clusters WHERE run_id = ?", (run_id,))
    conn.execute("DELETE FROM runs WHERE id = ?", (run_id,))


def replace_inspect_run(
    conn: sqlite3.Connection,
    *,
    run_id: str,
    target_date: str,
    started_at: datetime,
    finished_at: datetime,
    clusters: list[Cluster],
    provider: str,
    model: str,
) -> None:
    cluster_ids = [cluster.id for cluster in clusters]
    delete_run(conn, run_id)
    if cluster_ids:
        placeholders = ",".join("?" * len(cluster_ids))
        conn.execute(
            f"DELETE FROM memories WHERE cluster_id IN ({placeholders})",
            cluster_ids,
        )
        conn.execute(f"DELETE FROM clusters WHERE id IN ({placeholders})", cluster_ids)
    conn.execute(
        """
        INSERT INTO runs (id, target_date, started_at, finished_at, status, provider, model)
        VALUES (?, ?, ?, ?, 'ok', ?, ?)
        """,
        (
            run_id,
            target_date,
            started_at.isoformat(),
            finished_at.isoformat(),
            provider,
            model,
        ),
    )
    for cluster in clusters:
        lat = lon = None
        if cluster.centroid is not None:
            lat, lon = cluster.centroid
        conn.execute(
            """
            INSERT INTO clusters (
                id, run_id, start_ts, end_ts, place, lat, lon, asset_count, decision
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'unreviewed')
            """,
            (
                cluster.id,
                run_id,
                cluster.start.isoformat(),
                cluster.end.isoformat(),
                cluster.place,
                lat,
                lon,
                len(cluster.assets),
            ),
        )
    conn.commit()


def persist_baseline_run(
    conn: sqlite3.Connection,
    *,
    run_id: str,
    target_date: str,
    started_at: datetime,
    finished_at: datetime,
    labeled: list[LabeledCluster],
    memories: list[MemoryDraft],
    discarded: list[DiscardDraft],
    assets: list[Asset],
    provider: str,
    model: str,
    prompt_version: str,
    images_sent: int,
    input_tokens: int,
    output_tokens: int,
    cost_usd: float,
    status: str = "ok",
    turns: int = 1,
) -> None:
    cluster_ids = [item.cluster.id for item in labeled]
    delete_run(conn, run_id)
    if cluster_ids:
        placeholders = ",".join("?" * len(cluster_ids))
        conn.execute(
            f"DELETE FROM memories WHERE cluster_id IN ({placeholders})",
            cluster_ids,
        )
        conn.execute(f"DELETE FROM clusters WHERE id IN ({placeholders})", cluster_ids)
    conn.execute(
        """
        INSERT INTO runs (
            id, target_date, started_at, finished_at, status, turns, images_sent,
            input_tokens, output_tokens, cost_usd, provider, model, prompt_version
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            run_id,
            target_date,
            started_at.isoformat(),
            finished_at.isoformat(),
            status,
            turns,
            images_sent,
            input_tokens,
            output_tokens,
            cost_usd,
            provider,
            model,
            prompt_version,
        ),
    )
    by_label = {item.label: item.cluster for item in labeled}
    decisions = {memory.cluster_id: ("kept", None) for memory in memories}
    for discard in discarded:
        decisions[discard.cluster_id] = ("discarded", discard.reason)
    for item in labeled:
        cluster = item.cluster
        lat = lon = None
        if cluster.centroid is not None:
            lat, lon = cluster.centroid
        decision, reason = decisions.get(item.label, ("unreviewed", None))
        conn.execute(
            """
            INSERT INTO clusters (
                id, run_id, start_ts, end_ts, place, lat, lon, asset_count, decision, reason
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                cluster.id,
                run_id,
                cluster.start.isoformat(),
                cluster.end.isoformat(),
                cluster.place,
                lat,
                lon,
                len(cluster.assets),
                decision,
                reason,
            ),
        )
    created = finished_at.isoformat()
    for memory in memories:
        cluster = by_label[memory.cluster_id]
        conn.execute(
            """
            INSERT INTO memories (
                id, cluster_id, title, body, category, confidence, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                f"mem_{cluster.id}",
                cluster.id,
                memory.title,
                memory.body,
                memory.category,
                memory.confidence,
                created,
            ),
        )
    for asset in assets:
        conn.execute(
            """
            INSERT INTO assets_seen (uuid, added_at, run_id) VALUES (?, ?, ?)
            ON CONFLICT(uuid) DO UPDATE SET added_at = excluded.added_at, run_id = excluded.run_id
            """,
            (asset.uuid, asset.date_added.isoformat(), run_id),
        )
    conn.commit()


def fetch_run(conn: sqlite3.Connection, run_id: str) -> sqlite3.Row | None:
    return conn.execute("SELECT * FROM runs WHERE id = ?", (run_id,)).fetchone()


def fetch_clusters(conn: sqlite3.Connection, run_id: str) -> list[sqlite3.Row]:
    return conn.execute(
        "SELECT * FROM clusters WHERE run_id = ? ORDER BY start_ts, id",
        (run_id,),
    ).fetchall()


def fetch_memories(conn: sqlite3.Connection, run_id: str) -> list[sqlite3.Row]:
    return conn.execute(
        """
        SELECT m.* FROM memories m
        JOIN clusters c ON c.id = m.cluster_id
        WHERE c.run_id = ?
        ORDER BY m.created_at, m.id
        """,
        (run_id,),
    ).fetchall()


def set_memory_event_id(conn: sqlite3.Connection, cluster_id: str, event_id: str) -> None:
    conn.execute(
        "UPDATE memories SET event_id = ? WHERE cluster_id = ?",
        (event_id, cluster_id),
    )
