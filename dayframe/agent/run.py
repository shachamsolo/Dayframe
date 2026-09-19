from __future__ import annotations

import sqlite3
import time
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

from langchain_core.messages import AIMessage, ToolMessage
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.errors import GraphRecursionError

from dayframe.agent.budget import Budget, recursion_limit
from dayframe.agent.graph import build_graph
from dayframe.agent.state import AgentContext, labeled_from_state, load_asset
from dayframe.agent.tools import TOOLS
from dayframe.agent.trace import (
    RawDump,
    add_tool_results,
    append_jsonl,
    ensure_traces_dir,
    trace_path,
    turn_from_ai,
    wire_path,
)
from dayframe.baseline.anthropic import api_model_name
from dayframe.baseline.digest import label_clusters
from dayframe.baseline.models import DiscardDraft, LabeledCluster, MemoryDraft
from dayframe.baseline.run import apply_decisions
from dayframe.cluster.sessions import Cluster
from dayframe.config import Config
from dayframe.models import build_model
from dayframe.paths import db_path
from dayframe.photos.models import Asset
from dayframe.photos.reader import PhotosReader
from dayframe.pipeline import inspect_run_id
from dayframe.prefilter.rules import Dropped
from dayframe.window import parse_target_date


@dataclass
class AgentResult:
    run_id: str
    target_date: date
    labeled: list[LabeledCluster]
    memories: list[MemoryDraft]
    discarded: list[DiscardDraft]
    skipped_unreviewed: list[str]
    images_sent: int
    input_tokens: int
    output_tokens: int
    cost_usd: float
    wall_seconds: float
    turns: int
    model: str
    raw: list[Asset]
    dropped: list[Dropped]
    clusters: list[Cluster]
    status: str = "ok"
    resumed: bool = False
    interrupted: bool = False
    prompt_version: str = "system_v1"
    trace_path: Path | None = None

    @property
    def kept(self) -> list[Asset]:
        dropped_ids = {item.asset.uuid for item in self.dropped}
        return [asset for asset in self.raw if asset.uuid not in dropped_ids]


@dataclass
class _TurnBuffer:
    pending: dict[str, Any] | None = field(default=None)


def open_checkpointer(path: Path | None = None) -> tuple[sqlite3.Connection, SqliteSaver]:
    target = path or db_path()
    target.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(target), check_same_thread=False)
    saver = SqliteSaver(conn)
    saver.setup()
    return conn, saver


def initial_state(run_id: str, target_date: date, seen: set[str] | None) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "target_date": target_date.isoformat(),
        "seen": sorted(seen or []),
        "messages": [],
        "memories": [],
        "discarded": [],
        "viewed": [],
        "images_sent": 0,
        "cost_usd": 0.0,
        "input_tokens": 0,
        "output_tokens": 0,
        "turns": 0,
        "status": "running",
    }


def run_config(run_id: str, cfg: Config, *, callbacks: list[Any] | None = None) -> dict[str, Any]:
    config: dict[str, Any] = {
        "configurable": {"thread_id": run_id},
        "recursion_limit": recursion_limit(cfg.budget.max_turns),
    }
    if callbacks:
        config["callbacks"] = callbacks
    return config


def run_agent(
    *,
    target: str,
    cfg: Config,
    reader: PhotosReader,
    seen: set[str] | None = None,
    api_key: str | None = None,
    model: Any | None = None,
    checkpointer: Any | None = None,
    dump_wire: bool = True,
    now: Any | None = None,
) -> AgentResult:
    day = parse_target_date(target, now=now)
    run_id = inspect_run_id(day)
    model_name = api_model_name(cfg.provider.model, cfg.provider.name)
    if model is not None:
        bound = model.bind_tools(TOOLS)
    else:
        bound = build_model(cfg, api_key=api_key, tools=TOOLS)

    own_conn: sqlite3.Connection | None = None
    saver = checkpointer
    if saver is None:
        own_conn, saver = open_checkpointer()

    try:
        return _run(
            day=day,
            run_id=run_id,
            model_name=model_name,
            bound=bound,
            cfg=cfg,
            reader=reader,
            seen=seen,
            saver=saver,
            dump_wire=dump_wire,
        )
    finally:
        if own_conn is not None:
            own_conn.close()


def _run(
    *,
    day: date,
    run_id: str,
    model_name: str,
    bound: Any,
    cfg: Config,
    reader: PhotosReader,
    seen: set[str] | None,
    saver: Any,
    dump_wire: bool,
) -> AgentResult:
    ensure_traces_dir()
    config = run_config(run_id, cfg)
    graph = build_graph(saver, approval_mode=cfg.calendar.approval_mode)

    snapshot = graph.get_state(config)
    resumed = bool(snapshot.next)
    if snapshot.values and not snapshot.next:
        saver.delete_thread(run_id)
        snapshot = graph.get_state(config)
        resumed = False

    jsonl = trace_path(run_id)
    raw_path = wire_path(run_id)
    if not resumed:
        if jsonl.exists():
            jsonl.unlink()
        if raw_path.exists():
            raw_path.unlink()
    callbacks = [RawDump(raw_path)] if dump_wire else []
    if callbacks:
        config = run_config(run_id, cfg, callbacks=callbacks)

    seed_sent = int((snapshot.values or {}).get("images_sent") or 0) if resumed else 0
    context = AgentContext(
        cfg=cfg,
        reader=reader,
        model=bound,
        model_name=model_name,
        budget=Budget(
            max_images=cfg.budget.max_images,
            max_turns=cfg.budget.max_turns,
            max_cost_usd=cfg.budget.max_cost_usd,
            timeout_seconds=cfg.budget.timeout_seconds,
            images_sent=seed_sent,
        ),
        started_monotonic=time.monotonic(),
        dump_wire=dump_wire,
    )
    payload = None if resumed else initial_state(run_id, day, seen)
    buffer = _TurnBuffer()
    started = time.perf_counter()
    status = "ok"
    try:
        for update in graph.stream(payload, config, context=context, stream_mode="updates"):
            _trace_update(jsonl, buffer, update)
        if buffer.pending is not None:
            append_jsonl(jsonl, buffer.pending)
            buffer.pending = None
    except GraphRecursionError:
        status = "budget_exceeded"

    final = graph.get_state(config)
    values = final.values or {}
    interrupted = bool(final.next)
    if status == "ok" and values.get("status"):
        status = str(values["status"])

    labeled = labeled_from_state(values) if values.get("clusters") else label_clusters([])
    memories = [MemoryDraft.model_validate(item) for item in values.get("memories") or []]
    discards = [DiscardDraft.model_validate(item) for item in values.get("discarded") or []]
    kept, discarded, omitted = apply_decisions(
        labeled,
        memories,
        discards,
        min_confidence=cfg.calendar.min_confidence,
    )
    raw = [load_asset(item) for item in values.get("raw") or []]
    by_uuid = {asset.uuid: asset for asset in raw}
    dropped = [
        Dropped(asset=by_uuid[item["uuid"]], reason=item["reason"])
        for item in values.get("dropped") or []
        if item["uuid"] in by_uuid
    ]
    return AgentResult(
        run_id=run_id,
        target_date=day,
        labeled=labeled,
        memories=kept,
        discarded=discarded,
        skipped_unreviewed=omitted,
        images_sent=int(values.get("images_sent") or 0),
        input_tokens=int(values.get("input_tokens") or 0),
        output_tokens=int(values.get("output_tokens") or 0),
        cost_usd=float(values.get("cost_usd") or 0.0),
        wall_seconds=time.perf_counter() - started,
        turns=int(values.get("turns") or 0),
        model=model_name,
        raw=raw,
        dropped=dropped,
        clusters=[item.cluster for item in labeled],
        status=status,
        resumed=resumed,
        interrupted=interrupted,
        trace_path=jsonl if jsonl.is_file() else None,
    )


def _trace_update(path: Path, buffer: _TurnBuffer, update: dict[str, Any]) -> None:
    if "agent" in update:
        messages = _messages_from_update(update["agent"])
        ai = next((msg for msg in messages if isinstance(msg, AIMessage)), None)
        if ai is None:
            return
        if buffer.pending is not None:
            append_jsonl(path, buffer.pending)
        turn = turn_from_ai(ai)
        if ai.tool_calls:
            buffer.pending = turn
        else:
            append_jsonl(path, turn)
            buffer.pending = None
        return
    if "tools" in update and buffer.pending is not None:
        messages = _messages_from_update(update["tools"])
        tool_messages = [msg for msg in messages if isinstance(msg, ToolMessage)]
        add_tool_results(buffer.pending, tool_messages)
        append_jsonl(path, buffer.pending)
        buffer.pending = None


def _messages_from_update(payload: Any) -> list[Any]:
    if payload is None:
        return []
    if isinstance(payload, dict):
        return list(payload.get("messages") or [])
    if isinstance(payload, list):
        messages: list[Any] = []
        for item in payload:
            if isinstance(item, dict):
                messages.extend(item.get("messages") or [])
            elif isinstance(item, ToolMessage):
                messages.append(item)
        return messages
    return []
