from __future__ import annotations

import json
import time
from datetime import date
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.runtime import Runtime

from dayframe.agent.budget import BUDGET_EXHAUSTED
from dayframe.agent.state import (
    AgentContext,
    DayframeState,
    decided_labels,
    dump_asset,
    dump_cluster,
    labeled_from_state,
    load_asset,
)
from dayframe.agent.trace import message_to_json, wire_path
from dayframe.baseline.anthropic import cost_usd
from dayframe.baseline.digest import build_session_digest, label_clusters
from dayframe.cluster.sessions import cluster_assets
from dayframe.prefilter.rules import prefilter
from dayframe.prompt import load_prompt
from dayframe.window import calendar_day_window


def load_photos(state: DayframeState, runtime: Runtime[AgentContext]) -> dict[str, Any]:
    day = date.fromisoformat(state["target_date"])
    start, end = calendar_day_window(day)
    raw = runtime.context.reader.assets_added_between(start, end)
    return {"raw": [dump_asset(asset) for asset in raw]}


def prefilter_node(state: DayframeState, runtime: Runtime[AgentContext]) -> dict[str, Any]:
    del runtime
    raw = [load_asset(item) for item in state.get("raw") or []]
    seen = set(state.get("seen") or [])
    result = prefilter(raw, seen)
    return {
        "kept": [dump_asset(asset) for asset in result.kept],
        "dropped": [{"uuid": item.asset.uuid, "reason": item.reason} for item in result.dropped],
        "photo_count": len(result.kept),
    }


def cluster_node(state: DayframeState, runtime: Runtime[AgentContext]) -> dict[str, Any]:
    cfg = runtime.context.cfg
    kept = [load_asset(item) for item in state.get("kept") or []]
    clusters = cluster_assets(kept, cfg.cluster)
    labeled = label_clusters(clusters)
    updates: dict[str, Any] = {"clusters": [dump_cluster(cluster) for cluster in clusters]}
    if not clusters:
        return updates
    digest = build_session_digest(
        date.fromisoformat(state["target_date"]),
        labeled,
        photo_count=int(state.get("photo_count") or len(kept)),
        send_coordinates=cfg.privacy.send_coordinates,
    )
    updates["messages"] = [HumanMessage(content=digest)]
    return updates


def agent_node(state: DayframeState, runtime: Runtime[AgentContext]) -> dict[str, Any]:
    ctx = runtime.context
    cfg = ctx.cfg
    turns = int(state.get("turns") or 0)
    cost = float(state.get("cost_usd") or 0.0)
    elapsed = time.monotonic() - ctx.started_monotonic
    if (
        turns >= cfg.budget.max_turns
        or cost >= cfg.budget.max_cost_usd
        or elapsed >= cfg.budget.timeout_seconds
    ):
        return {
            "status": "budget_exceeded",
            "messages": [AIMessage(content="Stopping: budget exhausted.")],
        }

    prompt = [SystemMessage(load_prompt("system_v1"))]
    if int(state.get("images_sent") or 0) >= cfg.budget.max_images:
        prompt.append(SystemMessage(BUDGET_EXHAUSTED))
    messages = prompt + list(state.get("messages") or [])
    if ctx.dump_wire:
        _dump_request(state["run_id"], messages)
    msg = ctx.model.invoke(messages)
    usage = getattr(msg, "usage_metadata", None) or {}
    input_tokens = int(usage.get("input_tokens") or 0)
    output_tokens = int(usage.get("output_tokens") or 0)
    return {
        "messages": [msg],
        "cost_usd": cost_usd(ctx.model_name, input_tokens, output_tokens),
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "turns": 1,
    }


def write_calendar(state: DayframeState, runtime: Runtime[AgentContext]) -> dict[str, Any]:
    del runtime
    extras: list[dict[str, Any]] = []
    decided = decided_labels(state)
    for item in labeled_from_state(state):
        if item.label not in decided:
            extras.append({"cluster_id": item.label, "reason": "model_omitted"})
    status = state.get("status") or "running"
    if status == "running":
        status = "ok"
    return {"discarded": extras, "status": status}


def route_after_cluster(state: DayframeState) -> str:
    if not state.get("clusters"):
        return "write_calendar"
    return "agent"


def _dump_request(run_id: str, messages: list[Any]) -> None:
    path = wire_path(run_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = [message_to_json(message) for message in messages]
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
