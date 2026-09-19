from __future__ import annotations

import base64
from pathlib import Path
from typing import Any, Literal

from langchain.tools import ToolRuntime, tool
from langchain_core.messages import ToolMessage
from langgraph.types import Command
from pydantic import ValidationError

from dayframe.agent.state import AgentContext, decided_labels, resolve_cluster
from dayframe.baseline.digest import people_label
from dayframe.baseline.models import DiscardDraft, MemoryDraft
from dayframe.baseline.select import pick_representatives
from dayframe.cluster.sessions import Cluster
from dayframe.config import Config
from dayframe.images.prepare import asset_image_path, prepare_image
from dayframe.photos.models import Asset
from dayframe.window import ensure_aware

Category = Literal[
    "trip",
    "family",
    "meal",
    "celebration",
    "outing",
    "hobby",
    "nature",
    "other",
]


def expand_text(
    item_label: str,
    cluster: Cluster,
    *,
    send_coordinates: bool = False,
) -> str:
    n = len(cluster.assets)
    photo_word = "photo" if n == 1 else "photos"
    place = cluster.place or "unknown"
    people = people_label(cluster.people)
    start = cluster.start.strftime("%H:%M")
    end = cluster.end.strftime("%H:%M")
    lines = [
        f"[{item_label}] {cluster.id}",
        f"{start}–{end} · {place} · {people} · {n} {photo_word}",
    ]
    if send_coordinates and cluster.centroid is not None:
        lat, lon = cluster.centroid
        lines.append(f"centroid: {lat:.4f},{lon:.4f}")
    lines.append("")
    for asset in cluster.assets:
        lines.append(_asset_line(asset, send_coordinates=send_coordinates))
    return "\n".join(lines)


def _asset_line(asset: Asset, *, send_coordinates: bool) -> str:
    captured = ensure_aware(asset.date).strftime("%H:%M")
    place = asset.place_name or "unknown"
    people = people_label(asset.persons)
    score = f"{asset.score_overall:.2f}" if asset.score_overall is not None else "-"
    path = asset_image_path(asset)
    if path is not None:
        filename = path.name
    elif asset.path:
        filename = Path(asset.path).name
    else:
        filename = asset.uuid
    line = (
        f"{asset.uuid}  {captured}  {place}  people: {len(asset.persons)} ({people})  "
        f"score: {score}  file: {filename}"
    )
    if send_coordinates and asset.coords is not None:
        lat, lon = asset.coords
        line += f"  {lat:.4f},{lon:.4f}"
    return line


def view_content(
    item_label: str,
    cluster: Cluster,
    *,
    count: int,
    cfg: Config,
    remaining: int,
    already: set[str],
) -> tuple[list[dict[str, Any]], list[str]]:
    take = min(max(count, 0), 4, remaining, len(cluster.assets))
    if take <= 0:
        return [], []
    candidates = [asset for asset in cluster.assets if asset.uuid not in already] or list(
        cluster.assets
    )
    chosen = pick_representatives(candidates, take)
    blocks: list[dict[str, Any]] = []
    sent: list[str] = []
    captions: list[str] = []
    for index, asset in enumerate(chosen, start=1):
        path = asset_image_path(asset)
        if path is None:
            continue
        try:
            prepared = prepare_image(
                path,
                max_px=cfg.privacy.max_image_px,
                strip_exif=cfg.privacy.strip_exif,
            )
        except (OSError, ValueError):
            continue
        place = asset.place_name or cluster.place or "unknown"
        captured = ensure_aware(asset.date).strftime("%H:%M")
        captions.append(f"Image {index}: [{item_label}] {captured} · {place} · {asset.uuid}")
        blocks.append(
            {
                "type": "image",
                "source_type": "base64",
                "mime_type": prepared.media_type,
                "data": base64.b64encode(prepared.jpeg_bytes).decode("ascii"),
            }
        )
        sent.append(asset.uuid)
    if not blocks:
        return [], []
    content: list[dict[str, Any]] = [
        {
            "type": "text",
            "text": f"Attached {len(blocks)} photo(s) from [{item_label}].\n" + "\n".join(captions),
        },
        *blocks,
    ]
    return content, sent


@tool(
    description=(
        "Return more text metadata for a cluster (per-photo timestamps, filenames, "
        "face counts, exact places) without spending image budget."
    )
)
def expand_cluster(cluster_id: str, runtime: ToolRuntime) -> str:
    item = resolve_cluster(runtime.state, cluster_id)
    if item is None:
        return f"Unknown cluster_id: {cluster_id}"
    ctx: AgentContext = runtime.context
    return expand_text(
        item.label,
        item.cluster,
        send_coordinates=ctx.cfg.privacy.send_coordinates,
    )


@tool(
    description=(
        "Attach up to 4 representative photos from a cluster to the conversation "
        "for visual inspection. Costs image budget. Call this only for clusters "
        "that plausibly represent a real experience."
    )
)
def view_photos(cluster_id: str, count: int = 3, *, runtime: ToolRuntime) -> Command | str:
    item = resolve_cluster(runtime.state, cluster_id)
    if item is None:
        return f"Unknown cluster_id: {cluster_id}"
    ctx: AgentContext = runtime.context
    requested = min(max(int(count or 3), 1), 4)
    remaining = ctx.budget.images_remaining
    if remaining <= 0:
        return (
            f"Image budget exhausted ({ctx.budget.images_sent}/{ctx.budget.max_images}). "
            "Decide from text metadata; do not retry view_photos."
        )
    already = set(runtime.state.get("viewed") or [])
    granted = ctx.budget.consume_images(min(requested, remaining))
    if granted <= 0:
        return (
            f"Image budget exhausted ({ctx.budget.images_sent}/{ctx.budget.max_images}). "
            "Decide from text metadata; do not retry view_photos."
        )
    content, sent = view_content(
        item.label,
        item.cluster,
        count=granted,
        cfg=ctx.cfg,
        remaining=granted,
        already=already,
    )
    unused = granted - len(sent)
    if unused:
        ctx.budget.release_images(unused)
    if not sent:
        return f"No readable image files for [{item.label}]; decide from metadata."
    return Command(
        update={
            "images_sent": len(sent),
            "viewed": sent,
            "messages": [
                ToolMessage(
                    content=content,
                    tool_call_id=runtime.tool_call_id or "",
                    name="view_photos",
                )
            ],
        }
    )


@tool(
    description=(
        "Record a memory for a cluster. Call exactly once per kept cluster. "
        "Titles name the place when known. Confidence below 0.7 is stored locally "
        "but never written to the calendar."
    )
)
def write_memory(
    cluster_id: str,
    title: str,
    body: str,
    category: Category,
    confidence: float,
    runtime: ToolRuntime,
) -> Command | str:
    item = resolve_cluster(runtime.state, cluster_id)
    if item is None:
        return f"Unknown cluster_id: {cluster_id}"
    if item.label in decided_labels(runtime.state):
        return f"Cluster {item.label} already decided."
    try:
        draft = MemoryDraft(
            cluster_id=item.label,
            title=title,
            body=body,
            category=category,
            confidence=confidence,
        )
    except ValidationError as exc:
        return f"Invalid memory: {exc.errors()[0]['msg']}"
    ctx: AgentContext = runtime.context
    below = draft.confidence < ctx.cfg.calendar.min_confidence
    note = (
        f"Recorded locally with confidence {draft.confidence:.2f} "
        f"(below {ctx.cfg.calendar.min_confidence}; will not be written to the calendar)."
        if below
        else f"Recorded memory for [{item.label}]: {draft.title}"
    )
    return Command(
        update={
            "memories": [draft.model_dump()],
            "messages": [
                ToolMessage(
                    content=note,
                    tool_call_id=runtime.tool_call_id or "",
                    name="write_memory",
                )
            ],
        }
    )


@tool(description="Skip a cluster and record why. Call exactly once per discarded cluster.")
def discard_cluster(cluster_id: str, reason: str, runtime: ToolRuntime) -> Command | str:
    item = resolve_cluster(runtime.state, cluster_id)
    if item is None:
        return f"Unknown cluster_id: {cluster_id}"
    if item.label in decided_labels(runtime.state):
        return f"Cluster {item.label} already decided."
    draft = DiscardDraft(cluster_id=item.label, reason=reason)
    return Command(
        update={
            "discarded": [draft.model_dump()],
            "messages": [
                ToolMessage(
                    content=f"Discarded [{item.label}]: {draft.reason}",
                    tool_call_id=runtime.tool_call_id or "",
                    name="discard_cluster",
                )
            ],
        }
    )


TOOLS = [expand_cluster, view_photos, write_memory, discard_cluster]
