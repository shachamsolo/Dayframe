from __future__ import annotations

import operator
from dataclasses import dataclass
from datetime import datetime
from typing import Annotated, Any, TypedDict

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages

from dayframe.agent.budget import Budget
from dayframe.baseline.digest import label_clusters
from dayframe.baseline.models import LabeledCluster
from dayframe.cluster.sessions import Cluster
from dayframe.config import Config
from dayframe.photos.models import Asset
from dayframe.photos.reader import PhotosReader
from dayframe.window import ensure_aware


class DayframeState(TypedDict, total=False):
    run_id: str
    target_date: str
    seen: list[str]
    raw: list[dict[str, Any]]
    kept: list[dict[str, Any]]
    dropped: list[dict[str, Any]]
    clusters: list[dict[str, Any]]
    photo_count: int
    messages: Annotated[list[AnyMessage], add_messages]
    memories: Annotated[list[dict[str, Any]], operator.add]
    discarded: Annotated[list[dict[str, Any]], operator.add]
    viewed: Annotated[list[str], operator.add]
    images_sent: Annotated[int, operator.add]
    cost_usd: Annotated[float, operator.add]
    input_tokens: Annotated[int, operator.add]
    output_tokens: Annotated[int, operator.add]
    turns: Annotated[int, operator.add]
    status: str


@dataclass
class AgentContext:
    cfg: Config
    reader: PhotosReader
    model: BaseChatModel
    model_name: str
    budget: Budget
    started_monotonic: float
    dump_wire: bool = True


def dump_asset(asset: Asset) -> dict[str, Any]:
    return asset.model_dump(mode="json")


def load_asset(data: dict[str, Any]) -> Asset:
    return Asset.model_validate(data)


def dump_cluster(cluster: Cluster) -> dict[str, Any]:
    return {
        "id": cluster.id,
        "start": ensure_aware(cluster.start).isoformat(),
        "end": ensure_aware(cluster.end).isoformat(),
        "place": cluster.place,
        "centroid": list(cluster.centroid) if cluster.centroid is not None else None,
        "people": list(cluster.people),
        "assets": [dump_asset(asset) for asset in cluster.assets],
    }


def load_cluster(data: dict[str, Any]) -> Cluster:
    centroid = data.get("centroid")
    return Cluster(
        id=data["id"],
        assets=[load_asset(item) for item in data.get("assets") or []],
        start=datetime.fromisoformat(data["start"]),
        end=datetime.fromisoformat(data["end"]),
        place=data.get("place"),
        centroid=(float(centroid[0]), float(centroid[1])) if centroid else None,
        people=list(data.get("people") or []),
    )


def labeled_from_state(state: DayframeState) -> list[LabeledCluster]:
    clusters = [load_cluster(item) for item in state.get("clusters") or []]
    return label_clusters(clusters)


def resolve_cluster(state: DayframeState, cluster_id: str) -> LabeledCluster | None:
    needle = cluster_id.strip()
    for item in labeled_from_state(state):
        if needle in {item.label, item.cluster.id}:
            return item
    return None


def decided_labels(state: DayframeState) -> set[str]:
    labels = {item["cluster_id"] for item in state.get("memories") or []}
    labels.update(item["cluster_id"] for item in state.get("discarded") or [])
    return labels
