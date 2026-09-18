from __future__ import annotations

from dayframe.cluster.sessions import Cluster
from dayframe.photos.models import Asset
from dayframe.window import ensure_aware

M2_MAX_IMAGES = 8


def cluster_label(seq: int) -> str:
    return f"c{seq:03d}"


def pick_representatives(assets: list[Asset], k: int) -> list[Asset]:
    """Highest score.overall in each time bucket so frames are spread, not duplicates."""
    if k <= 0 or not assets:
        return []
    ordered = sorted(assets, key=lambda asset: (ensure_aware(asset.date), asset.uuid))
    if k >= len(ordered):
        return ordered
    picked: list[Asset] = []
    used: set[str] = set()
    n = len(ordered)
    for index in range(k):
        start = (index * n) // k
        end = max(start + 1, ((index + 1) * n) // k)
        bucket = ordered[start:end]
        choice = _best_unused(bucket, used) or _best_unused(ordered, used)
        if choice is None:
            break
        used.add(choice.uuid)
        picked.append(choice)
    return sorted(picked, key=lambda asset: (ensure_aware(asset.date), asset.uuid))


def allocate_slots(clusters: list[Cluster], budget: int = M2_MAX_IMAGES) -> dict[str, int]:
    """One slot per cluster while budget lasts; leftover goes to the most promising sessions."""
    if budget <= 0 or not clusters:
        return {}
    ranked = sorted(clusters, key=_priority, reverse=True)
    if len(ranked) >= budget:
        return {cluster.id: 1 for cluster in ranked[:budget]}
    slots = {cluster.id: 1 for cluster in ranked}
    remaining = budget - len(ranked)
    while remaining > 0:
        progressed = False
        for cluster in ranked:
            if remaining <= 0:
                break
            if slots[cluster.id] >= len(cluster.assets):
                continue
            slots[cluster.id] += 1
            remaining -= 1
            progressed = True
        if not progressed:
            break
    return slots


def _priority(cluster: Cluster) -> tuple[int, int, int, float, str]:
    people = 1 if cluster.people else 0
    place = 1 if cluster.place else 0
    score = max((asset.score_overall or 0.0) for asset in cluster.assets) if cluster.assets else 0.0
    return (people, len(cluster.assets), place, score, cluster.id)


def _best_unused(assets: list[Asset], used: set[str]) -> Asset | None:
    candidates = [asset for asset in assets if asset.uuid not in used]
    if not candidates:
        return None
    return max(
        candidates,
        key=lambda asset: (asset.score_overall or 0.0, asset.score_well_timed or 0.0, asset.uuid),
    )
