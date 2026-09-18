from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from math import asin, cos, radians, sin, sqrt

from dayframe.config import ClusterConfig
from dayframe.photos.models import Asset
from dayframe.window import ensure_aware

EARTH_RADIUS_KM = 6371.0


@dataclass
class Cluster:
    id: str
    assets: list[Asset]
    start: datetime
    end: datetime
    place: str | None
    centroid: tuple[float, float] | None
    people: list[str] = field(default_factory=list)


def haversine_km(a: tuple[float, float], b: tuple[float, float]) -> float:
    lat1, lon1 = a
    lat2, lon2 = b
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    h = sin(dlat / 2) ** 2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon / 2) ** 2
    return 2 * EARTH_RADIUS_KM * asin(sqrt(min(h, 1.0)))


def _should_split(prev: Asset, current: Asset, cfg: ClusterConfig) -> bool:
    gap = ensure_aware(current.date) - ensure_aware(prev.date)
    if gap > timedelta(minutes=cfg.time_gap_minutes):
        return True
    coords_a, coords_b = prev.coords, current.coords
    if coords_a is None or coords_b is None:
        return False
    return haversine_km(coords_a, coords_b) > cfg.geo_gap_km and gap > timedelta(
        minutes=cfg.min_gap_for_geo_split_minutes
    )


def _modal_place(assets: list[Asset]) -> str | None:
    names = [asset.place_name for asset in assets if asset.place_name]
    if not names:
        return None
    counts = Counter(names)
    top = counts.most_common()
    best_count = top[0][1]
    tied = [name for name, count in top if count == best_count]
    if len(tied) == 1:
        return tied[0]
    for asset in assets:
        if asset.place_name in tied:
            return asset.place_name
    return tied[0]


def _centroid(assets: list[Asset]) -> tuple[float, float] | None:
    points = [asset.coords for asset in assets if asset.coords is not None]
    if not points:
        return None
    lat = sum(point[0] for point in points) / len(points)
    lon = sum(point[1] for point in points) / len(points)
    return (lat, lon)


def _people(assets: list[Asset]) -> list[str]:
    seen: set[str] = set()
    names: list[str] = []
    for asset in assets:
        for person in asset.persons:
            if person and person not in seen:
                seen.add(person)
                names.append(person)
    return names


def _build_cluster(cluster_id: str, assets: list[Asset]) -> Cluster:
    start = min(ensure_aware(asset.date) for asset in assets)
    end = max(ensure_aware(asset.date) for asset in assets)
    return Cluster(
        id=cluster_id,
        assets=assets,
        start=start,
        end=end,
        place=_modal_place(assets),
        centroid=_centroid(assets),
        people=_people(assets),
    )


def cluster_assets(assets: list[Asset], cfg: ClusterConfig | None = None) -> list[Cluster]:
    """Sort by capture time; split on a large time gap, or a geo jump after a shorter gap."""
    config = cfg or ClusterConfig()
    ordered = sorted(assets, key=lambda asset: (ensure_aware(asset.date), asset.uuid))
    if not ordered:
        return []

    groups: list[list[Asset]] = [[ordered[0]]]
    for asset in ordered[1:]:
        if _should_split(groups[-1][-1], asset, config):
            groups.append([asset])
        else:
            groups[-1].append(asset)

    per_day = Counter()
    clusters: list[Cluster] = []
    for group in groups:
        start = min(ensure_aware(asset.date) for asset in group)
        day = start.date().isoformat()
        per_day[day] += 1
        clusters.append(_build_cluster(f"{day}-{per_day[day]:03d}", group))
    return clusters
