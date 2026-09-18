from __future__ import annotations

from datetime import date

from dayframe.baseline.models import AttachedImage, LabeledCluster
from dayframe.baseline.select import cluster_label
from dayframe.cluster.sessions import Cluster


def people_label(names: list[str]) -> str:
    return ", ".join(names) if names else "no people"


def format_digest_line(seq: int, cluster: Cluster, *, send_coordinates: bool = False) -> str:
    n = len(cluster.assets)
    photo_word = "photo" if n == 1 else "photos"
    place = cluster.place or "unknown"
    start = cluster.start.strftime("%H:%M")
    end = cluster.end.strftime("%H:%M")
    line = (
        f"[{cluster_label(seq)}] {start}–{end} · "
        f"{n} {photo_word} · {place} · {people_label(cluster.people)}"
    )
    if send_coordinates and cluster.centroid is not None:
        lat, lon = cluster.centroid
        line += f" · {lat:.4f},{lon:.4f}"
    return line


def label_clusters(clusters: list[Cluster]) -> list[LabeledCluster]:
    return [
        LabeledCluster(label=cluster_label(seq), seq=seq, cluster=cluster)
        for seq, cluster in enumerate(clusters, start=1)
    ]


def build_digest(
    target_date: date,
    labeled: list[LabeledCluster],
    *,
    photo_count: int,
    images: list[AttachedImage],
    send_coordinates: bool = False,
) -> str:
    session_word = "session" if len(labeled) == 1 else "sessions"
    lines = [
        f"{target_date.isoformat()}. {photo_count} photos across "
        f"{len(labeled)} candidate {session_word}.",
        "",
    ]
    for item in labeled:
        lines.append(format_digest_line(item.seq, item.cluster, send_coordinates=send_coordinates))
    if images:
        lines.append("")
        lines.append(f"Attached {len(images)} representative photos:")
        for index, image in enumerate(images, start=1):
            place = image.place or "unknown"
            lines.append(
                f"Image {index}: [{image.label}] {image.captured_at} · {place} · {image.asset_uuid}"
            )
    else:
        lines.append("")
        lines.append("No photos could be attached; decide from the digest alone.")
    lines.append("")
    lines.append("Call submit_day now. Cover every cluster exactly once.")
    return "\n".join(lines)
