from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

from dayframe.photos.models import Asset

PDF_UTIS = frozenset({"com.adobe.pdf", "public.pdf"})
PNG_UTIS = frozenset({"public.png"})


@dataclass(frozen=True)
class Dropped:
    asset: Asset
    reason: str


@dataclass
class PrefilterResult:
    kept: list[Asset]
    dropped: list[Dropped]


def _uti_norm(uti: str | None) -> str:
    return (uti or "").strip().lower()


def _is_pdf(uti: str | None) -> bool:
    value = _uti_norm(uti)
    return value in PDF_UTIS or value.endswith("pdf")


def _is_png(uti: str | None) -> bool:
    value = _uti_norm(uti)
    return value in PNG_UTIS or value.endswith("png")


def reject_reason(asset: Asset, seen: set[str]) -> str | None:
    if asset.screenshot:
        return "screenshot"
    if _is_pdf(asset.uti):
        return "pdf"
    if _is_png(asset.uti) and not asset.has_camera_exif:
        return "png_no_camera"
    if asset.uuid in seen:
        return "already_seen"
    return None


def collapse_bursts(assets: list[Asset]) -> PrefilterResult:
    kept: list[Asset] = []
    dropped: list[Dropped] = []
    groups: dict[str, list[Asset]] = defaultdict(list)
    singles: list[Asset] = []
    for asset in assets:
        if asset.burst and asset.burst_id:
            groups[asset.burst_id].append(asset)
        else:
            singles.append(asset)
    kept.extend(singles)
    for members in groups.values():
        best = max(members, key=lambda item: (item.score_overall or 0.0, item.uuid))
        kept.append(best)
        dropped.extend(
            Dropped(member, "burst_collapse") for member in members if member.uuid != best.uuid
        )
    kept.sort(key=lambda asset: (asset.date, asset.uuid))
    return PrefilterResult(kept=kept, dropped=dropped)


def prefilter(assets: list[Asset], seen: set[str] | None = None) -> PrefilterResult:
    """Conservative rule pass. A false drop is unrecoverable; junk that survives is cheap."""
    known = seen or set()
    dropped: list[Dropped] = []
    surviving: list[Asset] = []
    for asset in assets:
        reason = reject_reason(asset, known)
        if reason is not None:
            dropped.append(Dropped(asset, reason))
        else:
            surviving.append(asset)
    collapsed = collapse_bursts(surviving)
    return PrefilterResult(kept=collapsed.kept, dropped=[*dropped, *collapsed.dropped])
