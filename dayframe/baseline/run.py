from __future__ import annotations

import time

from dayframe.baseline.anthropic import (
    BaselineProviderError,
    api_model_name,
    complete,
    cost_usd,
    usage_tokens,
)
from dayframe.baseline.digest import build_digest, label_clusters
from dayframe.baseline.models import (
    AttachedImage,
    BaselineResult,
    DiscardDraft,
    LabeledCluster,
    MemoryDraft,
)
from dayframe.baseline.select import M2_MAX_IMAGES, allocate_slots, pick_representatives
from dayframe.config import Config
from dayframe.images.prepare import asset_image_path, prepare_image
from dayframe.pipeline import PipelineResult
from dayframe.prompt import load_prompt
from dayframe.window import ensure_aware


def run_baseline(
    result: PipelineResult,
    cfg: Config,
    *,
    api_key: str | None = None,
    max_images: int = M2_MAX_IMAGES,
    complete_fn=complete,
) -> BaselineResult:
    labeled = label_clusters(result.clusters)
    images = collect_images(labeled, cfg, max_images=max_images)[0]
    digest = build_digest(
        result.target_date,
        labeled,
        photo_count=len(result.kept),
        images=images,
        send_coordinates=cfg.privacy.send_coordinates,
    )
    model = api_model_name(cfg.provider.model, cfg.provider.name)
    started = time.perf_counter()
    submission, raw = complete_fn(
        system=load_prompt("baseline_v1"),
        digest=digest,
        images=images,
        model=model,
        api_key=api_key,
        timeout_seconds=cfg.budget.timeout_seconds,
    )
    wall = time.perf_counter() - started
    input_tokens, output_tokens = usage_tokens(raw)
    memories, discarded, omitted = apply_decisions(
        labeled,
        submission.memories,
        submission.discards,
        min_confidence=cfg.calendar.min_confidence,
    )
    return BaselineResult(
        labeled=labeled,
        memories=memories,
        discarded=discarded,
        skipped_unreviewed=omitted,
        images_sent=len(images),
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cost_usd=cost_usd(model, input_tokens, output_tokens),
        wall_seconds=wall,
        model=model,
        raw_response=raw,
    )


def collect_images(
    labeled: list[LabeledCluster],
    cfg: Config,
    *,
    max_images: int = M2_MAX_IMAGES,
) -> tuple[list[AttachedImage], list[str]]:
    clusters = [item.cluster for item in labeled]
    slots = allocate_slots(clusters, max_images)
    by_id = {item.cluster.id: item for item in labeled}
    attached: list[AttachedImage] = []
    skipped: list[str] = []
    for cluster_id, count in slots.items():
        item = by_id[cluster_id]
        chosen = pick_representatives(item.cluster.assets, count)
        prepared_any = False
        for asset in chosen:
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
            attached.append(
                AttachedImage(
                    label=item.label,
                    asset_uuid=asset.uuid,
                    captured_at=ensure_aware(asset.date).strftime("%H:%M"),
                    place=asset.place_name or item.cluster.place,
                    jpeg_bytes=prepared.jpeg_bytes,
                    media_type=prepared.media_type,
                )
            )
            prepared_any = True
        if not prepared_any and count > 0:
            skipped.append(item.label)
    return attached, skipped


def apply_decisions(
    labeled: list[LabeledCluster],
    memories: list[MemoryDraft],
    discards: list[DiscardDraft],
    *,
    min_confidence: float,
) -> tuple[list[MemoryDraft], list[DiscardDraft], list[str]]:
    known = {item.label: item for item in labeled}
    kept: list[MemoryDraft] = []
    discarded: list[DiscardDraft] = []
    assigned: set[str] = set()

    for memory in memories:
        label = memory.cluster_id
        if label not in known or label in assigned:
            continue
        assigned.add(label)
        if memory.confidence < min_confidence:
            discarded.append(DiscardDraft(cluster_id=label, reason="discarded_low_confidence"))
        else:
            kept.append(memory)

    for discard in discards:
        label = discard.cluster_id
        if label not in known or label in assigned:
            continue
        assigned.add(label)
        discarded.append(discard)

    omitted = [item.label for item in labeled if item.label not in assigned]
    for label in omitted:
        discarded.append(DiscardDraft(cluster_id=label, reason="model_omitted"))
    return kept, discarded, omitted


def require_anthropic(cfg: Config) -> None:
    if cfg.provider.name != "anthropic":
        raise BaselineProviderError(f"M2 baseline supports anthropic only, got {cfg.provider.name}")
