from datetime import date
from pathlib import Path

from PIL import Image

from dayframe.baseline.anthropic import cost_usd, format_http_error, parse_submission
from dayframe.baseline.digest import build_digest, format_digest_line, label_clusters
from dayframe.baseline.models import DaySubmission, DiscardDraft, MemoryDraft
from dayframe.baseline.run import apply_decisions, run_baseline
from dayframe.baseline.select import allocate_slots, pick_representatives
from dayframe.config import Config
from dayframe.photos.reader import FixturePhotosReader
from dayframe.pipeline import run_for_date
from dayframe.prompt import load_prompt
from tests.factories import make_asset

BEACH = Path(__file__).parent / "fixtures" / "days" / "beach.json"


def test_pick_representatives_spreads_across_time() -> None:
    assets = [
        make_asset("a", 16, 0, score_overall=0.1),
        make_asset("b", 16, 20, score_overall=0.9),
        make_asset("c", 17, 0, score_overall=0.2),
        make_asset("d", 18, 0, score_overall=0.95),
        make_asset("e", 19, 0, score_overall=0.3),
        make_asset("f", 19, 40, score_overall=0.4),
    ]
    picked = pick_representatives(assets, 3)
    assert [asset.uuid for asset in picked] == ["b", "d", "f"]


def test_allocate_slots_covers_every_cluster_then_favours_rich_ones() -> None:
    result = run_for_date(FixturePhotosReader.from_path(BEACH), "2026-09-17")
    slots = allocate_slots(result.clusters, 8)
    assert set(slots) == {cluster.id for cluster in result.clusters}
    assert sum(slots.values()) == 8
    palmachim = next(
        cluster.id for cluster in result.clusters if cluster.place == "Palmachim Beach"
    )
    assert slots[palmachim] >= 2


def test_digest_uses_c_labels() -> None:
    result = run_for_date(FixturePhotosReader.from_path(BEACH), "2026-09-17")
    labeled = label_clusters(result.clusters)
    line = format_digest_line(4, result.clusters[3])
    assert line == "[c004] 16:12–19:40 · 4 photos · Palmachim Beach · Maya, Noa"
    text = build_digest(date(2026, 9, 17), labeled, photo_count=14, images=[])
    assert "2026-09-17. 14 photos across 6 candidate sessions." in text
    assert "[c004]" in text
    assert "No photos could be attached" in text


def test_parse_submit_day_and_cost() -> None:
    body = {
        "content": [
            {
                "type": "tool_use",
                "name": "submit_day",
                "input": {
                    "memories": [
                        {
                            "cluster_id": "c004",
                            "title": "Sunset at Palmachim Beach",
                            "body": "Swimming with Maya and Noa, then ice cream.",
                            "category": "outing",
                            "confidence": 0.86,
                        }
                    ],
                    "discards": [{"cluster_id": "c003", "reason": "single context-free image"}],
                },
            }
        ],
        "usage": {"input_tokens": 1000, "output_tokens": 200},
    }
    submission = parse_submission(body)
    assert submission.memories[0].title == "Sunset at Palmachim Beach"
    assert cost_usd("claude-sonnet-4-5", 1_000_000, 0) == 3.0


def test_format_http_error_extracts_anthropic_message() -> None:
    raw = (
        '{"type":"error","error":{"type":"invalid_request_error",'
        '"message":"Your credit balance is too low to access the Anthropic API."}}'
    )
    assert format_http_error(400, raw) == (
        "anthropic: Your credit balance is too low to access the Anthropic API."
    )


def test_confidence_gate_and_omitted_clusters() -> None:
    result = run_for_date(FixturePhotosReader.from_path(BEACH), "2026-09-17")
    labeled = label_clusters(result.clusters)
    memories, discarded, omitted = apply_decisions(
        labeled,
        [
            MemoryDraft(
                cluster_id="c004",
                title="Sunset at Palmachim Beach",
                body="Afternoon at the beach.",
                category="outing",
                confidence=0.86,
            ),
            MemoryDraft(
                cluster_id="c001",
                title="Coffee at home",
                body="Morning coffee.",
                category="other",
                confidence=0.4,
            ),
        ],
        [DiscardDraft(cluster_id="c002", reason="office walk-through")],
        min_confidence=0.7,
    )
    assert [item.cluster_id for item in memories] == ["c004"]
    reasons = {item.cluster_id: item.reason for item in discarded}
    assert reasons["c001"] == "discarded_low_confidence"
    assert reasons["c002"] == "office walk-through"
    assert "c003" in omitted
    assert reasons["c003"] == "model_omitted"


def test_run_baseline_uses_complete_fn(tmp_path: Path) -> None:
    jpeg = tmp_path / "beach.jpg"
    Image.new("RGB", (640, 480), "orange").save(jpeg, format="JPEG")
    reader = FixturePhotosReader.from_path(BEACH)
    photos = run_for_date(reader, "2026-09-17")
    for asset in photos.clusters[3].assets:
        asset.path = str(jpeg)

    def fake_complete(**kwargs):
        assert "Palmachim Beach" in kwargs["digest"]
        assert kwargs["images"]
        assert kwargs["system"] == load_prompt("baseline_v1")
        submission = DaySubmission(
            memories=[
                MemoryDraft(
                    cluster_id="c004",
                    title="Sunset at Palmachim Beach",
                    body="Swimming, ice cream, sunset.",
                    category="outing",
                    confidence=0.9,
                )
            ],
            discards=[
                DiscardDraft(cluster_id=label, reason="not memorable")
                for label in ("c001", "c002", "c003", "c005", "c006")
            ],
        )
        return submission, {"usage": {"input_tokens": 500, "output_tokens": 80}}

    baseline = run_baseline(photos, Config(), complete_fn=fake_complete)
    assert baseline.images_sent == 2
    assert baseline.memories[0].title == "Sunset at Palmachim Beach"
    assert baseline.cost_usd > 0
    assert len(baseline.discarded) == 5
