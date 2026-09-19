from __future__ import annotations

import warnings
from datetime import date
from pathlib import Path

import pytest
from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import InMemorySaver
from PIL import Image

from dayframe.agent.budget import Budget
from dayframe.agent.run import open_checkpointer, run_agent
from dayframe.agent.tools import expand_text, view_content
from dayframe.agent.trace import RawDump, format_replay, load_trace, redact_content, trace_path
from dayframe.baseline.digest import build_session_digest, label_clusters
from dayframe.config import CalendarConfig, Config
from dayframe.photos.reader import FixturePhotosReader
from dayframe.pipeline import run_for_date

warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")

BEACH = Path(__file__).parent / "fixtures" / "days" / "beach.json"


class ScriptedModel:
    def __init__(self, responses: list[AIMessage], *, fail_at: int | None = None) -> None:
        self.responses = list(responses)
        self.fail_at = fail_at
        self.calls = 0

    def bind_tools(self, tools, **kwargs):  # noqa: ANN001
        del tools, kwargs
        return self

    def invoke(self, messages, **kwargs):  # noqa: ANN001
        del messages, kwargs
        if self.fail_at is not None and self.calls == self.fail_at:
            self.calls += 1
            raise RuntimeError("killed")
        if self.calls >= len(self.responses):
            return AIMessage(content="done")
        msg = self.responses[self.calls]
        self.calls += 1
        return msg


class CountingReader:
    def __init__(self, inner: FixturePhotosReader) -> None:
        self.inner = inner
        self.calls = 0

    def assets_added_between(self, start, end):  # noqa: ANN001
        self.calls += 1
        return self.inner.assets_added_between(start, end)

    def asset(self, uuid: str):
        return self.inner.asset(uuid)


def _tc(name: str, args: dict, call_id: str) -> dict:
    return {"name": name, "args": args, "id": call_id, "type": "tool_call"}


def _keep_and_skip() -> list[dict]:
    keep = _tc(
        "write_memory",
        {
            "cluster_id": "c004",
            "title": "Sunset at Palmachim Beach",
            "body": "Swimming with Maya and Noa, then ice cream at sunset.",
            "category": "outing",
            "confidence": 0.9,
        },
        "w1",
    )
    discards = [
        _tc("discard_cluster", {"cluster_id": f"c{i:03d}", "reason": "not memorable"}, f"d{i}")
        for i in (1, 2, 3, 5, 6)
    ]
    return [keep, *discards]


def test_session_digest_is_text_only() -> None:
    result = run_for_date(FixturePhotosReader.from_path(BEACH), "2026-09-17")
    labeled = label_clusters(result.clusters)
    text = build_session_digest(date(2026, 9, 17), labeled, photo_count=14)
    assert text.startswith("Yesterday: 2026-09-17. 14 photos across 6 candidate sessions.")
    assert "[c004]" in text
    assert "submit_day" not in text
    assert "Attached" not in text


def test_expand_text_lists_per_photo_metadata() -> None:
    result = run_for_date(FixturePhotosReader.from_path(BEACH), "2026-09-17")
    cluster = result.clusters[3]
    text = expand_text("c004", cluster)
    assert "[c004] 2026-09-17-004" in text
    assert "people: 2 (Maya, Noa)" in text
    assert "palmachim-1" in text


def test_view_content_attaches_jpegs_and_respects_remaining(tmp_path: Path) -> None:
    jpeg = tmp_path / "beach.jpg"
    Image.new("RGB", (640, 480), "orange").save(jpeg, format="JPEG")
    result = run_for_date(FixturePhotosReader.from_path(BEACH), "2026-09-17")
    cluster = result.clusters[3]
    for asset in cluster.assets:
        asset.path = str(jpeg)
    content, sent = view_content("c004", cluster, count=4, cfg=Config(), remaining=2, already=set())
    assert len(sent) == 2
    assert content[0]["type"] == "text"
    assert sum(1 for block in content if block.get("type") == "image") == 2


def test_budget_consume_does_not_overshoot() -> None:
    budget = Budget(max_images=3, max_turns=12, max_cost_usd=1, timeout_seconds=30)
    assert budget.consume_images(2) == 2
    assert budget.consume_images(4) == 1
    assert budget.consume_images(1) == 0
    budget.release_images(1)
    assert budget.images_remaining == 1


def test_agent_loop_writes_memories_and_trace(isolated_home: Path) -> None:
    model = ScriptedModel(
        [
            AIMessage(
                content="",
                tool_calls=[_tc("expand_cluster", {"cluster_id": "c004"}, "e1")],
                usage_metadata={
                    "input_tokens": 1000,
                    "output_tokens": 40,
                    "total_tokens": 1040,
                },
            ),
            AIMessage(content="", tool_calls=_keep_and_skip()),
            AIMessage(content="done"),
        ]
    )
    result = run_agent(
        target="2026-09-17",
        cfg=Config(),
        reader=FixturePhotosReader.from_path(BEACH),
        model=model,
        checkpointer=InMemorySaver(),
        dump_wire=True,
    )
    assert result.status == "ok"
    assert result.turns == 3
    assert [memory.title for memory in result.memories] == ["Sunset at Palmachim Beach"]
    skipped = {item.cluster_id for item in result.discarded}
    assert skipped == {"c001", "c002", "c003", "c005", "c006"}
    assert result.cost_usd > 0
    rows = load_trace(result.trace_path)
    assert len(rows) == 3
    assert rows[0]["tool_calls"][0]["name"] == "expand_cluster"
    assert "Palmachim Beach" in rows[0]["tool_results"][0]["content"]
    replay = format_replay(rows, run_id=result.run_id)
    assert "Turn 1" in replay
    assert "expand_cluster" in replay
    wire = isolated_home / "traces" / "run_2026-09-17.raw_request.json"
    assert wire.is_file()


def test_view_photos_enforces_image_budget(tmp_path: Path, isolated_home: Path) -> None:
    jpeg = tmp_path / "beach.jpg"
    Image.new("RGB", (32, 32), "blue").save(jpeg, format="JPEG")
    reader = FixturePhotosReader.from_path(BEACH)
    photos = run_for_date(reader, "2026-09-17")
    for asset in photos.clusters[3].assets:
        asset.path = str(jpeg)
    cfg = Config()
    cfg.budget.max_images = 1
    model = ScriptedModel(
        [
            AIMessage(
                content="",
                tool_calls=[_tc("view_photos", {"cluster_id": "c004", "count": 4}, "v1")],
            ),
            AIMessage(content="", tool_calls=_keep_and_skip()),
            AIMessage(content="done"),
        ]
    )
    result = run_agent(
        target="2026-09-17",
        cfg=cfg,
        reader=reader,
        model=model,
        checkpointer=InMemorySaver(),
        dump_wire=False,
    )
    assert result.images_sent == 1
    rows = load_trace(trace_path(result.run_id))
    view = rows[0]["tool_results"][0]["content"]
    images = [block for block in view if isinstance(block, dict) and block.get("type") == "image"]
    assert len(images) == 1
    assert images[0].get("bytes", 0) > 0 or "data" not in images[0]


def test_resume_skips_completed_nodes(isolated_home: Path) -> None:
    conn, saver = open_checkpointer(isolated_home / "dayframe.db")
    inner = FixturePhotosReader.from_path(BEACH)
    try:
        first_reader = CountingReader(inner)
        killer = ScriptedModel(
            [
                AIMessage(
                    content="",
                    tool_calls=[_tc("expand_cluster", {"cluster_id": "c004"}, "e1")],
                )
            ],
            fail_at=1,
        )
        with pytest.raises(RuntimeError, match="killed"):
            run_agent(
                target="2026-09-17",
                cfg=Config(),
                reader=first_reader,
                model=killer,
                checkpointer=saver,
                dump_wire=False,
            )
        assert first_reader.calls == 1

        second_reader = CountingReader(inner)
        continuation = ScriptedModel(
            [
                AIMessage(content="", tool_calls=_keep_and_skip()),
                AIMessage(content="done"),
            ]
        )
        result = run_agent(
            target="2026-09-17",
            cfg=Config(),
            reader=second_reader,
            model=continuation,
            checkpointer=saver,
            dump_wire=False,
        )
        assert result.resumed is True
        assert second_reader.calls == 0
        assert result.memories[0].title == "Sunset at Palmachim Beach"
        assert result.turns >= 2
    finally:
        conn.close()


def test_review_mode_interrupts_until_resume() -> None:
    cfg = Config(calendar=CalendarConfig(approval_mode="review"))
    saver = InMemorySaver()
    model = ScriptedModel(
        [
            AIMessage(
                content="",
                tool_calls=[_tc("expand_cluster", {"cluster_id": "c004"}, "e1")],
            ),
            AIMessage(content="", tool_calls=_keep_and_skip()),
            AIMessage(content="done"),
        ]
    )
    first = run_agent(
        target="2026-09-17",
        cfg=cfg,
        reader=FixturePhotosReader.from_path(BEACH),
        model=model,
        checkpointer=saver,
        dump_wire=False,
    )
    assert first.interrupted is True
    assert first.status == "pending_review"
    assert first.memories[0].title == "Sunset at Palmachim Beach"
    calls = model.calls

    parked = run_agent(
        target="2026-09-17",
        cfg=cfg,
        reader=FixturePhotosReader.from_path(BEACH),
        model=model,
        checkpointer=saver,
        dump_wire=False,
    )
    assert parked.interrupted is True
    assert parked.resumed is True
    assert model.calls == calls

    done = run_agent(
        target="2026-09-17",
        cfg=cfg,
        reader=FixturePhotosReader.from_path(BEACH),
        model=model,
        checkpointer=saver,
        dump_wire=False,
        resume=True,
    )
    assert done.interrupted is False
    assert done.status == "ok"
    assert done.memories[0].title == "Sunset at Palmachim Beach"


def test_raw_dump_redacts_images(tmp_path: Path) -> None:
    path = tmp_path / "raw.json"
    handler = RawDump(path)
    messages = [
        [
            {
                "type": "human",
                "content": [
                    {"type": "text", "text": "hi"},
                    {
                        "type": "image",
                        "source_type": "base64",
                        "data": "abc" * 20,
                        "mime_type": "image/jpeg",
                    },
                ],
            }
        ]
    ]
    handler.on_chat_model_start({}, messages)
    dumped = path.read_text(encoding="utf-8")
    assert "abcabc" not in dumped
    assert '"bytes"' in dumped
    redacted = redact_content(messages[0][0]["content"])
    assert redacted[1]["type"] == "image"
    assert redacted[1]["bytes"] == 60
