from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path

from googleapiclient.errors import HttpError
from tenacity import wait_none

from dayframe.baseline.digest import label_clusters
from dayframe.baseline.models import MemoryDraft
from dayframe.calendar.auth import TOKEN_MODE, run_auth, save_credentials
from dayframe.calendar.events import build_memory_event, event_body, event_id_for
from dayframe.calendar.google import GoogleWriter, persist_calendar_id
from dayframe.calendar.ics import write_ics
from dayframe.calendar.write import publish_memories, undo_run
from dayframe.cluster.sessions import Cluster, cluster_assets
from dayframe.config import Config
from dayframe.paths import config_path, google_token_path
from tests.factories import TZ, make_asset

EXAMPLE = Path(__file__).resolve().parents[1] / "config.example.toml"


def _http_error(status: int) -> HttpError:
    resp = type("Resp", (), {"status": status, "reason": "error"})()
    payload = json.dumps({"error": {"code": status, "message": "error"}}).encode()
    return HttpError(resp, payload)


class FakeAPI:
    def __init__(self) -> None:
        self.calendars: dict[str, dict] = {}
        self.events: dict[tuple[str, str], dict] = {}
        self.list_items: list[dict] = []
        self.inserts = 0
        self.updates = 0
        self.deletes = 0
        self.fail_inserts = 0
        self.fail_status = 429

    def calendars_get(self, calendar_id: str) -> dict | None:
        return self.calendars.get(calendar_id)

    def calendars_insert(self, body: dict) -> dict:
        cal_id = "cal_dayframe"
        item = {"id": cal_id, **body}
        self.calendars[cal_id] = item
        self.list_items.append({"id": cal_id, "summary": body["summary"]})
        return {"id": cal_id}

    def calendar_list(self) -> list[dict]:
        return list(self.list_items)

    def events_insert(self, calendar_id: str, body: dict) -> dict:
        if self.fail_inserts > 0:
            self.fail_inserts -= 1
            raise _http_error(self.fail_status)
        event_id = str(body["id"])
        key = (calendar_id, event_id)
        if key in self.events:
            raise _http_error(409)
        self.events[key] = dict(body)
        self.inserts += 1
        return {"id": event_id}

    def events_update(self, calendar_id: str, event_id: str, body: dict) -> dict:
        self.events[(calendar_id, event_id)] = dict(body)
        self.updates += 1
        return {"id": event_id}

    def events_list_by_run(self, calendar_id: str, run_id: str) -> list[dict]:
        found: list[dict] = []
        for (cid, event_id), body in self.events.items():
            if cid != calendar_id:
                continue
            private = (body.get("extendedProperties") or {}).get("private") or {}
            if private.get("dayframe_run_id") == run_id:
                found.append({"id": event_id, **body})
        return found

    def events_delete(self, calendar_id: str, event_id: str) -> None:
        self.events.pop((calendar_id, event_id), None)
        self.deletes += 1


def _sample(place: str = "Palmachim Beach") -> tuple[MemoryDraft, Cluster, Config]:
    asset = make_asset(
        "p1",
        16,
        12,
        place_name=place,
        persons=["Maya"],
        latitude=31.933,
        longitude=34.706,
    )
    later = make_asset(
        "p2",
        19,
        40,
        place_name=place,
        persons=["Maya", "Noa"],
        latitude=31.933,
        longitude=34.706,
    )
    cluster = cluster_assets([asset, later])[0]
    memory = MemoryDraft(
        cluster_id="c001",
        title="Sunset at Palmachim Beach",
        body="Swimming with Maya and Noa, then ice cream at sunset.",
        category="outing",
        confidence=0.86,
    )
    return memory, cluster, Config()


def test_event_id_is_sha1_hex() -> None:
    cluster_id = "2026-09-17-004"
    expected = hashlib.sha1(f"dayframe:{cluster_id}".encode()).hexdigest()
    assert event_id_for(cluster_id) == expected
    assert set(expected) <= set("0123456789abcdef")


def test_event_body_matches_spec() -> None:
    memory, cluster, cfg = _sample()
    event = build_memory_event(memory, cluster, cfg.calendar, "run_2026-09-17")
    body = event_body(event)
    assert body["id"] == event_id_for(cluster.id)
    assert body["summary"] == "Dayframe: Sunset at Palmachim Beach"
    assert body["location"] == "Palmachim Beach"
    assert body["transparency"] == "transparent"
    start = body["start"]
    assert start["timeZone"] == "Asia/Jerusalem"
    assert start["dateTime"].startswith("2026-09-17T16:12:00")
    private = body["extendedProperties"]["private"]
    assert private["dayframe_run_id"] == "run_2026-09-17"
    assert private["dayframe_cluster_id"] == cluster.id
    assert private["dayframe_confidence"] == "0.86"


def test_zero_duration_cluster_gets_one_minute() -> None:
    asset = make_asset("x", 10, 0, place_name="home")
    cluster = cluster_assets([asset])[0]
    memory = MemoryDraft(
        cluster_id="c001",
        title="At home",
        body="A quiet morning.",
        category="other",
        confidence=0.8,
    )
    event = build_memory_event(memory, cluster, Config().calendar, "run_2026-09-17")
    assert (event.end - event.start).total_seconds() == 60


def test_ics_fallback_is_deterministic(isolated_home: Path) -> None:
    memory, cluster, cfg = _sample()
    labeled = label_clusters([cluster])
    first = publish_memories([memory], labeled, cfg, "run_2026-09-17", credentials=False)
    second = publish_memories([memory], labeled, cfg, "run_2026-09-17", credentials=False)
    assert first.backend == "ics"
    assert first.ics_path == isolated_home / "out" / "run_2026-09-17.ics"
    text = first.ics_path.read_text(encoding="utf-8")
    assert "BEGIN:VEVENT" in text
    assert "TRANSP:TRANSPARENT" in text
    assert "Dayframe: Sunset at Palmachim Beach" in text
    assert f"UID:{event_id_for(cluster.id)}@dayframe.local" in text
    first_uid = first.ics_path.read_text(encoding="utf-8").split("DTSTAMP")[0]
    second_uid = second.ics_path.read_text(encoding="utf-8").split("DTSTAMP")[0]
    assert first_uid == second_uid


def test_google_insert_then_409_updates() -> None:
    memory, cluster, cfg = _sample()
    labeled = label_clusters([cluster])
    api = FakeAPI()
    first = publish_memories([memory], labeled, cfg, "run_2026-09-17", api=api, wait=wait_none())
    second = publish_memories([memory], labeled, cfg, "run_2026-09-17", api=api, wait=wait_none())
    assert first.backend == "google"
    assert api.inserts == 1
    assert second.backend == "google"
    assert api.inserts == 1
    assert api.updates == 1
    stored = next(iter(api.events.values()))
    assert stored["id"] == event_id_for(cluster.id)


def test_google_retries_then_succeeds() -> None:
    memory, cluster, cfg = _sample()
    labeled = label_clusters([cluster])
    api = FakeAPI()
    api.fail_inserts = 2
    result = publish_memories([memory], labeled, cfg, "run_2026-09-17", api=api, wait=wait_none())
    assert result.backend == "google"
    assert api.inserts == 1


def test_ensure_calendar_creates_then_reuses() -> None:
    api = FakeAPI()
    cfg = Config().calendar
    writer = GoogleWriter(api, cfg, wait=wait_none())
    first = writer.ensure_calendar()
    second = writer.ensure_calendar()
    assert first == second == "cal_dayframe"
    assert len(api.list_items) == 1


def test_ensure_calendar_creates_when_list_forbidden() -> None:
    class NoListAPI(FakeAPI):
        def calendar_list(self) -> list[dict]:
            raise _http_error(403)

    api = NoListAPI()
    writer = GoogleWriter(api, Config().calendar, wait=wait_none())
    assert writer.ensure_calendar() == "cal_dayframe"


def test_persist_calendar_id_updates_config(isolated_home: Path) -> None:
    target = config_path()
    target.write_text(EXAMPLE.read_text(encoding="utf-8"), encoding="utf-8")
    cfg = Config.load(target)
    assert persist_calendar_id("abc@group.calendar.google.com", cfg.calendar)
    assert cfg.calendar.calendar_id == "abc@group.calendar.google.com"
    reloaded = Config.load(target)
    assert reloaded.calendar.calendar_id == "abc@group.calendar.google.com"


def test_undo_deletes_google_events_and_ics(isolated_home: Path) -> None:
    memory, cluster, cfg = _sample()
    labeled = label_clusters([cluster])
    api = FakeAPI()
    publish_memories([memory], labeled, cfg, "run_2026-09-17", api=api, wait=wait_none())
    write_ics(
        [build_memory_event(memory, cluster, cfg.calendar, "run_2026-09-17")],
        isolated_home / "out" / "run_2026-09-17.ics",
    )
    from dayframe.store.db import connect, init_db, persist_baseline_run

    conn = init_db(connect())
    try:
        persist_baseline_run(
            conn,
            run_id="run_2026-09-17",
            target_date="2026-09-17",
            started_at=datetime(2026, 9, 18, 7, 0, tzinfo=TZ),
            finished_at=datetime(2026, 9, 18, 7, 1, tzinfo=TZ),
            labeled=labeled,
            memories=[memory],
            discarded=[],
            assets=cluster.assets,
            provider="anthropic",
            model="claude-sonnet-4-5",
            prompt_version="system_v1",
            images_sent=0,
            input_tokens=0,
            output_tokens=0,
            cost_usd=0.0,
        )
    finally:
        conn.close()

    result = undo_run("run_2026-09-17", cfg, api=api, wait=wait_none())
    assert result.google_deleted == 1
    assert result.ics_deleted is True
    assert result.local_deleted is True
    assert api.events == {}
    assert not (isolated_home / "out" / "run_2026-09-17.ics").exists()


def test_auth_saves_token_mode_0600(isolated_home: Path) -> None:
    secret = isolated_home / "client.json"
    secret.write_text("{}", encoding="utf-8")

    class _Creds:
        def to_json(self) -> str:
            return json.dumps(
                {
                    "token": "abc",
                    "refresh_token": "def",
                    "token_uri": "https://oauth2.googleapis.com/token",
                    "client_id": "id",
                    "client_secret": "sec",
                }
            )

    class _Flow:
        @classmethod
        def from_client_secrets_file(cls, path, scopes):  # noqa: ANN001
            inst = cls()
            inst.path = path
            inst.scopes = scopes
            return inst

        def run_local_server(self, **kwargs):  # noqa: ANN003
            del kwargs
            return _Creds()

    path = run_auth(secret=secret, flow_cls=_Flow)
    assert path == google_token_path()
    assert path.stat().st_mode & 0o777 == TOKEN_MODE
    assert "refresh_token" in path.read_text(encoding="utf-8")


def test_save_credentials_roundtrip_json(isolated_home: Path) -> None:
    class _Creds:
        def to_json(self) -> str:
            return '{"token":"x"}'

    path = save_credentials(_Creds())  # type: ignore[arg-type]
    assert path.read_text(encoding="utf-8") == '{"token":"x"}'
