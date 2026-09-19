from __future__ import annotations

import re
from typing import Any

from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from tenacity import Retrying, retry_if_exception, stop_after_attempt, wait_exponential_jitter
from tenacity.wait import wait_base

from dayframe.calendar.events import CalendarError, MemoryEvent, event_body
from dayframe.config import CalendarConfig
from dayframe.paths import config_path

_CALENDAR_ID_LINE = re.compile(r'^(\s*calendar_id\s*=\s*)".*?"', re.MULTILINE)


def http_status(exc: BaseException) -> int | None:
    if not isinstance(exc, HttpError):
        return None
    try:
        return int(exc.resp.status)
    except (TypeError, ValueError, AttributeError):
        return None


def _retryable(exc: BaseException) -> bool:
    return http_status(exc) in {429, 500, 502, 503}


class GoogleCalendarAPI:
    def __init__(self, service: Any) -> None:
        self.service = service

    def calendars_get(self, calendar_id: str) -> dict[str, Any] | None:
        try:
            return self.service.calendars().get(calendarId=calendar_id).execute()
        except HttpError as exc:
            if http_status(exc) == 404:
                return None
            raise

    def calendars_insert(self, body: dict[str, Any]) -> dict[str, Any]:
        return self.service.calendars().insert(body=body).execute()

    def calendar_list(self) -> list[dict[str, Any]]:
        items: list[dict[str, Any]] = []
        page_token = None
        while True:
            resp = self.service.calendarList().list(pageToken=page_token).execute()
            items.extend(resp.get("items") or [])
            page_token = resp.get("nextPageToken")
            if not page_token:
                return items

    def events_insert(self, calendar_id: str, body: dict[str, Any]) -> dict[str, Any]:
        return self.service.events().insert(calendarId=calendar_id, body=body).execute()

    def events_update(
        self, calendar_id: str, event_id: str, body: dict[str, Any]
    ) -> dict[str, Any]:
        return (
            self.service.events()
            .update(calendarId=calendar_id, eventId=event_id, body=body)
            .execute()
        )

    def events_list_by_run(self, calendar_id: str, run_id: str) -> list[dict[str, Any]]:
        items: list[dict[str, Any]] = []
        page_token = None
        while True:
            resp = (
                self.service.events()
                .list(
                    calendarId=calendar_id,
                    privateExtendedProperty=f"dayframe_run_id={run_id}",
                    showDeleted=False,
                    maxResults=2500,
                    pageToken=page_token,
                )
                .execute()
            )
            items.extend(resp.get("items") or [])
            page_token = resp.get("nextPageToken")
            if not page_token:
                return items

    def events_delete(self, calendar_id: str, event_id: str) -> None:
        try:
            self.service.events().delete(calendarId=calendar_id, eventId=event_id).execute()
        except HttpError as exc:
            if http_status(exc) in {404, 410}:
                return
            raise


def build_api(creds: Credentials) -> GoogleCalendarAPI:
    service = build("calendar", "v3", credentials=creds, cache_discovery=False)
    return GoogleCalendarAPI(service)


def persist_calendar_id(calendar_id: str, cfg: CalendarConfig | None = None) -> bool:
    if cfg is not None:
        cfg.calendar_id = calendar_id
    target = config_path()
    if not target.is_file():
        return False
    text = target.read_text(encoding="utf-8")
    escaped = calendar_id.replace("\\", "\\\\").replace('"', '\\"')
    new, n = _CALENDAR_ID_LINE.subn(rf'\1"{escaped}"', text, count=1)
    if n == 0 or new == text:
        return False
    target.write_text(new, encoding="utf-8")
    return True


class GoogleWriter:
    def __init__(
        self,
        api: Any,
        cfg: CalendarConfig,
        *,
        wait: wait_base | None = None,
    ) -> None:
        self.api = api
        self.cfg = cfg
        self._wait = wait or wait_exponential_jitter(initial=1, max=16)

    def find_calendar(self) -> str | None:
        calendar_id = self.cfg.calendar_id.strip()
        if calendar_id:
            existing = self.api.calendars_get(calendar_id)
            if existing is not None:
                return calendar_id
        found = self._find_listed_calendar()
        if found:
            persist_calendar_id(found, self.cfg)
            return found
        return None

    def _find_listed_calendar(self) -> str | None:
        # calendar.app.created cannot call calendarList.list (needs calendar.calendarlist).
        try:
            items = self.api.calendar_list()
        except HttpError as exc:
            if http_status(exc) == 403:
                return None
            raise
        for item in items:
            if item.get("summary") == self.cfg.name and item.get("id"):
                return str(item["id"])
        return None

    def ensure_calendar(self) -> str:
        found = self.find_calendar()
        if found:
            return found
        created = self.api.calendars_insert(
            {
                "summary": self.cfg.name,
                "timeZone": self.cfg.timezone,
                "description": "Memories written by Dayframe.",
            }
        )
        new_id = created.get("id")
        if not new_id:
            raise CalendarError("Google Calendar insert returned no id")
        persist_calendar_id(str(new_id), self.cfg)
        return str(new_id)

    def upsert_event(self, calendar_id: str, event: MemoryEvent) -> str:
        body = event_body(event)
        for attempt in Retrying(
            retry=retry_if_exception(_retryable),
            stop=stop_after_attempt(5),
            wait=self._wait,
            reraise=True,
        ):
            with attempt:
                return self._upsert_once(calendar_id, body)
        raise CalendarError(f"failed to write event {event.event_id}")

    def _upsert_once(self, calendar_id: str, body: dict[str, Any]) -> str:
        try:
            created = self.api.events_insert(calendar_id, body)
            return str(created.get("id") or body["id"])
        except HttpError as exc:
            if http_status(exc) != 409:
                raise
            updated = self.api.events_update(calendar_id, str(body["id"]), body)
            return str(updated.get("id") or body["id"])

    def delete_run_events(self, calendar_id: str, run_id: str) -> int:
        events = self.api.events_list_by_run(calendar_id, run_id)
        deleted = 0
        for event in events:
            event_id = event.get("id")
            if not event_id:
                continue
            self._delete_once(calendar_id, str(event_id))
            deleted += 1
        return deleted

    def _delete_once(self, calendar_id: str, event_id: str) -> None:
        for attempt in Retrying(
            retry=retry_if_exception(_retryable),
            stop=stop_after_attempt(5),
            wait=self._wait,
            reraise=True,
        ):
            with attempt:
                self.api.events_delete(calendar_id, event_id)
                return
