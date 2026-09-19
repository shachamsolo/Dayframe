from __future__ import annotations

from pathlib import Path
from typing import Any

from google.auth.exceptions import RefreshError
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow

from dayframe.calendar.events import CalendarError
from dayframe.paths import google_client_secret_path, google_token_path

SCOPES = ("https://www.googleapis.com/auth/calendar.app.created",)
TOKEN_MODE = 0o600


def credentials_available(path: Path | None = None) -> bool:
    target = path or google_token_path()
    return target.is_file()


def save_credentials(creds: Credentials, path: Path | None = None) -> Path:
    target = path or google_token_path()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(creds.to_json(), encoding="utf-8")
    target.chmod(TOKEN_MODE)
    return target


def load_credentials(path: Path | None = None) -> Credentials | None:
    target = path or google_token_path()
    if not target.is_file():
        return None
    creds = Credentials.from_authorized_user_file(str(target), list(SCOPES))
    if creds.expired and creds.refresh_token:
        try:
            creds.refresh(Request())
        except RefreshError as exc:
            raise CalendarError(f"Google token refresh failed; run: dayframe auth ({exc})") from exc
        save_credentials(creds, target)
    if not creds.valid:
        raise CalendarError(f"Google token at {target} is invalid; run: dayframe auth")
    return creds


def run_auth(
    *,
    secret: Path | None = None,
    token_path: Path | None = None,
    flow_cls: Any = InstalledAppFlow,
) -> Path:
    source = secret if secret is not None else google_client_secret_path()
    if source is None:
        raise CalendarError(
            "DAYFRAME_GOOGLE_CLIENT_SECRET is not set; export it to your OAuth desktop client JSON"
        )
    if not source.is_file():
        raise CalendarError(f"client secret not found: {source}")
    flow = flow_cls.from_client_secrets_file(str(source), list(SCOPES))
    # Google's native-app loopback redirect is 127.0.0.1, not localhost.
    # localhost in the authorize URL is a common cause of the generic
    # "400 … request because it is malformed" accounts.google.com page.
    creds = flow.run_local_server(
        host="127.0.0.1",
        port=0,
        redirect_uri_trailing_slash=False,
        open_browser=True,
        access_type="offline",
        prompt="consent",
        authorization_prompt_message=(
            "If a browser did not open, paste this entire URL into Chrome or Safari "
            "(not Cursor's Simple Browser, and do not click a truncated terminal link):\n{url}\n"
        ),
    )
    return save_credentials(creds, token_path)
