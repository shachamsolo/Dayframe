from __future__ import annotations

import json
import os
from typing import Any

import httpx

from dayframe.baseline.models import AttachedImage, DaySubmission
from dayframe.paths import api_key as env_api_key

ANTHROPIC_URL = "https://api.anthropic.com/v1/messages"
ANTHROPIC_VERSION = "2023-06-01"
TOOL_NAME = "submit_day"

# USD per million tokens. Image tokens are already in input_tokens.
_PRICES: dict[str, tuple[float, float]] = {
    "claude-sonnet-4-5": (3.0, 15.0),
    "claude-sonnet-4-20250514": (3.0, 15.0),
    "claude-3-5-sonnet-latest": (3.0, 15.0),
    "claude-3-5-sonnet-20241022": (3.0, 15.0),
    "claude-3-haiku-20240307": (0.25, 1.25),
    "claude-haiku-4-5": (1.0, 5.0),
}

SUBMIT_DAY_TOOL: dict[str, Any] = {
    "name": TOOL_NAME,
    "description": (
        "Submit the day's memories and discards. Call exactly once. "
        "Every cluster from the digest must appear in memories or discards, never both."
    ),
    "input_schema": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "memories": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "cluster_id": {"type": "string"},
                        "title": {"type": "string"},
                        "body": {"type": "string"},
                        "category": {
                            "type": "string",
                            "enum": [
                                "trip",
                                "family",
                                "meal",
                                "celebration",
                                "outing",
                                "hobby",
                                "nature",
                                "other",
                            ],
                        },
                        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                    },
                    "required": ["cluster_id", "title", "body", "category", "confidence"],
                },
            },
            "discards": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "cluster_id": {"type": "string"},
                        "reason": {"type": "string"},
                    },
                    "required": ["cluster_id", "reason"],
                },
            },
        },
        "required": ["memories", "discards"],
    },
}


class BaselineProviderError(RuntimeError):
    pass


def api_model_name(model: str, provider: str = "anthropic") -> str:
    prefix = f"{provider}:"
    if model.startswith(prefix):
        return model[len(prefix) :]
    return model


def cost_usd(model: str, input_tokens: int, output_tokens: int) -> float:
    in_price, out_price = _PRICES.get(model, (3.0, 15.0))
    return (input_tokens / 1_000_000) * in_price + (output_tokens / 1_000_000) * out_price


def complete(
    *,
    system: str,
    digest: str,
    images: list[AttachedImage],
    model: str,
    api_key: str | None = None,
    timeout_seconds: int = 180,
    temperature: float = 0.3,
    max_tokens: int = 2048,
) -> tuple[DaySubmission, dict[str, Any]]:
    key = api_key if api_key is not None else env_api_key()
    if not key:
        raise BaselineProviderError(
            "DAYFRAME_API_KEY is not set; add it to .env or export it (never config.toml)"
        )
    payload = {
        "model": model,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "system": system,
        "tools": [SUBMIT_DAY_TOOL],
        "tool_choice": {"type": "tool", "name": TOOL_NAME},
        "messages": [{"role": "user", "content": _user_content(digest, images)}],
    }
    url = os.environ.get("DAYFRAME_ANTHROPIC_BASE_URL", ANTHROPIC_URL)
    try:
        response = httpx.post(
            url,
            headers={
                "x-api-key": key,
                "anthropic-version": ANTHROPIC_VERSION,
                "content-type": "application/json",
            },
            json=payload,
            timeout=timeout_seconds,
        )
    except httpx.HTTPError as exc:
        raise BaselineProviderError(f"anthropic request failed: {exc}") from exc
    if response.status_code >= 400:
        raise BaselineProviderError(format_http_error(response.status_code, response.text))
    try:
        body = response.json()
    except json.JSONDecodeError as exc:
        raise BaselineProviderError("anthropic returned non-JSON") from exc
    return parse_submission(body), body


def parse_submission(body: dict[str, Any]) -> DaySubmission:
    for block in body.get("content") or []:
        if not isinstance(block, dict):
            continue
        if block.get("type") == "tool_use" and block.get("name") == TOOL_NAME:
            raw = block.get("input") or {}
            if isinstance(raw, str):
                try:
                    raw = json.loads(raw)
                except json.JSONDecodeError as exc:
                    raise BaselineProviderError("submit_day input was not JSON") from exc
            if not isinstance(raw, dict):
                raise BaselineProviderError("submit_day input was not an object")
            return DaySubmission.model_validate(raw)
    raise BaselineProviderError("anthropic response had no submit_day tool call")


def usage_tokens(body: dict[str, Any]) -> tuple[int, int]:
    usage = body.get("usage") or {}
    return int(usage.get("input_tokens") or 0), int(usage.get("output_tokens") or 0)


def _user_content(digest: str, images: list[AttachedImage]) -> list[dict[str, Any]]:
    content: list[dict[str, Any]] = [{"type": "text", "text": digest}]
    for image in images:
        content.append(
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": image.media_type,
                    "data": _b64(image.jpeg_bytes),
                },
            }
        )
    return content


def _b64(data: bytes) -> str:
    import base64

    return base64.b64encode(data).decode("ascii")


def format_http_error(status: int, text: str) -> str:
    message = _error_message(text)
    if message:
        return f"anthropic: {message}"
    return f"anthropic HTTP {status}: {_trim(text)}"


def _error_message(text: str) -> str | None:
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return None
    error = payload.get("error") if isinstance(payload, dict) else None
    if isinstance(error, dict):
        message = error.get("message")
        if isinstance(message, str) and message.strip():
            return message.strip()
    if isinstance(error, str) and error.strip():
        return error.strip()
    return None


def _trim(text: str, limit: int = 400) -> str:
    compact = " ".join(text.split())
    if len(compact) <= limit:
        return compact
    return compact[: limit - 1] + "…"
