from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from langchain_core.callbacks.base import BaseCallbackHandler
from langchain_core.messages import AIMessage, BaseMessage, ToolMessage

from dayframe.paths import traces_dir


def trace_path(run_id: str) -> Path:
    return traces_dir() / f"{run_id}.jsonl"


def wire_path(run_id: str) -> Path:
    return traces_dir() / f"{run_id}.raw_request.json"


def ensure_traces_dir() -> Path:
    path = traces_dir()
    path.mkdir(parents=True, exist_ok=True)
    return path


def message_to_json(message: BaseMessage | dict[str, Any]) -> dict[str, Any]:
    if isinstance(message, dict):
        dumped = dict(message)
    elif hasattr(message, "model_dump"):
        dumped = message.model_dump()
    elif hasattr(message, "dict"):
        dumped = message.dict()
    else:
        dumped = {"repr": repr(message)}
    dumped["content"] = redact_content(dumped.get("content"))
    return dumped


def redact_content(content: object) -> object:
    if isinstance(content, list):
        return [redact_block(block) for block in content]
    return content


def redact_block(block: object) -> object:
    if not isinstance(block, dict):
        return block
    kind = block.get("type")
    if kind in {"image", "image_url"}:
        data = block.get("data") or ""
        url = ""
        image_url = block.get("image_url")
        if isinstance(image_url, dict):
            url = str(image_url.get("url") or "")
        elif isinstance(image_url, str):
            url = image_url
        payload = data or url
        return {
            "type": "image",
            "mime_type": block.get("mime_type") or block.get("media_type") or "image/jpeg",
            "bytes": len(payload) if isinstance(payload, str) else 0,
        }
    if "source" in block and isinstance(block.get("source"), dict):
        source = dict(block["source"])
        if source.get("type") == "base64" and "data" in source:
            data = source.get("data") or ""
            source = {
                "type": "base64",
                "media_type": source.get("media_type", "image/jpeg"),
                "bytes": len(data) if isinstance(data, str) else 0,
            }
            return {**block, "source": source}
    return block


class RawDump(BaseCallbackHandler):
    """Write the last chat-model request so the wire format is inspectable."""

    def __init__(self, path: Path) -> None:
        self.path = path

    def on_chat_model_start(self, serialized, messages, **kwargs) -> None:  # noqa: ANN001
        del serialized, kwargs
        payload = [[message_to_json(message) for message in group] for group in messages]
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, default=str) + "\n")


def load_trace(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append(json.loads(line))
    return rows


def turn_from_ai(ai: AIMessage, *, request: list[Any] | None = None) -> dict[str, Any]:
    usage = ai.usage_metadata or {}
    return {
        "request_messages": [message_to_json(item) for item in request or []],
        "completion": message_to_json(ai),
        "tool_calls": list(ai.tool_calls or []),
        "tool_results": [],
        "usage": {
            "input_tokens": int(usage.get("input_tokens") or 0),
            "output_tokens": int(usage.get("output_tokens") or 0),
        },
    }


def add_tool_results(turn: dict[str, Any], messages: list[Any]) -> None:
    results = []
    for message in messages:
        if isinstance(message, ToolMessage) or (
            isinstance(message, dict) and message.get("type") == "tool"
        ):
            results.append(message_to_json(message))
    turn["tool_results"] = results


def format_replay(rows: list[dict[str, Any]], *, run_id: str | None = None) -> str:
    lines: list[str] = []
    if run_id:
        lines.append(f"Replay {run_id}  {len(rows)} turn{'s' if len(rows) != 1 else ''}")
        lines.append("")
    images = 0
    in_tok = out_tok = 0
    for index, row in enumerate(rows, start=1):
        usage = row.get("usage") or {}
        inp = int(usage.get("input_tokens") or 0)
        out = int(usage.get("output_tokens") or 0)
        in_tok += inp
        out_tok += out
        lines.append(f"Turn {index}  {inp} in / {out} out")
        for call in row.get("tool_calls") or []:
            name = call.get("name", "tool")
            args = call.get("args") or {}
            summary = ", ".join(f"{key}={value!r}" for key, value in list(args.items())[:4])
            lines.append(f"  -> {name}({summary})")
            if name == "view_photos":
                images += int(args.get("count") or 0)
        for result in row.get("tool_results") or []:
            content = result.get("content")
            preview = _preview_result(content)
            lines.append(f"  <- {preview}")
        text = _completion_text(row.get("completion") or {})
        if text and not (row.get("tool_calls") or []):
            lines.append(f"  text: {text}")
        lines.append("")
    lines.append(f"Done. {len(rows)} turns, ~{images} image requests, {in_tok} in / {out_tok} out")
    return "\n".join(lines).rstrip() + "\n"


def _completion_text(completion: dict[str, Any]) -> str:
    content = completion.get("content")
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = [
            block.get("text", "")
            for block in content
            if isinstance(block, dict) and block.get("type") == "text"
        ]
        return " ".join(part for part in parts if part).strip()
    return ""


def _preview_result(content: object, limit: int = 120) -> str:
    if isinstance(content, list):
        n_img = sum(
            1
            for block in content
            if isinstance(block, dict) and block.get("type") in {"image", "image_url"}
        )
        texts = [
            block.get("text", "")
            for block in content
            if isinstance(block, dict) and block.get("type") == "text"
        ]
        prefix = f"{n_img} image(s). " if n_img else ""
        text = " ".join(texts).strip()
        return _clip(prefix + text, limit)
    if isinstance(content, str):
        return _clip(content.replace("\n", " "), limit)
    return _clip(str(content), limit)


def _clip(text: str, limit: int) -> str:
    compact = " ".join(text.split())
    if len(compact) <= limit:
        return compact
    return compact[: limit - 1] + "…"
