from __future__ import annotations

from typing import Any

from langchain.chat_models import init_chat_model
from langchain_core.language_models.chat_models import BaseChatModel

from dayframe.config import Config
from dayframe.paths import api_key as env_api_key


def model_string(cfg: Config) -> str:
    model = cfg.provider.model.strip()
    if ":" in model:
        return model
    return f"{cfg.provider.name}:{model}"


def build_model(
    cfg: Config,
    *,
    api_key: str | None = None,
    tools: list[Any] | None = None,
) -> BaseChatModel:
    key = api_key if api_key is not None else env_api_key()
    if not key:
        raise RuntimeError(
            "DAYFRAME_API_KEY is not set; add it to .env or export it (never config.toml)"
        )
    model = init_chat_model(
        model_string(cfg),
        temperature=0.3,
        timeout=cfg.budget.timeout_seconds,
        api_key=key,
    )
    if tools:
        return model.bind_tools(tools)
    return model
