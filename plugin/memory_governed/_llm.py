# -*- coding: utf-8 -*-
"""最小共享 LLM 调用（consolidate / kb.enrich 共用）。

端点配置复用 ``config.synthesis`` 段（``base_url`` / ``model`` /
``api_key_env``，模型字段留空时由 ``load_governed_config`` 继承 agent 的
config.yaml）—— 与 ``_synthesize`` 同一来源，不新增配置面。

约定：**失败一律返回 ``None``，绝不抛异常** —— 调用方（治理巡检、入库
富化）都把 LLM 当作可选增强，模型不可用时优雅降级是唯一正确行为
（WeKnora：model unavailable = abort that phase with a reason）。
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from ._config import env_secret

logger = logging.getLogger(__name__)


def chat_completion(messages: List[Dict[str, Any]],
                    config: Any,
                    *,
                    temperature: float = 0.0,
                    max_tokens: int = 1000,
                    timeout: float = 60.0) -> Optional[str]:
    """一次 chat/completions 调用 → assistant content；任何失败返回 ``None``。

    ``temperature`` 默认 0：合并/富化这类结构化产出要的是可复现，不是花样。
    """
    syn = getattr(config, "synthesis", None)
    if syn is None:
        return None
    base_url = str(getattr(syn, "base_url", "") or "").strip()
    model = str(getattr(syn, "model", "") or "").strip()
    api_key = env_secret(str(getattr(syn, "api_key_env", "") or ""))
    if not base_url or not model or not api_key:
        logger.debug("llm chat unavailable (base_url/model/api_key missing)")
        return None
    try:
        import httpx  # type: ignore
    except ImportError:
        logger.debug("llm chat requires httpx; skipping")
        return None
    try:
        r = httpx.post(
            base_url.rstrip("/") + "/chat/completions",
            headers={"Authorization": f"Bearer {api_key}"},
            json={"model": model, "messages": messages,
                  "temperature": temperature, "max_tokens": max_tokens},
            timeout=timeout,
        )
        if r.status_code >= 400:
            logger.warning("llm chat HTTP %s: %s", r.status_code, r.text[:200])
            return None
        return str(r.json()["choices"][0]["message"]["content"])
    except Exception as e:  # noqa: BLE001 — 可选增强，失败降级不抛栈
        logger.debug("llm chat failed: %s", e)
        return None
