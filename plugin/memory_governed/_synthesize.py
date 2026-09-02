# -*- coding: utf-8 -*-
"""Session 归纳：对话 → 知识卡片（插件侧调 LLM，模型对齐 agent 对话模型）。

职责边界：本模块只做「调 LLM 把一段对话归纳成结构化知识卡片候选」，
落库的**治理**（密钥闸门 / 置信分流 / 链接补全）由
``KnowledgeBase.add(confidence=...)`` 承担 —— 归纳与治理分离，与 ``_ingest``
的「只拿内容、不落库」边界一致。

归纳用 LLM 与 agent 对话模型**同款**：``config.synthesis`` 的模型字段留空时，
运行时自动继承 ``config.yaml`` 的 model 段（任何 hermes 环境都自动对齐 agent
的对话模型）。这不是「调用 agent」，而是复用同一个模型名与端点。

失败 / 未配置 / 未启用时一律返回 ``[]``，绝不抛异常、绝不写垃圾。
"""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Any, Dict, List

logger = logging.getLogger(__name__)

#: 单次归纳最多喂给模型的对话字符数（超长 session 截断，避免打爆 prompt）
_MAX_TRANSCRIPT_CHARS = 24000

_SYNTH_SYSTEM_PROMPT = (
    "You distill durable knowledge from a conversation into atomic notes. "
    "Output ONLY a JSON array (no markdown fences, no prose). Each element is "
    '{"title": string, "body": string (markdown, 1-4 sentences), '
    '"tags": [string], "concepts": [string], "confidence": number 0-1}. '
    "Extract only facts / decisions / preferences worth keeping long-term. "
    "Skip greetings, chitchat, one-off task details, and anything containing "
    "secrets or credentials. confidence = how sure you are this is durable "
    "knowledge (>=0.7 high, <0.7 uncertain). Return [] if nothing is worth "
    "keeping."
)


def _as_strs(value: Any) -> List[str]:
    """归一化 tags/concepts 到 List[str]（与 _kb._as_list 同语义）。"""
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return [str(v).strip() for v in value if str(v).strip()]
    return [str(value).strip()] if str(value).strip() else []


def _content_to_text(content: Any) -> str:
    """把 message 的 content 归一化成纯文本。

    hermes 真实 transcript 里 content 可能是 ``str``（纯文本）或 ``list``
    （OpenAI 多模态 parts，如 ``[{"type": "text", "text": ...},
    {"type": "image_url", ...}]``）。多模态时只保留 text 文本，图片/音频等
    非文本 parts 跳过。
    """
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: List[str] = []
        for p in content:
            if isinstance(p, dict):
                text = p.get("text")
                if isinstance(text, str) and text.strip():
                    parts.append(text)
            elif isinstance(p, str) and p.strip():
                parts.append(p)
        return "\n".join(parts)
    return str(content)


def _build_transcript(messages: List[Dict[str, Any]]) -> str:
    """把 messages 拼成 ``role: content`` 文本，截断到上限。"""
    lines: List[str] = []
    total = 0
    for msg in messages or []:
        role = str(msg.get("role", "") or "").strip()
        content = _content_to_text(msg.get("content")).strip()
        if not content or role not in ("user", "assistant"):
            continue
        line = f"{role}: {content}"
        if total + len(line) > _MAX_TRANSCRIPT_CHARS:
            remaining = _MAX_TRANSCRIPT_CHARS - total
            if remaining > 0:
                lines.append(line[:remaining])
            break
        lines.append(line)
        total += len(line)
    result = "\n".join(lines)
    return result[:_MAX_TRANSCRIPT_CHARS]


def _salvage_truncated_array(s: str) -> List[Any]:
    """从被截断的 JSON 数组里抢救已完整的对象。

    reasoning 模型（mimo-v2.5）偶发把 content 截断（max_tokens 被思考占满），
    导致 JSON 数组不完整。用 ``raw_decode`` 逐个解析，截断处停止，保留前面的
    完整对象，丢弃最后那个不完整的。
    """
    decoder = json.JSONDecoder()
    s = (s or "").strip()
    if not s.startswith("["):
        return []
    out: List[Any] = []
    idx = 1
    n = len(s)
    while True:
        # 跳过空白和逗号
        while idx < n and s[idx] in " \t\r\n,":
            idx += 1
        if idx >= n or s[idx] == "]":
            break
        try:
            obj, end = decoder.raw_decode(s, idx)
        except json.JSONDecodeError:
            break
        out.append(obj)
        idx = end
    return out


def _parse_candidates(text: str) -> List[Dict[str, Any]]:
    """解析 LLM 输出的 JSON 数组，逐条清洗，坏条目丢弃。"""
    text = (text or "").strip()
    # 去掉可能的 markdown 围栏
    fence = re.search(r"```(?:json)?\s*(.*?)```", text, re.S)
    if fence:
        text = fence.group(1).strip()
    # 截取第一个 [ ... ] 块（模型可能在 JSON 前后加解释）
    start = text.find("[")
    if start == -1:
        return []
    end = text.rfind("]")
    if end == -1 or end <= start:
        # 数组被截断（缺右括号），抢救已完整的对象
        raw = _salvage_truncated_array(text[start:])
    else:
        try:
            raw = json.loads(text[start:end + 1])
        except Exception as e:  # noqa: BLE001
            logger.warning("synthesis JSON parse failed, salvaging: %s", e)
            raw = _salvage_truncated_array(text[start:end + 1])

    if not isinstance(raw, list):
        return []

    out: List[Dict[str, Any]] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        title = str(item.get("title", "") or "").strip()
        body = str(item.get("body", "") or "").strip()
        if not title or not body:
            continue
        try:
            confidence = float(item.get("confidence", 0.5))
        except (TypeError, ValueError):
            confidence = 0.5
        confidence = max(0.0, min(1.0, confidence))
        out.append({
            "title": title,
            "body": body,
            "tags": _as_strs(item.get("tags")),
            "concepts": _as_strs(item.get("concepts")),
            "confidence": confidence,
        })
    return out


def synthesize_notes(
    messages: List[Dict[str, Any]],
    config: Any,
    timeout: float = 60.0,
) -> List[Dict[str, Any]]:
    """对话 → 知识卡片候选列表。失败 / 未启用 / 未配置返回 ``[]``。

    Returns:
        ``[{title, body, tags, concepts, confidence}]`` —— 交给
        ``KnowledgeBase.add(confidence=...)`` 做治理落库。
    """
    syn = getattr(config, "synthesis", None)
    if syn is None or not getattr(syn, "enabled", False):
        return []

    base_url = str(getattr(syn, "base_url", "") or "").strip()
    model = str(getattr(syn, "model", "") or "").strip()
    if not base_url or not model:
        logger.warning("synthesis not configured (base_url/model); skipping")
        return []

    api_key_env = str(getattr(syn, "api_key_env", "") or "")
    api_key = os.environ.get(api_key_env, "") if api_key_env else ""
    if not api_key:
        logger.warning("synthesis api key env not set: %s; skipping", api_key_env)
        return []

    transcript = _build_transcript(messages)
    if not transcript.strip():
        return []

    try:
        import httpx  # type: ignore
    except ImportError:
        logger.warning("synthesis requires httpx; skipping")
        return []

    max_candidates = 5
    try:
        max_candidates = int(getattr(syn, "max_candidates", 5) or 5)
    except (TypeError, ValueError):
        max_candidates = 5
    if max_candidates < 1:
        max_candidates = 1

    endpoint = base_url.rstrip("/") + "/chat/completions"
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": _SYNTH_SYSTEM_PROMPT},
            {"role": "user", "content": transcript},
        ],
        "temperature": 0.2,
        "max_tokens": 8000,
    }
    try:
        r = httpx.post(
            endpoint,
            headers={"Authorization": f"Bearer {api_key}"},
            json=payload,
            timeout=timeout,
        )
    except Exception as e:  # noqa: BLE001
        logger.warning("synthesis request failed: %s", e)
        return []

    if r.status_code >= 400:
        logger.warning("synthesis HTTP %s: %s", r.status_code, r.text[:300])
        return []

    try:
        data = r.json()
        content = data["choices"][0]["message"]["content"]
    except Exception as e:  # noqa: BLE001
        logger.warning("synthesis bad response: %s", e)
        return []

    return _parse_candidates(str(content))[:max_candidates]
