#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Knowledge processor with LLM-based classification and link generation."""

from __future__ import annotations

import hashlib
import json
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any

import httpx

from hermes_env import bootstrap, env_path, setup_logging
from wiki_utils import StateManager, retry

logger = setup_logging("knowledge_processor")

HERMES_HOME = bootstrap(__file__)
WIKI_DIR = env_path("WIKI_DIR", HERMES_HOME / "wiki")
KNOWLEDGE_DIR = WIKI_DIR / "knowledge"

CATEGORIES = ["技术", "工作项目", "产品", "学习", "生活", "其他"]


def get_state() -> StateManager:
    return StateManager(KNOWLEDGE_DIR, "processor")

CATEGORY_PROMPT = """你是一个知识分类专家。根据以下内容，判断它属于哪个类别。

类别选项: {categories}

内容:
{content}

请返回 JSON 格式:
{{"category": "类别名", "confidence": 0.0-1.0, "reason": "简短理由"}}"""

EXTRACT_PROMPT = """你是一个知识提取专家。从以下内容中提取关键信息。

内容:
{content}

请返回 JSON 格式:
{{
  "title": "简洁标题",
  "summary": "100字以内摘要",
  "concepts": ["概念1", "概念2", ...],
  "tags": ["标签1", "标签2", ...],
  "key_points": ["要点1", "要点2", ...]
}}"""


@retry(max_attempts=3, delay=2.0, exceptions=(httpx.HTTPStatusError, httpx.ConnectError))
def call_llm(prompt: str) -> str:
    api_key = os.getenv("LLM_API_KEY") or os.getenv("PERSONA_LLM_API_KEY") or os.getenv("SILICONFLOW_API_KEY")
    base_url = os.getenv("LLM_BASE_URL") or os.getenv("PERSONA_LLM_BASE_URL")
    model = os.getenv("LLM_MODEL") or os.getenv("PERSONA_LLM_MODEL")
    if not api_key or not base_url or not model:
        raise RuntimeError("LLM not configured. Set LLM_API_KEY, LLM_BASE_URL, LLM_MODEL")
    if not api_key:
        raise RuntimeError("LLM API key not configured")

    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 1000,
        "temperature": 0.3,
    }
    with httpx.Client(timeout=60) as client:
        response = client.post(
            f"{base_url.rstrip('/')}/chat/completions",
            json=payload,
            headers={"Authorization": f"Bearer {api_key}"},
        )
        response.raise_for_status()
        data = response.json()
    return data["choices"][0]["message"]["content"]


def parse_json_response(text: str) -> dict[str, Any]:
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        text = text[start:end + 1]
    return json.loads(text)


def classify_by_rules(content: str) -> str:
    content_lower = content.lower()
    tech_keywords = ["python", "docker", "linux", "api", "代码", "函数", "数据库", "服务器", "配置", "部署", "git", "script"]
    work_keywords = ["项目", "需求", "进度", "会议", "任务", "完成", "deadline", "排期"]
    life_keywords = ["生活", "健康", "运动", "旅行", "读书", "电影", "音乐"]
    learning_keywords = ["学习", "教程", "课程", "论文", "研究", "原理", "算法"]

    scores = {
        "技术": sum(1 for kw in tech_keywords if kw in content_lower),
        "工作项目": sum(1 for kw in work_keywords if kw in content_lower),
        "生活": sum(1 for kw in life_keywords if kw in content_lower),
        "学习": sum(1 for kw in learning_keywords if kw in content_lower),
    }
    best = max(scores, key=scores.get)  # type: ignore
    return best if scores[best] > 0 else "其他"


def classify_content(content: str) -> str:
    try:
        api_key = os.getenv("PERSONA_LLM_API_KEY") or os.getenv("SILICONFLOW_API_KEY") or os.getenv("EMBED_API_KEY")
        if not api_key:
            logger.info("no LLM API key, using rule-based classification")
            return classify_by_rules(content)
        prompt = CATEGORY_PROMPT.format(categories=", ".join(CATEGORIES), content=content[:2000])
        response = call_llm(prompt)
        result = parse_json_response(response)
        category = result.get("category", "其他")
        if category not in CATEGORIES:
            category = "其他"
        logger.info("classified as: %s (confidence: %.2f)", category, result.get("confidence", 0))
        return category
    except Exception as exc:
        logger.warning("classification failed, using rules: %s", exc)
        return classify_by_rules(content)


def extract_info_basic(content: str) -> dict[str, Any]:
    lines = content.strip().split("\n")
    title = ""
    for line in lines[:5]:
        line = line.strip()
        if line.startswith("#"):
            title = line.lstrip("#").strip()
            break
    if not title and lines:
        title = lines[0][:50]

    summary = content[:200].replace("\n", " ").strip()

    concepts = []
    concept_patterns = [r"[\[【](.+?)[\]】]", r"「(.+?)」"]
    for pattern in concept_patterns:
        matches = re.findall(pattern, content)
        concepts.extend(matches[:5])
    if not concepts:
        words = re.findall(r"[\u4e00-\u9fa5]{2,4}", content)
        from collections import Counter
        word_counts = Counter(words)
        concepts = [w for w, c in word_counts.most_common(5) if c >= 2]

    tags = []
    tag_keywords = ["python", "docker", "linux", "api", "git", "hermes", "agent", "llm"]
    for kw in tag_keywords:
        if kw in content.lower():
            tags.append(kw)

    key_points = []
    for line in lines:
        line = line.strip()
        if line.startswith(("- ", "* ", "• ")):
            key_points.append(line[2:])
        elif re.match(r"^\d+[\.\)]\s", line):
            key_points.append(re.sub(r"^\d+[\.\)]\s*", "", line))

    return {
        "title": title,
        "summary": summary,
        "concepts": concepts[:10],
        "tags": tags[:10],
        "key_points": key_points[:10],
    }


def extract_info(content: str) -> dict[str, Any]:
    try:
        api_key = os.getenv("PERSONA_LLM_API_KEY") or os.getenv("SILICONFLOW_API_KEY") or os.getenv("EMBED_API_KEY")
        if not api_key:
            logger.info("no LLM API key, using basic extraction")
            return extract_info_basic(content)
        prompt = EXTRACT_PROMPT.format(content=content[:3000])
        response = call_llm(prompt)
        result = parse_json_response(response)
        logger.info("extracted: title=%s, concepts=%d, tags=%d",
                     result.get("title", ""), len(result.get("concepts", [])), len(result.get("tags", [])))
        return result
    except Exception as exc:
        logger.warning("extraction failed, using basic: %s", exc)
        return extract_info_basic(content)


def generate_note(item: dict[str, Any], info: dict[str, Any], category: str) -> str:
    now = datetime.now().strftime("%Y-%m-%d")
    source_type = item.get("type", "unknown")
    source_path = item.get("path") or item.get("url") or "unknown"
    concepts = info.get("concepts", [])
    tags = info.get("tags", [])

    concept_links = ", ".join(f"[[{c}]]" for c in concepts)
    tag_links = " ".join(f"#{t}" for t in tags)

    frontmatter = f"""---
title: {info.get('title', item.get('name', 'Untitled'))}
type: {source_type}
source: {source_path}
date: {now}
category: {category}
concepts: [{concept_links}]
tags: [{tag_links}]
---"""

    key_points = "\n".join(f"- {p}" for p in info.get("key_points", []))

    note = f"""{frontmatter}

# {info.get('title', item.get('name', 'Untitled'))}

## 摘要
{info.get('summary', '')}

## 关键要点
{key_points}

## 概念关联
{concept_links}

---
*来源: {source_path}*
*处理时间: {now}*
"""
    return note


def process_item(item: dict[str, Any]) -> dict[str, Any] | None:
    content = item.get("content", "")
    if not content:
        if item.get("path"):
            try:
                content = Path(item["path"]).read_text(encoding="utf-8", errors="replace")[:5000]
            except Exception:
                pass
    if not content:
        logger.warning("no content for item: %s", item.get("name", ""))
        return None

    category = classify_content(content)
    info = extract_info(content)
    note = generate_note(item, info, category)

    output_dir = KNOWLEDGE_DIR / "documents" if item["type"] == "document" else KNOWLEDGE_DIR / item["type"]
    output_dir.mkdir(parents=True, exist_ok=True)
    safe_name = re.sub(r'[^\w.-]', '_', info.get("title", item.get("name", "untitled")))[:50]
    output_path = output_dir / f"{safe_name}.md"
    output_path.write_text(note, encoding="utf-8")

    logger.info("generated note: %s", output_path)
    return {
        "path": str(output_path),
        "title": info.get("title", ""),
        "category": category,
        "concepts": info.get("concepts", []),
        "tags": info.get("tags", []),
    }


def process_batch(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    state = get_state()

    for item in items:
        item_hash = item.get("hash") or item.get("url") or hashlib.md5(item.get("content", "").encode()).hexdigest()
        if state.is_processed(item_hash):
            continue
        result = process_item(item)
        if result:
            results.append(result)
            state.mark_processed(item_hash)

    state.last_run = datetime.now().isoformat()
    logger.info("processed %d items", len(results))
    return results


def main() -> int:
    import argparse
    parser = argparse.ArgumentParser(description="Knowledge processor")
    parser.add_argument("--input", help="JSON file with items to process")
    args = parser.parse_args()

    if args.input:
        items = json.loads(Path(args.input).read_text(encoding="utf-8", errors="replace"))
    else:
        items = []

    results = process_batch(items)
    for result in results:
        print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
