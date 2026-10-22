# -*- coding: utf-8 -*-
"""Autolink must never rewrite URLs or code."""

from __future__ import annotations

from plugin.memory_governed._kb import KnowledgeBase


def test_url_is_never_corrupted():
    body = "MiMo API at https://token-plan-cn.xiaomimimo.com/v1 accepts OpenAI"
    out = KnowledgeBase._autolink(body, ["mimo"])
    assert "https://token-plan-cn.xiaomimimo.com/v1" in out
    assert "xiaomi[[mimo]]" not in out


def test_only_prose_is_linked_and_casing_is_preserved():
    body = "inline `MiMo` then prose MiMo"
    out = KnowledgeBase._autolink(body, ["mimo"])
    assert "`MiMo`" in out
    assert "prose [[MiMo]]" in out


def test_fenced_code_is_not_linked():
    body = "```\nMiMo\n```\nprose MiMo"
    out = KnowledgeBase._autolink(body, ["MiMo"])
    assert "```\nMiMo\n```" in out
    assert "prose [[MiMo]]" in out


def test_existing_wikilink_is_not_doubled():
    body = "already [[MiMo]] linked"
    assert KnowledgeBase._autolink(body, ["MiMo"]) == body
