# -*- coding: utf-8 -*-
"""One implementation of surrogate hygiene, and it guards the path that matters.

The history this file pins down
-------------------------------
A lone surrogate (``\\udcae``) cannot be encoded as UTF-8, and every layer below
the write pipeline fails on it differently and quietly:

* the embedding call raises, and ``embed_one`` turns that into ``None`` — so the
  semantic channel drops to lexical without a word;
* Arrow demands strict UTF-8, so the L2 row cannot be written;
* SQLite encodes text parameters as UTF-8 too, so the archive and its FTS mirror
  raise on the same input.

The fix for that had grown **three identical copies** — ``_embedding._sanitize``,
``hgm_mcp._ensure_utf8`` and an inline block in ``memory_cli.cmd_remember`` — and
**none of them covered the plugin's own extraction path**, which is where a fact
actually reaches L2. Three copies of a guard is three chances to drift, and the
unguarded entry point was the one that mattered.

So the tests below are in two halves: the function behaves, and there is exactly
one of it.
"""
from __future__ import annotations

import ast
import importlib.util
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from plugin.memory_governed import _text                      # noqa: E402
from plugin.memory_governed._embedding import EmbeddingService  # noqa: E402

SURROGATE = "地址是 10.1.1.1 \udcae 结束"

#: Every place that used to hold a copy of the logic.
FORMER_COPYISTS = (
    REPO / "plugin" / "memory_governed" / "_embedding.py",
    REPO / "plugin" / "memory_governed" / "__init__.py",
    REPO / "scripts" / "hgm_mcp.py",
    REPO / "memory_cli.py",
)


class TestSanitizeUtf8:
    def test_clean_text_is_returned_untouched(self):
        for text in ("普通中文", "plain ascii", "混合 mixed 123", "emoji 🙂 也可以"):
            assert _text.sanitize_utf8(text) == text

    def test_a_lone_surrogate_is_replaced(self):
        out = _text.sanitize_utf8(SURROGATE)
        assert "\udcae" not in out
        assert "?" in out
        assert "地址是" in out and "结束" in out, "只该丢掉坏的那个字符"
        out.encode("utf-8")            # the whole point: this must not raise

    def test_the_result_is_always_encodable(self):
        for bad in ("\udcae", "a\ud800b", "\udfff", "汉字\udcae汉字"):
            _text.sanitize_utf8(bad).encode("utf-8")

    @pytest.mark.parametrize("value, expected", [(None, ""), (123, "123"),
                                                 (True, "True"), ("", "")])
    def test_non_strings_do_not_explode(self, value, expected):
        assert _text.sanitize_utf8(value) == expected

    def test_it_matches_what_the_embedding_layer_does(self):
        """``_sanitize`` must stay an alias, not become a second copy."""
        assert EmbeddingService._sanitize(SURROGATE) == _text.sanitize_utf8(SURROGATE)


class TestSanitizeMessages:
    def test_content_is_sanitized(self):
        out = _text.sanitize_messages([{"role": "user", "content": SURROGATE}])
        assert "\udcae" not in out[0]["content"]
        out[0]["content"].encode("utf-8")

    def test_other_keys_survive_untouched(self):
        msg = {"role": "user", "content": "好", "timestamp": 12.5, "extra": {"a": 1}}
        out = _text.sanitize_messages([msg])[0]
        assert out["role"] == "user" and out["timestamp"] == 12.5
        assert out["extra"] == {"a": 1}

    def test_the_input_is_not_mutated(self):
        """The caller may still be holding the original turn."""
        original = {"role": "user", "content": SURROGATE}
        _text.sanitize_messages([original])
        assert original["content"] == SURROGATE

    @pytest.mark.parametrize("messages", [None, [], [None], ["not a dict"]])
    def test_odd_input_passes_through(self, messages):
        out = _text.sanitize_messages(messages)
        assert isinstance(out, list)


class TestThereIsOnlyOneImplementation:
    """The guard against the guard multiplying again.

    Same idea as ``test_persona_sections.py``: duplication that cannot be
    removed is duplication that must be pinned. Here it *can* be removed, so the
    pin is "the pattern appears exactly once".
    """

    def test_the_pattern_lives_only_in_text_py(self):
        pattern = 'encode("utf-8", errors="replace").decode'
        offenders = []
        for path in FORMER_COPYISTS:
            for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
                if pattern in line:
                    offenders.append(f"{path.name}:{i}")
        assert not offenders, (
            f"surrogate 清洗又出现第二份实现: {offenders} —— 请改成调用 _text.sanitize_utf8")

    def test_the_embedding_layer_delegates_rather_than_reimplements(self):
        tree = ast.parse((REPO / "plugin/memory_governed/_embedding.py")
                         .read_text(encoding="utf-8"))
        calls = [n for n in ast.walk(tree)
                 if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
                 and n.func.id == "sanitize_utf8"]
        assert calls, "_embedding 里没有调用共享实现"

    def test_the_mcp_server_delegates_rather_than_reimplements(self):
        spec = importlib.util.spec_from_file_location(
            "hgm_mcp_hygiene", REPO / "scripts" / "hgm_mcp.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        assert module._ensure_utf8(SURROGATE) == _text.sanitize_utf8(SURROGATE), (
            "MCP 侧没有走共享实现")

    def test_sync_turn_sanitizes_at_the_ingress(self):
        """The choke point: one call covers archive, FTS mirror and extraction."""
        tree = ast.parse((REPO / "plugin/memory_governed/__init__.py")
                         .read_text(encoding="utf-8"))
        sync_turn = next((n for n in ast.walk(tree)
                          if isinstance(n, ast.FunctionDef) and n.name == "sync_turn"), None)
        assert sync_turn is not None, "没找到 sync_turn"
        called = [n for n in ast.walk(sync_turn)
                  if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
                  and n.func.id == "sanitize_messages"]
        assert called, (
            "sync_turn 入口没有清洗 —— 插件自己的抽取路径（事实真正落进 L2 的那条）"
            "会对代理对无能为力")
