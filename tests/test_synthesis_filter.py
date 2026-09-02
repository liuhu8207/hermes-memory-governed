# -*- coding: utf-8 -*-
"""session 归纳前置过滤（_should_synthesize）测试。

验证零成本启发式：有 durable 信号 → 归纳；无 → 跳过（省 LLM token）。
"""

from __future__ import annotations

from plugin.memory_governed import __init__ as provider_mod
from plugin.memory_governed.__init__ import GovernedMemoryProvider


def _should(messages):
    # 直接调未绑定方法（不依赖完整 provider 初始化）
    return GovernedMemoryProvider._should_synthesize(None, messages)


class TestDurableSignal:
    def test_preference_hits(self):
        assert _should([{"role": "user", "content": "我以后都用 Linux Mint 不用 CachyOS"}]) is True

    def test_decision_hits(self):
        assert _should([{"role": "user", "content": "我们决定知识库归纳用 mimo 模型"}]) is True

    def test_change_hits(self):
        assert _should([{"role": "user", "content": "改用 XingChen Ultra 这个 ASR 模型"}]) is True

    def test_config_done_hits(self):
        assert _should([{"role": "user", "content": "github token 已经保存好了"}]) is True

    def test_negative_decision_hits(self):
        assert _should([{"role": "user", "content": "哦，那不用切旧版了"}]) is True

    def test_converge_decision_hits(self):
        assert _should([{"role": "user", "content": "暂时先设置这些吧"}]) is True
        assert _should([{"role": "user", "content": "够用，那 tts 呢"}]) is True

    def test_multimodal_content_hits(self):
        msgs = [
            {"role": "user", "content": [
                {"type": "text", "text": "以后都记住这个偏好"},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAA"}},
            ]},
        ]
        assert _should(msgs) is True

    def test_only_user_messages_counted(self):
        # durable 信号只出现在 assistant 消息里，不应命中
        msgs = [
            {"role": "assistant", "content": "我会记住这个偏好的"},
        ]
        assert _should(msgs) is False


class TestNoSignal:
    def test_chitchat_skipped(self):
        assert _should([{"role": "user", "content": "你好"}]) is False
        assert _should([{"role": "user", "content": "今天天气怎么样"}]) is False

    def test_pure_query_skipped(self):
        assert _should([{"role": "user", "content": "现在几点"}]) is False
        assert _should([{"role": "user", "content": "帮我查一下 Linux 的版本号"}]) is False

    def test_pure_task_skipped(self):
        assert _should([{"role": "user", "content": "帮我运行 pytest"}]) is False
        assert _should([{"role": "user", "content": "把文件 A 移到 B 目录"}]) is False

    def test_question_not_durable(self):
        # 含"以后"但是提问，不算 durable
        assert _should([{"role": "user", "content": "你以后都用什么模型？"}]) is False

    def test_empty_skipped(self):
        assert _should([]) is False
        assert _should([{"role": "user", "content": ""}]) is False
        assert _should([{"role": "system", "content": "偏好"}]) is False
