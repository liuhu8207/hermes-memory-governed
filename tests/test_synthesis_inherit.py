# -*- coding: utf-8 -*-
"""synthesis 自动继承 agent 对话模型（config.yaml model 段）测试。

验证：归纳模型不再硬编码 mimo-v2.5，而是运行时从 hermes config.yaml 的
model 段自动继承（任何环境都对齐 agent 的对话模型），显式配置优先。
"""

from __future__ import annotations

from plugin.memory_governed._config import (
    GovernedMemoryConfig,
    _load_agent_model_config,
    _inherit_synthesis_model,
)


def _write_config_yaml(tmp_path, text):
    (tmp_path / "config.yaml").write_text(text, encoding="utf-8")
    return tmp_path


class TestLoadAgentModelConfig:
    def test_parses_model_section(self, tmp_path):
        _write_config_yaml(tmp_path, (
            "model:\n"
            "  base_url: https://token-plan-cn.xiaomimimo.com/v1\n"
            "  default: mimo-v2.5\n"
            "  provider: xiaomi\n"
            "database:\n"
            "  journal_mode: wal\n"
        ))
        assert _load_agent_model_config(tmp_path) == {
            "base_url": "https://token-plan-cn.xiaomimimo.com/v1",
            "default": "mimo-v2.5",
            "provider": "xiaomi",
        }

    def test_ignores_nested_model_keys(self, tmp_path):
        # tts 子段的缩进 model: 不应误匹配顶层的 model 段
        _write_config_yaml(tmp_path, (
            "tts:\n"
            "  provider: mimo-tts\n"
            "  providers:\n"
            "    - name: base\n"
            "      model: base\n"
            "model:\n"
            "  base_url: https://example.com/v1\n"
            "  default: agent-model\n"
            "  provider: openrouter\n"
        ))
        result = _load_agent_model_config(tmp_path)
        assert result["default"] == "agent-model"
        assert result["provider"] == "openrouter"

    def test_missing_file_returns_none(self, tmp_path):
        assert _load_agent_model_config(tmp_path) is None

    def test_empty_section_returns_none(self, tmp_path):
        _write_config_yaml(tmp_path, "model:\n")
        assert _load_agent_model_config(tmp_path) is None


class TestInheritSynthesisModel:
    def test_fills_empty_from_agent(self, tmp_path):
        _write_config_yaml(tmp_path, (
            "model:\n"
            "  base_url: https://example.com/v1\n"
            "  default: agent-model\n"
            "  provider: xiaomi\n"
        ))
        cfg = GovernedMemoryConfig()
        cfg.synthesis.enabled = True
        _inherit_synthesis_model(cfg, tmp_path)
        assert cfg.synthesis.model == "agent-model"
        assert cfg.synthesis.base_url == "https://example.com/v1"
        assert cfg.synthesis.provider == "xiaomi"
        assert cfg.synthesis.api_key_env == "XIAOMI_API_KEY"

    def test_keeps_explicit_config(self, tmp_path):
        _write_config_yaml(tmp_path, (
            "model:\n"
            "  base_url: https://agent.example/v1\n"
            "  default: agent-model\n"
            "  provider: xiaomi\n"
        ))
        cfg = GovernedMemoryConfig()
        cfg.synthesis.model = "explicit-model"
        cfg.synthesis.base_url = "https://explicit.example/v1"
        cfg.synthesis.provider = "custom"
        cfg.synthesis.api_key_env = "CUSTOM_KEY"
        _inherit_synthesis_model(cfg, tmp_path)
        assert cfg.synthesis.model == "explicit-model"
        assert cfg.synthesis.base_url == "https://explicit.example/v1"
        assert cfg.synthesis.provider == "custom"
        assert cfg.synthesis.api_key_env == "CUSTOM_KEY"

    def test_no_config_yaml_keeps_empty(self, tmp_path):
        cfg = GovernedMemoryConfig()
        _inherit_synthesis_model(cfg, tmp_path)
        assert cfg.synthesis.model == ""
        assert cfg.synthesis.base_url == ""
        assert cfg.synthesis.api_key_env == ""
