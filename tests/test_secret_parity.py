"""密钥规则：两份实现必须一致，且 CSV 形状要拦得住。

背景（2026-09-23 实测）：``memory_cli.SECRET_PATTERNS`` 与
``_bridge.SECRET_PATTERNS`` 是两份副本（CLI 要在裸解释器上也能跑写路径），
**此前已经漂移** —— CLI 只有 4 条，缺 PEM 私钥与 bearer token，key=value 那条
也更松。后果是：桥拦住了某条内容，会让人误以为写路径也拦住了。**同一个漏洞
有两个答案**，正是本项目反复出现的故障形态。

另一条新增规则来自 2026-09-22 的事故：CSV 形状的凭据记录
``SSH-<名>,<用户>,<口令>,ssh://<host>`` 漏进了 L2。key=value 那条要求
``:``/``=`` 分隔符，而 CSV 用逗号 —— 不是规则不够宽，是**形状不同**。
"""

from __future__ import annotations

import importlib.util
import re
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import memory_cli as cli  # noqa: E402
from plugin.memory_governed import _bridge as bridge  # noqa: E402


def _load_script(name: str):
    """按路径加载 scripts/ 下的脚本（它们不是包）。"""
    # l2_rebuild.py 顶层要 import 同目录的 hermes_env，所以 scripts/ 也得在
    # sys.path 上 —— 缺了它加载就会以 ModuleNotFoundError 失败。
    scripts = str(REPO / "scripts")
    if scripts not in sys.path:
        sys.path.insert(0, scripts)
    path = REPO / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


# --------------------------------------------------------------------------
# 1. 两份实现必须一致
# --------------------------------------------------------------------------

class TestTheTwoCopiesAreIdentical:
    def test_same_number_of_patterns(self):
        assert len(cli.SECRET_PATTERNS) == len(bridge.SECRET_PATTERNS)

    def test_every_pattern_matches_one_for_one(self):
        left = [p.pattern for p in cli.SECRET_PATTERNS]
        right = [p.pattern for p in bridge.SECRET_PATTERNS]
        assert left == right, (
            "两份 SECRET_PATTERNS 漂移了：桥拦得住、CLI 拦不住（或反之），"
            "同一个漏洞有两个答案。请同步两侧。"
        )

    def test_the_bridge_names_stay_index_aligned(self):
        # 名字表用于报错；多一条规则而少一个名字，报错就会退化成索引号。
        assert len(bridge.SECRET_PATTERN_NAMES) == len(bridge.SECRET_PATTERNS)


# --------------------------------------------------------------------------
# 2. CSV 形状的凭据必须拦得住
# --------------------------------------------------------------------------

class TestCsvCredentialShape:
    # 只用占位符，绝不出现真实凭据。
    @pytest.mark.parametrize("text", [
        "SSH-云服务器,root,password123,ssh://192.168.1.1",
        "SSH-NAS2,admin,password456,ssh://192.168.1.2",
        "ssh-example,user,aaaaaaaaaaaa,ssh://host",
    ])
    def test_caught_by_both_sides(self, text):
        assert any(p.search(text) for p in cli.SECRET_PATTERNS)
        assert any(p.search(text) for p in bridge.SECRET_PATTERNS)

    def test_the_bridge_names_it(self):
        names = bridge.matched_secret_patterns(
            "SSH-云服务器,root,password123,ssh://192.168.1.1")
        assert "csv_credential_record" in names

    @pytest.mark.parametrize("text", [
        "vaultwarden 备份路径在 /vol2/1000/Docker/vaultwarden",
        "LanceDB 的 add_columns 接受 pa.Field 参数，传映射会抛 TypeError",
        "HGM 的 L2 写入门禁阈值是 strong_signal>=2",
        "内网 SSH 免密：私钥集中在 C:/Users/lionlliu/.ssh/, 两把无口令密钥",
        "公司设备 → Tailscale → 家里 NAS Vaultwarden",
    ])
    def test_no_false_positive_on_real_facts(self, text):
        # 高精度是硬要求：fail-closed 下误报会**静默**丢掉一条真记忆。
        assert bridge.matched_secret_patterns(text) == []


# --------------------------------------------------------------------------
# 3. 补齐的那两条（CLI 此前完全不拦）
# --------------------------------------------------------------------------

class TestTheGapThatWasClosed:
    @pytest.mark.parametrize("text", [
        "-----BEGIN RSA PRIVATE KEY-----",
        "Authorization: bearer abcdefghij0123456789",
    ])
    def test_cli_now_catches_what_it_used_to_miss(self, text):
        assert any(p.search(text) for p in cli.SECRET_PATTERNS)


# --------------------------------------------------------------------------
# 4. 回灌必须过门禁
# --------------------------------------------------------------------------

class TestRebuildAppliesTheGate:
    """2026-09-22 的事故：回灌不走门禁，一次灌进 2023 行碎片 + 明文口令。"""

    @staticmethod
    def _msgs(contents):
        return [{"content": c, "role": "user", "timestamp": 1700000000.0,
                 "_l3_rowid": 1} for c in contents]

    def test_fragment_and_credential_are_dropped_and_counted(self):
        mod = _load_script("l2_rebuild")
        if not mod._GATE_AVAILABLE:
            pytest.skip("门禁模块不可用（插件导入失败）")
        msgs = self._msgs([
            # 分隔线：短且无信息
            "------------------------------------------",
            # CSV 凭据：必须被拦
            "SSH-云服务器,root,password123,ssh://192.168.1.1",
            # 像事实的句子：应当留下
            "HGM 的 L2 表 memories 新增 project 列，NULL 表示全局事实。",
        ])
        facts, stats = mod.extract_facts(msgs, gate=True)
        for f in facts:
            assert "password123" not in f["content"]
            assert not re.fullmatch(r"-{10,}", f["content"])
        assert any(k.startswith("secret:") for k in stats)

    def test_without_the_gate_the_old_behaviour_returns(self):
        mod = _load_script("l2_rebuild")
        if not mod._GATE_AVAILABLE:
            pytest.skip("门禁模块不可用（插件导入失败）")
        msgs = self._msgs(["SSH-云服务器,root,password123,ssh://192.168.1.1"])
        facts, stats = mod.extract_facts(msgs, gate=False)
        assert any("password123" in f["content"] for f in facts)
        assert stats == {}

    def test_stats_are_reported_not_swallowed(self):
        mod = _load_script("l2_rebuild")
        if not mod._GATE_AVAILABLE:
            pytest.skip("门禁模块不可用（插件导入失败）")
        _, stats = mod.extract_facts(
            self._msgs(["------------------------------------------"]), gate=True)
        # 拦下多少、因为什么必须可数：静默丢弃和成功长得一样。
        assert stats and all(isinstance(v, int) for v in stats.values())
