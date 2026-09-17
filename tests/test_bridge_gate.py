# -*- coding: utf-8 -*-
"""Bridge 准入闸门回归测试（2026-09-16 统一治理入口）。

背景：Bridge 候选会被 ``import_approved`` promote 进 **L1**（每轮注入的规则
层），所以它的门槛必须比 L2 更严 —— L2 存错只是多一条误导性提示，L1 存错
就是一条永久规则。实测候选池 8 条里 6 条是噪声，其中
「我儿子的准考证，考试前一天记得提醒我」一旦 promote 就会让 Hermes 永远
记得提醒一场早已结束的考试。

本文件钉死四件事：
1. 时效性一次性待办进不了池；
2. 模板骨架（USER.md / MEMORY.md 的脚手架）进不了池，且不会顺着
   persona → 候选 → promote 的环路回流 L1；
3. 候选抽取用的是**统一**的强信号尺子，而不是另起一套词表；
4. 该「统一尺子」是**逐字调用**共享定义 ``_sync.dialogue_fact_admits``，
   而不是在此处内联一行 ``_fact_signal_score(..., strong_only=True)`` ——
   2026-09-17 的内联版本没去角色先验、也不消解疑问句式，把整类「能不能…」
   疑问句当规则放行（见 ``TestBridgeSharesTheDialogueRuler``）。
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from plugin.memory_governed import GovernedMemoryProvider
from plugin.memory_governed._bridge import screen_bridge_content
from plugin.memory_governed._sync import (
    _fact_signal_score,
    dialogue_fact_admits,
    external_structural_evidence,
    has_template_scaffold,
    is_ephemeral_content,
    is_memory_aggregate,
    is_template_placeholder,
    strip_template_fragments,
)


class _SessionStub:
    """只借 ``_extract_session_candidates`` 用到的那个 staticmethod。"""

    _passes_structural_filter = staticmethod(
        GovernedMemoryProvider._passes_structural_filter
    )


# ---------------------------------------------------------------------------
# 1) 时效性（一次性、有时间点的待办）
# ---------------------------------------------------------------------------

class TestEphemeral:
    @pytest.mark.parametrize("text", [
        "我儿子的准考证，考试前一天记得提醒我",
        "明天记得买牛奶",
        "下周一提醒我交报表",
        "3天后别忘了给车保养",
        "考试前一天记得提醒我",
        "remind me next week about the invoice",
    ])
    def test_one_off_reminders_are_ephemeral(self, text):
        assert is_ephemeral_content(text), text

    @pytest.mark.parametrize("text", [
        # 周期性偏好 —— 这恰恰是 L1 该记住的东西，构成否决票
        "每次要输密码太麻烦了，能不能做成免密",
        "都配吧，免得以后每次都弹窗",
        "以后每次都要提醒我检查日志",
        "我总是先跑测试再提交",
        # 与时间/提醒无关的持久事实
        "我在NAS有装secretstore，但本地布署不能同步给其它地方装的secretstore",
        "我家里用的是虚拟机软路由，我打算换回硬件路由器",
    ])
    def test_durable_content_is_not_ephemeral(self, text):
        assert not is_ephemeral_content(text), text

    def test_time_without_action_is_not_enough(self):
        """只有时间词、没有提醒动作 → 不判时效（可能是日程描述）。"""
        assert not is_ephemeral_content("下周我要出差三天")

    def test_action_without_time_is_not_enough(self):
        assert not is_ephemeral_content("记得把日志级别调成 debug")


# ---------------------------------------------------------------------------
# 2) 模板骨架
# ---------------------------------------------------------------------------

_USER_TEMPLATE = """# User Profile
_Generated: 2026-09-16T00:54:43.681897_

## User
# User Profile

> 手写用户信息。直接编辑此文件。

## 身份
<!-- 你的名字、角色、时区等 -->

## 偏好
<!-- 交流风格、使用的工具等 -->
"""

_PERSONA_WITH_REAL_CONTENT = """# User Profile
_Generated: 2026-09-16T03:00:41.838865_

## User
# User Profile

> 手写用户信息。直接编辑此文件。

## Known Facts
- SecretStore 使用 `ROCKET_TLS` 而不是 `SSL_CERT_FILE`
- 必须同步 `rsa.key`，否则两边加密的数据不兼容
"""


class TestTemplate:
    def test_bare_template_is_detected(self):
        assert is_template_placeholder(_USER_TEMPLATE)

    def test_real_content_beside_template_is_kept(self):
        assert not is_template_placeholder(_PERSONA_WITH_REAL_CONTENT)

    def test_short_sentence_is_not_mistaken_for_template(self):
        """短句字符本来就少 —— 不能被「实质字符数」判据误伤成模板。"""
        for text in ["都配吧，免得以后每次都弹窗", "能不能做成免密", "我先看看"]:
            assert not is_template_placeholder(text), text

    def test_strip_removes_scaffolding(self):
        out = strip_template_fragments(_PERSONA_WITH_REAL_CONTENT)
        assert "直接编辑此文件" not in out
        assert "_Generated" not in out
        assert "<!--" not in out
        # 真实内容必须原样保留
        assert "ROCKET_TLS" in out
        assert "rsa.key" in out

    def test_strip_drops_empty_sections(self):
        """空章节标题（模板遗留）清掉；有内容的标题保留。"""
        out = strip_template_fragments(_USER_TEMPLATE)
        assert "## 身份" not in out
        assert "## 偏好" not in out

    def test_strip_keeps_populated_heading(self):
        text = "## 配置\n- SecretStore 用 ROCKET_TLS\n"
        out = strip_template_fragments(text)
        assert "## 配置" in out
        assert "ROCKET_TLS" in out


# ---------------------------------------------------------------------------
# 3) 统一闸门
# ---------------------------------------------------------------------------

class TestScreenBridgeContent:
    @pytest.mark.parametrize("text, reason", [
        ("我儿子的准考证，考试前一天记得提醒我", "ephemeral"),
        # 夹具含 `_Generated:` 等强痕迹 → 走更严的 template_scaffold
        # （原因见 has_template_scaffold 的 docstring）
        (_USER_TEMPLATE, "template_scaffold"),
        ("[Image]\n为什么你每次要弹这个", "media"),
        ("[Image attached at: C:\\Users\\example\\cache\\images\\img_c68.jpg", "media"),
        ("C:\\Users\\example\\AppData\\Local\\hermes\\plugins\\deepseek\\__init__.py",
         "abs_path"),
        ("", "empty"),
    ])
    def test_rejections_carry_a_reason(self, text, reason):
        assert screen_bridge_content(text) == reason

    def test_weak_only_template_is_still_rejected(self):
        """只剩弱痕迹（H1 标题）时没有强痕迹，走 ``template`` 而非
        ``template_scaffold`` —— 两级判据各司其职。"""
        weak_only = "# User Profile\n\n## 身份\n<!-- x -->\n\n## 偏好\n"
        assert not has_template_scaffold(weak_only)
        assert screen_bridge_content(weak_only) == "template"

    def test_credentials_are_left_to_the_quarantine_channel(self):
        """凭据不归本闸门管 —— 它走 quarantine（脱敏 + 计数 + 审计）。

        如果这里把凭据一并拒掉，安全事件就只剩一个 ``skipped`` 计数器，
        从观测上等于消失了。
        """
        assert screen_bridge_content("passwd: hunter2xyz") is None

    @pytest.mark.parametrize("text", [
        "我家里用的是虚拟机软路由，我打算换回硬件路由器，还要装tailscale进行组网",
        "我在NAS有装secretstore，但本地布署不能同步给其它地方装的secretstore",
        "SecretStore 使用 `ROCKET_TLS` 而不是 `SSL_CERT_FILE`",
        "能不能做成免密？每次要输密码太麻烦了",
    ])
    def test_real_knowledge_passes(self, text):
        assert screen_bridge_content(text) is None, text


# ---------------------------------------------------------------------------
# 3b) 脚手架残留 / 记忆系统自身聚合快照（2026-09-16 追加）
# ---------------------------------------------------------------------------

#: 生产环境候选池里的真实一行（2026-09-16 抓取，id 前缀 hermes-ddfa3ccc）。
#: 它是 persona.md 被整篇导出的产物：USER.md + Memory Rules 两套模板骨架，
#: 后面跟着 24 条 L2 事实转储、Knowledge Areas 计数与 Stats 统计。
#: 一旦 promote，这些陈旧诊断（"L1手写规则层空的"、"刚才在修复 … bug 时
#: 网关重启了"）会变成永久 L1 规则。
_PERSONA_DUMP_CANDIDATE = """# User Profile
_Generated: 2026-09-16T00:54:43.681897_

## User
# User Profile

> 手写用户信息。直接编辑此文件。

## 身份
<!-- 你的名字、角色、时区等 -->

## Rules & Preferences
# Memory Rules

> 手写规则层。直接编辑此文件。

## 项目规则
<!-- 在此添加项目相关的规则 -->

测试：governed memory system 与原 memory 系统对接正常。

SecretStore 已配置为 Hermes 凭证库：
- 服务器: https://192.0.2.62:8787 (本地IP HTTPS)

## Knowledge Areas
- other: 24 facts
- life: 3 facts

## Known Facts
- ⚠ 问题: L1作为最高信任层，目前没有实际的手写规则
- 刚才在修复 `governed_health` 的 bug 时网关重启了

## Stats
- Conversations archived: 261
- Messages archived (all roles): 657
"""


class TestScaffoldAndAggregate:
    def test_persona_dump_is_rejected(self):
        """整篇 persona 转储必须被拦下 —— 这是本轮修复的那条真实脏数据。"""
        assert screen_bridge_content(_PERSONA_DUMP_CANDIDATE) in (
            "template_scaffold", "aggregate",
        )

    def test_scaffold_is_detected_even_beside_real_content(self):
        """关键回归：夹带真实内容**不能**成为放行理由。

        旧判据（is_template_placeholder）只看「剥掉骨架后还剩不剩东西」，
        所以这条会被放行 —— 24 条转储就跟着进来了。
        """
        assert not is_template_placeholder(_PERSONA_WITH_REAL_CONTENT)
        assert has_template_scaffold(_PERSONA_WITH_REAL_CONTENT)
        assert screen_bridge_content(_PERSONA_WITH_REAL_CONTENT) == "template_scaffold"

    @pytest.mark.parametrize("text", [
        "## Stats\n- Conversations archived: 261\n- Messages archived: 657",
        "## Knowledge Areas\n- other: 24 facts",
        "## Known Facts\n- SecretStore 用 ROCKET_TLS\n",
    ])
    def test_memory_aggregates_are_detected(self, text):
        assert is_memory_aggregate(text), text

    @pytest.mark.parametrize("text", [
        "SecretStore 使用 `ROCKET_TLS` 而不是 `SSL_CERT_FILE`",
        "我在NAS有装secretstore，但本地布署不能同步给其它地方装的secretstore",
        # 真实知识笔记可以有标题 —— 不能因为像文档就判成聚合
        "## 代理方案对比\n- 香港节点组需要支持手动指定，不可用时才回退自动",
    ])
    def test_real_content_is_not_an_aggregate(self, text):
        assert not is_memory_aggregate(text), text

    def test_persona_is_not_a_bridge_source(self):
        """persona.md 是派生聚合，不该再作为候选来源（整篇导出＝闭环污染）。"""
        import importlib.util
        from pathlib import Path

        src = (Path(__file__).resolve().parent.parent
               / "scripts" / "scope_recall_bridge.py").read_text(encoding="utf-8")
        # collect_candidates 里不得再出现 persona 的 source_record 调用
        body = src.split("def collect_candidates", 1)[1]
        assert 'source_record(persona_path' not in body
        # 守卫必须存在，避免将来被静默加回
        assert 'if source_kind == "persona":' in src


# ---------------------------------------------------------------------------
# 4) 候选抽取用统一的强信号尺子
# ---------------------------------------------------------------------------

class TestSessionCandidates:
    def _extract(self, text: str, role: str = "user"):
        return GovernedMemoryProvider._extract_session_candidates(
            _SessionStub(), [{"role": role, "content": text}]
        )

    @pytest.mark.parametrize("text", [
        "我儿子的准考证，考试前一天记得提醒我",
        "[Image]\n为什么你每次要弹这个",
        "我已经移了两个，把剩下两个移进去，其它不要动",
        "都配吧，免得以后每次都弹窗",
        "暂时不用，我通过代理测试一下，现在发现个问题",
        "不用了，每次搞你都被拦截了，估计大模型有风控",
    ])
    def test_noise_is_rejected(self, text):
        assert self._extract(text) == [], text

    @pytest.mark.parametrize("text", [
        "我在NAS有装secretstore，但本地布署不能同步给其它地方装的secretstore",
        "我家里用的是虚拟机软路由，我打算换回硬件路由器，还要装tailscale进行组网",
    ])
    def test_real_preferences_survive(self, text):
        got = self._extract(text)
        assert len(got) == 1, text
        assert got[0]["target"] == "memory"

    def test_a_polar_question_is_no_longer_a_preference(self):
        """放宽前的这条「真实偏好」实为疑问句，2026-09-17 起按规则拒收。

        旧内联门槛把「能不能做成免密？每次要输密码太麻烦了」当偏好放行，因为
        「能不能」字面含「不能」+ 角色先验加满。它是一条**请求**（问能不能做
        成免密），不是承诺；写入路径与回放已按同一判断把它剔掉（见
        ``TestBridgeSharesTheDialogueRuler``）。此处只把这一条的预期如实改掉，
        不含其它两条真实部署事实。
        """
        assert self._extract("能不能做成免密？每次要输密码太麻烦了") == []

    def test_assistant_turns_are_never_candidates(self):
        """Bridge 只从 user 轮抽候选 —— 助手叙述没有「用户意愿」可言。"""
        assert self._extract("SecretStore 使用 ROCKET_TLS 而不是 SSL_CERT_FILE",
                             role="assistant") == []

    def test_multimodal_content_does_not_crash(self):
        """content 是 list（多模态）时不能崩 —— 这正是 9-15 那次 P0 的成因。"""
        msgs = [{"role": "user", "content": [
            {"type": "text", "text": "我家里用的是虚拟机软路由，我打算换回硬件路由器，还要装tailscale进行组网"},
        ]}]
        assert len(GovernedMemoryProvider._extract_session_candidates(_SessionStub(), msgs)) == 1

    def test_empty_multimodal_content_is_skipped(self):
        msgs = [{"role": "user", "content": [{"type": "image_url", "image_url": {}}]}]
        assert GovernedMemoryProvider._extract_session_candidates(_SessionStub(), msgs) == []


# ---------------------------------------------------------------------------
# 5) 桥接入口与写入 / 回放共用同一把尺子（2026-09-17）
# ---------------------------------------------------------------------------

#: 直呼 ``_fact_signal_score(..., strong_only=True)`` 曾让本入口把整类疑问句放行：
#: 「能不能…」字面含「不能」，而旧内联表达式既不传 ``include_role=False``（于是
#: 「谁说的」替「说了什么」作了证），也不消解疑问句式。修复前实测这五条全部
#: ADMIT。Bridge 候选会 promote 进 L1（每轮注入的规则层），代价比 L2 更重。
CHATTER_AT_BRIDGE = [
    "能不能帮我改一下配置",
    "能不能做成免密",
    "这个能不能行",
    "我能不能先看看",
    "我一般都用这个",
    # 非承诺型虚词句：「我的」已被移出强信号池（Defect C），只剩通用虚词。
    "我的意思是再想想",
]

#: 真承诺：独立的模态信号（我需要…）必须照样放行 —— 收紧不得误伤规则本身。
COMMITMENT_AT_BRIDGE = [
    "我需要把网关换成 schtasks 启动",
]

#: 真结论：单条强信号 + 命名佐证（因为…PostgreSQL）。它只有一条强信号，靠
#: ``dialogue_fact_admits`` 的佐证分支进来 —— 去掉该分支会在收紧时连带删掉
#: 一整族「…因为…」结论，所以这里要在入口层面钉住。
CORROBORATED_AT_BRIDGE = [
    "我们决定用 PostgreSQL 因为它更公平稳定",
]


class TestBridgeSharesTheDialogueRuler:
    """入口判定必须逐条等于共享尺子，疑问句不再借角色先验通行。"""

    def _extract(self, text: str, role: str = "user"):
        return GovernedMemoryProvider._extract_session_candidates(
            _SessionStub(), [{"role": role, "content": text}]
        )

    @pytest.mark.parametrize("text", CHATTER_AT_BRIDGE)
    def test_chatter_no_longer_becomes_an_l1_rule(self, text):
        assert self._extract(text) == [], text

    @pytest.mark.parametrize("text", COMMITMENT_AT_BRIDGE + CORROBORATED_AT_BRIDGE)
    def test_real_commitments_still_become_candidates(self, text):
        got = self._extract(text)
        assert len(got) == 1, text
        assert got[0]["target"] == "memory"

    @pytest.mark.parametrize(
        "text", CHATTER_AT_BRIDGE + COMMITMENT_AT_BRIDGE + CORROBORATED_AT_BRIDGE)
    def test_entry_verdict_equals_the_shared_ruler_verbatim(self, text):
        """不是「差不多」：入口布尔值必须逐条等于 ``dialogue_fact_admits``。"""
        assert bool(self._extract(text)) == dialogue_fact_admits(text, "user"), text

    def test_an_ip_port_fact_is_structured_but_names_no_pool(self):
        """如实钉住当前边界（已知缺口，**不放宽门槛**）。

        ``示例主路由 192.0.2.1 的 SSH 端口是 8022`` 结构证据齐全（IP + 数字 +
        拉丁，struct>=1），却因强信号池为 0 而够不着佐证分支要求的
        ``score >= _USER_SIGNAL_WEIGHT``，本入口因此仍拒收它。这是「IP:port 这类
        命名事实不在强词表里」的缺口，不是本闸门的放行错误；补它需要另一套独立
        工作（见任务报告）。此处只把现状钉住：将来口径若修正，这条会红，提醒复核。
        """
        text = "示例主路由 192.0.2.1 的 SSH 端口是 8022"
        assert external_structural_evidence(text) >= 1           # 结构：有
        assert _fact_signal_score(text, "", strong_only=True) == 0  # 强信号池：无
        assert not dialogue_fact_admits(text, "user")            # 结论：拒收
        assert self._extract(text) == []


# ---------------------------------------------------------------------------
# 6) 族守卫：``plugin/`` 下除尺子模块外不得再出现「私有尺子」
# ---------------------------------------------------------------------------

_PLUGIN_DIR = Path(__file__).resolve().parent.parent / "plugin"


def _strong_only_calls(path: Path):
    """返回行号列表：``path`` 里手写的 ``_fact_signal_score(..., strong_only=True)``。

    用 AST 而非子串匹配，避免把 docstring / 注释里的示例误判成真实调用。
    """
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    hits = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        name = (func.attr if isinstance(func, ast.Attribute)
                else getattr(func, "id", None))
        if name != "_fact_signal_score":
            continue
        for kw in node.keywords:
            if (kw.arg == "strong_only"
                    and isinstance(kw.value, ast.Constant)
                    and kw.value.value is True):
                hits.append(node.lineno)
    return hits


class TestNoPrivateRuler:
    def test_only_the_ruler_module_may_call_strong_only(self):
        """``_sync.py`` 是唯一允许比较强信号分数的地方。

        三个消费者（写入 ``_extract_atomic_facts`` / 回放 ``l2_apply_gate.py`` /
        Bridge 候选 ``_extract_session_candidates``）必须共享
        ``dialogue_fact_admits``。任何模块自己写
        ``_fact_signal_score(..., strong_only=True)`` 比较式，都会长出第二把尺子
        —— 本轮的 Bridge 漏点正是这样来的，这条守卫让它变成测试失败而非静默污染。
        """
        offenders = {}
        for py in sorted(_PLUGIN_DIR.rglob("*.py")):
            if py.name == "_sync.py":
                continue
            hits = _strong_only_calls(py)
            if hits:
                offenders[str(py.relative_to(_PLUGIN_DIR.parent))] = hits
        assert offenders == {}, (
            "发现 _sync.py 之外的私有强信号准入尺子，请改用 "
            f"dialogue_fact_admits()：{offenders}"
        )
