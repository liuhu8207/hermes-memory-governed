# -*- coding: utf-8 -*-
"""插件侧 vault 遍历越界的回归测试（2026-09-17）。

**缺陷**：``KnowledgeBase.scan_vault_paths()``（喂索引同步）和
``KnowledgeBase._iter_notes()``（喂检索打分）各自手写了一份
``self._vault.rglob("*.md")``，只硬编码跳过 ``.obsidian``，没有任何链接 /
包含性检查。于是：

- 一个指向库外的 junction 会让**域外 .md** 被收进 ``scan_vault_paths()``，
  再经 ``scan_vault_mtimes`` → ``_diff_index_state`` → ``sync_index`` 被嵌入、
  落进 LanceDB 的 ``kb_index``、并在 ``kb-search`` 里被召回 —— 这不是「多列了
  一个文件」，是域外内容能进共享检索层。
- 同一个仓库，CLI（``memory_cli.iter_notes``，跳过所有点目录 + realpath 包含性
  检查）和插件对「库里有哪些笔记」给出**不同答案**。共享存储只能有一个答案。

**边界（如实）**：真实 vault（27 篇）里链接数为 0，所以这是**潜在暴露，不是已
实现暴露**。判 P1 是因为路径可达且插件每会话都跑，不是因为已有数据被读出去。

**本文件守住的三件事**：
1. junction 越界的域外文件不出现在任何一条遍历结果里；
2. 所有点目录都被跳过（不只是 ``.obsidian``）；
3. 两条路**必须经过同一个遍历器** —— 一旦有人再写第二份 ``rglob``，AST 断言
   立刻变红。
"""

from __future__ import annotations

import ast
import os
import subprocess
from pathlib import Path

import pytest

from plugin.memory_governed import _kb as _kb_mod
from plugin.memory_governed._config import GovernedMemoryConfig
from plugin.memory_governed._kb import (
    KnowledgeBase,
    _is_link,
    _is_within,
    iter_vault_notes,
)

NOTE_A = "---\ntitle: A\n---\nalpha body"
SECRET = "---\ntitle: Secret\n---\n域外机密 leak-canary"
JUNK = "---\ntitle: Junk\n---\n不该被收进来的内容"

#: 点目录逐个验证：``.obsidian`` 是旧代码唯一跳过的；``.git`` / ``.trash`` 是
#: 旧代码会收、CLI 侧会跳的 —— 两边不一致正是这个缺陷的形状。
DOT_DIRS = (".git", ".trash", ".obsidian", ".hidden")


def _make_kb(wiki: Path, tmp_path: Path) -> KnowledgeBase:
    """沙箱 KB：``l2_db_path`` 也指向沙箱。

    必须覆盖：``KnowledgeBase.__init__`` 里 ``_index_db`` 由
    ``Path(l2_db_path).parent / "kb_index"`` 推导，不覆盖就会解析到**生产**
    kb_index —— 那等于把测试笔记写进真实向量库。
    """
    cfg = GovernedMemoryConfig()
    cfg.wiki_dir = str(wiki)
    cfg.l2_db_path = str(tmp_path / "sandbox" / "memory" / "l2")
    cfg.vector.backend = "none"
    return KnowledgeBase(cfg)


def _make_dir_link(link: Path, target: Path) -> None:
    """建一个 ``rglob`` 会跟出去的目录链接。

    Windows 上**必须**是 junction：``Path.symlink_to(target_is_directory=True)``
    在没有 Developer Mode 的机器上**不报错但什么都不创建**，基于 symlink 的测试
    会因此假绿。junction 无需特权，也正是生产环境的真实形态。POSIX 上用 symlink。

    无论哪条路，调用方都会复核结果：没建起来的链接必须让测试**响亮地失败**，
    绝不能悄悄变成「没发现越界」。
    """
    if os.name == "nt":
        subprocess.run(
            ["powershell", "-NoProfile", "-NonInteractive", "-Command",
             "New-Item -ItemType Junction -Path '%s' -Target '%s' | Out-Null"
             % (str(link), str(target))],
            check=False, capture_output=True, timeout=120,
        )
    else:
        link.symlink_to(target, target_is_directory=True)


@pytest.fixture
def escaped_vault(tmp_path: Path):
    """``(vault, tmp_path)``：vault 内有一个指向库外的目录链接 ``leak``。"""
    vault = tmp_path / "vault"
    (vault / "notes").mkdir(parents=True)
    (vault / "notes" / "a.md").write_text(NOTE_A, encoding="utf-8")

    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "secret.md").write_text(SECRET, encoding="utf-8")

    for dot in DOT_DIRS:
        d = vault / dot
        d.mkdir()
        (d / ("%s.md" % dot.strip("."))).write_text(JUNK, encoding="utf-8")

    link = vault / "leak"
    _make_dir_link(link, outside)

    # 链接必须真的建起来、真的越界 —— 否则整组测试是假绿。
    if not link.exists():
        pytest.fail(
            "目录链接没建起来：%s —— 无法验证越界行为，此测试不能算通过"
            "（Windows 无 Developer Mode 时 symlink_to 会静默不建，请用 junction）"
            % link)
    real = Path(os.path.realpath(str(link)))
    vault_real = Path(os.path.realpath(str(vault)))
    if _is_within(real, vault_real):
        pytest.fail("链接没有越界：%s -> %s 仍在 %s 之内，测不到缺陷"
                    % (link, real, vault_real))
    return vault, tmp_path


# ---------------------------------------------------------------------------
# 越界链接
# ---------------------------------------------------------------------------

class TestWalkRefusesOutOfVaultLinks:
    """域外 .md 绝不能出现在任何一条遍历结果里。"""

    def test_scan_vault_paths_excludes_the_junction_escape(self, escaped_vault):
        vault, tmp = escaped_vault
        kb = _make_kb(vault, tmp)

        rels = sorted(kb.scan_vault_paths())

        assert "notes/a.md" in rels, "正常笔记必须还在，修复不能把库清空"
        assert not any(r.startswith("leak") for r in rels), (
            "域外文件进了索引同步的输入 —— 它会被嵌入、落进 kb_index、"
            "并在 kb-search 里被召回：%s" % rels)

    def test_iter_notes_excludes_the_junction_escape(self, escaped_vault):
        vault, tmp = escaped_vault
        kb = _make_kb(vault, tmp)

        rels = sorted(kb._rel(n.path) for n in kb._iter_notes())

        assert "notes/a.md" in rels, "正常笔记必须还在"
        assert not any(r.startswith("leak") for r in rels), (
            "域外文件出现在检索打分集里 —— 域外内容会被搜出来：%s" % rels)

    def test_the_escape_is_refused_visibly_not_swallowed(self, escaped_vault):
        """拒绝必须可见：静默丢弃让「库是空的」和「遍历被拒」长得一模一样。"""
        vault, tmp = escaped_vault
        kb = _make_kb(vault, tmp)
        kb.scan_vault_paths()

        refusals = kb.last_walk_refusals
        assert refusals, "越界被拒了却没有任何痕迹 —— 又是一次静默失败"
        assert any("leak" in r for r in refusals), (
            "拒绝记录里没点名越界的那个链接：%s" % refusals)
        assert any("outside" in r for r in refusals), (
            "拒绝记录里没说清它跑到库外哪儿去了：%s" % refusals)

    def test_a_link_whose_target_has_no_notes_is_still_reported(self, tmp_path):
        """The refusal must not depend on ``rglob`` having descended the link.

        ``pathlib`` will not descend a directory **symlink** (cycle guard), while
        a Windows **junction** is transparent to ``scandir`` and *is* descended.
        So on POSIX an out-of-vault link produced no hits and, with them, no
        refusal: the escape was excluded silently, and "that link is ignored"
        looked identical to "there is nothing there".

        Pointing the link at a directory that holds no notes reproduces that
        situation on Windows as well — the walk finds nothing, so only the
        explicit link check can report it. The test therefore has teeth on both
        platforms rather than only on the one that happened to fail.
        """
        vault = tmp_path / "vault"
        (vault / "notes").mkdir(parents=True)
        (vault / "notes" / "a.md").write_text(NOTE_A, encoding="utf-8")
        outside = tmp_path / "outside"          # deliberately contains no .md
        outside.mkdir()

        link = vault / "leak"
        _make_dir_link(link, outside)
        if not link.exists():
            pytest.fail("目录链接没建起来：%s —— 无法验证越界行为，此测试不能算通过"
                        % link)

        kb = _make_kb(vault, tmp_path)
        assert "notes/a.md" in kb.scan_vault_paths()

        refusals = kb.last_walk_refusals
        assert refusals, "越界链接被静默忽略 —— 又是一次没有任何痕迹的失败"
        assert any("leak" in r for r in refusals), (
            "拒绝记录里没点名越界的那个链接：%s" % refusals)
        assert any("outside" in r for r in refusals), (
            "拒绝记录里没说清它跑到库外哪儿去了：%s" % refusals)

    def test_domain_content_never_reaches_the_note_set(self, escaped_vault):
        """端到端：域外文件内容一条都不能进检索集（不是只挡路径）。"""
        vault, tmp = escaped_vault
        kb = _make_kb(vault, tmp)
        texts = [n.full_text for n in kb._iter_notes()]
        assert not any("leak-canary" in t for t in texts), (
            "域外正文被读进了检索集 —— 共享检索层会被域外内容污染")


# ---------------------------------------------------------------------------
# 点目录
# ---------------------------------------------------------------------------

class TestWalkSkipsEveryDotDirectory:
    """所有点目录都跳过，与 CLI 侧 ``iter_notes`` 同规则。"""

    @pytest.mark.parametrize("dot", DOT_DIRS)
    def test_dot_dir_is_skipped_by_scan_vault_paths(self, escaped_vault, dot):
        vault, tmp = escaped_vault
        kb = _make_kb(vault, tmp)
        rels = sorted(kb.scan_vault_paths())
        assert not any(r.startswith(dot + "/") for r in rels), (
            "%s/ 被收进来了 —— 只有 .obsidian 被跳过的旧代码正是这个形状" % dot)

    @pytest.mark.parametrize("dot", DOT_DIRS)
    def test_dot_dir_is_skipped_by_iter_notes(self, escaped_vault, dot):
        vault, tmp = escaped_vault
        kb = _make_kb(vault, tmp)
        rels = sorted(kb._rel(n.path) for n in kb._iter_notes())
        assert not any(r.startswith(dot + "/") for r in rels), (
            "%s/ 被收进检索集了" % dot)

    def test_real_note_is_still_there(self, escaped_vault):
        vault, tmp = escaped_vault
        kb = _make_kb(vault, tmp)
        assert sorted(kb.scan_vault_paths()) == ["notes/a.md"]


# ---------------------------------------------------------------------------
# 单一遍历器
# ---------------------------------------------------------------------------

class TestThereIsOnlyOneVaultWalk:
    """两条路必须共用同一个遍历谓词 —— 复制第二份，这里就红。"""

    def test_module_declares_exactly_one_rglob(self):
        """AST 级守卫：源码里只允许一处 ``rglob`` 调用，且必须在共享遍历器里。

        用 AST 而不是字符串计数：注释和 docstring 里也会出现 ``rglob`` 字样，
        字符串计数会把它们算进去（本机实测把 1 处真实调用数成 4 处）。
        """
        src = Path(_kb_mod.__file__).read_text(encoding="utf-8")
        tree = ast.parse(src)

        calls = [n for n in ast.walk(tree)
                 if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                 and n.func.attr == "rglob"]
        assert len(calls) == 1, (
            "_kb.py 里有 %d 处 rglob 调用 —— 库遍历只能有一份实现，"
            "多一份就意味着索引同步侧和检索侧会各自漂移" % len(calls))

        owner: dict[int, str] = {}
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                for child in ast.walk(node):
                    owner[id(child)] = node.name
        assert owner.get(id(calls[0])) == "iter_vault_notes", (
            "唯一的 rglob 必须住在 iter_vault_notes 里，实际在 %r"
            % owner.get(id(calls[0])))

    def test_both_entry_points_go_through_the_shared_walker(
            self, escaped_vault, monkeypatch):
        """运行时守卫：两条路都真的调用了 :func:`iter_vault_notes`。"""
        vault, tmp = escaped_vault
        kb = _make_kb(vault, tmp)

        seen: list[Path] = []
        original = _kb_mod.iter_vault_notes

        def spy(v, *, errors=None):
            seen.append(Path(v))
            return original(v, errors=errors)

        monkeypatch.setattr(_kb_mod, "iter_vault_notes", spy)

        kb.scan_vault_paths()
        kb._iter_notes()

        assert len(seen) == 2, (
            "两条遍历路径里有 %d 条没走共享遍历器 —— 任何一条自己 rglob，"
            "插件就会对「库里有什么」给出第二个答案" % (2 - len(seen)))

    def test_walker_is_usable_directly(self, escaped_vault):
        """共享遍历器本身的行为：收库内、拒库外、跳点目录。"""
        vault, _tmp = escaped_vault
        rels = sorted(str(p.relative_to(vault)).replace("\\", "/")
                      for p in iter_vault_notes(vault))
        assert rels == ["notes/a.md"], rels


# ---------------------------------------------------------------------------
# 链接识别
# ---------------------------------------------------------------------------

class TestLinkDetection:
    """``_is_link`` 必须同时认得 symlink 和 junction。"""

    def test_junction_is_detected(self, escaped_vault):
        vault, _tmp = escaped_vault
        link = vault / "leak"
        assert _is_link(link) is True, "junction 没被认出来 —— 它会整个绕过检查"
        if os.name == "nt":
            # 事实记录：islink 对 junction 返回 False，只判 islink 等于没判
            assert os.path.islink(str(link)) is False

    def test_plain_directory_is_not_a_link(self, escaped_vault):
        vault, _tmp = escaped_vault
        assert _is_link(vault / "notes") is False

    def test_link_staying_inside_the_vault_is_allowed(self, tmp_path):
        """留在库内的链接不该被拒 —— 重要的是方向，不是这种结构。"""
        vault = tmp_path / "vault"
        (vault / "notes").mkdir(parents=True)
        (vault / "notes" / "a.md").write_text(NOTE_A, encoding="utf-8")
        inner = tmp_path / "vault" / "inner"
        inner.mkdir()
        (inner / "deep.md").write_text(NOTE_A, encoding="utf-8")
        _make_dir_link(vault / "alias", inner)

        rels = sorted(str(p.relative_to(vault)).replace("\\", "/")
                      for p in iter_vault_notes(vault))
        assert "notes/a.md" in rels, "库内正常笔记必须还在"
