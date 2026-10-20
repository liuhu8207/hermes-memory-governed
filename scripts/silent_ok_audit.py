#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""审计 ``# silent-ok:`` 豁免标记 —— CI 阻塞闸门 + 人工复核清单。

为什么需要它（以及为什么它此前等于不存在）
----------------------------------------
护栏此前**只活在 pytest 里**（``tests/test_recall_degraded_visibility.py``），
CI 只跑测试，没人跑独立的审计器 —— "没人跑的检查等于不存在"（本项目刚吃过这个
亏：棘轮从 ``38526e3`` 起就是红的，因为没人跑、没人发现）。本脚本因此被接进
``.github/workflows/ci.yml`` 作为**独立阻塞 job**：``--check`` 非零即 CI 红。

三轮都挂在哪儿 / 这一版怎么修
----------------------------
**问题 1：没有调用方。** 见上 —— 现已接进 CI（``guardrail`` job，且 ``test``
job 用 ``needs: guardrail`` 显式依赖它，红了就跑不到测试）。

**问题 2：干净树上 ``--check`` 恒红。** 旧版按**原始文本行**扫 ``# silent-ok:``，
于是把 docstring / 模块注释里"讲解这个标记怎么用"的示例也当成真实豁免，连它自己
脚下的 ``scripts/silent_ok_audit.py`` 都被扫到了（第 2 行 docstring）。这一版
**只扫 AST**：标记只有在 ``ExceptHandler`` 的**行区间内**（或 suppress 提及行上）
才生效；docstring 覆盖的行一律置空（``guardrail_rules.code_lines``）。文档里的
示例因此被正确识别为"不生效"，并在输出里单独列出（不静默丢弃）。

**问题 3：判不了理由的真伪。** 旧版只查"理由是否 ≥8 字符"。这一版验证**可验证的
那部分**：
* ``already-reported`` 的理由必须**点名上报者**（上报机制 ``log_degraded`` /
  ``logger.warning`` …，或可见渠道 ``errors`` / ``返回值`` / ``report`` …）。
  ``调用方会上报`` 这种无主语的话**直接判红**。
* 理由里**引用到的标识必须真实存在**（``--reason-key`` 可注入运行时才知道的
  ``_diag.stats()`` 键 / logger 名 / 函数名）。引用不存在的键 = 编造 ⇒ 判红。
* **验证不了的不粉饰**：只点名机制/渠道、却给不出可解析标识的，单列进
  "需人工确认"清单（默认不阻断；``--strict`` 升级为阻断）。

判定逻辑**不在这里实现**：全部来自仓库根的 ``guardrail_rules.py`` —— 与
``tests/test_recall_degraded_visibility.py`` **同一个模块、同一份代码**，改一次
两边同步变化（不存在第二套判定悄悄漂移的可能）。

用法
----
    python scripts/silent_ok_audit.py               # 列出全部生效标记
    python scripts/silent_ok_audit.py --check       # 闸门模式：有阻断问题即非零
    python scripts/silent_ok_audit.py --check --strict
    python scripts/silent_ok_audit.py --json
    python scripts/silent_ok_audit.py --reason-key ingest::fetch_failed
    python scripts/silent_ok_audit.py --reason-keys-file ci/reason-keys.txt
    python scripts/silent_ok_audit.py --scope all   # 附带列出 legacy 区（棘轮管辖）

退出码
------
``0`` 干净；``1`` 有阻断级问题（或 ``--strict`` 下有需人工确认项）；``2`` 用法/
IO 错误。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import guardrail_rules as rules  # noqa: E402  （先补 sys.path 再导入）

#: 退出码。
EXIT_OK = 0
EXIT_BLOCKED = 1
EXIT_USAGE = 2

#: ``--scope`` 的两个取值。
SCOPES = ("package", "all")


def _targets(scope: str, root: Path) -> list:
    """本次扫描的文件集合（``root`` 由调用方显式传入，便于测试替换）。

    * ``package``：``plugin/memory_governed/**/*.py``（**零容忍**，阻断）—— 与
      pytest 闸门同一范围。
    * ``all``：额外带上 ``memory_cli.py`` + ``scripts/*.py``。这一区**不由本闸门
      管辖**（它有 201 处历史存量，由 ``tests/test_guardrail_ratchet.py`` 的按文件
      棘轮看管），故这里**只列出、不阻断** —— 不然两个闸门会互相打架。
    """
    files = rules.package_files(root)
    if scope == "all":
        files = files + [REPO / "memory_cli.py"] + sorted((REPO / "scripts").glob("*.py"))
    return [p for p in files if p.exists()]


def _legacy_files(scope: str) -> set:
    """``--scope all`` 里"只列不阻断"的那部分（按绝对路径）。"""
    if scope != "all":
        return set()
    return {p.resolve() for p in ([REPO / "memory_cli.py"]
                                 + sorted((REPO / "scripts").glob("*.py")))}


def _load_reason_keys(explicit: list, key_file: str | None) -> list:
    """汇总 ``--reason-key`` 与 ``--reason-keys-file`` 注入的已知标识。"""
    keys = list(explicit or [])
    if key_file:
        path = Path(key_file)
        if not path.exists():
            raise SystemExit("找不到 --reason-keys-file: %s" % path)
        for raw in path.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if line and not line.startswith("#"):
                keys.append(line)
    return keys


def _build_index(paths: list, extra_keys: list) -> rules.SymbolIndex:
    """建立"仓库里真实存在哪些标识"的静态索引（用于理由的存在性检查）。"""
    sources = []
    for path in paths:
        try:
            sources.append((path.stem, path.read_text(encoding="utf-8")))
        except (OSError, UnicodeDecodeError):  # silent-ok: value-fallback — 读不出的旁路文件不进索引，per-file 非子系统故障
            continue
    return rules.SymbolIndex.from_sources(sources, extra_keys=extra_keys)


def _split(findings: list, legacy: set) -> tuple:
    """把 findings 拆成 ``(阻断, 需人工确认, legacy 区参考项)``。"""
    blocking, human, legacy_only = [], [], []
    for f in findings:
        if (REPO / f.file).resolve() in legacy:
            legacy_only.append(f)
        elif f.blocking:
            blocking.append(f)
        else:
            human.append(f)
    return blocking, human, legacy_only


def _print_text(result: rules.AuditResult, blocking: list, human: list,
                legacy_only: list, *, strict: bool, scope: str,
                extra_keys: list, list_all: bool = False) -> None:
    by_kind: dict = {}
    for site in result.sites:
        by_kind[site.kind] = by_kind.get(site.kind, 0) + 1
    kinds = " / ".join("%s %d" % (k, by_kind[k]) for k in sorted(by_kind)) or "无"

    print("silent-ok 审计 · 范围 %s/**/*.py%s"
          % ("plugin/memory_governed", " (+ legacy 区，只列不阻断)" if scope == "all" else ""))
    print("  扫描文件     : %d" % len(result.files))
    for name in result.files[:6]:
        print("      · %s" % name)
    if len(result.files) > 6:
        print("      · …（共 %d 个）" % len(result.files))
    print("  ExceptHandler: %d 个" % result.handler_count)
    print("  生效标记     : %d（%s）" % (len(result.sites), kinds))
    print("  未附着标记   : %d（文档/注释示例，不生效）" % len(result.unattached))
    for file_label, lineno, kind, marker_kind in result.unattached:
        print("      · %s:%d  [%s] 标记种类=%r" % (file_label, lineno, kind, marker_kind))
    print("  注入的已知标识: %s" % (", ".join(extra_keys) if extra_keys else "无"))

    print("\n生效标记清单（file:line · 种类 · 验证状态 · 理由）：")
    if not list_all:
        print("  （共 %d 条；只列出需要处置的，全部清单见 --list-all）" % len(result.sites))
    for site in result.sites:
        if not list_all:
            verdict = site.verdict
            quiet = verdict is None or (not verdict.problems and not verdict.needs_human)
            if quiet:
                continue
        verdict = site.verdict
        if verdict is None:
            state = "未验证"
        elif verdict.problems:
            # ``value-fallback`` 的理由问题只是提示（不阻断），标签要说清楚。
            prefix = "阻断: " if site.kind == "already-reported" else "待确认: "
            state = prefix + verdict.problems[0][:28]
        elif verdict.verified:
            state = "已验证(%s)" % ",".join((verdict.refs + verdict.diag_keys)[:3])
        elif verdict.needs_human:
            state = "需人工确认"
        else:
            state = "无引用可校验（理由可选）"
        print("  %-46s:%-5d %-16s %-22s %s"
              % (site.file, site.line, site.kind, state,
                 (site.reason or "").strip()))

    print("\n违规（阻断级）: %d" % len(blocking))
    for f in blocking:
        print("  FAIL %s:%d [%s] %s" % (f.file, f.line, f.category, f.detail))
    print("需人工确认（未自动验证，默认不阻断%s）: %d"
          % ("；--strict 已开启，视同阻断" if strict else "", len(human)))
    for f in human:
        print("  HUMAN %s:%d [%s] %s" % (f.file, f.line, f.category, f.detail))
    if legacy_only:
        print("\nlegacy 区（memory_cli.py / scripts，由棘轮看管，本闸门不阻断）: %d"
              % len(legacy_only))
        for f in legacy_only[:20]:
            print("  INFO %s:%d [%s] %s" % (f.file, f.line, f.category, f.detail))
        if len(legacy_only) > 20:
            print("  …（共 %d 项）" % len(legacy_only))


def _print_json(result: rules.AuditResult, blocking: list, human: list,
                legacy_only: list) -> None:
    payload = {
        "files": result.files,
        "handler_count": result.handler_count,
        "markers": [
            {
                "file": s.file, "line": s.line, "function": s.function,
                "kind": s.kind, "reason": s.reason, "caught": list(s.caught),
                "verified": bool(s.verdict and s.verdict.verified),
                "needs_human": bool(s.verdict and s.verdict.needs_human),
                "refs": list(s.verdict.refs) if s.verdict else [],
                "diag_keys": list(s.verdict.diag_keys) if s.verdict else [],
            }
            for s in result.sites
        ],
        "unattached_markers": [
            {"file": f, "line": n, "where": w, "kind": k}
            for f, n, w, k in result.unattached
        ],
        "blocking": [
            {"file": f.file, "line": f.line, "function": f.function,
             "category": f.category, "detail": f.detail}
            for f in blocking
        ],
        "needs_human": [
            {"file": f.file, "line": f.line, "function": f.function,
             "category": f.category, "detail": f.detail}
            for f in human
        ],
        "legacy_info": [
            {"file": f.file, "line": f.line, "category": f.category}
            for f in legacy_only
        ],
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def main(argv: list | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="审计 # silent-ok: 豁免标记（判定逻辑与 pytest 闸门共用）")
    parser.add_argument("--check", action="store_true",
                        help="闸门模式：存在阻断级问题时退出码非 0")
    parser.add_argument("--strict", action="store_true",
                        help="把「需人工确认」也升级为阻断（需与 --check 合用）")
    parser.add_argument("--json", action="store_true", help="输出 JSON")
    parser.add_argument("--list-all", action="store_true",
                        help="列出**全部**生效标记（默认只列需要处置的）")
    parser.add_argument("--scope", choices=SCOPES, default="package",
                        help="package=只扫插件包（默认，零容忍）；"
                             "all=附带列出 legacy 区（只列不阻断）")
    parser.add_argument(
        "--reason-key", action="append", default=[], metavar="IDENT",
        help="注入一个**已知存在**的标识（_diag 计数键 comp::reason、logger 名、"
             "函数名…），让理由引用它能通过存在性检查；可重复")
    parser.add_argument(
        "--reason-keys-file", default=None, metavar="PATH",
        help="从文件按行读取要注入的已知标识（# 开头为注释）")
    args = parser.parse_args(argv)

    # 扫描根在**调用时**读取（不是 import 时的默认值），这样守卫可被测试替换验证。
    root = rules.PACKAGE_DIR
    anchor_err = rules.package_anchor_error(root)
    if anchor_err:
        print("扫描集合自检失败：%s" % anchor_err, file=sys.stderr)
        return EXIT_USAGE

    paths = _targets(args.scope, root)
    if not paths:
        print("扫描目标为空（路径写错？）", file=sys.stderr)
        return EXIT_USAGE

    extra_keys = _load_reason_keys(args.reason_key, args.reason_keys_file)
    symbols = _build_index(paths, extra_keys)
    result = rules.scan_paths(paths, root=REPO, symbols=symbols)
    blocking, human, legacy_only = _split(result.findings, _legacy_files(args.scope))

    if args.json:
        _print_json(result, blocking, human, legacy_only)
    else:
        _print_text(result, blocking, human, legacy_only, strict=args.strict,
                    scope=args.scope, extra_keys=extra_keys,
                    list_all=args.list_all)

    if not args.check:
        return EXIT_OK
    if blocking:
        if not args.json:
            print("\nFAIL: %d 项阻断级问题（silent-ok 审计未通过）" % len(blocking))
        else:
            print(json.dumps({"result": "fail", "blocking": len(blocking)},
                             ensure_ascii=False), file=sys.stderr)
        return EXIT_BLOCKED
    if args.strict and human:
        if not args.json:
            print("\nFAIL(--strict): %d 项需人工确认（未自动验证）" % len(human))
        return EXIT_BLOCKED
    if not args.json:
        print("\nOK: 干净树，0 违规（需人工确认 %d 项，不阻断）" % len(human))
    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
