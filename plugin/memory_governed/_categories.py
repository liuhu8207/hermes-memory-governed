# -*- coding: utf-8 -*-
"""Controlled categories shared by L2 facts and KB navigation."""

from __future__ import annotations

from ._vault import KB_CATEGORIES, normalize_category

CONTROLLED_CATEGORIES = tuple(KB_CATEGORIES)

_META = (
    "hermes-memory-governed", "hgm", "memory_cli", "共享记忆", "mcp",
    "governed", "plugin", "检索", "知识库", "l2", "l3", "写入质量门",
)
_USER = (
    "我喜欢", "我偏好", "我习惯", "用户偏好", "用户画像", "偏好",
    "用户常驻", "我的主力", "我的", "我去", "我不能", "我必须",
    "i like", "i prefer", "i went",
)
_ENVIRONMENT = (
    "nas", "unraid", "truenas", "zfs", "wireguard", "routeros", "vpn",
    "服务器", "主机", "本机", "硬件", "cpu", "gpu", "内存", "硬盘", "存储",
    "网络", "路由", "主板", "python 3.", "windows", "linux", "容器",
)
_PROJECT = (
    "项目", "project", "dsh-memory-governed", "workbuddy2api-hub",
    "opencode", "zcode", "octop", "仓库", "分支", "版本",
)
_WORKFLOW = (
    "流程", "步骤", "约定", "规则", "不能", "必须", "授权", "审批",
    "workflow", "policy", "注意", "禁止",
)
_REFERENCE = (
    "规格", "标准", "文档", "参考", "api", "协议", "端点", "型号",
    "兼容", "版本号", "reference",
)
_TOOLING = (
    "配置", "安装", "升级", "环境变量", "oauth", "token", "provider",
    "model", "embedding", "rerank", "docker", "watchtower", "trim-cli",
    "browser-skill", "mimo", "workbuddy", "codex", "claude", "命令",
    "工具", "客户端", "mcp__", "路径", "python",
)

_LEGACY = {
    "tech": "tooling",
    "work": "project",
    "life": "user",
    "ops-rule": "workflow",
    "dsh-setup": "tooling",
    "hgm-ops": "meta",
    "hgm-architecture": "meta",
    "ops-network": "environment",
    "ops-setup": "environment",
    "hgm-setup": "meta",
    "开源项目评估": "reference",
}


def _has_any(text: str, needles: tuple[str, ...]) -> bool:
    return any(needle in text for needle in needles)


def normalize_legacy_category(value: str) -> str:
    raw = str(value or "").strip().lower()
    return normalize_category(raw) or _LEGACY.get(raw, "")


def classify_fact_category(text: str) -> str:
    """Return one controlled category, or ``other`` when evidence is weak."""
    lower = str(text or "").lower()
    if _has_any(lower, _META):
        return "meta"
    if _has_any(lower, _USER):
        return "user"
    if _has_any(lower, _ENVIRONMENT):
        return "environment"
    if _has_any(lower, _TOOLING):
        return "tooling"
    if _has_any(lower, _PROJECT):
        return "project"
    if _has_any(lower, _WORKFLOW):
        return "workflow"
    if _has_any(lower, _REFERENCE):
        return "reference"
    return "other"
