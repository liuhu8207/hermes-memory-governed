#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Dashboard generator - create HTML recap dashboard.

Features:
- Weekly knowledge dashboard
- Meeting summary view
- Knowledge growth stats
"""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from hermes_env import bootstrap, env_path, setup_logging

logger = setup_logging("dashboard_gen")

HERMES_HOME = bootstrap(__file__)
WIKI_DIR = env_path("WIKI_DIR", HERMES_HOME / "wiki")
KNOWLEDGE_DIR = WIKI_DIR / "knowledge"
DASHBOARD_DIR = HERMES_HOME / "dashboard"


def count_notes_by_type() -> dict[str, int]:
    counts: dict[str, int] = {}
    if not KNOWLEDGE_DIR.exists():
        return counts
    for subdir in KNOWLEDGE_DIR.iterdir():
        if subdir.is_dir() and not subdir.name.startswith("."):
            md_count = len(list(subdir.rglob("*.md")))
            if md_count > 0:
                counts[subdir.name] = md_count
    return counts


def count_notes_by_category() -> dict[str, int]:
    counts: dict[str, int] = {}
    if not KNOWLEDGE_DIR.exists():
        return counts
    for md_file in KNOWLEDGE_DIR.rglob("*.md"):
        if md_file.name.startswith("."):
            continue
        try:
            content = md_file.read_text(encoding="utf-8", errors="replace")
            for line in content.split("\n"):
                if line.startswith("category:"):
                    category = line.split(":", 1)[1].strip()
                    counts[category] = counts.get(category, 0) + 1
                    break
        except Exception:
            pass
    return counts


def get_recent_meetings(days: int = 7) -> list[dict[str, Any]]:
    meetings_dir = KNOWLEDGE_DIR / "meetings"
    if not meetings_dir.exists():
        return []

    cutoff = datetime.now() - timedelta(days=days)
    meetings = []
    for md_file in meetings_dir.rglob("*.md"):
        try:
            content = md_file.read_text(encoding="utf-8", errors="replace")
            title = ""
            date = ""
            summary = ""
            for line in content.split("\n"):
                if line.startswith("title:"):
                    title = line.split(":", 1)[1].strip()
                elif line.startswith("date:"):
                    date = line.split(":", 1)[1].strip()
                elif line.startswith("## 摘要"):
                    summary = content.split("## 摘要")[1].split("##")[0].strip()[:200]
                    break
            # `days` 曾形同虚设（cutoff 计算后从未使用），导致"近期会议"其实
            # 返回了全部历史。这里按 frontmatter 的 `date`（YYYY-MM-DD）过滤；
            # 无法解析日期的条目保守保留（宁可多显示也不误删）。
            if date:
                try:
                    if datetime.strptime(date, "%Y-%m-%d") < cutoff:
                        continue
                except ValueError:
                    pass
            meetings.append({"title": title, "date": date, "summary": summary})
        except Exception:
            pass
    return sorted(meetings, key=lambda x: x.get("date", ""), reverse=True)[:10]


def generate_html_dashboard() -> str:
    type_counts = count_notes_by_type()
    category_counts = count_notes_by_category()
    meetings = get_recent_meetings()
    now = datetime.now().strftime("%Y-%m-%d %H:%M")

    type_rows = "\n".join(
        f"<tr><td>{k}</td><td>{v}</td></tr>"
        for k, v in sorted(type_counts.items(), key=lambda x: x[1], reverse=True)
    )
    category_rows = "\n".join(
        f"<tr><td>{k}</td><td>{v}</td></tr>"
        for k, v in sorted(category_counts.items(), key=lambda x: x[1], reverse=True)
    )
    meeting_rows = "\n".join(
        f"<tr><td>{m.get('date', '')}</td><td>{m.get('title', '')}</td><td>{m.get('summary', '')[:100]}</td></tr>"
        for m in meetings
    )
    total_notes = sum(type_counts.values())

    return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>知识库看板</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background: #f5f5f5; padding: 20px; }}
        .container {{ max-width: 1200px; margin: 0 auto; }}
        h1 {{ color: #333; margin-bottom: 20px; }}
        .stats {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px; margin-bottom: 30px; }}
        .stat-card {{ background: white; border-radius: 12px; padding: 20px; box-shadow: 0 2px 8px rgba(0,0,0,0.1); }}
        .stat-card h3 {{ color: #666; font-size: 14px; margin-bottom: 8px; }}
        .stat-card .value {{ font-size: 32px; font-weight: bold; color: #333; }}
        .section {{ background: white; border-radius: 12px; padding: 20px; margin-bottom: 20px; box-shadow: 0 2px 8px rgba(0,0,0,0.1); }}
        .section h2 {{ color: #333; margin-bottom: 15px; font-size: 18px; }}
        table {{ width: 100%; border-collapse: collapse; }}
        th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #eee; }}
        th {{ background: #f9f9f9; font-weight: 600; }}
        tr:hover {{ background: #f5f5f5; }}
        .footer {{ text-align: center; color: #999; margin-top: 30px; font-size: 12px; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>知识库看板</h1>

        <div class="stats">
            <div class="stat-card">
                <h3>总笔记数</h3>
                <div class="value">{total_notes}</div>
            </div>
            <div class="stat-card">
                <h3>分类数</h3>
                <div class="value">{len(category_counts)}</div>
            </div>
            <div class="stat-card">
                <h3>近期会议</h3>
                <div class="value">{len(meetings)}</div>
            </div>
        </div>

        <div class="section">
            <h2>按类型统计</h2>
            <table>
                <tr><th>类型</th><th>数量</th></tr>
                {type_rows}
            </table>
        </div>

        <div class="section">
            <h2>按分类统计</h2>
            <table>
                <tr><th>分类</th><th>数量</th></tr>
                {category_rows}
            </table>
        </div>

        <div class="section">
            <h2>近期会议</h2>
            <table>
                <tr><th>日期</th><th>标题</th><th>摘要</th></tr>
                {meeting_rows}
            </table>
        </div>

        <div class="footer">
            生成时间: {now}
        </div>
    </div>
</body>
</html>"""


def generate_dashboard() -> Path:
    DASHBOARD_DIR.mkdir(parents=True, exist_ok=True)
    html = generate_html_dashboard()
    output_path = DASHBOARD_DIR / "dashboard.html"
    output_path.write_text(html, encoding="utf-8")
    logger.info("generated dashboard: %s", output_path)
    return output_path


def main() -> int:
    import argparse
    parser = argparse.ArgumentParser(description="Dashboard generator")
    parser.add_argument("--open", action="store_true", help="Open dashboard in browser")
    args = parser.parse_args()

    output_path = generate_dashboard()
    print(f"Dashboard: {output_path}")

    if args.open:
        import webbrowser
        webbrowser.open(f"file://{output_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
