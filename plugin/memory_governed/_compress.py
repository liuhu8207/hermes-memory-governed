"""Mermaid short-term compression for long tasks.

When context is about to be compressed, extract key decisions and
tool outputs into a compact Mermaid task canvas instead of keeping
the full verbose logs.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List

from ._config import GovernedMemoryConfig

logger = logging.getLogger(__name__)


class MermaidCompressor:
    """Compress long tool outputs and conversation history into
    a compact Mermaid task canvas.

    This is used in on_pre_compress() to preserve key information
    from messages that are about to be discarded by context compression.
    """

    def __init__(self, config: GovernedMemoryConfig):
        self._config = config
        self._max_tokens = config.mermaid_compress.canvas_max_tokens

    def compress(self, messages: List[Dict[str, Any]]) -> str:
        """Compress messages into a Mermaid task canvas.

        Returns a compact text summary that fits within the token budget.
        """
        if not self._config.mermaid_compress.enabled:
            return ""

        if not messages:
            return ""

        # Extract key events from messages
        events = self._extract_events(messages)
        if not events:
            return ""

        # Build Mermaid flowchart
        canvas = self._build_canvas(events)

        # Truncate to token budget
        max_chars = self._max_tokens * 4
        if len(canvas) > max_chars:
            canvas = canvas[:max_chars] + "\n..."

        return canvas

    def _extract_events(self, messages: List[Dict[str, Any]]) -> List[dict]:
        """Extract key events from messages: tool calls, decisions, errors."""
        events = []
        for msg in messages:
            role = msg.get("role", "")
            content = msg.get("content", "")

            if role == "assistant" and content:
                # Look for decision patterns
                if self._looks_like_decision(content):
                    events.append({
                        "type": "decision",
                        "text": self._truncate(content, 120),
                    })

            elif role == "tool":
                # Tool output: extract key info
                tool_name = msg.get("name", "tool")
                summary = self._summarize_tool_output(content, tool_name)
                if summary:
                    events.append({
                        "type": "tool",
                        "tool": tool_name,
                        "text": summary,
                    })

            elif role == "user" and content:
                # User message: extract intent
                if len(content) > 20:  # Skip trivial
                    events.append({
                        "type": "user",
                        "text": self._truncate(content, 100),
                    })

        return events[-20:]  # Keep last 20 events

    def _build_canvas(self, events: List[dict]) -> str:
        """Build a Mermaid flowchart from events."""
        lines = ["graph TD"]

        for i, event in enumerate(events):
            node_id = f"N{i}"
            etype = event["type"]
            text = event["text"].replace('"', "'")

            if etype == "decision":
                lines.append(f'    {node_id}{{"{text}"}}')
            elif etype == "tool":
                tool = event.get("tool", "tool")
                lines.append(f'    {node_id}[/"{tool}: {text}"/]')
            elif etype == "user":
                lines.append(f'    {node_id}("{text}")')
            else:
                lines.append(f'    {node_id}["{text}"]')

            if i > 0:
                lines.append(f"    N{i-1} --> {node_id}")

        return "\n".join(lines)

    @staticmethod
    def _looks_like_decision(content: str) -> bool:
        """Check if content looks like a decision or conclusion."""
        patterns = [
            r"decided to",
            r"chose to",
            r"will use",
            r"plan is to",
            r"the fix is",
            r"root cause",
            r"决定",
            r"方案是",
            r"原因是",
        ]
        content_lower = content.lower()
        return any(re.search(p, content_lower) for p in patterns)

    @staticmethod
    def _summarize_tool_output(content: str, tool_name: str) -> str:
        """Summarize a tool output into a short line."""
        if not content:
            return ""
        # Take first meaningful line
        lines = content.strip().split("\n")
        for line in lines:
            line = line.strip()
            if line and len(line) > 5:
                return line[:120]
        return ""

    @staticmethod
    def _truncate(text: str, limit: int) -> str:
        if len(text) <= limit:
            return text
        return text[:limit - 3] + "..."
