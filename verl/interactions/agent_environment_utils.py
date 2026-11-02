# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Shared helpers for text-based long-horizon environment interactions."""

from __future__ import annotations

import json
import re
from typing import Any


def latest_assistant_text(messages: list[dict[str, Any]]) -> str:
    """Return the most recent plain-text assistant message."""
    for message in reversed(messages):
        if message.get("role") != "assistant":
            continue
        content = message.get("content", "")
        if isinstance(content, str):
            return content.strip()
        return json.dumps(content, ensure_ascii=False)
    return ""


def extract_action(response: str) -> str:
    """Extract ``ACTION: ...`` while accepting an exact one-line fallback."""
    text = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL | re.IGNORECASE).strip()
    matches = re.findall(r"(?:^|\n)\s*ACTION\s*:\s*([^\n]+)", text, flags=re.IGNORECASE)
    if matches:
        return matches[-1].strip().strip("`")
    nonempty = [line.strip().strip("`") for line in text.splitlines() if line.strip()]
    return nonempty[-1] if len(nonempty) == 1 else ""


def match_admissible_action(action: str, admissible_actions: list[str]) -> str | None:
    """Return the environment's canonical spelling for an exact action."""
    normalized = " ".join(action.lower().split())
    for candidate in admissible_actions:
        if normalized == " ".join(candidate.lower().split()):
            return candidate
    return None


def normalize_rubric(value: Any) -> str:
    """Normalize parquet/JSON rubric representations to compact guidance text."""
    if value is None:
        return ""
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return ""
        try:
            decoded = json.loads(stripped)
        except json.JSONDecodeError:
            return stripped
        return normalize_rubric(decoded)
    if isinstance(value, dict):
        for key in ("rubrics", "criteria", "items"):
            if key in value:
                return normalize_rubric(value[key])
        return "\n".join(f"- {key}: {item}" for key, item in value.items())
    if isinstance(value, (list, tuple)):
        lines = []
        for item in value:
            if isinstance(item, dict):
                text = item.get("criterion") or item.get("rubric") or item.get("text") or str(item)
            else:
                text = str(item)
            text = " ".join(text.split())
            if text:
                lines.append(f"- {text}")
        return "\n".join(lines)
    return str(value)


def render_environment_observation(
    *,
    benchmark: str,
    observation: str,
    admissible_actions: list[str],
    rubric: str = "",
    search_available: bool = False,
) -> str:
    """Render a consistent observation/action contract for the policy."""
    if search_available:
        actions = ["search[<keywords>]"] + admissible_actions
    else:
        actions = admissible_actions
    action_text = "\n".join(f"- {action}" for action in actions) or "- (no valid action)"
    sections = [
        f"[{benchmark} observation]",
        observation.strip(),
        "[currently valid actions]",
        action_text,
    ]
    if rubric:
        sections.extend(
            [
                "[task-specific decision rubric]",
                rubric,
                (
                    "Apply a rubric item only when its IF condition matches the current observation and its quoted "
                    "action appears verbatim in [currently valid actions]. Otherwise ignore that item. The current "
                    "valid-action list and environment reward are authoritative."
                ),
            ]
        )
    sections.append("Reply with exactly one line: ACTION: <one currently valid action>")
    return "\n\n".join(sections)
