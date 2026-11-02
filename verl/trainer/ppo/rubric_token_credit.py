# Copyright 2026
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
"""Rubric generation and privileged teacher-input construction for token credit."""

from __future__ import annotations

import json
import os
import random
import re
import time
import urllib.error
import urllib.request
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
from typing import Any

import torch

from verl.experimental.agent_loop.turn_context import TURN_REPLAY_KEY, validate_trajectory_record

PROBLEM_RUBRIC_SYSTEM = """\
You are a rubric design expert. Given a math problem, generate exactly {k} concise rubric items that describe the
critical reasoning steps and key decision points needed to solve it correctly.

Requirements:
- Focus on decision points where a solver could go wrong: algebraic manipulations, substitutions, sign choices,
  constraints, or case analysis.
- Each item must be a short, verifiable statement about one specific reasoning step.
- State what the correct step is; avoid generic advice.
- Do not reveal or hint at the final numerical answer.
- Keep the items non-redundant.

Return JSON only: {{"rubrics": ["rubric 1", "rubric 2", ...]}}.
"""

PROBLEM_RUBRIC_USER = """\
Problem:
{problem}

Generate exactly {k} rubric items describing the key steps needed to solve this problem.
"""

AGENT_RUBRIC_PROMPT_VERSION = "agent_generic_v1"
ALFWORLD_RUBRIC_PROMPT_VERSION = "alfworld_state_criteria_v6"
RUBRIC_FILTER_VERSION = "mixed_outcome_item_matrix_v1"
TASK_GROUP_HINT_PROMPT_VERSION = "task_group_process_rubric_v2"

# V1 is the exact prompt used by the already successful ALFWorld experiment.
# Keep it intact so the experiment can be reproduced or restored without
# recovering prompt text from Git history.
TASK_GROUP_HINT_SYSTEM_V1 = """\
You are a task coach for an interactive environment.

You will see one task and several previous attempts at the same task.
Use them to write useful hints for a NEW agent that has never seen those attempts.

Write 3 to {k} short hints.

Useful hints can describe:
- important subgoals needed to complete the task,
- useful ordering of subgoals,
- mistakes or no-progress behavior to avoid,
- what must be true before the task is finished.

Rules:
- Use only information supported by the task or the attempts.
- Do not mention attempts, trajectories, rollouts, rewards, sources, evidence, or IDs.
- Do not copy a complete action sequence.
- Do not invent object locations or hidden state.
- Keep each hint short, concrete, and useful by itself.
- Do not explain your reasoning.

Output only numbered hints:
1. ...
2. ...
3. ...
"""

TASK_GROUP_HINT_USER_V1 = """\
Task:
{problem}

Previous attempts:
{attempts}

Write useful hints for a new agent solving this same task.
"""

# Active process-rubric wording restored for the new whole-bundle comparison.
# It uses the original short numbered format, without demonstrations or JSON,
# while enforcing the current 5--8 criterion contract.
TASK_GROUP_HINT_SYSTEM_V2 = """\
You are a task coach who writes process rubrics for an interactive environment.

You will see one task and several previous attempts at the same task.
Use them to write useful process criteria for a NEW agent that has never seen those attempts.

Write 5 to {k} short rubric criteria.

Useful rubric criteria can describe:
- important subgoals needed to complete the task,
- useful ordering of subgoals,
- mistakes or no-progress behavior to avoid,
- what must be true before the task is finished.

Rules:
- Use only information supported by the task or the attempts.
- Do not mention attempts, trajectories, rollouts, rewards, sources, evidence, or IDs.
- Do not copy a complete action sequence.
- Do not invent object locations or hidden state.
- Keep each rubric criterion short, concrete, and useful by itself.
- Do not explain your reasoning.

Output only a numbered process rubric:
1. ...
2. ...
3. ...
4. ...
5. ...
"""

# Exact process-rubric wording used by the archived WebShop-small milestone.
TASK_GROUP_HINT_SYSTEM_LEGACY_V2 = """\
You are a task coach who writes process rubrics for an interactive environment.

You will see one task and several previous attempts at the same task.
Use them to write useful process criteria for a NEW agent that has never seen those attempts.

Write 3 to {k} short rubric criteria.

Useful rubric criteria can describe:
- important subgoals needed to complete the task,
- useful ordering of subgoals,
- mistakes or no-progress behavior to avoid,
- what must be true before the task is finished.

Rules:
- Use only information supported by the task or the attempts.
- Do not mention attempts, trajectories, rollouts, rewards, sources, evidence, or IDs.
- Do not copy a complete action sequence.
- Do not invent object locations or hidden state.
- Keep each rubric criterion short, concrete, and useful by itself.
- Do not explain your reasoning.

Output only a numbered process rubric:
1. ...
2. ...
3. ...
"""

TASK_GROUP_HINT_USER_V2 = """\
Task:
{problem}

Previous attempts:
{attempts}

Write a useful process rubric for a new agent solving this same task.
"""

# The group-hint route uses this prompt directly; it does not append few-shot
# examples from the later rubric-pool generator.
TASK_GROUP_HINT_SYSTEM = TASK_GROUP_HINT_SYSTEM_V2
TASK_GROUP_HINT_USER = TASK_GROUP_HINT_USER_V2

AGENT_RUBRIC_SYSTEM = """\
You are the current policy reflecting on its own long-horizon environment
rollouts. Extract exactly {k} concise, typed action-selection rubrics from the
task, policy-visible observations and valid actions, trajectory transitions,
and terminal environment observations below.

Use only these rubric types:
- <SUGGEST>: a state-conditioned action rule supported by a visible, non-invalid
  trajectory transition;
- <PITFALL>: a concrete bad action, invalid repetition, or ordering mistake
  demonstrated by a failed or lower-reward transition;
- <RECOVER>: an action rule for escaping an observed failed or looping state;
- <VERIFY>: a visible check that establishes whether a subgoal or the full task
  has actually been completed;
- <BONUS>: a distinct alternative strategy, allowed only when another
  successful trajectory directly demonstrates it.

Grounding requirements:
- Every item must cite one supplied evidence_id as "SOURCE <evidence_id>".
- Use only state and actions visible to the acting policy. Never invent an
  object location, hidden identifier, oracle plan, or unobserved success.
- Never say an action worked unless its next visible observation supports that
  claim.
- Keep each item atomic and reusable. Do not copy a complete action sequence.
- For mixed outcomes, include at least one <SUGGEST> and one <PITFALL>.
- If every rollout failed, emit only <PITFALL> and <RECOVER> items; do not
  fabricate a successful route.
- If every rollout succeeded, do not invent a <PITFALL>; use <SUGGEST> and
  evidence-backed <VERIFY> items instead.

For ALFWorld, respect the observable task stages: search -> acquire -> optional
transform -> deliver/examine. A useful item names the visible condition, one
action-selection rule, and the expected visible transition. A <PITFALL> names
the demonstrated failure and what to avoid; it need not invent a positive next
step.

Return JSON only, with each rubric stored as one tagged string:
{{"rubrics": [
  "<SUGGEST> IF ...; THEN ...; EXPECT ...; SOURCE rollout_1",
  "<PITFALL> IF ...; AVOID ...; BECAUSE ...; SOURCE rollout_4"
]}}
The array must contain exactly {k} distinct strings.
"""

ALFWORLD_AGENT_RUBRIC_SYSTEM = """\
You are an ALFWorld trajectory critic. Read the task and ONE rollout. Extract
observable STATE criteria, not a fixed action script. These criteria will be
scored against other rollouts of the same task.

Output 2 to {k} lines and nothing else. Use only these three line shapes:
<MILESTONE> PASS WHEN <one required subgoal is visibly achieved>; SOURCE <id>
<AVOID> FAIL WHEN <one observed invalid action or no-progress loop occurs>; SOURCE <id>
<VERIFY> PASS WHEN <the complete task state is visibly achieved>; SOURCE <id>

Rules:
1. Describe a world state, inventory state, or repeated behavior. Do not give
   ACTION commands and do not prescribe an exact route.
2. Use task entities and facts visible in the supplied rollout only. Never use
   hidden PDDL, a reference plan, an outcome label, or an invented object id.
3. A milestone must preserve progress: never advise dropping, undoing, or
   moving an acquired/transformed object back to an earlier location.
4. An AVOID line is allowed only for an invalid action or repeated transition
   actually visible in the supplied rollout.
5. VERIFY must express the whole task, not a partial step. For examine-in-light
   tasks, holding the target and turning on the lamp are not sufficient unless
   the target is examined. For two-object tasks, both objects must be placed.
6. Every line cites the supplied evidence id. Do not repeat or contradict a
   criterion. If only two criteria are supported, output two; never invent
   extra rules to fill the quota.
"""

AGENT_RUBRIC_JUDGE_SYSTEM = """\
You are a trajectory-rubric judge. Score EACH candidate rubric against the
single supplied on-policy trajectory. You are judging whether the visible
trajectory follows or supports the action-selection guidance; you are NOT
judging the final answer and you are never given a correctness label.

For every rubric return one score:
- 1.0: the visible trajectory clearly follows/supports the rubric;
- 0.0: the visible trajectory clearly violates/contradicts the rubric;
- 0.5: the rubric is not applicable or the trajectory lacks enough visible
  evidence to decide.

For <MILESTONE> and <VERIFY>, 1.0 means the stated state is visibly achieved,
0.0 means the trajectory visibly ends without achieving it, and 0.5 means the
evidence is genuinely insufficient. For <AVOID>, 1.0 means the named behavior
does not occur and 0.0 means it occurs. Use only the supplied observations and
actions. Preserve the candidate order and return JSON only:
{{"scores": [0.0, 1.0, 0.5]}}
The score array must contain exactly {m} numbers.
"""

AGENT_RUBRIC_JUDGE_USER = """\
Task shown to the agent:
{problem}

Candidate rubrics:
{rubrics}

One on-policy trajectory:
{trajectory}

Return exactly {m} scores in candidate order.
"""

AGENT_RUBRIC_USER = """\
Benchmark:
{benchmark}

Task shown to the agent:
{problem}

Compact on-policy trajectory evidence:
{trajectories}

Allowed evidence ids for THIS task only:
{evidence_ids}

Generate no more than {k} distinct action-selection criteria. Prefer grounded
coverage of the task's decision stages over filling the quota.
"""

ALFWORLD_AGENT_RUBRIC_USER = """\
Task:
{problem}

Evidence id: {evidence_id}
One rollout:
{trajectory}

Write 2 to {k} short state criteria using only that evidence id.
"""

_THINK_RE = re.compile(r"<think>.*?</think>", flags=re.DOTALL | re.IGNORECASE)
_THINK_OPEN_RE = re.compile(r"<think>", flags=re.IGNORECASE)
_THINK_CLOSE_RE = re.compile(r"</think>", flags=re.IGNORECASE)
_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", flags=re.DOTALL | re.IGNORECASE)


def strip_thinking(text: str) -> str:
    """Remove Qwen thinking traces while preserving a following rubric JSON object."""
    cleaned = _THINK_RE.sub("", str(text or ""))
    # Qwen chat templates may place the opening <think> token in the prompt,
    # so generated text can contain reasoning followed only by </think>.
    # In that case everything through the final unmatched closing tag is
    # private reasoning and must not enter the parsed rubric.
    unmatched_closes = list(_THINK_CLOSE_RE.finditer(cleaned))
    if unmatched_closes:
        cleaned = cleaned[unmatched_closes[-1].end() :]
    while match := _THINK_OPEN_RE.search(cleaned):
        tail = cleaned[match.end() :]
        rubric_json = re.search(r'(?:```(?:json)?\s*)?\{\s*"(?:rubrics|criteria)"\s*:', tail, re.IGNORECASE)
        if rubric_json:
            cleaned = f"{cleaned[: match.start()]}{tail[rubric_json.start() :]}"
        else:
            cleaned = cleaned[: match.start()]
    return _THINK_CLOSE_RE.sub("", cleaned).strip()


def _get(config: Any, key: str, default: Any = None) -> Any:
    if hasattr(config, "get"):
        return config.get(key, default)
    return getattr(config, key, default)


def normalize_messages(raw_prompt: Any) -> list[dict[str, str]]:
    """Normalize VERL's object-array prompt representation to chat messages."""
    if hasattr(raw_prompt, "tolist") and not isinstance(raw_prompt, (str, bytes, dict, list, tuple)):
        raw_prompt = raw_prompt.tolist()
    if isinstance(raw_prompt, str):
        return [{"role": "user", "content": raw_prompt}]
    if isinstance(raw_prompt, dict):
        if "messages" in raw_prompt:
            raw_prompt = raw_prompt["messages"]
        elif "role" in raw_prompt and "content" in raw_prompt:
            raw_prompt = [raw_prompt]
        else:
            return [{"role": "user", "content": json.dumps(raw_prompt, ensure_ascii=False)}]
    if isinstance(raw_prompt, tuple):
        raw_prompt = list(raw_prompt)
    if not isinstance(raw_prompt, list):
        return [{"role": "user", "content": str(raw_prompt)}]

    messages = []
    for message in raw_prompt:
        if isinstance(message, dict):
            role = str(message.get("role", "user"))
            content = message.get("content", "")
            if isinstance(content, list):
                text_parts = [
                    str(part.get("text", ""))
                    for part in content
                    if isinstance(part, dict) and part.get("type") == "text"
                ]
                content = "\n".join(part for part in text_parts if part)
            messages.append({"role": role, "content": str(content)})
        else:
            messages.append({"role": "user", "content": str(message)})
    return messages or [{"role": "user", "content": ""}]


def extract_problem_text(raw_prompt: Any) -> str:
    """Extract the final user message used as the problem-only rubric input."""
    messages = normalize_messages(raw_prompt)
    for message in reversed(messages):
        if message["role"] == "user":
            return message["content"].strip()
    return messages[-1]["content"].strip()


def build_rubric_generation_messages(problem: str, rubrics_per_prompt: int) -> list[dict[str, str]]:
    """Build the problem-only messages shared by online and offline rubric generation."""
    return [
        {
            "role": "system",
            "content": PROBLEM_RUBRIC_SYSTEM.format(k=rubrics_per_prompt),
        },
        {
            "role": "user",
            "content": PROBLEM_RUBRIC_USER.format(problem=problem, k=rubrics_per_prompt),
        },
    ]


def build_agent_rubric_generation_messages(
    problem: str,
    trajectory_evidence: list[dict[str, Any]],
    rubrics_per_prompt: int,
    benchmark: str = "",
) -> list[dict[str, str]]:
    """Build an online self-rubric prompt from the current actor's rollout group."""
    normalized_benchmark = str(benchmark or "").strip().lower()
    is_alfworld = "alfworld" in normalized_benchmark
    system_prompt = ALFWORLD_AGENT_RUBRIC_SYSTEM if is_alfworld else AGENT_RUBRIC_SYSTEM
    evidence_ids = [str(item.get("evidence_id", "")) for item in trajectory_evidence if item.get("evidence_id")]
    visible_evidence = [
        {
            "evidence_id": str(item.get("evidence_id", "")),
            "transcript": str(item.get("transcript", "")),
        }
        for item in trajectory_evidence
        if item.get("evidence_id")
    ]
    messages = [
        {
            "role": "system",
            "content": system_prompt.format(k=rubrics_per_prompt),
        },
    ]
    if is_alfworld and visible_evidence:
        user_content = ALFWORLD_AGENT_RUBRIC_USER.format(
            problem=problem,
            evidence_id=visible_evidence[0]["evidence_id"],
            trajectory=visible_evidence[0]["transcript"],
            k=rubrics_per_prompt,
        )
    else:
        user_content = AGENT_RUBRIC_USER.format(
            benchmark=benchmark or "generic long-horizon agent",
            problem=problem,
            trajectories=json.dumps(visible_evidence, ensure_ascii=False, indent=2),
            evidence_ids=", ".join(evidence_ids) or "(none)",
            k=rubrics_per_prompt,
        )
    messages.append({"role": "user", "content": user_content})
    return messages


def compact_online_trajectory(text: str, *, max_chars: int) -> str:
    """Bound an online trajectory while retaining its beginning and terminal evidence."""
    cleaned = strip_thinking(text)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()
    if len(cleaned) <= max_chars:
        return cleaned
    head_budget = max_chars * 2 // 3
    tail_budget = max_chars - head_budget
    return f"{cleaned[:head_budget]}\n...[middle omitted]...\n{cleaned[-tail_budget:]}"


def build_task_group_hint_messages(
    problem: str,
    trajectory_evidence: list[dict[str, Any]],
    *,
    max_hints: int,
    max_trajectories: int,
    max_trajectory_chars: int | None,
    correctness_threshold: float = 0.5,
    prompt_style: str = "process_rubric_5_to_k",
) -> list[dict[str, str]]:
    """Build the successful task-group hint prompt used by ``agent_rubric2``.

    Every selected on-policy attempt is shown in one request. Outcome labels
    are privileged generation-time evidence only; the resulting actor-visible
    hints are required not to mention them.
    """
    if max_hints < 1:
        raise ValueError("max_hints must be at least 1")
    if max_trajectories < 1:
        raise ValueError("max_trajectories must be at least 1")
    prompt_systems = {
        "process_rubric_5_to_k": TASK_GROUP_HINT_SYSTEM_V2,
        "legacy_process_rubric_3_to_k": TASK_GROUP_HINT_SYSTEM_LEGACY_V2,
    }
    if prompt_style not in prompt_systems:
        raise ValueError(f"Unsupported task-group hint prompt style: {prompt_style!r}")

    attempts = []
    for attempt_index, evidence in enumerate(trajectory_evidence[:max_trajectories], start=1):
        transcript = strip_thinking(str(evidence.get("transcript", "")))
        transcript = re.sub(r"\n{3,}", "\n\n", transcript).strip()
        if max_trajectory_chars is not None:
            transcript = compact_online_trajectory(transcript, max_chars=max_trajectory_chars)
        # Callers that have benchmark-native task success should pass it
        # explicitly.  The threshold fallback preserves compatibility with
        # binary math rewards and older cached evidence.
        if "success" in evidence:
            result = "success" if bool(evidence["success"]) else "failure"
        else:
            outcome = float(evidence.get("outcome", 0.0))
            result = "success" if outcome > correctness_threshold else "failure"
        attempts.append(f"Attempt {attempt_index}:\n{transcript}\nResult: {result}")

    return [
        {"role": "system", "content": prompt_systems[prompt_style].format(k=max_hints)},
        {
            "role": "user",
            "content": TASK_GROUP_HINT_USER.format(
                problem=str(problem).strip(),
                attempts="\n\n".join(attempts),
            ),
        },
    ]


def order_task_group_trajectory_pairs(
    pairs,
    *,
    seed: str,
    mode: str = "outcome_anchored_random",
):
    """Order evidence using the active route or the archived milestone rule."""
    pairs = list(pairs)
    if not pairs:
        return []
    if mode == "legacy_alternating_extremes":
        ranked = sorted(pairs, key=lambda pair: (pair[1], pair[0]))
        ordered = []
        left, right = 0, len(ranked) - 1
        while left <= right:
            ordered.append(ranked[left])
            left += 1
            if left <= right:
                ordered.append(ranked[right])
                right -= 1
        return ordered
    if mode != "outcome_anchored_random":
        raise ValueError(f"Unsupported task-group evidence order: {mode!r}")
    lowest = min(pairs, key=lambda pair: (pair[1], pair[0]))
    highest = max(pairs, key=lambda pair: (pair[1], pair[0]))
    ordered = [highest]
    if lowest[0] != highest[0]:
        ordered.append(lowest)
    anchored = {pair[0] for pair in ordered}
    remaining = [pair for pair in pairs if pair[0] not in anchored]
    random.Random(seed).shuffle(remaining)
    return ordered + remaining


def fit_task_group_hint_messages(
    problem: str,
    trajectory_evidence: list[dict[str, Any]],
    tokenizer,
    *,
    max_hints: int,
    max_trajectories: int,
    max_trajectory_chars: int,
    max_input_tokens: int,
    correctness_threshold: float = 0.5,
    enable_thinking: bool = False,
    prompt_style: str = "process_rubric_5_to_k",
) -> tuple[list[dict[str, str]] | None, dict[str, Any]]:
    """Fit an aggregate group-hint prompt to the model context.

    The caller orders trajectories as highest reward, lowest reward, then a
    shuffled remainder.  Each trajectory first uses the historical mechanical
    6000-character-style head/tail cap.  If the resulting aggregate remains
    too long, only the end of the user message is removed and an explicit
    truncation notice is appended.  No model-generated summary is introduced.
    """
    if max_input_tokens < 1:
        raise ValueError("max_input_tokens must be positive")
    if max_trajectory_chars < 1:
        raise ValueError("max_trajectory_chars must be positive")

    def token_count(messages):
        token_ids = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            enable_thinking=enable_thinking,
        )
        if hasattr(token_ids, "shape"):
            token_count = int(token_ids.shape[-1])
        elif token_ids and isinstance(token_ids[0], (list, tuple)):
            token_count = len(token_ids[0])
        else:
            return len(token_ids)
        return token_count

    messages = build_task_group_hint_messages(
        problem,
        trajectory_evidence,
        max_hints=max_hints,
        max_trajectories=max_trajectories,
        max_trajectory_chars=max_trajectory_chars,
        correctness_threshold=correctness_threshold,
        prompt_style=prompt_style,
    )
    full_token_count = token_count(messages)
    if full_token_count <= max_input_tokens:
        return messages, {
            "prompt_tokens": full_token_count,
            "prompt_token_budget": max_input_tokens,
            "context_truncated": False,
            "context_overflow": False,
        }

    truncation_notice = (
        "\n\n[TRUNCATED: The later part of the aggregated trajectories was omitted "
        "because the context limit was reached.]"
    )
    original_user_content = messages[-1]["content"]

    def render_prefix(prefix_chars: int):
        candidate = [dict(message) for message in messages]
        candidate[-1]["content"] = original_user_content[:prefix_chars].rstrip() + truncation_notice
        return candidate, token_count(candidate)

    minimum_messages, minimum_tokens = render_prefix(0)
    if minimum_tokens > max_input_tokens:
        return None, {
            "prompt_tokens": minimum_tokens,
            "prompt_token_budget": max_input_tokens,
            "context_truncated": True,
            "context_overflow": True,
        }

    low, high = 0, len(original_user_content)
    best_messages, best_tokens, best_chars = minimum_messages, minimum_tokens, 0
    while low <= high:
        middle = (low + high) // 2
        candidate_messages, candidate_tokens = render_prefix(middle)
        if candidate_tokens <= max_input_tokens:
            best_messages, best_tokens, best_chars = candidate_messages, candidate_tokens, middle
            low = middle + 1
        else:
            high = middle - 1
    return best_messages, {
        "prompt_tokens": best_tokens,
        "prompt_token_budget": max_input_tokens,
        "retained_user_chars": best_chars,
        "original_user_chars": len(original_user_content),
        "context_truncated": True,
        "context_overflow": False,
    }


def parse_task_group_hint_response(
    raw_text: str,
    *,
    max_items: int,
    min_items: int = 1,
    max_item_chars: int = 280,
    max_rubric_chars: int = 1600,
) -> tuple[str, list[str]]:
    """Parse group hints while removing thinking and legacy rubric metadata."""
    cleaned = strip_thinking(raw_text)
    fence_match = _FENCE_RE.search(cleaned)
    if fence_match:
        cleaned = fence_match.group(1).strip()
    cleaned = re.sub(r"(?<!^)(?<!\n)\s+(?=\d+\s*[.)]\s+)", "\n", cleaned)

    raw_items: list[Any] = []
    candidates = [cleaned]
    for left, right in (("{", "}"), ("[", "]")):
        start, end = cleaned.find(left), cleaned.rfind(right)
        if 0 <= start < end:
            candidates.append(cleaned[start : end + 1])
    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except (json.JSONDecodeError, TypeError):
            continue
        if isinstance(parsed, dict):
            values = parsed.get("rubrics", parsed.get("hints", parsed.get("criteria", [])))
        elif isinstance(parsed, list):
            values = parsed
        else:
            values = []
        if isinstance(values, list):
            for item in values:
                if isinstance(item, str):
                    raw_items.append(item)
                elif isinstance(item, dict):
                    raw_items.append(item.get("text") or item.get("description") or item.get("criterion") or "")
            if raw_items:
                break
    if not raw_items:
        raw_items = cleaned.splitlines()

    items: list[str] = []
    seen: set[str] = set()
    for raw_item in raw_items:
        item = re.sub(r"^\s*(?:[-*•]|\d+\s*[.)])\s*", "", str(raw_item).strip()).strip()
        item = re.sub(
            r"^<(?:MILESTONE|AVOID|VERIFY|SUGGEST|PITFALL|RECOVER|BONUS)>\s*",
            "",
            item,
            flags=re.IGNORECASE,
        )
        item = re.sub(r"\s*;\s*SOURCE\s*:?\s*[\w.:-]+\s*$", "", item, flags=re.IGNORECASE)
        item = re.sub(r"^(?:PASS|FAIL)\s+WHEN\s+", "", item, flags=re.IGNORECASE)
        item = re.sub(r"\s+", " ", item).strip(" -\t")
        if len(item) < 8:
            continue
        item = item[:max_item_chars].rstrip()
        key = item.casefold()
        if key in seen:
            continue
        seen.add(key)
        items.append(item)
        if len(items) >= max_items:
            break

    if len(items) < min_items:
        return "", []
    # Never slice a numbered criterion in half: teacher replay and offline
    # snapshots consume the complete bundle, not a partially cut final item.
    while items:
        rubric = "\n".join(f"{index}. {item}" for index, item in enumerate(items, start=1))
        if len(rubric) <= max_rubric_chars:
            break
        items.pop()
    if len(items) < min_items:
        return "", []
    return rubric, items


def parse_rubric_response(raw_text: str, max_items: int, max_chars: int) -> str:
    """Parse vLLM output into a numbered rubric block suitable for a teacher prompt."""
    cleaned = strip_thinking(raw_text)
    fence_match = _FENCE_RE.search(cleaned)
    if fence_match:
        cleaned = fence_match.group(1).strip()

    candidates = [cleaned]
    for left, right in (("{", "}"), ("[", "]")):
        start, end = cleaned.find(left), cleaned.rfind(right)
        if 0 <= start < end:
            candidates.append(cleaned[start : end + 1])

    parsed: Any = None
    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
            break
        except (json.JSONDecodeError, TypeError):
            continue

    if isinstance(parsed, dict):
        items = parsed.get("rubrics", parsed.get("criteria", []))
    elif isinstance(parsed, list):
        items = parsed
    else:
        items = []

    descriptions = []
    for item in items:
        if isinstance(item, str):
            description = item
        elif isinstance(item, dict):
            description = item.get("description", item.get("criterion", item.get("text", "")))
            if not description:
                item_type = str(item.get("type", "")).strip().upper()
                pass_when = str(item.get("pass_when", "")).strip()
                fail_when = str(item.get("fail_when", "")).strip()
                source = str(item.get("source", item.get("evidence_id", ""))).strip()
                if item_type in {"MILESTONE", "VERIFY"} and pass_when and source:
                    description = f"<{item_type}> PASS WHEN {pass_when}; SOURCE {source}"
                elif item_type == "AVOID" and fail_when and source:
                    description = f"<AVOID> FAIL WHEN {fail_when}; SOURCE {source}"
        else:
            description = ""
        description = re.sub(r"\s+", " ", str(description)).strip()
        if description and description not in descriptions:
            descriptions.append(description)
        if len(descriptions) >= max_items:
            break

    if not descriptions:
        for line in cleaned.splitlines():
            description = re.sub(r"^\s*(?:[-*]|\d+[.)])\s*", "", line).strip()
            if description and len(description) >= 8:
                descriptions.append(re.sub(r"\s+", " ", description))
            if len(descriptions) >= max_items:
                break

    rubric_text = "\n".join(f"{idx + 1}. {description}" for idx, description in enumerate(descriptions))
    return rubric_text[:max_chars].strip()


def rubric_text_to_items(rubric_text: str) -> list[str]:
    """Split a numbered rubric block into canonical, non-empty items."""
    items: list[str] = []
    for raw_line in str(rubric_text or "").splitlines():
        item = re.sub(r"^\s*(?:[-*]|\d+[.)])\s*", "", raw_line).strip()
        item = re.sub(r"\s+", " ", item)
        if item and item not in items:
            items.append(item)
    return items


def format_rubric_items(items: list[str], *, max_chars: int | None = None) -> str:
    """Format rubric items for the privileged teacher prompt."""
    text = "\n".join(f"{index}. {item}" for index, item in enumerate(items, start=1))
    return text[:max_chars].strip() if max_chars is not None else text


def build_agent_rubric_judge_messages(
    problem: str,
    trajectory: str,
    rubric_items: list[str],
) -> list[dict[str, str]]:
    """Build one label-free rubric-satisfaction request for one rollout."""
    rendered_rubrics = format_rubric_items(rubric_items)
    return [
        {
            "role": "system",
            "content": AGENT_RUBRIC_JUDGE_SYSTEM.format(m=len(rubric_items)),
        },
        {
            "role": "user",
            "content": AGENT_RUBRIC_JUDGE_USER.format(
                problem=problem,
                rubrics=rendered_rubrics,
                trajectory=trajectory,
                m=len(rubric_items),
            ),
        },
    ]


def parse_rubric_judge_response(raw_text: str, expected_items: int) -> list[float] | None:
    """Parse an exact-length rubric satisfaction vector; fail closed on malformed output."""
    cleaned = strip_thinking(raw_text)
    fence_match = _FENCE_RE.search(cleaned)
    if fence_match:
        cleaned = fence_match.group(1).strip()
    candidates = [cleaned]
    start, end = cleaned.find("{"), cleaned.rfind("}")
    if 0 <= start < end:
        candidates.append(cleaned[start : end + 1])
    parsed: Any = None
    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
            break
        except (json.JSONDecodeError, TypeError):
            continue
    scores = parsed.get("scores") if isinstance(parsed, dict) else None
    if not isinstance(scores, list) or len(scores) != expected_items:
        return None
    normalized: list[float] = []
    for value in scores:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return None
        numeric = float(value)
        if not 0.0 <= numeric <= 1.0:
            return None
        normalized.append(numeric)
    return normalized


def filter_rubrics_by_outcome_correlation(
    rubric_items: list[str],
    score_matrix: list[list[float]],
    outcomes: list[float],
    *,
    reliability_threshold: float = 0.5,
    correctness_threshold: float = 0.5,
    min_score_range: float = 1e-6,
) -> tuple[list[str], dict[str, Any]]:
    """Select rubrics that separate correct from incorrect rollouts.

    ``score_matrix`` is item-major: M rubric items by N rollouts. Uniform
    outcome groups are deliberately ineligible because they contain no
    within-prompt credit-assignment signal.
    """
    labels = [float(value) > correctness_threshold for value in outcomes]
    stats: dict[str, Any] = {
        "eligible_group": False,
        "outcome_group": "empty",
        "candidate_items": len(rubric_items),
        "reliable_items": 0,
        "reliability_scores": [],
        "pearson_scores": [],
        "selected_indices": [],
        "constant_items": 0,
    }
    if not labels:
        return [], stats
    if all(labels):
        stats["outcome_group"] = "all_correct"
        return [], stats
    if not any(labels):
        stats["outcome_group"] = "all_wrong"
        return [], stats
    stats["outcome_group"] = "mixed"
    if len(score_matrix) != len(rubric_items) or any(len(row) != len(labels) for row in score_matrix):
        stats["matrix_shape_valid"] = False
        return [], stats
    stats["matrix_shape_valid"] = True
    stats["eligible_group"] = True

    positive_indices = [index for index, label in enumerate(labels) if label]
    negative_indices = [index for index, label in enumerate(labels) if not label]
    label_mean = sum(labels) / len(labels)
    label_var = sum((float(label) - label_mean) ** 2 for label in labels)
    ranked: list[tuple[float, float, int]] = []
    reliability_scores: list[float] = []
    pearson_scores: list[float] = []
    constant_items = 0
    for item_index, row in enumerate(score_matrix):
        values = [float(value) for value in row]
        positive_mean = sum(values[index] for index in positive_indices) / len(positive_indices)
        negative_mean = sum(values[index] for index in negative_indices) / len(negative_indices)
        reliability = positive_mean - negative_mean
        value_mean = sum(values) / len(values)
        value_var = sum((value - value_mean) ** 2 for value in values)
        if value_var > 0.0 and label_var > 0.0:
            covariance = sum(
                (value - value_mean) * (float(label) - label_mean)
                for value, label in zip(values, labels, strict=True)
            )
            pearson = covariance / (value_var * label_var) ** 0.5
        else:
            pearson = 0.0
        reliability_scores.append(reliability)
        pearson_scores.append(pearson)
        if max(values) - min(values) <= min_score_range:
            constant_items += 1
            continue
        if reliability >= reliability_threshold:
            ranked.append((reliability, pearson, item_index))

    ranked.sort(key=lambda value: (value[0], value[1]), reverse=True)
    selected_indices = [item_index for _, _, item_index in ranked]
    selected = [rubric_items[index] for index in selected_indices]
    stats.update(
        {
            "reliable_items": len(selected),
            "reliability_scores": reliability_scores,
            "pearson_scores": pearson_scores,
            "selected_indices": selected_indices,
            "constant_items": constant_items,
        }
    )
    return selected, stats


_ALFWORLD_STATE_CRITERION_RE = re.compile(
    r"^\s*<?(MILESTONE|AVOID|VERIFY)>?\s*:?\s+(PASS|FAIL)\s+WHEN\s+(.+?)\s*;?\s*"
    r'SOURCE\s*:?\s*"?([A-Za-z0-9_.:-]+)"?\s*$',
    re.IGNORECASE,
)
_ALFWORLD_ENTITY_ID_RE = re.compile(r"\b[a-z][a-z0-9_-]*\s+\d+\b", re.IGNORECASE)


def validate_alfworld_rubric_text(
    rubric_text: str,
    trajectory_evidence: list[dict[str, Any]],
    *,
    max_items: int,
) -> tuple[str, dict[str, Any]]:
    """Keep simple ALFWorld state criteria grounded in one visible trajectory."""
    evidence_by_id = {
        str(item.get("evidence_id")): item for item in trajectory_evidence if str(item.get("evidence_id", ""))
    }
    rejection_reasons: dict[str, int] = {}

    def reject(reason: str) -> None:
        rejection_reasons[reason] = rejection_reasons.get(reason, 0) + 1

    accepted: list[str] = []
    seen: set[str] = set()
    input_items = 0
    for raw_line in str(rubric_text or "").splitlines():
        description = re.sub(r"^\s*\d+[.)]\s*", "", raw_line).strip()
        if not description:
            continue
        input_items += 1

        match = _ALFWORLD_STATE_CRITERION_RE.fullmatch(description)
        if match is None:
            reject("invalid_state_criterion_schema")
            continue
        item_type, verdict, statement, source_id = match.groups()
        item_type = item_type.upper()
        verdict = verdict.upper()
        statement = " ".join(statement.strip().split())
        evidence = evidence_by_id.get(source_id)
        if evidence is None:
            reject("unknown_source")
            continue
        if (item_type == "AVOID") != (verdict == "FAIL"):
            reject("type_verdict_mismatch")
            continue

        transcript = str(evidence.get("transcript", ""))
        combined_visible_text = transcript.lower()
        executed_actions = [
            " ".join(action.lower().split())
            for action in re.findall(
                r"^[ \t]*(?:CHOSEN\s+)?ACTION\s*:\s*([^\r\n<]+)",
                transcript,
                re.IGNORECASE | re.MULTILINE,
            )
        ]
        demonstrated_loop = any(executed_actions.count(action) >= 2 for action in set(executed_actions))
        unknown_entities = {
            entity.lower()
            for entity in _ALFWORLD_ENTITY_ID_RE.findall(statement)
            if entity.lower() not in combined_visible_text
        }
        if unknown_entities:
            reject("invented_entity_id")
            continue
        lowered_statement = statement.lower()
        if re.search(r"\b(?:action|expect|source)\s*:", lowered_statement) or '"' in statement:
            reject("contains_action_script")
            continue
        if re.search(r"\b(?:move|moved|put|drop|dropped|return|returned)\b.+\bback\b", lowered_statement):
            reject("regresses_task_progress")
            continue
        if item_type == "AVOID" and not (
            re.search(r"\binvalid action\b", transcript, re.IGNORECASE)
            or demonstrated_loop
        ):
            reject("avoid_not_demonstrated")
            continue
        if len(statement) < 12:
            reject("criterion_too_short")
            continue
        canonical = f"<{item_type}> {verdict} WHEN {statement}; SOURCE {source_id}"
        semantic_key = re.sub(r"\s+", " ", canonical.rsplit("; SOURCE", 1)[0]).lower()
        if semantic_key in seen:
            reject("duplicate_state_criterion")
            continue
        seen.add(semantic_key)
        accepted.append(canonical)
        if len(accepted) >= max_items:
            break

    filtered = "\n".join(f"{index}. {item}" for index, item in enumerate(accepted, start=1))
    stats = {
        "input_items": input_items,
        "accepted_items": len(accepted),
        "rejected_items": max(0, input_items - len(accepted)),
        "rejection_reasons": rejection_reasons,
    }
    return filtered, stats


class RubricVLLMClient:
    """OpenAI-compatible client for a dedicated, fixed teacher vLLM server."""

    def __init__(self, config: Any):
        self.base_url = str(_get(config, "base_url", "http://127.0.0.1:8001/v1")).rstrip("/")
        self.model = _get(config, "model", None)
        self.api_key_env = str(_get(config, "api_key_env", "RUBRIC_API_KEY"))
        self.timeout = float(_get(config, "timeout", 120.0))
        self.rubrics_per_prompt = int(_get(config, "rubrics_per_prompt", 8))
        self.max_tokens = int(_get(config, "max_tokens", 2048))
        self.temperature = float(_get(config, "temperature", 1.0))
        self.top_p = float(_get(config, "top_p", 0.95))
        self.enable_thinking = bool(_get(config, "generator_enable_thinking", False))
        self.max_workers = max(1, int(_get(config, "max_workers", 8)))
        self.max_rubric_chars = int(_get(config, "max_rubric_chars", 4000))
        self.cache_size = max(0, int(_get(config, "cache_size", 10000)))
        self.fail_on_error = bool(_get(config, "fail_on_error", False))
        self._cache: OrderedDict[str, str] = OrderedDict()
        if not self.model:
            raise ValueError("algorithm.token_credit.rubric_pipeline.model must name the teacher vLLM model")

    def _request(self, problem: str) -> str:
        payload = {
            "model": self.model,
            "messages": build_rubric_generation_messages(problem, self.rubrics_per_prompt),
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "chat_template_kwargs": {"enable_thinking": self.enable_thinking},
        }
        api_key = os.environ.get(self.api_key_env, "EMPTY")
        request = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                response_json = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            try:
                response_body = exc.read().decode("utf-8", errors="replace")
            except Exception:  # noqa: BLE001
                response_body = ""
            detail = f": {response_body}" if response_body else ""
            raise RuntimeError(f"teacher vLLM rubric request failed with HTTP {exc.code}{detail}") from exc
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"teacher vLLM rubric request failed: {exc}") from exc

        try:
            raw_text = response_json["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError(f"unexpected teacher vLLM response: {response_json}") from exc
        rubric_text = parse_rubric_response(raw_text, self.rubrics_per_prompt, self.max_rubric_chars)
        if not rubric_text:
            raise RuntimeError("teacher vLLM returned no parseable rubric items")
        return rubric_text

    def _cache_put(self, problem: str, rubric_text: str) -> None:
        if self.cache_size <= 0:
            return
        self._cache[problem] = rubric_text
        self._cache.move_to_end(problem)
        while len(self._cache) > self.cache_size:
            self._cache.popitem(last=False)

    def generate_batch(self, raw_prompts: list[Any]) -> tuple[list[str], torch.Tensor, dict[str, float]]:
        """Generate once per unique problem and expand rubrics back to the rollout batch."""
        start_time = time.perf_counter()
        problems = [extract_problem_text(prompt) for prompt in raw_prompts]
        unique_problems = list(dict.fromkeys(problems))
        results: dict[str, str] = {}
        failures: dict[str, Exception] = {}
        cache_hits = 0

        pending = []
        for problem in unique_problems:
            if problem in self._cache:
                results[problem] = self._cache[problem]
                self._cache.move_to_end(problem)
                cache_hits += 1
            else:
                pending.append(problem)

        with ThreadPoolExecutor(max_workers=min(self.max_workers, max(1, len(pending)))) as executor:
            future_to_problem = {executor.submit(self._request, problem): problem for problem in pending}
            for future in as_completed(future_to_problem):
                problem = future_to_problem[future]
                try:
                    rubric_text = future.result()
                    results[problem] = rubric_text
                    self._cache_put(problem, rubric_text)
                except Exception as exc:  # noqa: BLE001
                    failures[problem] = exc

        if failures and self.fail_on_error:
            first_problem, first_error = next(iter(failures.items()))
            raise RuntimeError(f"rubric generation failed for problem {first_problem[:120]!r}") from first_error

        rubric_texts = [results.get(problem, "") for problem in problems]
        eligible = torch.tensor([bool(text) for text in rubric_texts], dtype=torch.bool)
        successful_unique = len(unique_problems) - len(failures)
        rubric_lengths = [len(text) for text in rubric_texts if text]
        metrics = {
            "rubric_pipeline/success_rate": successful_unique / max(1, len(unique_problems)),
            "rubric_pipeline/unique_prompts": float(len(unique_problems)),
            "rubric_pipeline/cache_hit_rate": cache_hits / max(1, len(unique_problems)),
            "rubric_pipeline/latency_s": time.perf_counter() - start_time,
            "rubric_pipeline/rubric_chars_mean": sum(rubric_lengths) / max(1, len(rubric_lengths)),
        }
        if failures:
            first_error = next(iter(failures.values()))
            print(
                f"[Rubric Pipeline] {len(failures)}/{len(unique_problems)} unique prompts failed; "
                f"those rollouts use ordinary GRPO. First error: {first_error}",
                flush=True,
            )
        return rubric_texts, eligible, metrics


def load_precomputed_rubrics(
    non_tensor_batch: dict[str, Any],
    raw_prompts: list[Any],
    *,
    rubric_column: str = "rubric",
    valid_column: str = "rubric_valid",
    fail_on_error: bool = True,
) -> tuple[list[str], torch.Tensor, dict[str, float]]:
    """Read precomputed problem rubrics from a rollout batch."""
    if rubric_column not in non_tensor_batch:
        raise KeyError(
            f"Precomputed rubric column {rubric_column!r} is absent from the training batch. "
            f"Available non-tensor keys: {list(non_tensor_batch.keys())}"
        )

    raw_rubrics = list(non_tensor_batch[rubric_column])
    rubric_texts = [str(text).strip() if text is not None else "" for text in raw_rubrics]
    if valid_column and valid_column in non_tensor_batch:
        explicit_valid = [bool(value) for value in list(non_tensor_batch[valid_column])]
    else:
        explicit_valid = [True] * len(rubric_texts)
    eligible_values = [bool(text) and valid for text, valid in zip(rubric_texts, explicit_valid, strict=True)]
    eligible = torch.tensor(eligible_values, dtype=torch.bool)

    if len(rubric_texts) != len(raw_prompts):
        raise ValueError(
            f"Rubric batch size {len(rubric_texts)} does not match raw prompt batch size {len(raw_prompts)}"
        )
    if fail_on_error and not all(eligible_values):
        invalid_count = len(eligible_values) - sum(eligible_values)
        raise RuntimeError(
            f"{invalid_count}/{len(eligible_values)} trajectories have an empty or invalid precomputed rubric"
        )

    problems = [extract_problem_text(prompt) for prompt in raw_prompts]
    rubric_lengths = [len(text) for text, valid in zip(rubric_texts, eligible_values, strict=True) if valid]
    metrics = {
        "rubric_pipeline/success_rate": sum(eligible_values) / max(1, len(eligible_values)),
        "rubric_pipeline/unique_prompts": float(len(set(problems))),
        "rubric_pipeline/cache_hit_rate": 1.0,
        "rubric_pipeline/latency_s": 0.0,
        "rubric_pipeline/rubric_chars_mean": sum(rubric_lengths) / max(1, len(rubric_lengths)),
    }
    return rubric_texts, eligible, metrics


def build_rubric_conditioned_inputs(
    batch: Any,
    tokenizer: Any,
    rubric_texts: list[str],
    *,
    teacher_max_prompt_length: int,
    teacher_enable_thinking: bool,
) -> dict[str, torch.Tensor]:
    """Build rubric-conditioned teacher inputs over the exact single- or multi-turn response."""
    raw_prompts = list(batch.non_tensor_batch["raw_prompt"])
    if len(raw_prompts) != len(rubric_texts):
        raise ValueError(f"rubric count {len(rubric_texts)} does not match batch size {len(raw_prompts)}")

    initial_observations = batch.non_tensor_batch.get("interaction_initial_observation")
    if initial_observations is not None and len(initial_observations) != len(raw_prompts):
        raise ValueError(
            "interaction_initial_observation count does not match batch size: "
            f"{len(initial_observations)} != {len(raw_prompts)}"
        )

    prompt_texts = []
    for index, (raw_prompt, rubric_text) in enumerate(zip(raw_prompts, rubric_texts, strict=True)):
        messages = deepcopy(normalize_messages(raw_prompt))
        if initial_observations is not None:
            initial_observation = initial_observations[index]
            if initial_observation:
                messages.append({"role": "user", "content": str(initial_observation)})
        if rubric_text:
            suffix = f"\n\nKey criteria for solving:\n{rubric_text}"
            for message in reversed(messages):
                if message["role"] == "user":
                    message["content"] = f"{message['content']}{suffix}"
                    break
            else:
                messages.append({"role": "user", "content": suffix.strip()})
        template_kwargs = {
            "tokenize": False,
            "add_generation_prompt": True,
            "enable_thinking": teacher_enable_thinking,
        }
        try:
            prompt_text = tokenizer.apply_chat_template(messages, **template_kwargs)
        except TypeError:
            template_kwargs.pop("enable_thinking")
            prompt_text = tokenizer.apply_chat_template(messages, **template_kwargs)
        prompt_texts.append(prompt_text)

    old_padding_side = getattr(tokenizer, "padding_side", "right")
    old_truncation_side = getattr(tokenizer, "truncation_side", "right")
    tokenizer.padding_side = "left"
    # Keep the final user problem and appended rubric if an unusually long
    # system/history prefix exceeds the privileged-teacher prompt budget.
    tokenizer.truncation_side = "left"
    try:
        encoded = tokenizer(
            prompt_texts,
            padding="max_length",
            truncation=True,
            max_length=teacher_max_prompt_length,
            add_special_tokens=False,
            return_tensors="pt",
        )
    finally:
        tokenizer.padding_side = old_padding_side
        tokenizer.truncation_side = old_truncation_side

    responses = batch.batch["responses"]
    response_mask = batch.batch["response_mask"]
    device = responses.device
    teacher_prompts = encoded["input_ids"].to(device=device, dtype=responses.dtype)
    prompt_mask = encoded["attention_mask"].to(device=device, dtype=response_mask.dtype)
    # Multi-turn agent responses interleave policy tokens (response_mask=1)
    # with environment observations (response_mask=0). Those observation
    # tokens must remain visible in the teacher-forcing context even though no
    # policy loss is applied to them. The full attention mask preserves that
    # context; falling back to response_mask keeps single-turn compatibility.
    if "attention_mask" in batch.batch and "prompts" in batch.batch:
        student_prompt_length = batch.batch["prompts"].shape[-1]
        response_attention_mask = batch.batch["attention_mask"][:, student_prompt_length:]
        if response_attention_mask.shape != responses.shape:
            raise ValueError(
                "Student response attention mask must match responses, "
                f"got {tuple(response_attention_mask.shape)} and {tuple(responses.shape)}"
            )
        response_attention_mask = response_attention_mask.to(device=device, dtype=response_mask.dtype)
    else:
        response_attention_mask = response_mask
    attention_mask = torch.cat([prompt_mask, response_attention_mask], dim=-1)
    input_ids = torch.cat([teacher_prompts, responses], dim=-1)
    position_ids = attention_mask.long().cumsum(dim=-1) - 1
    position_ids.masked_fill_(attention_mask == 0, 0)

    return {
        "prompts": teacher_prompts,
        "responses": responses,
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "position_ids": position_ids,
        "response_mask": response_mask,
    }


def build_rubric_conditioned_turn_replay(
    batch: Any,
    tokenizer: Any,
    rubric_texts: list[str],
    *,
    teacher_max_prompt_length: int,
    teacher_enable_thinking: bool,
) -> list[dict[str, Any]]:
    """Apply a rubric transform independently to every saved generation context.

    The response tokens and their flattened trajectory positions are unchanged.
    Only the context of each turn is rebuilt. Empty-rubric fallback trajectories
    keep their exact rollout token ids without re-tokenization.
    """

    records = list(batch.non_tensor_batch[TURN_REPLAY_KEY])
    if len(records) != len(rubric_texts):
        raise ValueError(
            f"turn replay count {len(records)} does not match rubric count {len(rubric_texts)}"
        )

    transformed_records: list[dict[str, Any]] = []
    response_length = int(batch.batch["responses"].shape[-1])
    for trajectory_index, (record, rubric_text) in enumerate(zip(records, rubric_texts, strict=True)):
        validate_trajectory_record(record, response_length=response_length)
        transformed = deepcopy(record)
        rubric_text = str(rubric_text or "").strip()
        if not rubric_text:
            transformed_records.append(transformed)
            continue

        fixed_prefix_messages = int(transformed.get("fixed_prefix_messages", 3))
        suffix = f"\n\nKey criteria for solving:\n{rubric_text}"
        for turn in transformed.get("turns", []):
            messages = deepcopy(turn["context_messages"])
            injected = False
            prefix_end = min(fixed_prefix_messages, len(messages))
            for message_index in range(prefix_end - 1, -1, -1):
                if str(messages[message_index].get("role", "")) == "user":
                    messages[message_index]["content"] = f"{messages[message_index].get('content', '')}{suffix}"
                    injected = True
                    break
            if not injected:
                raise ValueError(
                    f"trajectory {trajectory_index} turn {turn.get('turn_index')} has no user message "
                    "in its fixed prefix for rubric injection"
                )
            template_kwargs = {
                "add_generation_prompt": True,
                "tokenize": True,
                "enable_thinking": teacher_enable_thinking,
            }
            try:
                context_token_ids = tokenizer.apply_chat_template(messages, **template_kwargs)
            except TypeError:
                template_kwargs.pop("enable_thinking")
                context_token_ids = tokenizer.apply_chat_template(messages, **template_kwargs)
            if hasattr(context_token_ids, "tolist"):
                context_token_ids = context_token_ids.tolist()
            context_token_ids = list(context_token_ids)
            if teacher_max_prompt_length > 0 and len(context_token_ids) > teacher_max_prompt_length:
                raise ValueError(
                    "rubric-conditioned turn context exceeds teacher_max_prompt_length; "
                    "increase the limit instead of truncating tokens because replay contexts must remain turn-aligned: "
                    f"trajectory={trajectory_index}, turn={turn.get('turn_index')}, "
                    f"tokens={len(context_token_ids)}, limit={teacher_max_prompt_length}"
                )
            turn["context_messages"] = messages
            turn["context_token_ids"] = context_token_ids
        transformed_records.append(transformed)
    return transformed_records
