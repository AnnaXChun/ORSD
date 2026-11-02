# Copyright 2026

import json
from types import SimpleNamespace

import torch

from verl.trainer.ppo.rubric_token_credit import (
    RubricVLLMClient,
    build_agent_rubric_generation_messages,
    build_agent_rubric_judge_messages,
    build_rubric_conditioned_inputs,
    build_rubric_generation_messages,
    build_task_group_hint_messages,
    compact_online_trajectory,
    extract_problem_text,
    fit_task_group_hint_messages,
    filter_rubrics_by_outcome_correlation,
    load_precomputed_rubrics,
    order_task_group_trajectory_pairs,
    parse_rubric_judge_response,
    parse_rubric_response,
    parse_task_group_hint_response,
    validate_alfworld_rubric_text,
)


def test_rubric_judge_prompt_is_label_free_and_parser_is_exact_length():
    messages = build_agent_rubric_judge_messages(
        "Put the apple in the fridge.",
        "CHOSEN ACTION: take apple 1 from table 1",
        [
            "<MILESTONE> PASS WHEN the apple is held; SOURCE rollout_1",
            "<AVOID> FAIL WHEN look repeats without new state; SOURCE rollout_1",
        ],
    )
    rendered = "\n".join(message["content"] for message in messages)
    assert "official_outcome" not in rendered
    assert "ground_truth" not in rendered
    assert "take apple 1 from table 1" in rendered
    assert parse_rubric_judge_response('{"scores": [1, 0.5]}', 2) == [1.0, 0.5]
    assert parse_rubric_judge_response('{"scores": [1]}', 2) is None


def test_correlation_filter_accepts_only_discriminative_mixed_group_items():
    items = ["correct-only", "constant", "wrong-only"]
    matrix = [
        [1.0, 0.0, 1.0, 0.0],
        [1.0, 1.0, 1.0, 1.0],
        [0.0, 1.0, 0.0, 1.0],
    ]
    selected, stats = filter_rubrics_by_outcome_correlation(
        items,
        matrix,
        [1.0, 0.0, 1.0, 0.0],
        reliability_threshold=0.5,
    )
    assert selected == ["correct-only"]
    assert stats["outcome_group"] == "mixed"
    assert stats["eligible_group"] is True
    assert stats["reliability_scores"] == [1.0, 0.0, -1.0]
    assert stats["constant_items"] == 1


def test_correlation_filter_disables_uniform_outcome_groups():
    for outcomes, expected_group in (([1, 1, 1, 1], "all_correct"), ([0, 0, 0, 0], "all_wrong")):
        selected, stats = filter_rubrics_by_outcome_correlation(
            ["item"],
            [[1.0, 0.0, 1.0, 0.0]],
            outcomes,
        )
        assert selected == []
        assert stats["eligible_group"] is False
        assert stats["outcome_group"] == expected_group


def test_parse_rubric_json_and_strip_thinking():
    raw = '<think>private reasoning</think>```json\n{"rubrics": ["Check the domain.", "Split the two cases."]}\n```'
    rubric_text = parse_rubric_response(raw, max_items=8, max_chars=4000)
    assert rubric_text == "1. Check the domain.\n2. Split the two cases."
    assert "private reasoning" not in rubric_text


def test_parse_rubric_json_strips_unclosed_thinking_prefix():
    raw = '<think>private reasoning without a closing tag\n{"rubrics": ["<SUGGEST> Check the visible state."]}'
    rubric_text = parse_rubric_response(raw, max_items=4, max_chars=4000)
    assert rubric_text == "1. <SUGGEST> Check the visible state."
    assert "private reasoning" not in rubric_text


def test_parse_rubric_json_strips_prompt_supplied_think_opening():
    raw = 'private reasoning emitted after a prompt-side opening</think>{"rubrics": ["<VERIFY> Check success."]}'
    rubric_text = parse_rubric_response(raw, max_items=4, max_chars=4000)
    assert rubric_text == "1. <VERIFY> Check success."
    assert "private reasoning" not in rubric_text


def test_extract_problem_text_uses_final_user_message():
    prompt = [
        {"role": "system", "content": "Solve carefully."},
        {"role": "user", "content": "What is 2+2?"},
    ]
    assert extract_problem_text(prompt) == "What is 2+2?"


def test_build_rubric_generation_messages_are_problem_only():
    messages = build_rubric_generation_messages("What is 2+2?", 8)
    assert [message["role"] for message in messages] == ["system", "user"]
    assert "What is 2+2?" in messages[-1]["content"]
    assert "trajectory" not in messages[-1]["content"].lower()


def test_build_online_agent_rubric_messages_are_trajectory_only_without_oracle():
    messages = build_agent_rubric_generation_messages(
        "Put the apple in the fridge.",
        [
            {
                "evidence_id": "rollout_1",
                "official_outcome": 1.0,
                "transcript": "ACTION: take apple 1 from table 1",
            }
        ],
        6,
        benchmark="alfworld",
    )
    rendered = "\n".join(message["content"] for message in messages)
    assert "Put the apple in the fridge." in rendered
    assert "official_outcome" not in rendered
    assert "take apple 1 from table 1" in rendered
    assert "Output 2 to 6 lines" in rendered
    assert "reference plan" in rendered
    assert "<MILESTONE>" in rendered
    assert "<AVOID>" in rendered
    assert "<VERIFY>" in rendered
    assert "Evidence id: rollout_1" in rendered
    assert "One rollout:" in rendered
    assert "turning on the lamp are not sufficient" in rendered
    assert "both objects must be placed" in rendered
    assert [message["role"] for message in messages] == ["system", "user"]


def test_task_group_hint_prompt_uses_whole_contrastive_group():
    messages = build_task_group_hint_messages(
        "Put the apple in the fridge.",
        [
            {"transcript": "ACTION: take apple\nOBSERVATION: apple held", "outcome": 1.0},
            {"transcript": "ACTION: look\nOBSERVATION: no progress", "outcome": 0.0},
            {"transcript": "ACTION: open fridge\nOBSERVATION: fridge open", "outcome": 1.0},
        ],
        max_hints=5,
        max_trajectories=3,
        max_trajectory_chars=6000,
    )
    rendered = "\n".join(message["content"] for message in messages)
    assert [message["role"] for message in messages] == ["system", "user"]
    assert "Attempt 1:" in rendered and "Attempt 3:" in rendered
    assert "Result: success" in rendered and "Result: failure" in rendered
    assert "new agent" in rendered.lower()
    assert "Do not mention attempts" in rendered
    assert "numbered process rubric" in rendered
    assert "Write 5 to 5 short rubric criteria" in rendered


def test_task_group_hint_prefers_explicit_success_over_continuous_outcome():
    messages = build_task_group_hint_messages(
        "Buy the exact requested product.",
        [
            {"transcript": "ACTION: click[Buy Now]", "outcome": 0.95, "success": False},
            {"transcript": "ACTION: click[Buy Now]", "outcome": 0.80, "success": True},
        ],
        max_hints=4,
        max_trajectories=2,
        max_trajectory_chars=6000,
    )
    rendered = "\n".join(message["content"] for message in messages)
    assert rendered.count("Result: failure") == 1
    assert rendered.count("Result: success") == 1


class _CharTokenizer:
    def apply_chat_template(self, messages, **kwargs):
        del kwargs
        return list("\n".join(message["content"] for message in messages))


def test_task_group_hint_orders_reward_extremes_then_shuffles_remainder():
    pairs = [(index, float(index)) for index in range(8)]
    first = order_task_group_trajectory_pairs(pairs, seed="step:task")
    second = order_task_group_trajectory_pairs(pairs, seed="step:task")
    assert first == second
    assert first[:2] == [(7, 7.0), (0, 0.0)]
    assert sorted(first) == pairs


def test_task_group_hint_legacy_order_alternates_low_and_high_rewards():
    pairs = [(index, float(index)) for index in range(8)]
    ordered = order_task_group_trajectory_pairs(
        pairs,
        seed="unused-for-legacy-order",
        mode="legacy_alternating_extremes",
    )
    assert ordered[:4] == [(0, 0.0), (7, 7.0), (1, 1.0), (6, 6.0)]


def test_task_group_hint_legacy_prompt_restores_three_to_five_contract():
    messages = build_task_group_hint_messages(
        "Buy the requested product.",
        [
            {"transcript": "ACTION: search", "success": False},
            {"transcript": "ACTION: buy", "success": True},
        ],
        max_hints=5,
        max_trajectories=4,
        max_trajectory_chars=6000,
        prompt_style="legacy_process_rubric_3_to_k",
    )
    rendered = "\n".join(message["content"] for message in messages)
    assert "Write 3 to 5 short rubric criteria" in rendered
    assert "4. ..." not in rendered


def test_task_group_hint_context_fit_truncates_only_the_end():
    evidence = [
        {"transcript": f"start {index} " + "long observation " * 100 + f" end {index}", "success": index % 2 == 0}
        for index in range(8)
    ]
    messages, audit = fit_task_group_hint_messages(
        "Complete the task.",
        evidence,
        _CharTokenizer(),
        max_hints=8,
        max_trajectories=8,
        max_trajectory_chars=6000,
        max_input_tokens=1800,
    )
    assert messages is not None
    rendered = "\n".join(message["content"] for message in messages)
    assert audit["prompt_tokens"] <= audit["prompt_token_budget"]
    assert audit["context_truncated"] is True
    assert audit["context_overflow"] is False
    assert "start 0" in rendered
    assert "[TRUNCATED:" in rendered


def test_task_group_hint_context_overflow_returns_fallback_signal():
    messages, audit = fit_task_group_hint_messages(
        "task " * 100,
        [{"transcript": "trajectory", "success": True} for _ in range(8)],
        _CharTokenizer(),
        max_hints=8,
        max_trajectories=8,
        max_trajectory_chars=6000,
        max_input_tokens=10,
    )
    assert messages is None
    assert audit["context_overflow"] is True


def test_task_group_hint_parser_matches_agent_rubric2_contract():
    raw = """<think>private reasoning</think>
1. Inspect likely receptacles before repeating navigation.
2. <SUGGEST> Acquire the target before opening its destination; SOURCE rollout_1
3. Inspect likely receptacles before repeating navigation.
"""
    rubric, items = parse_task_group_hint_response(raw, max_items=5)
    assert items == [
        "Inspect likely receptacles before repeating navigation.",
        "Acquire the target before opening its destination",
    ]
    assert rubric == (
        "1. Inspect likely receptacles before repeating navigation.\n"
        "2. Acquire the target before opening its destination"
    )
    assert "private reasoning" not in rubric
    assert "SOURCE" not in rubric


def test_task_group_hint_parser_enforces_five_item_bundle_when_requested():
    raw = "\n".join(f"{index}. criterion number {index}" for index in range(1, 5))
    rubric, items = parse_task_group_hint_response(raw, max_items=8, min_items=5)
    assert rubric == ""
    assert items == []


def test_generic_agent_rubric_prompt_keeps_two_message_contract():
    messages = build_agent_rubric_generation_messages(
        "Complete a generic web task.",
        [{"evidence_id": "rollout_1", "official_outcome": 0.0, "transcript": "failed"}],
        4,
    )
    assert [message["role"] for message in messages] == ["system", "user"]
    assert "exactly 4" in messages[0]["content"]


def test_alfworld_grounding_gate_keeps_state_criteria_and_rejects_scripts():
    evidence = [
        {
            "evidence_id": "rollout_1",
            "official_outcome": 1.0,
            "transcript": (
                "CHOSEN ACTION: take alarmclock 1 from dresser 1\n"
                "NEXT OBSERVATION: You pick up the alarmclock 1.\n"
                "CHOSEN ACTION: use desklamp 1\n"
                "NEXT OBSERVATION: The desklamp 1 is now on.\n"
                "CHOSEN ACTION: examine alarmclock 1\n"
                "NEXT OBSERVATION: You won the game!"
            ),
        },
        {
            "evidence_id": "rollout_2",
            "official_outcome": 0.0,
            "transcript": (
                "CHOSEN ACTION: take desklamp 1 from dresser 1\n"
                "NEXT OBSERVATION: Invalid action: take desklamp 1 from dresser 1.\n"
                "CHOSEN ACTION: go to shelf 1\n"
                "NEXT OBSERVATION: You arrive at shelf 1."
            ),
        },
    ]
    rubric = "\n".join(
        [
            "1. <MILESTONE> PASS WHEN alarmclock 1 is held in inventory; SOURCE rollout_1",
            "2. <VERIFY> PASS WHEN alarmclock 1 is examined while desklamp 1 is on; SOURCE rollout_1",
            "3. <AVOID> FAIL WHEN an invalid attempt to take desklamp 1 occurs; SOURCE rollout_2",
            "4. <MILESTONE> PASS WHEN apple 1 is held; SOURCE rollout_1",
            '5. <MILESTONE> PASS WHEN ACTION: "go to shelf 1" succeeds; SOURCE rollout_2',
            "6. <MILESTONE> PASS WHEN alarmclock 1 is moved back to dresser 1; SOURCE rollout_1",
        ]
    )

    filtered, stats = validate_alfworld_rubric_text(rubric, evidence, max_items=6)

    assert filtered.count("\n") + 1 == 3
    assert "alarmclock 1 is held" in filtered
    assert "alarmclock 1 is examined" in filtered
    assert "invalid attempt to take desklamp 1" in filtered
    assert "apple 1" not in filtered
    assert "ACTION:" not in filtered
    assert "moved back" not in filtered
    assert stats["input_items"] == 6
    assert stats["accepted_items"] == 3
    assert stats["rejection_reasons"]["invented_entity_id"] == 1
    assert stats["rejection_reasons"]["contains_action_script"] == 1
    assert stats["rejection_reasons"]["regresses_task_progress"] == 1


def test_alfworld_grounding_gate_allows_task_state_without_hidden_outcome():
    evidence = [
        {
            "evidence_id": "rollout_1",
            "official_outcome": 0.0,
            "transcript": "CHOSEN ACTION: go to desk 1\nNEXT OBSERVATION: You arrive at desk 1.",
        }
    ]
    rubric = "1. <MILESTONE> PASS WHEN desk 1 has been inspected for the target; SOURCE rollout_1"

    filtered, stats = validate_alfworld_rubric_text(rubric, evidence, max_items=4)

    assert "desk 1 has been inspected" in filtered
    assert stats["accepted_items"] == 1


def test_alfworld_parser_accepts_compact_json_criteria():
    evidence = [
        {
            "evidence_id": "rollout_1",
            "official_outcome": 1.0,
            "transcript": (
                "CHOSEN ACTION: examine desk 1\n"
                "NEXT OBSERVATION: On the desk 1, you see an alarmclock 1.\n"
                "CHOSEN ACTION: take alarmclock 1 from desk 1\n"
                "NEXT OBSERVATION: You pick up the alarmclock 1. You won the game!"
            ),
        }
    ]
    raw = json.dumps(
        {
            "criteria": [
                {"type": "MILESTONE", "pass_when": "desk 1 has been inspected", "source": "rollout_1"},
                {"type": "VERIFY", "pass_when": "alarmclock 1 is held", "evidence_id": "rollout_1"},
            ]
        }
    )
    rubric = parse_rubric_response(raw, max_items=4, max_chars=4000)

    filtered, stats = validate_alfworld_rubric_text(rubric, evidence, max_items=4)

    assert stats["accepted_items"] == 2
    assert stats["rejected_items"] == 0
    assert "<MILESTONE> PASS WHEN desk 1 has been inspected" in filtered
    assert "<VERIFY> PASS WHEN alarmclock 1 is held" in filtered
    assert "SOURCE rollout_1" in filtered


def test_compact_online_trajectory_removes_thinking_and_keeps_terminal_evidence():
    text = "<think>private chain</think>" + "a" * 30 + "SUCCESS"
    compact = compact_online_trajectory(text, max_chars=24)
    assert "private chain" not in compact
    assert "SUCCESS" in compact
    assert "middle omitted" in compact


def test_vllm_client_deduplicates_and_caches_problem_requests():
    client = RubricVLLMClient({"model": "teacher", "max_workers": 2})
    calls = []

    def fake_request(problem):
        calls.append(problem)
        return "1. Use a valid derivation."

    client._request = fake_request
    prompts = [
        [{"role": "user", "content": "Problem A"}],
        [{"role": "user", "content": "Problem A"}],
        [{"role": "user", "content": "Problem B"}],
    ]

    rubric_texts, eligible, metrics = client.generate_batch(prompts)
    assert sorted(calls) == ["Problem A", "Problem B"]
    assert rubric_texts == [
        "1. Use a valid derivation.",
        "1. Use a valid derivation.",
        "1. Use a valid derivation.",
    ]
    assert eligible.tolist() == [True, True, True]
    assert metrics["rubric_pipeline/unique_prompts"] == 2.0

    client.generate_batch(prompts)
    assert len(calls) == 2


def test_load_precomputed_rubrics_expands_with_rollout_batch():
    prompts = [
        [{"role": "user", "content": "Problem A"}],
        [{"role": "user", "content": "Problem A"}],
        [{"role": "user", "content": "Problem B"}],
    ]
    non_tensor_batch = {
        "rubric": ["1. Criterion A", "1. Criterion A", "1. Criterion B"],
        "rubric_valid": [True, True, True],
    }
    rubric_texts, eligible, metrics = load_precomputed_rubrics(non_tensor_batch, prompts)
    assert rubric_texts == non_tensor_batch["rubric"]
    assert eligible.tolist() == [True, True, True]
    assert metrics["rubric_pipeline/success_rate"] == 1.0
    assert metrics["rubric_pipeline/unique_prompts"] == 2.0
    assert metrics["rubric_pipeline/cache_hit_rate"] == 1.0


class _FakeTokenizer:
    padding_side = "right"

    def apply_chat_template(self, messages, **kwargs):
        del kwargs
        return messages[-1]["content"]

    def __call__(self, prompt_texts, **kwargs):
        max_length = kwargs["max_length"]
        assert all("Key criteria for solving:" in prompt for prompt in prompt_texts)
        input_ids = torch.tensor([[0, 0, 11, 12], [0, 0, 21, 22]], dtype=torch.long)
        attention_mask = torch.tensor([[0, 0, 1, 1], [0, 0, 1, 1]], dtype=torch.long)
        assert input_ids.shape[-1] == max_length
        return {"input_ids": input_ids, "attention_mask": attention_mask}


def test_build_teacher_inputs_reuses_exact_student_responses():
    responses = torch.tensor([[31, 32, 0], [41, 0, 0]], dtype=torch.long)
    response_mask = torch.tensor([[1, 1, 0], [1, 0, 0]], dtype=torch.long)
    batch = SimpleNamespace(
        batch={"responses": responses, "response_mask": response_mask},
        non_tensor_batch={
            "raw_prompt": [
                [{"role": "user", "content": "Problem A"}],
                [{"role": "user", "content": "Problem B"}],
            ]
        },
    )

    teacher_inputs = build_rubric_conditioned_inputs(
        batch,
        _FakeTokenizer(),
        ["1. Criterion A", "1. Criterion B"],
        teacher_max_prompt_length=4,
        teacher_enable_thinking=True,
    )

    assert torch.equal(teacher_inputs["responses"], responses)
    assert torch.equal(teacher_inputs["input_ids"][:, -3:], responses)
    assert teacher_inputs["input_ids"].shape == (2, 7)
    assert teacher_inputs["position_ids"].shape == (2, 7)


class _AgentFakeTokenizer:
    padding_side = "right"
    truncation_side = "right"

    def apply_chat_template(self, messages, **kwargs):
        del kwargs
        assert messages[-1]["role"] == "user"
        assert "[ALFWorld observation]" in messages[-1]["content"]
        assert "Key criteria for solving:" in messages[-1]["content"]
        return messages[-1]["content"]

    def __call__(self, prompt_texts, **kwargs):
        assert len(prompt_texts) == 1
        max_length = kwargs["max_length"]
        return {
            "input_ids": torch.tensor([[0, 0, 11, 12]], dtype=torch.long)[:, :max_length],
            "attention_mask": torch.tensor([[0, 0, 1, 1]], dtype=torch.long)[:, :max_length],
        }


def test_agent_teacher_keeps_initial_observation_and_environment_tokens_visible():
    responses = torch.tensor([[31, 91, 32, 0]], dtype=torch.long)
    # Tokens 31/32 are actor actions; token 91 is an environment observation.
    response_mask = torch.tensor([[1, 0, 1, 0]], dtype=torch.long)
    student_prompts = torch.tensor([[1, 2, 3]], dtype=torch.long)
    full_attention_mask = torch.tensor([[1, 1, 1, 1, 1, 1, 0]], dtype=torch.long)
    batch = SimpleNamespace(
        batch={
            "prompts": student_prompts,
            "responses": responses,
            "response_mask": response_mask,
            "attention_mask": full_attention_mask,
        },
        non_tensor_batch={
            "raw_prompt": [[{"role": "user", "content": "Put the apple in the fridge."}]],
            "interaction_initial_observation": ["[ALFWorld observation]\nYou are in a kitchen."],
        },
    )

    teacher_inputs = build_rubric_conditioned_inputs(
        batch,
        _AgentFakeTokenizer(),
        ["1. Check the current admissible actions."],
        teacher_max_prompt_length=4,
        teacher_enable_thinking=False,
    )

    # The environment token remains part of causal context, but still carries
    # no actor loss/credit mask.
    assert torch.equal(teacher_inputs["attention_mask"][:, -4:], torch.tensor([[1, 1, 1, 0]]))
    assert torch.equal(teacher_inputs["response_mask"], response_mask)
