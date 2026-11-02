"""Opt-in trainer adapter for per-trajectory rubric pooling.

Only replaces online rubric creation. Environment rollout, plain/privileged
turn replay, reward, GRPO loss, and optimizer ordering stay in RayPPOTrainer.
"""

import json
import math
import time
import uuid
from collections import Counter
from pathlib import Path

import numpy as np

from verl.trainer.ppo.rubric_generator_loss import (
    REPLAY_KEY,
    align_item_spans_to_tokens,
    expand_item_advantages,
    extract_replay_records,
    loss_settings,
)
from verl.trainer.ppo.rubric_pool import (
    POOL_VERSION,
    CloudRubricJudge,
    build_pool,
    clean_transcript,
    fit_generation_messages,
    generator_item_quality,
    generator_quality,
    has_outcome_contrast,
    leave_one_out_generator_quality,
    parse_candidates_with_spans,
    reinforcepp_batch_advantages,
    reinforcepp_item_advantages,
    reject_fewshot_copy,
    select_pool,
    strip_reasoning,
)
from verl.trainer.ppo.rubric_token_credit import compact_online_trajectory, extract_problem_text


def _object_array(values):
    result = np.empty(len(values), dtype=object)
    result[:] = values
    return result


def _compact_group_audit(record):
    """Keep reproducibility-critical scalar results without raw model I/O."""
    item_keys = (
        "id",
        "text",
        "sources",
        "evaluation_indices",
        "judge_coverage",
        "spearman_correlation",
        "pearson_correlation",
        "score_std",
        "score_range",
        "max_selected_similarity",
        "redundant_with",
        "valid",
        "selected",
        "reason",
    )
    return {
        "uid": record["uid"],
        "trajectory_indices": record["trajectory_indices"],
        "outcomes": record["outcomes"],
        "contrast": record["contrast"],
        "bundles": record["bundles"],
        "generation_validation": record["generation_validation"],
        "items": [{key: item.get(key) for key in item_keys} for item in record["items"]],
        "selected_rubric": record["selected_rubric"],
        "generator_reward": record["generator_reward"],
        "generator_valid_fraction": record["generator_valid_fraction"],
        "generator_selected_fraction": record["generator_selected_fraction"],
        "generator_quality_advantage": record["generator_quality_advantage"],
        "generator_item_quality": record.get("generator_item_quality", []),
        "generator_item_advantage": record.get("generator_item_advantage", []),
        "generator_loo_stats": record.get("generator_loo_stats", {}),
        "requests": record["requests"],
        "failed_requests": record.get("failed_requests", 0),
        "failure_reasons": record.get("failure_reasons", {}),
        "error": record["error"],
        "fallback_reason": record["fallback_reason"],
    }


def generate_online_rubric_pool(trainer, rollout_output, rollout_metadata=None):
    if not trainer.async_rollout_mode:
        raise RuntimeError("rubric pool requires async actor rollout")
    config = trainer.config.algorithm.token_credit.rubric_pipeline
    generator_settings = loss_settings(config)
    item_credit_enabled = bool(generator_settings and generator_settings["item_credit"])
    if generator_settings is not None:
        # SingleTurnAgentLoop retains at most rollout.response_length tokens.
        # Never train on a silently sliced rubric generation.
        if int(config.get("max_tokens", 4096)) > trainer.config.actor_rollout_ref.rollout.response_length:
            raise ValueError("Rubric max_tokens must not exceed rollout.response_length when training the generator")
    # Instantiate before any GPU generation so missing credentials fail early.
    judge = getattr(trainer, "_cloud_rubric_pool_judge", None)
    if judge is None:
        judge = CloudRubricJudge(config)
        trainer._cloud_rubric_pool_judge = judge
    audit_mode = str(config.get("pool_audit_mode", "compact")).lower()
    audit_every = int(config.get("pool_audit_every_n_steps", 1))
    write_audit = audit_mode != "off" and trainer.global_steps % audit_every == 0
    full_audit = write_audit and audit_mode == "full"
    metadata = rollout_output.non_tensor_batch if rollout_metadata is None else rollout_metadata
    n = len(rollout_output)
    if any(key not in metadata or len(metadata[key]) != n for key in ("uid", "raw_prompt")):
        raise ValueError("rubric pool requires row-aligned uid and raw_prompt from repeated rollout metadata")
    if n == 0:
        raise ValueError("Cannot pool rubrics from an empty rollout")
    uids = [str(uid) for uid in metadata["uid"]]
    groups = {}
    for index, uid in enumerate(uids):
        groups.setdefault(uid, []).append(index)
    outcome_metadata = dict(rollout_output.non_tensor_batch)
    if "data_source" in metadata:
        outcome_metadata["data_source"] = metadata["data_source"]
    outcomes = [trainer._rubric_outcome_score(outcome_metadata, i) for i in range(n)]
    if not all(math.isfinite(r) for r in outcomes):
        raise ValueError("Non-finite environment outcome in rubric pool")
    threshold = float(config.get("correctness_threshold", 0.6))
    epsilon = float(config.get("outcome_contrast_epsilon", 1e-6))
    max_items = int(config.get("rubrics_per_prompt", 8))
    min_items = int(config.get("pool_min_rubrics_per_bundle", 5))
    max_item_chars = int(config.get("max_rubric_item_chars", 280))
    max_chars = int(config.get("max_trajectory_chars", 6000))
    full_trajectory = bool(config.get("pool_full_trajectory", True))
    generator_include_trajectory = bool(config.get("pool_generator_include_trajectory", True))
    generator_prompt_style = str(config.get("pool_generator_prompt_style", "structured_json"))
    generator_reward_source = str(config.get("pool_generator_reward_source", "valid")).lower()
    thinking = bool(config.get("generator_enable_thinking", False))
    rollout_config = trainer.config.actor_rollout_ref.rollout
    model_limit = getattr(rollout_config, "max_model_len", None)
    if model_limit is None:
        model_limit = int(rollout_config.prompt_length) + int(rollout_config.response_length)
    reserve = int(config.get("pool_generator_context_reserve_tokens", 256))
    input_budget = int(model_limit) - int(config.get("max_tokens", 4096)) - reserve
    configured_budget = int(config.get("pool_generator_max_input_tokens", 0))
    if configured_budget > 0:
        input_budget = min(input_budget, configured_budget)
    # Match SingleTurnAgentLoop's template, including any configured tools.
    data_config = trainer.config.data
    template_kwargs = dict(data_config.get("apply_chat_template_kwargs", {}))
    template_kwargs["enable_thinking"] = thinking
    template_kwargs["tools"] = []
    tool_path = data_config.get("tool_config_path")
    if tool_path:
        from verl.tools.utils.tool_registry import initialize_tools_from_config

        template_kwargs["tools"] = [
            tool.tool_schema.model_dump(exclude_unset=True, exclude_none=True)
            for tool in initialize_tools_from_config(tool_path)
        ]
    custom_template = trainer.config.actor_rollout_ref.model.get("custom_chat_template")
    if custom_template is not None:
        template_kwargs["chat_template"] = custom_template
    raw_transcripts = rollout_output.non_tensor_batch.get("interaction_trajectory_transcript")
    if raw_transcripts is None or len(raw_transcripts) != n:
        # A flattened token stream is not a reliable action/observation record.
        raise ValueError("rubric pool requires interaction_trajectory_transcript from the agent loop")
    initials = rollout_output.non_tensor_batch.get("interaction_initial_observation", [""] * n)
    transcripts, transcript_truncated, problems, messages = [], [], [], []
    generator_transcripts, generator_context_audit, judge_truncated = [], [], []
    for index in range(n):
        transcript = clean_transcript(str(raw_transcripts[index] or ""))
        initial = clean_transcript(str(initials[index] or ""))
        if initial and initial not in transcript:
            transcript = f"INITIAL OBSERVATION: {initial}\n{transcript}"
        legacy_truncated = not full_trajectory and len(transcript) > max_chars
        if not full_trajectory:
            transcript = compact_online_trajectory(transcript, max_chars=max_chars)
        # Keep the full visible trajectory for the cloud. Never reuse the
        # actor's context-limited view as judge evidence in full mode.
        transcripts.append(transcript)
        judge_truncated.append(legacy_truncated)
        problem = extract_problem_text(metadata["raw_prompt"][index])
        problems.append(problem)
        prompt, generator_transcript, context_audit = fit_generation_messages(
            problem,
            transcript,
            max_items,
            trainer.tokenizer,
            input_budget,
            template_kwargs,
            min_count=min_items,
            include_trajectory=generator_include_trajectory,
            prompt_style=generator_prompt_style,
        )
        messages.append(prompt)
        if full_audit:
            generator_transcripts.append(generator_transcript)
        generator_context_audit.append(context_audit)
        transcript_truncated.append(legacy_truncated or context_audit["truncated"])
    for indices in groups.values():
        if any(problems[i] != problems[indices[0]] for i in indices):
            raise ValueError("A rubric uid group contains different tasks")

    # One actual generation per trajectory, including uniform-outcome groups.
    # Existing helper handles worker padding and removes the padded rows again.
    generation_batch = rollout_output.select_idxs(list(range(n)))
    generation_batch.non_tensor_batch["raw_prompt"] = _object_array(messages)
    generation_batch.non_tensor_batch["agent_name"] = _object_array(["single_turn_agent"] * n)
    generation_batch.non_tensor_batch["chat_template_kwargs_override"] = _object_array(
        [{"enable_thinking": thinking} for _ in range(n)]
    )
    generation_batch.meta_info = trainer._auxiliary_actor_generation_meta_info(
        {
            "temperature": float(config.get("temperature", 0.3)),
            # Auxiliary PPO replays the temperature-scaled full distribution,
            # not a top-p truncated distribution. Pool-only behavior is intact.
            "top_p": 1.0 if generator_settings is not None else float(config.get("top_p", 0.9)),
            "top_k": -1,
            "max_tokens": int(config.get("max_tokens", 4096)),
        }
    )
    start = time.monotonic()
    if config.get("pool_console_progress", False):
        print(f"[rubric pool] generating {n} rubric bundles for {len(groups)} groups", flush=True)
    generated, padding = trainer._generate_auxiliary_actor_sequences(generation_batch)
    if len(generated) != n:
        raise ValueError("Auxiliary rubric output is not row-aligned after unpadding")
    generator_records = (
        extract_replay_records(generated, generator_settings["temperature"]) if generator_settings is not None else None
    )
    prompt_width = generated.batch["prompts"].shape[-1]
    for index, audit in enumerate(generator_context_audit):
        actual = int(generated.batch["attention_mask"][index, :prompt_width].sum().item())
        audit["actual_input_prompt_tokens"] = actual
        if actual != audit["input_prompt_tokens"]:
            raise ValueError(
                f"Rubric generator prompt token count differs from budgeted template: row={index}, "
                f"expected={audit['input_prompt_tokens']}, actual={actual}"
            )
    attention = generated.batch["attention_mask"][:, prompt_width:]
    generator_output_token_count = int(attention.sum().item())
    generator_input_token_count = sum(a["actual_input_prompt_tokens"] for a in generator_context_audit)
    bundles, cleaned_generations, generation_validation, item_token_masks = [], [], [], []
    for index, (ids, mask) in enumerate(zip(generated.batch["responses"], attention, strict=True)):
        decode_kwargs = {"skip_special_tokens": True}
        if item_credit_enabled:
            # Offset mapping must see the exact decoded sampler text; tokenizer
            # whitespace cleanup can otherwise break the token-ID round trip.
            decode_kwargs["clean_up_tokenization_spaces"] = False
        raw = trainer.tokenizer.decode(ids[mask.bool()].detach().cpu().tolist(), **decode_kwargs)
        if full_audit:
            cleaned_generations.append(strip_reasoning(raw))
        parsed, item_spans = parse_candidates_with_spans(
            raw,
            min_items=min_items,
            max_items=max_items,
            max_item_chars=max_item_chars,
        )
        accepted, validation = reject_fewshot_copy(
            parsed,
            similarity_threshold=float(config.get("pool_fewshot_copy_similarity", 0.82)),
        )
        validation.update(parsed_item_count=len(parsed), accepted_item_count=len(accepted))
        bundles.append(accepted)
        generation_validation.append(validation)
        if item_credit_enabled and accepted:
            item_token_masks.append(
                align_item_spans_to_tokens(
                    trainer.tokenizer,
                    generator_records[index]["full_response_ids"],
                    generator_records[index]["response_loss_mask"],
                    raw,
                    item_spans,
                )
            )
        else:
            item_token_masks.append([])
    generation_seconds = time.monotonic() - start
    # The cloud step can be long. Do not retain auxiliary generation tensors
    # (or their full prompt messages) while waiting for network requests.
    del attention, generated, generation_batch, messages

    jobs, records = [], []
    for uid, indices in groups.items():
        rewards = [outcomes[i] for i in indices]
        group_bundles = [bundles[i] for i in indices]
        pool = build_pool(group_bundles)
        contrast = has_outcome_contrast(rewards, epsilon)
        record = {
            "uid": uid,
            "trajectory_indices": indices,
            "outcomes": rewards,
            "contrast": contrast,
            "bundles": group_bundles,
            "generation_validation": [generation_validation[i] for i in indices],
            "pool": pool,
            "matrix": None,
            "requests": 0,
            "error": None,
            "fallback_reason": "" if contrast and pool else "no_contrast_or_no_candidates",
        }
        if full_audit:
            record.update(
                task=problems[indices[0]],
                generation_without_thinking=[cleaned_generations[i] for i in indices],
                transcripts=[transcripts[i] for i in indices],
                transcript_truncated=[transcript_truncated[i] for i in indices],
                judge_transcript_truncated=[judge_truncated[i] for i in indices],
                generator_transcripts=[generator_transcripts[i] for i in indices],
                generator_context_audit=[generator_context_audit[i] for i in indices],
            )
        records.append(record)
        if contrast and pool:
            jobs.append(
                {
                    "uid": uid,
                    "problem": problems[indices[0]],
                    "transcripts": [transcripts[i] for i in indices],
                    "pool": pool,
                }
            )
    if config.get("pool_console_progress", False):
        print(f"[rubric pool] cloud scoring {len(jobs)}/{len(groups)} contrastive groups", flush=True)
    start = time.monotonic()
    judged = judge.score_groups(jobs)
    judge_seconds = time.monotonic() - start
    judge_request_count = sum(result["requests"] for result in judged.values())
    judge_transcript_chars = sum(map(len, transcripts))
    del jobs
    postprocess_start = time.monotonic()
    rubric_texts, valid = [""] * n, [False] * n
    all_quality, all_advantages = [0.0] * n, [0.0] * n
    all_valid_fraction, all_selected_fraction = [0.0] * n, [0.0] * n
    all_item_quality = [[] for _ in range(n)]
    generator_advantage_groups = []
    selected_count = 0
    for record in records:
        result = judged.get(record["uid"])
        if result:
            record.update(result)
        pool = record["pool"]
        matrix = record["matrix"]
        # Legacy/all-request failures may still provide no matrix. Partial
        # matrices retain successful cells and are screened by coverage below.
        if matrix is None:
            matrix = [[None] * len(pool) for _ in record["outcomes"]]
        text, diagnostics = select_pool(
            pool,
            matrix,
            record["outcomes"],
            min_correlation=float(config.get("pool_min_correlation", 0.2)),
            min_score_std=float(config.get("pool_min_score_std", 0.08)),
            min_score_range=float(config.get("pool_min_score_range", 0.2)),
            min_judge_coverage=float(config.get("pool_min_judge_coverage", 1.0)),
            behavior_dedup_enabled=bool(config.get("pool_behavior_dedup_enable", False)),
            max_behavior_similarity=float(config.get("pool_max_behavior_similarity", 0.9)),
            max_selected=int(config.get("pool_max_selected", 8)),
            min_selected=int(config.get("pool_min_selected", 1)),
            max_chars=int(config.get("max_rubric_chars", 1600)),
            epsilon=epsilon,
        )
        if record["error"] and not text:
            record["fallback_reason"] = "cloud_judge_error"
        elif not text and not record["fallback_reason"]:
            record["fallback_reason"] = "no_outcome_validated_criterion_passed"
        valid_group = bool(record["contrast"] and result and any(item["valid"] for item in diagnostics))
        valid_fraction = generator_quality(
            record["bundles"],
            pool,
            diagnostics,
            min_items=min_items,
            max_items=max_items,
            valid_group=valid_group,
            reward_source="valid",
        )
        selected_fraction = generator_quality(
            record["bundles"],
            pool,
            diagnostics,
            min_items=min_items,
            max_items=max_items,
            valid_group=valid_group,
            reward_source="selected",
        )
        loo_stats = {}
        if generator_reward_source == "selected_loo_spearman_relu":
            quality, item_quality, loo_stats = leave_one_out_generator_quality(
                record["bundles"],
                pool,
                matrix,
                record["outcomes"],
                diagnostics,
                min_correlation=float(config.get("pool_min_correlation", 0.2)),
                support_threshold=float(config.get("pool_generator_support_threshold", 0.5)),
                min_satisfied=int(config.get("pool_generator_min_satisfied", 2)),
                min_unsatisfied=int(config.get("pool_generator_min_unsatisfied", 2)),
                valid_group=valid_group,
            )
        elif generator_reward_source == "valid":
            quality = valid_fraction
        elif generator_reward_source == "selected":
            quality = selected_fraction
        else:
            quality = generator_quality(
                record["bundles"],
                pool,
                diagnostics,
                min_items=min_items,
                max_items=max_items,
                valid_group=valid_group,
                reward_source=generator_reward_source,
            )
        if generator_reward_source != "selected_loo_spearman_relu":
            item_quality = (
                generator_item_quality(
                    record["bundles"],
                    pool,
                    diagnostics,
                    valid_group=valid_group,
                    reward_source=generator_reward_source,
                )
                if item_credit_enabled
                else [[] for _ in record["bundles"]]
            )
        record.update(
            items=diagnostics,
            selected_rubric=text,
            generator_reward=quality,
            generator_quality=quality,
            generator_valid_fraction=valid_fraction,
            generator_selected_fraction=selected_fraction,
            generator_quality_advantage=[0.0] * len(quality),
            generator_item_quality=item_quality,
            generator_item_advantage=[[] for _ in item_quality],
            generator_loo_stats=loo_stats,
        )
        selected_count += sum(item["selected"] for item in diagnostics)
        for local, index in enumerate(record["trajectory_indices"]):
            rubric_texts[index], valid[index] = text, bool(text)
            all_quality[index] = quality[local]
            all_valid_fraction[index] = valid_fraction[local]
            all_selected_fraction[index] = selected_fraction[local]
            all_item_quality[index] = item_quality[local]
        if valid_group:
            generator_advantage_groups.append(record["trajectory_indices"])
        del record["pool"]  # Item dataclasses are already serialized in diagnostics.

    # REINFORCE++-baseline adaptation for rubric generation: subtract the mean
    # terminal bundle reward within each task, then normalize all informative
    # bundle advantages together across this actor batch.
    all_advantages, advantage_stats = reinforcepp_batch_advantages(
        all_quality, generator_advantage_groups, epsilon=epsilon
    )
    if item_credit_enabled:
        all_item_advantages, item_advantage_stats = reinforcepp_item_advantages(
            all_item_quality, generator_advantage_groups, epsilon=epsilon
        )
    else:
        all_item_advantages = [[] for _ in range(n)]
        item_advantage_stats = {
            "active_group_count": 0,
            "active_bundle_count": 0,
            "active_item_count": 0,
            "centered_mean": 0.0,
            "centered_std": 0.0,
        }
    for record in records:
        record["generator_quality_advantage"] = [
            all_advantages[index] for index in record["trajectory_indices"]
        ]
        record["generator_item_advantage"] = [
            all_item_advantages[index] for index in record["trajectory_indices"]
        ]
    postprocess_seconds = time.monotonic() - postprocess_start

    # Existing teacher logprob and outcome-alignment gate consume these keys.
    rollout_output.non_tensor_batch["online_rubric"] = _object_array(rubric_texts)
    rollout_output.non_tensor_batch["online_rubric_valid"] = np.array(valid, dtype=bool)
    # Quality advantage belongs ONLY to the independent rubric-response PPO
    # branch. It is never added to environment rewards or action advantages.
    rollout_output.non_tensor_batch["rubric_generator_quality"] = np.array(all_quality, dtype=float)
    rollout_output.non_tensor_batch["rubric_generator_quality_advantage"] = np.array(all_advantages, dtype=float)
    if generator_records is not None:
        for i, record in enumerate(generator_records):
            record.update(uid=uids[i], quality=all_quality[i], advantage=all_advantages[i])
            if item_credit_enabled:
                record["token_advantages"] = expand_item_advantages(
                    item_token_masks[i],
                    all_item_advantages[i],
                    len(record["full_response_ids"]),
                )
        rollout_output.non_tensor_batch[REPLAY_KEY] = _object_array(generator_records)
    # A scheduled compact/full audit is useful for routine inspection, but a
    # request failure must be recorded immediately even between audit steps.
    # The failure audit remains compact unless this step was already selected
    # for a full audit, and never contains the API key or HTTP response body.
    request_failure_detected = any(int(record.get("failed_requests", 0)) > 0 for record in records)
    if audit_mode != "off" and request_failure_detected:
        write_audit = True

    audit_file = None
    audit_seconds = 0.0
    audit_bytes = 0
    if write_audit:
        audit_start = time.monotonic()
        audit = {
            "version": POOL_VERSION,
            "step": trainer.global_steps,
            "judge_model": config.get("pool_judge_model"),
            "judge_concurrency": int(config.get("pool_judge_concurrency", 8)),
            "judge_retries": int(config.get("pool_judge_retries", 2)),
            "judge_max_attempts": int(config.get("pool_judge_retries", 2)) + 1,
            "full_trajectory": full_trajectory,
            "generator_context_limit": int(model_limit),
            "generator_input_token_budget": input_budget,
            # These are actor-tokenizer counts, NOT a claim about the cloud model's
            # tokenizer/context size. Cloud overflow is an API error, not truncation.
            "transcript_token_counter": "actor_tokenizer",
            "judge_input_policy": "full_visible_trajectory" if full_trajectory else "legacy_character_budget",
            # This audit is written before the optimizer update; actual application
            # is reported by the actor, not inferred from successful generation.
            "generator_update_status": "pending" if generator_settings else "disabled",
            "generator_loss_enabled": generator_settings is not None,
            "generator_loss_coef": generator_settings["coef"] if generator_settings else 0.0,
            "generator_item_credit_enabled": item_credit_enabled,
            "generator_include_trajectory": generator_include_trajectory,
            "generator_sampling_top_p": 1.0 if generator_settings else float(config.get("top_p", 0.9)),
            "outcome_source": config.get("outcome_score_source"),
            # Retained only for the downstream action-token outcome/gap gate. Pool
            # correlation itself uses the continuous reward vector without binning.
            "downstream_outcome_positive_threshold": threshold,
            "correlation_method": "spearman",
            "criterion_validity": (
                "judge_coverage>=min_judge_coverage_and_outcome_contrast_and_"
                "score_std>=min_score_std_and_score_range>=min_score_range_and_spearman>min_correlation"
            ),
            "generator_reward": generator_reward_source,
            "generator_reward_source": generator_reward_source,
            "generator_advantage": (
                "selected_item_same_task_mean_baseline_then_batch_std"
                if item_credit_enabled
                else "bundle_same_task_mean_baseline_then_batch_std"
            ),
            "generator_leave_one_out": generator_reward_source == "selected_loo_spearman_relu",
            "generator_support": {
                "threshold": float(config.get("pool_generator_support_threshold", 0.5)),
                "min_satisfied": int(config.get("pool_generator_min_satisfied", 2)),
                "min_unsatisfied": int(config.get("pool_generator_min_unsatisfied", 2)),
            },
            "rubrics_per_bundle": {"min": min_items, "max": max_items},
            "min_correlation": float(config.get("pool_min_correlation", 0.2)),
            "min_score_std": float(config.get("pool_min_score_std", 0.08)),
            "min_score_range": float(config.get("pool_min_score_range", 0.2)),
            "min_judge_coverage": float(config.get("pool_min_judge_coverage", 1.0)),
            "behavior_dedup_enabled": bool(config.get("pool_behavior_dedup_enable", False)),
            "max_behavior_similarity": float(config.get("pool_max_behavior_similarity", 0.9)),
            "fewshot_copy_similarity": float(config.get("pool_fewshot_copy_similarity", 0.82)),
            "generator_prompt_style": generator_prompt_style,
            "generator_advantage_stats": advantage_stats,
            "generator_item_advantage_stats": item_advantage_stats,
            "audit_mode": audit_mode,
            "groups": records if full_audit else [_compact_group_audit(record) for record in records],
        }
        if generator_settings is None:
            audit["generator_loss_applied"] = False
        audit_dir = config.get("pool_audit_dir") or str(
            Path(trainer.config.trainer.default_local_dir) / "rubric_pool_audit"
        )
        audit_dir = Path(audit_dir)
        audit_dir.mkdir(parents=True, exist_ok=True)
        audit_file = audit_dir / f"step_{trainer.global_steps:06d}_{uuid.uuid4().hex[:8]}.json"
        with audit_file.open("x", encoding="utf-8") as output:
            json.dump(
                audit,
                output,
                ensure_ascii=False,
                indent=2 if full_audit else None,
                separators=None if full_audit else (",", ":"),
                allow_nan=False,
            )
        audit_bytes = audit_file.stat().st_size
        audit_seconds = time.monotonic() - audit_start
    degraded_groups = sum(bool(record["error"]) for record in records)
    failures = sum(bool(record["error"] and not record["selected_rubric"]) for record in records)
    failed_requests = sum(int(record.get("failed_requests", 0)) for record in records)
    failure_reasons = Counter()
    for record in records:
        failure_reasons.update(record.get("failure_reasons", {}))
    http_429_failures = failure_reasons["Cloud rubric judge HTTP 429"]
    http_5xx_failures = sum(
        count
        for reason, count in failure_reasons.items()
        if reason.startswith("Cloud rubric judge HTTP 5")
    )
    http_failures = sum(
        count for reason, count in failure_reasons.items() if reason.startswith("Cloud rubric judge HTTP ")
    )
    attempted_groups = sum(int(record.get("requests", 0)) > 0 for record in records)
    successful_requests = max(judge_request_count - failed_requests, 0)
    if judge_request_count > 0 and failed_requests == judge_request_count:
        reason_summary = ", ".join(
            f"{reason} x{count}" for reason, count in sorted(failure_reasons.items())
        ) or "unknown failure"
        print(
            f"[rubric pool] all {judge_request_count} cloud-judge requests failed: {reason_summary}; "
            f"audit={audit_file or 'disabled'}",
            flush=True,
        )
    if degraded_groups and config.get("fail_on_error", False):
        suffix = f"; inspect audit {audit_file}" if audit_file else ""
        raise RuntimeError(f"Cloud rubric judging had request failures in {degraded_groups} groups{suffix}")
    count = len(groups)
    eligible = sum(bool(record["selected_rubric"]) for record in records)
    parsed_item_count = sum(item["parsed_item_count"] for item in generation_validation)
    generated_item_count = sum(len(bundle) for bundle in bundles)
    valid_item_count = sum(item["valid"] for record in records for item in record["items"])
    candidate_item_count = sum(len(record["items"]) for record in records)
    judged_cell_coverage = sum(
        item["judge_coverage"] for record in records for item in record["items"]
    ) / max(candidate_item_count, 1)
    rejection_counts = {
        reason: sum(item["reason"] == reason for record in records for item in record["items"])
        for reason in (
            "insufficient_score_std",
            "insufficient_score_range",
            "weak_or_reversed_spearman",
            "redundant_behavior",
            "insufficient_judge_coverage",
            "insufficient_outcome_contrast",
        )
    }
    selected_spearman = [
        item["spearman_correlation"] for record in records for item in record["items"] if item["selected"]
    ]
    all_spearman = [
        item["spearman_correlation"]
        for record in records
        for item in record["items"]
        if item["spearman_correlation"] is not None
    ]
    all_pearson = [
        item["pearson_correlation"]
        for record in records
        for item in record["items"]
        if item["pearson_correlation"] is not None
    ]
    loo_generated = sum(record.get("generator_loo_stats", {}).get("generated_unique_items", 0) for record in records)
    loo_selected = sum(record.get("generator_loo_stats", {}).get("selected_items", 0) for record in records)
    loo_support = sum(record.get("generator_loo_stats", {}).get("support_pass_items", 0) for record in records)
    loo_positive = sum(record.get("generator_loo_stats", {}).get("positive_quality_items", 0) for record in records)
    metrics = {
        "rubric_pipeline/online_actor": 1.0,
        "rubric_pipeline/task_group_hints": 1.0,
        "rubric_pipeline/correlation_filter": 1.0,
        "rubric_pipeline/auxiliary_prompt_count": float(n),
        "rubric_pipeline/auxiliary_padding_rows": float(padding),
        "rubric_pipeline/generator_thinking": float(thinking),
        "rubric_pipeline/success_rate": sum(valid) / n,
        "rubric_pipeline/teacher_eligible_group_frac": eligible / count,
        "rubric_pipeline/contrastive_outcome_group_frac": sum(r["contrast"] for r in records) / count,
        "rubric_pipeline/items_mean": selected_count / count,
        "rubric_pipeline/chars_mean": sum(map(len, rubric_texts)) / n,
        "rubric_pipeline/outcome_mean": sum(outcomes) / n,
        "rubric_pool/enabled": 1.0,
        "rubric_pool/generation_count": float(n),
        "rubric_pool/parse_success_rate": sum(
            item["parsed_item_count"] > 0 for item in generation_validation
        )
        / n,
        "rubric_pool/generator_accepted_bundle_frac": sum(
            item["accepted"] for item in generation_validation
        )
        / n,
        "rubric_pool/generator_fewshot_copy_bundle_frac": sum(
            item["reason"] == "fewshot_copy" for item in generation_validation
        )
        / n,
        "rubric_pool/generator_fewshot_copy_items_mean": sum(
            item["fewshot_copy_count"] for item in generation_validation
        )
        / n,
        "rubric_pool/generator_bundle_rejected_frac": sum(
            not item["accepted"] for item in generation_validation
        )
        / n,
        "rubric_pool/generator_max_fewshot_similarity_mean": sum(
            item["max_fewshot_similarity"] for item in generation_validation
        )
        / n,
        "rubric_pool/generated_items_mean": parsed_item_count / n,
        "rubric_pool/generator_accepted_items_mean": generated_item_count / n,
        "rubric_pool/candidate_items_mean": sum(len(r["items"]) for r in records) / count,
        "rubric_pool/valid_items_mean": valid_item_count / count,
        "rubric_pool/valid_candidate_frac": valid_item_count
        / max(candidate_item_count, 1),
        "rubric_pool/rejected_low_score_std_frac": rejection_counts["insufficient_score_std"]
        / max(candidate_item_count, 1),
        "rubric_pool/rejected_low_score_range_frac": rejection_counts["insufficient_score_range"]
        / max(candidate_item_count, 1),
        "rubric_pool/rejected_weak_correlation_frac": rejection_counts["weak_or_reversed_spearman"]
        / max(candidate_item_count, 1),
        "rubric_pool/rejected_redundant_behavior_frac": rejection_counts["redundant_behavior"]
        / max(candidate_item_count, 1),
        "rubric_pool/rejected_insufficient_judge_coverage_frac": rejection_counts["insufficient_judge_coverage"]
        / max(candidate_item_count, 1),
        "rubric_pool/behavior_dedup_enabled": float(config.get("pool_behavior_dedup_enable", False)),
        "rubric_pool/selected_items_mean": selected_count / count,
        "rubric_pool/teacher_eligible_group_frac": eligible / count,
        "rubric_pool/judge_failed_group_frac": failures / count,
        "rubric_pool/judge_degraded_group_frac": degraded_groups / count,
        "rubric_pool/judge_attempted_group_frac": attempted_groups / count,
        "rubric_pool/judge_failed_attempted_group_frac": failures / max(attempted_groups, 1),
        "rubric_pool/judge_request_active": float(judge_request_count > 0),
        "rubric_pool/judge_successful_request_count": float(successful_requests),
        "rubric_pool/judge_successful_request_frac": successful_requests / max(judge_request_count, 1),
        "rubric_pool/judge_request_failure_frac": failed_requests / max(judge_request_count, 1),
        "rubric_pool/judge_failure_http_400_frac": failure_reasons["Cloud rubric judge HTTP 400"]
        / max(judge_request_count, 1),
        "rubric_pool/judge_failure_http_401_frac": failure_reasons["Cloud rubric judge HTTP 401"]
        / max(judge_request_count, 1),
        "rubric_pool/judge_failure_http_403_frac": failure_reasons["Cloud rubric judge HTTP 403"]
        / max(judge_request_count, 1),
        "rubric_pool/judge_failure_http_404_frac": failure_reasons["Cloud rubric judge HTTP 404"]
        / max(judge_request_count, 1),
        "rubric_pool/judge_failure_http_429_frac": http_429_failures / max(judge_request_count, 1),
        "rubric_pool/judge_failure_http_5xx_frac": http_5xx_failures / max(judge_request_count, 1),
        "rubric_pool/judge_failure_http_other_frac": (
            http_failures - http_429_failures - http_5xx_failures
        )
        / max(judge_request_count, 1),
        "rubric_pool/judge_failure_transport_timeout_frac": failure_reasons[
            "Cloud rubric judge transport_or_timeout"
        ]
        / max(judge_request_count, 1),
        "rubric_pool/judge_failure_invalid_response_frac": failure_reasons[
            "Cloud rubric judge invalid_response"
        ]
        / max(judge_request_count, 1),
        "rubric_pool/judge_failure_truncated_response_frac": failure_reasons[
            "Cloud rubric judge truncated_response"
        ]
        / max(judge_request_count, 1),
        "rubric_pool/judge_cell_coverage": judged_cell_coverage,
        "rubric_pool/judge_request_count": float(judge_request_count),
        "rubric_pool/judge_concurrency": float(config.get("pool_judge_concurrency", 8)),
        "rubric_pool/judge_max_attempts": float(config.get("pool_judge_retries", 2)) + 1.0,
        "rubric_pool/generation_seconds": generation_seconds,
        "rubric_pool/judge_seconds": judge_seconds,
        "rubric_pool/judge_seconds_per_request": judge_seconds / max(judge_request_count, 1),
        "rubric_pool/postprocess_seconds": postprocess_seconds,
        "rubric_pool/audit_write_seconds": audit_seconds,
        "rubric_pool/total_pipeline_seconds": generation_seconds
        + judge_seconds
        + postprocess_seconds
        + audit_seconds,
        "rubric_pool/audit_bytes": float(audit_bytes),
        "rubric_pool/audit_written": float(write_audit),
        "rubric_pool/transcript_truncated_frac": sum(transcript_truncated) / n,
        "rubric_pool/generator_transcript_truncated_frac": sum(transcript_truncated) / n,
        "rubric_pool/judge_transcript_truncated_frac": sum(judge_truncated) / n,
        "rubric_pool/generator_input_token_budget": float(input_budget),
        "rubric_pool/generator_trajectory_conditioned": float(generator_include_trajectory),
        "rubric_pool/generator_reward_selected_only": float(generator_reward_source.startswith("selected")),
        "rubric_pool/generator_reward_spearman_weighted": float(
            generator_reward_source == "selected_spearman_sum"
        ),
        "rubric_pool/generator_reward_continuous_loo": float(
            generator_reward_source == "selected_loo_spearman_relu"
        ),
        "rubric_pool/generator_loo_selected_frac": loo_selected / max(loo_generated, 1),
        "rubric_pool/generator_loo_support_pass_frac": loo_support / max(loo_selected, 1),
        "rubric_pool/generator_loo_positive_quality_frac": loo_positive / max(loo_support, 1),
        "rubric_pool/generator_input_tokens_mean": sum(a["input_prompt_tokens"] for a in generator_context_audit) / n,
        "rubric_pool/generator_input_tokens_max": float(max(a["input_prompt_tokens"] for a in generator_context_audit)),
        "rubric_pool/generator_input_token_count": float(generator_input_token_count),
        "rubric_pool/generator_output_token_count": float(generator_output_token_count),
        "rubric_pool/judge_transcript_chars_mean": judge_transcript_chars / n,
        "rubric_pool/generator_quality_mean": sum(all_quality) / n,
        "rubric_pool/generator_reward_mean": sum(all_quality) / n,
        "rubric_pool/generator_valid_fraction_mean": sum(all_valid_fraction) / n,
        "rubric_pool/generator_selected_fraction_mean": sum(all_selected_fraction) / n,
        "rubric_pool/generator_advantage_mean": sum(all_advantages) / n,
        "rubric_pool/generator_advantage_abs_mean": sum(map(abs, all_advantages)) / n,
        "rubric_pool/generator_advantage_batch_std": advantage_stats["centered_std"],
        "rubric_pool/generator_active_group_frac": advantage_stats["active_group_count"] / count,
        "rubric_pool/generator_active_bundle_frac": advantage_stats["active_bundle_count"] / n,
        "rubric_pool/generator_item_credit_enabled": float(item_credit_enabled),
        "rubric_pool/generator_item_advantage_abs_mean": (
            sum(abs(value) for row in all_item_advantages for value in row)
            / max(sum(len(row) for row in all_item_advantages), 1)
        ),
        "rubric_pool/generator_item_active_bundle_frac": item_advantage_stats["active_bundle_count"] / n,
        "rubric_pool/generator_item_active_frac": item_advantage_stats["active_item_count"]
        / max(sum(len(row) for row in all_item_advantages), 1),
        "rubric_pool/spearman_correlation_mean": sum(all_spearman) / max(len(all_spearman), 1),
        "rubric_pool/pearson_correlation_mean": sum(all_pearson) / max(len(all_pearson), 1),
        "rubric_pool/selected_spearman_mean": sum(selected_spearman) / max(len(selected_spearman), 1),
        "rubric_pool/generator_loss_enabled": float(generator_settings is not None),
    }
    if generator_settings is None:
        metrics["rubric_pool/generator_loss_applied"] = 0.0
    trainer_config = trainer.config.trainer
    config_get = trainer_config.get if hasattr(trainer_config, "get") else (
        lambda key, default=None: getattr(trainer_config, key, default)
    )
    if bool(config_get("analysis_log_enable", False)):
        analysis_every = int(config_get("analysis_log_every_n_steps", 1))
        if analysis_every <= 0:
            raise ValueError("trainer.analysis_log_every_n_steps must be positive")
        if trainer.global_steps % analysis_every == 0:
            analysis_dir = config_get("analysis_log_dir") or str(
                Path(trainer_config.default_local_dir) / "logs" / "analysis"
            )
            analysis_path = Path(analysis_dir) / "rubric_pool_updates.jsonl"
            analysis_path.parent.mkdir(parents=True, exist_ok=True)
            analysis_record = {
                "schema_version": 1,
                "step": trainer.global_steps,
                "metrics": metrics,
                "groups": [_compact_group_audit(record) for record in records],
            }
            with analysis_path.open("a", encoding="utf-8") as output:
                output.write(
                    json.dumps(
                        analysis_record,
                        ensure_ascii=False,
                        separators=(",", ":"),
                        allow_nan=False,
                    )
                    + "\n"
                )
    if config.get("pool_console_progress", False):
        print(
            f"[rubric pool] eligible={eligible}/{count}, selected={selected_count}, "
            f"generation={generation_seconds:.1f}s, judge={judge_seconds:.1f}s, "
            f"audit={audit_file or 'off'}",
            flush=True,
        )
    return rollout_output, metrics
