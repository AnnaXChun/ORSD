"""Independent rubric-response PPO objective for the shared FSDP actor.

No environment tokens or advantages are changed here. Records travel in a
row-aligned DataProto non-tensor column, independently of environment replay.
Torch is imported lazily so configuration and record checks also work on CPU
tools which do not import VERL/Ray.
"""

import math
from contextlib import contextmanager
from difflib import SequenceMatcher

REPLAY_KEY = "rubric_generator_replay"
SETTINGS_KEY = "rubric_generator_loss"


def post_start_linear_decay(global_step, total_steps, start_step, final_scale=0.0):
    """Return a plateau-then-linear scale and its post-start progress."""
    if any(isinstance(value, bool) or int(value) != value for value in (global_step, total_steps, start_step)):
        raise ValueError("linear decay steps must be integers")
    if total_steps < 1 or global_step < 0 or start_step < 0:
        raise ValueError("linear decay requires total_steps>=1 and nonnegative global/start steps")
    final_scale = float(final_scale)
    if not math.isfinite(final_scale) or not 0 <= final_scale <= 1:
        raise ValueError("linear decay final_scale must be in [0, 1]")
    progress = (
        0.0
        if global_step <= start_step
        else min(1.0, (global_step - start_step) / max(total_steps - start_step, 1))
    )
    return 1.0 - progress * (1.0 - final_scale), progress


def linear_warmup(global_step, warmup_steps, initial_scale=0.0):
    """Return an initial-to-one linear scale and its warmup progress."""
    if any(isinstance(value, bool) or int(value) != value for value in (global_step, warmup_steps)):
        raise ValueError("linear warmup steps must be integers")
    if global_step < 0 or warmup_steps < 1:
        raise ValueError("linear warmup requires global_step>=0 and warmup_steps>=1")
    initial_scale = float(initial_scale)
    if not math.isfinite(initial_scale) or not 0 <= initial_scale <= 1:
        raise ValueError("linear warmup initial_scale must be in [0, 1]")
    progress = min(1.0, global_step / warmup_steps)
    return initial_scale + progress * (1.0 - initial_scale), progress


def loss_settings(config):
    """Double gate: a stray auxiliary-loss flag cannot enable the pool."""
    if not config or not config.get("pool_enable", False) or not config.get("pool_generator_loss_enable", False):
        return None
    coef = float(config.get("pool_generator_loss_coef", 0.01))
    temperature = float(config.get("temperature", 0.3))
    clip_ratio = float(config.get("pool_generator_clip_ratio", 0.2))
    micro_size = config.get("pool_generator_micro_batch_size", 1)
    linear_decay_enable = config.get("pool_generator_linear_decay_enable", False)
    linear_decay_final_scale = float(config.get("pool_generator_linear_decay_final_scale", 0.0))
    linear_decay_start_step = config.get("pool_generator_linear_decay_start_step", 0)
    linear_warmup_enable = config.get("pool_generator_linear_warmup_enable", False)
    linear_warmup_steps = config.get("pool_generator_linear_warmup_steps", 0)
    linear_warmup_initial_scale = float(config.get("pool_generator_linear_warmup_initial_scale", 0.0))
    if not math.isfinite(coef) or coef < 0:
        raise ValueError("pool_generator_loss_coef must be finite and nonnegative")
    if coef == 0:
        return None
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("Rubric PPO requires a positive finite sampling temperature")
    if not math.isfinite(clip_ratio) or not 0 < clip_ratio < 1:
        raise ValueError("pool_generator_clip_ratio must be in (0, 1)")
    if isinstance(micro_size, bool) or int(micro_size) != micro_size or micro_size < 1:
        raise ValueError("pool_generator_micro_batch_size must be a positive integer")
    if not isinstance(linear_decay_enable, bool):
        raise ValueError("pool_generator_linear_decay_enable must be boolean")
    if not isinstance(linear_warmup_enable, bool):
        raise ValueError("pool_generator_linear_warmup_enable must be boolean")
    if linear_decay_enable and linear_warmup_enable:
        raise ValueError("generator linear decay and linear warmup are mutually exclusive")
    if not math.isfinite(linear_decay_final_scale) or not 0 <= linear_decay_final_scale <= 1:
        raise ValueError("pool_generator_linear_decay_final_scale must be in [0, 1]")
    if (
        isinstance(linear_decay_start_step, bool)
        or int(linear_decay_start_step) != linear_decay_start_step
        or linear_decay_start_step < 0
    ):
        raise ValueError("pool_generator_linear_decay_start_step must be a nonnegative integer")
    if (
        isinstance(linear_warmup_steps, bool)
        or int(linear_warmup_steps) != linear_warmup_steps
        or linear_warmup_steps < int(linear_warmup_enable)
    ):
        raise ValueError("pool_generator_linear_warmup_steps must be positive when warmup is enabled")
    if not math.isfinite(linear_warmup_initial_scale) or not 0 <= linear_warmup_initial_scale <= 1:
        raise ValueError("pool_generator_linear_warmup_initial_scale must be in [0, 1]")
    item_credit = config.get("pool_generator_item_credit_enable", False)
    if not isinstance(item_credit, bool):
        raise ValueError("pool_generator_item_credit_enable must be boolean")
    return {
        "coef": coef,
        "temperature": temperature,
        "clip_ratio": clip_ratio,
        "micro_batch_size": int(micro_size),
        "item_credit": item_credit,
        "linear_decay_enable": linear_decay_enable,
        "linear_decay_final_scale": linear_decay_final_scale,
        "linear_decay_start_step": int(linear_decay_start_step),
        "linear_warmup_enable": linear_warmup_enable,
        "linear_warmup_steps": int(linear_warmup_steps),
        "linear_warmup_initial_scale": linear_warmup_initial_scale,
    }


def validate_execution(settings, actor_config, rollout_config, legacy_worker_impl):
    if settings is None:
        return
    if actor_config.get("strategy", "fsdp") not in {"fsdp", "fsdp2"} or legacy_worker_impl == "disable":
        raise ValueError("Rubric generator loss currently requires the legacy FSDP/FSDP2 actor worker")
    if actor_config.get("use_dynamic_bsz", False):
        raise ValueError("Rubric generator loss requires actor.use_dynamic_bsz=False for equal FSDP forward counts")
    if rollout_config.get("mode", "async") != "async":
        raise ValueError("Rubric generator loss requires async actor rollout")


def extract_replay_records(generated, temperature):
    """Save actual sampler token IDs; never decode/re-encode the policy output."""
    batch = generated.batch
    for key in ("prompts", "responses", "attention_mask", "response_mask"):
        if key not in batch:
            raise KeyError(f"Rubric generator replay is missing {key}")
    prompt_width = batch["prompts"].shape[-1]
    response_width = batch["responses"].shape[-1]
    if batch["attention_mask"].shape[-1] != prompt_width + response_width:
        raise ValueError("Rubric generation attention mask does not align with prompt + response")
    if batch["response_mask"].shape != batch["responses"].shape:
        raise ValueError("Rubric response loss mask is not token-aligned")
    records = []
    for i in range(len(generated)):
        pm = batch["attention_mask"][i, :prompt_width].detach().cpu().tolist()
        rm = batch["attention_mask"][i, prompt_width:].detach().cpu().tolist()
        lm = batch["response_mask"][i].detach().cpu().tolist()
        if pm != sorted(pm) or rm != sorted(rm, reverse=True):
            raise ValueError("Rubric replay expects left-padded prompts and right-padded responses")
        if any(x not in (0, 1) for x in pm + rm + lm) or any(a > b for a, b in zip(lm, rm, strict=True)):
            raise ValueError("Invalid rubric generation attention/loss mask")
        p = batch["prompts"][i].detach().cpu().tolist()
        r = batch["responses"][i].detach().cpu().tolist()
        context = [int(x) for x, keep in zip(p, pm, strict=True) if keep]
        response = [int(x) for x, keep in zip(r, rm, strict=True) if keep]
        if not context:
            raise ValueError("Rubric generation context is empty")
        if "input_ids" in batch and batch["input_ids"][i].detach().cpu().tolist() != p + r:
            raise ValueError("Rubric generation input_ids differs from saved prompt + response")
        record = {
            "context_token_ids": context,
            "full_response_ids": response,
            "response_loss_mask": lm[: len(response)],
            "temperature": float(temperature),
            "quality": 0.0,
            "advantage": 0.0,
        }
        if "rollout_log_probs" in batch:
            record["rollout_log_probs"] = batch["rollout_log_probs"][i, : len(response)].detach().cpu().tolist()
        records.append(record)
    return records


def has_signal(records):
    for record in records:
        token_advantages = record.get("token_advantages")
        if token_advantages is not None:
            if len(token_advantages) != len(record["response_loss_mask"]):
                raise ValueError("Rubric token advantage/loss-mask length mismatch")
            aligned = zip(token_advantages, record["response_loss_mask"], strict=True)
            if any(float(advantage) != 0 and keep for advantage, keep in aligned):
                return True
        elif float(record["advantage"]) != 0 and any(record["response_loss_mask"]):
            return True
    return False


def align_item_spans_to_tokens(tokenizer, response_ids, response_loss_mask, raw_text, item_spans):
    """Map decoded rubric-string spans back to the exact sampled token rows."""
    if item_spans is None:
        raise ValueError("Parsed rubric response has no character spans for item-level credit")
    if len(response_ids) != len(response_loss_mask):
        raise ValueError("Rubric response/loss-mask length mismatch")
    try:
        encoded = tokenizer(raw_text, add_special_tokens=False, return_offsets_mapping=True)
        encoded_ids = list(encoded["input_ids"])
        offsets = [tuple(pair) for pair in encoded["offset_mapping"]]
    except (KeyError, TypeError, NotImplementedError) as error:
        raise ValueError("Item-level rubric credit requires a fast tokenizer with offset mappings") from error
    if len(encoded_ids) != len(offsets):
        raise ValueError("Rubric tokenizer returned inconsistent IDs and offset mappings")
    special_ids = set(getattr(tokenizer, "all_special_ids", ()))
    original_positions = [index for index, token_id in enumerate(response_ids) if token_id not in special_ids]
    original_ids = [response_ids[index] for index in original_positions]
    original_offsets = [None] * len(original_ids)
    matcher = SequenceMatcher(a=encoded_ids, b=original_ids, autojunk=False)
    for tag, encoded_start, encoded_stop, original_start, original_stop in matcher.get_opcodes():
        if tag == "equal":
            for encoded_index, original_index in zip(
                range(encoded_start, encoded_stop), range(original_start, original_stop), strict=True
            ):
                original_offsets[original_index] = offsets[encoded_index]
            continue
        if original_start == original_stop:
            continue

        # Decode only the mismatched sampled-token boundaries. This preserves
        # the actual rollout tokenization while using the re-encoded text solely
        # for exact-match regions and character coordinates.
        fallback_start = offsets[encoded_start][0] if encoded_start < len(offsets) else len(raw_text)
        fallback_stop = offsets[encoded_stop - 1][1] if encoded_stop > encoded_start else fallback_start
        for original_index in range(original_start, original_stop):
            token_start, token_stop = fallback_start, fallback_stop
            try:
                prefix_before = tokenizer.decode(
                    original_ids[:original_index],
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                )
                prefix_after = tokenizer.decode(
                    original_ids[: original_index + 1],
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                )
                if (
                    raw_text.startswith(prefix_before)
                    and raw_text.startswith(prefix_after)
                    and len(prefix_before) <= len(prefix_after)
                ):
                    token_start, token_stop = len(prefix_before), len(prefix_after)
            except (AttributeError, TypeError, NotImplementedError):
                pass
            original_offsets[original_index] = (token_start, token_stop)
    if any(pair is None for pair in original_offsets):
        raise ValueError("Could not align decoded rubric text to its sampled token IDs")
    masks = []
    previous_stop = -1
    for start, stop in item_spans:
        if not 0 <= start < stop <= len(raw_text) or start < previous_stop:
            raise ValueError("Rubric item character spans are invalid or overlapping")
        mask = [0] * len(response_ids)
        for original_index, (token_start, token_stop) in enumerate(original_offsets):
            if token_stop > start and token_start < stop:
                position = original_positions[original_index]
                if response_loss_mask[position]:
                    mask[position] = 1
        if not any(mask):
            raise ValueError("A parsed rubric item has no aligned sampled tokens")
        masks.append(mask)
        previous_stop = stop
    if any(sum(mask[position] for mask in masks) > 1 for position in range(len(response_ids))):
        raise ValueError("Rubric item token masks overlap after sampled-token alignment")
    return masks


def expand_item_advantages(item_masks, item_advantages, response_length):
    """Place each item advantage only on that item's sampled token span."""
    if len(item_masks) != len(item_advantages):
        raise ValueError("Rubric item mask/advantage count mismatch")
    token_advantages = [0.0] * response_length
    for mask, advantage in zip(item_masks, item_advantages, strict=True):
        if len(mask) != response_length or not math.isfinite(float(advantage)):
            raise ValueError("Invalid rubric item token mask or advantage")
        for index, keep in enumerate(mask):
            if keep:
                if token_advantages[index] != 0:
                    raise ValueError("Rubric item token masks overlap")
                token_advantages[index] = float(advantage)
    return token_advantages


def pack_records(records, pad_token_id, device, temperature, *, require_old=False):
    import torch

    if len(records) == 0:
        raise ValueError("Cannot replay an empty rubric micro batch")
    contexts = [r["context_token_ids"] for r in records]
    responses = [r["full_response_ids"] for r in records]
    if any(not c for c in contexts):
        raise ValueError("Empty rubric context")
    p, width = max(map(len, contexts)), max(1, max(map(len, responses)))
    prompts = torch.full((len(records), p), pad_token_id, dtype=torch.long, device=device)
    output_ids = torch.full((len(records), width), pad_token_id, dtype=torch.long, device=device)
    attention = torch.zeros((len(records), p + width), dtype=torch.long, device=device)
    mask = torch.zeros((len(records), width), dtype=torch.float32, device=device)
    old = torch.zeros_like(mask)
    token_mode = any("token_advantages" in record for record in records)
    if token_mode and not all("token_advantages" in record for record in records):
        raise ValueError("Cannot mix bundle-level and item-level rubric records in one replay batch")
    advantages = torch.zeros_like(mask) if token_mode else []
    for i, (r, context, response) in enumerate(zip(records, contexts, responses, strict=True)):
        if float(r["temperature"]) != float(temperature):
            raise ValueError("Rubric rollout/replay temperature mismatch")
        if len(r["response_loss_mask"]) != len(response):
            raise ValueError("Rubric response/loss-mask length mismatch")
        advantage = float(r["advantage"])
        if not math.isfinite(advantage):
            raise ValueError("Non-finite rubric quality advantage")
        if token_mode:
            token_advantages = r["token_advantages"]
            if len(token_advantages) != len(response) or not all(math.isfinite(float(x)) for x in token_advantages):
                raise ValueError("Rubric token advantages are missing, non-finite, or misaligned")
            if response:
                advantages[i, : len(response)] = torch.tensor(token_advantages, dtype=torch.float32, device=device)
        else:
            advantages.append(advantage)
        prompts[i, -len(context) :] = torch.tensor(context, device=device)
        attention[i, p - len(context) : p + len(response)] = 1
        if response:
            output_ids[i, : len(response)] = torch.tensor(response, device=device)
            mask[i, : len(response)] = torch.tensor(r["response_loss_mask"], device=device)
        if require_old:
            anchor = r.get("old_log_probs")
            if anchor is None or len(anchor) != len(response):
                raise ValueError("Rubric old logprob snapshot is missing or misaligned")
            old[i, : len(response)] = torch.tensor(anchor, dtype=torch.float32, device=device)
    positions = (attention.cumsum(-1) - 1).clamp_min(0) * attention
    return {
        "prompts": prompts,
        "responses": output_ids,
        "input_ids": torch.cat((prompts, output_ids), dim=-1),
        "attention_mask": attention,
        "position_ids": positions,
        "response_mask": mask,
        "old_log_probs": old.detach(),
        "advantages": (
            advantages.detach()
            if token_mode
            else torch.tensor(advantages, dtype=torch.float32, device=device).unsqueeze(-1).detach()
        ),
        "item_credit": token_mode,
    }


def clipped_generator_loss(log_probs, packed, clip_ratio):
    """Mean over bundles of their response-token means, independent of length."""
    import torch

    mask = packed["response_mask"].bool()
    old = packed["old_log_probs"].detach().float()
    if log_probs.shape != mask.shape or old.shape != mask.shape:
        raise ValueError("Rubric forward logprobs are not response-aligned")
    if not torch.isfinite(log_probs[mask]).all() or not torch.isfinite(old[mask]).all():
        raise ValueError("Non-finite rubric replay logprob")
    gap = torch.where(mask, log_probs.float() - old, 0.0)
    ratio = gap.clamp(-20, 20).exp()
    advantage = packed["advantages"].detach().float()
    objective = torch.minimum(ratio * advantage, ratio.clamp(1 - clip_ratio, 1 + clip_ratio) * advantage)
    objective_mask = mask & advantage.ne(0) if packed.get("item_credit", False) else mask
    denominator = objective_mask.sum(-1).clamp_min(1)
    loss = -(torch.where(objective_mask, objective, 0.0).sum(-1) / denominator).mean()
    objective_abs = (torch.where(objective_mask, objective.abs(), 0.0).sum(-1) / denominator).mean()
    count = objective_mask.sum().clamp_min(1)
    active_rows = (objective_mask & advantage.ne(0)).any(-1)
    return loss, {
        # Signed PPO losses can cancel across centered positive/negative
        # advantages.  This non-cancelling scale is the useful quantity when
        # choosing the auxiliary coefficient relative to the task objective.
        "objective_abs": objective_abs.detach().item(),
        "ratio_mean": ((ratio.detach() * objective_mask).sum() / count).item(),
        "clipfrac": (((ratio.detach() - 1).abs() > clip_ratio) * objective_mask).sum().div(count).item(),
        "approx_kl": ((ratio.detach() - 1 - gap.detach()) * objective_mask).sum().div(count).item(),
        "active_frac": active_rows.float().mean().item(),
        "item_credit": float(packed.get("item_credit", False)),
    }


@contextmanager
def _sampler_model_mode(actor):
    """Use deterministic inference mode for the frozen old-policy snapshot."""
    training = actor.actor_module.training
    actor.actor_module.eval()
    try:
        yield
    finally:
        actor.actor_module.train(training)


@contextmanager
def _training_model_mode(actor):
    """Enable training behavior for a gradient-bearing auxiliary forward.

    Hugging Face decoder layers only activate gradient checkpointing while the
    module is in training mode. Running this backward pass under ``eval()``
    still builds an autograd graph, but retains the full activations and can
    cause length-dependent OOMs.
    """
    training = actor.actor_module.training
    actor.actor_module.train()
    try:
        yield
    finally:
        actor.actor_module.train(training)


def cache_old_logprobs(actor, records, settings, pad_token_id, device):
    """Called once BEFORE all PPO epochs/minibatches, while theta == theta_old."""
    import torch

    size = settings["micro_batch_size"]
    discrepancy_sum, count = 0.0, 0
    with _sampler_model_mode(actor), torch.no_grad():
        for start in range(0, len(records), size):
            chunk = records[start : start + size]
            packed = pack_records(chunk, pad_token_id, device, settings["temperature"])
            log_probs = (
                actor._forward_micro_batch(
                    packed,
                    temperature=settings["temperature"],
                    calculate_entropy=False,
                )["log_probs"]
                .detach()
                .float()
            )
            if log_probs.shape != packed["response_mask"].shape:
                raise ValueError("Rubric old logprobs are not response-aligned")
            for i, record in enumerate(chunk):
                n = len(record["full_response_ids"])
                values = log_probs[i, :n].cpu().tolist()
                if not all(math.isfinite(v) for v in values):
                    raise ValueError("Non-finite rubric old logprob")
                record["old_log_probs"] = values
                if "rollout_log_probs" in record:
                    sampled = record["rollout_log_probs"]
                    if len(sampled) != n or not all(math.isfinite(v) for v in sampled):
                        raise ValueError("Invalid rubric rollout logprobs")
                    discrepancy_sum += sum(abs(a - b) for a, b in zip(values, sampled, strict=True))
                    count += n
    return {
        "rubric_generator/rollout_old_logprob_abs_diff_mean": discrepancy_sum / max(count, 1),
        "rubric_generator/rollout_logprob_available": float(count > 0),
    }


def backward_generator_loss(actor, records, settings, pad_token_id, device, task_scale):
    """Accumulate auxiliary gradients; NEVER step or zero the optimizer here.

    Iterate all rows, including zero-advantage rows, on every FSDP rank. Only a
    GLOBAL no-signal decision made by the driver may skip this branch entirely.
    """
    metrics = {
        "rubric_generator/loss": 0.0,
        "rubric_generator/loss_contribution": 0.0,
        "rubric_generator/objective_abs": 0.0,
        "rubric_generator/objective_abs_contribution": 0.0,
    }
    size = settings["micro_batch_size"]
    with _training_model_mode(actor):
        for start in range(0, len(records), size):
            chunk = records[start : start + size]
            packed = pack_records(chunk, pad_token_id, device, settings["temperature"], require_old=True)
            outputs = actor._forward_micro_batch(packed, temperature=settings["temperature"], calculate_entropy=False)
            loss, details = clipped_generator_loss(outputs["log_probs"], packed, settings["clip_ratio"])
            scale = task_scale * len(chunk) / len(records)
            weighted = loss * settings["coef"] * scale
            if actor.scaler is not None:
                actor.scaler.scale(weighted).backward()
            else:
                weighted.backward()
            metrics["rubric_generator/loss"] += loss.detach().item() * scale
            metrics["rubric_generator/loss_contribution"] += weighted.detach().item()
            metrics["rubric_generator/objective_abs"] += details["objective_abs"] * scale
            metrics["rubric_generator/objective_abs_contribution"] += (
                details["objective_abs"] * settings["coef"] * scale
            )
            for key, value in details.items():
                if key == "objective_abs":
                    continue
                name = f"rubric_generator/{key}"
                metrics[name] = metrics.get(name, 0.0) + value * len(chunk) / len(records)
    return metrics
