"""Lossless rollout capture; never reconstruct sampled tokens by re-tokenizing chat."""
import hashlib
import json
from pathlib import Path


def sampling_seed(seed, task_id, sibling):
    payload = f"{seed}:{task_id}:{sibling}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "big") % (2**31 - 1)


def checkpoint_fingerprint(model_path):
    """Hash local HF weights once per collection/replay process, not per turn."""
    root = Path(model_path)
    weights = sorted(root.glob("*.safetensors")) or sorted(root.glob("pytorch_model*.bin"))
    if not weights:
        raise ValueError("Expected a local HF checkpoint with safetensors or pytorch_model weights")
    files = [root / "config.json", *weights]
    digest = hashlib.sha256()
    for path in files:
        digest.update(path.name.encode())
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def tokenizer_fingerprint(tokenizer):
    payload = {"vocab": tokenizer.get_vocab(), "template": tokenizer.chat_template,
               "special": tokenizer.special_tokens_map}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def action_span(tokenizer, ids):
    """Select tokens overlapping the executed action, excluding reasoning/EOS.

    Prefix decoding uses the original IDs; no assumption that encode(decode(ids))
    preserves sampling boundaries. Boundary-straddling tokens are included.
    """
    from verl.interactions.agent_environment_utils import extract_action
    text = tokenizer.decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    action = extract_action(text)
    if not action:
        return 0, len(ids), action, "full_response_unparseable"
    start = text.rfind(action)
    end = start + len(action)
    lengths = [0] + [len(tokenizer.decode(ids[:i], skip_special_tokens=True,
                                          clean_up_tokenization_spaces=False))
                     for i in range(1, len(ids) + 1)]
    selected = [i for i in range(len(ids)) if lengths[i] < end and lengths[i + 1] > start]
    if not selected:
        raise ValueError("Executed action cannot be aligned to sampled tokens")
    return selected[0], selected[-1] + 1, action, "executed_action"


def observation_end(tokenizer, ids, template_kwargs):
    # Observation chunks from veRL include the NEXT assistant generation header.
    # Leave that header visible when masking the preceding interaction.
    messages = [{"role": "user", "content": "boundary probe"}]
    with_header = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
                                                **template_kwargs)
    without = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=False,
                                            **template_kwargs)
    if with_header[:len(without)] != without:
        raise ValueError("Chat template has no stable generation suffix")
    suffix = with_header[len(without):]
    if suffix and ids[-len(suffix):] != suffix:
        raise ValueError("Observation chunk does not end in the actual generation header")
    return len(ids) - len(suffix)


def finish_record(agent_data, tokenizer, config, kwargs, reward, details, sampling_params):
    turns = agent_data.extra_fields.pop("_attention_turns", [])
    identity = dict(agent_data.interaction_kwargs)
    benchmark = identity["name"]
    task_id = str(identity.get("task_id", identity.get("game_file", identity.get("goal_idx",
                  f"{identity.get('task_name')}:{identity.get('variation_idx')}"))))
    reason = agent_data.extra_fields.get("termination_reason") or (
        turns[-1].get("done_reason", "unknown") if turns else "no_actions")
    return {"schema_version": 1, "benchmark": benchmark, "task_id": task_id,
            "trajectory_id": agent_data.request_id, "trajectory_reward": float(reward),
            "environment_reward": float(details.get("official_reward", reward)),
            "success": bool(details.get("success", False)), "reward_details": details,
            "step_rewards": [t["reward"] for t in turns], "done_reason": reason,
            "grpo_advantage": None, "input_ids": list(agent_data.prompt_ids), "turns": turns,
            "messages": agent_data.messages, "interaction_kwargs": identity,
            "sampling_params": sampling_params, "model_path": str(config.model_path),
            "tokenizer_fingerprint": tokenizer_fingerprint(tokenizer),
            "tokenizer_chat_template": tokenizer.chat_template,
            "model_fingerprint": str(config.model_fingerprint),
            "tokenizer_path": str(config.tokenizer_path),
            "rollout_engine_seed": int(config.rollout_engine_seed),
            "chat_template_kwargs": dict(config.get("chat_template_kwargs", {}))}


def export_records(records, directory):
    """Called after validation unpadding; per-trajectory atomic files avoid Ray races."""
    destination = Path(directory)
    destination.mkdir(parents=True, exist_ok=True)
    for record in records:
        if not isinstance(record, dict):
            raise ValueError("Missing lossless attention_traceback_record in rollout output")
        path = destination / (record["trajectory_id"] + ".json")
        if path.exists():
            raise FileExistsError(path)
        temp = path.with_suffix(".tmp")
        temp.write_text(json.dumps(record, ensure_ascii=False, allow_nan=False), encoding="utf-8")
        temp.replace(path)
