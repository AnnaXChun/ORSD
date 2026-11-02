"""Top-1 proposals, paired controls and reward-root traversal."""
import hashlib
import math
import random
import statistics


TRACE_IMPLEMENTATION_VERSION = 2
TRUNCATION_REASONS = {"context_budget_exhausted", "length_limit"}


def reward_roots(record, timeout_window=3):
    count = len(record["turns"])
    if not count:
        return []
    roots = []
    if record["benchmark"] == "scienceworld":
        roots = [i for i, r in enumerate(record["step_rewards"]) if r != 0]
    if record["done_reason"] == "max_steps":
        roots += list(range(max(0, count - timeout_window), count))
    elif record["benchmark"] != "scienceworld" and record["done_reason"] == "terminal":
        roots = [count - 1]
    return sorted(set(roots))


def delta_stats(values):
    return {"mean_delta": statistics.mean(values), "mean_abs_delta": statistics.mean(map(abs, values)),
            "mean_positive_delta": statistics.mean(max(v, 0) for v in values),
            "max_abs_delta": max(map(abs, values))}


def entropy(masses):
    total = sum(masses)
    return -sum((v / total) * math.log(v / total) for v in masses if v > 0) if total else 0.0


def select_trace_roots(record, config):
    roots = reward_roots(record, config["timeout_root_window"])
    if (not roots and config.get("diagnostic_truncation_roots", False)
            and record["done_reason"] in TRUNCATION_REASONS):
        count = len(record["turns"])
        return list(range(max(0, count - config["timeout_root_window"]), count)), "truncation_diagnostic"
    return roots, "reward_protocol"


def trace_record(record, backend, config):
    roots, root_origin = select_trace_roots(record, config)
    normal = backend.run(record)
    for t, output in normal.items():
        record["turns"][t]["action_token_logps"] = output["logps"]
    edge_cache = {}
    traces = []
    for root in roots:
        target = root
        visited = set()
        edges = []
        stop = "max_depth"
        while len(edges) < config["max_trace_depth"]:
            if target in visited:
                stop = "visited"
                break
            visited.add(target)
            if target == 0:
                stop = "no_history"
                break
            if target not in edge_cache:
                before = normal[target]
                source = max(range(target), key=lambda k: before["attention"][k])
                seed_key = f"{config['seed']}:{record['task_id']}:{record['trajectory_id']}:{target}"
                rng = random.Random(int.from_bytes(hashlib.sha256(seed_key.encode()).digest()[:8], "big"))
                alternatives = [k for k in range(target) if k != source]
                random_source = rng.choice(alternatives) if alternatives else None
                controls = {}
                for k in {source, target - 1, random_source} - {None}:
                    controls[k] = backend.run(record, target=target, source=k, intervention=config["intervention"])[target]
                masked = controls[source]
                delta = [a - b for a, b in zip(before["logps"], masked["logps"], strict=True)]
                src, dst = record["turns"][source], record["turns"][target]
                edge = {"source_turn": source, "target_turn": target,
                        "source_action": src["action_text"], "source_observation": src["observation_text"],
                        "target_action": dst["action_text"], "token_selection": dst["token_selection"],
                        "attention_score": before["attention"][source],
                        "target_token_ids": dst["action_token_ids"],
                        "target_tokens": [backend.tokenizer.decode([i], clean_up_tokenization_spaces=False)
                                          for i in dst["action_token_ids"]],
                        "original_token_logps": before["logps"], "masked_token_logps": masked["logps"],
                        "token_delta": delta, **delta_stats(delta),
                        "random_source": random_source, "previous_source": target - 1,
                        "attention_before": before["attention"], "attention_after": masked["attention"],
                        "attention_token_mass_before": before["attention_token_mass"],
                        "attention_token_mass_after": masked["attention_token_mass"],
                        "original_top_source": source,
                        "new_top_source": max((k for k in range(target) if k != source),
                                              key=lambda k: masked["attention"][k], default=None),
                        "attention_entropy_before": entropy(before["attention"]),
                        "attention_entropy_after": entropy(masked["attention"]),
                        "intervention": config["intervention"]}
                for label, k in (("random", random_source), ("previous", target - 1)):
                    edge[f"{label}_token_delta"] = None if k is None else [
                        a - b for a, b in zip(before["logps"], controls[k]["logps"], strict=True)]
                    edge[f"{label}_masked_token_logps"] = None if k is None else controls[k]["logps"]
                threshold = config["delta_threshold"]
                edge["dependency_valid"] = None if threshold is None else edge["mean_abs_delta"] > threshold
                edge_cache[target] = edge
            edge = edge_cache[target]
            edges.append(edge)
            if edge["dependency_valid"] is False:
                stop = "below_threshold"
                break
            target = edge["source_turn"]
        traces.append({"benchmark": record["benchmark"], "task_id": record["task_id"],
                       "trajectory_id": record["trajectory_id"], "reward": record.get("environment_reward", record["trajectory_reward"]),
                       "trajectory_reward": record["trajectory_reward"],
                       "success": record["success"], "done_reason": record["done_reason"],
                       "grpo_advantage": record["grpo_advantage"], "root_turn": root,
                       "root_origin": root_origin,
                       "eligible_for_reward_hypotheses": root_origin == "reward_protocol",
                       "trace_depth": len(edges), "edges": edges, "stop_reason": stop,
                       "path_status": "candidate_unthresholded" if config["delta_threshold"] is None else "thresholded"})
    return traces
