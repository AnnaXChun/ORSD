"""Qwen3 teacher-forced replay with selected attention rows, O(action_tokens * L).

SDPA performs the forward; hooks reconstruct only the last layers' requested
attention rows from the exact normalized, rotary-embedded Q/K. No rollout model
is patched. A dense causal mask is needed only for mask_span interventions.
"""
from contextlib import contextmanager
import importlib
import math

import torch


def validate_record(record):
    ids = record["input_ids"]
    previous_end = 0
    for i, turn in enumerate(record["turns"]):
        s, e = turn["action_token_start"], turn["action_token_end"]
        a, b = turn["turn_token_start"], turn["turn_token_end"]
        os_, oe = turn["observation_token_start"], turn["observation_token_end"]
        assert turn["turn_idx"] == i
        assert previous_end <= a <= s < e <= os_ <= oe == b <= len(ids)
        assert s > 0 and ids[s:e] == turn["action_token_ids"], "token/action span mismatch"
        # Explicit causal-LM shift: the token at m is predicted by hidden[m-1].
        assert list(range(s - 1, e - 1)) == [m - 1 for m in range(s, e)]
        previous_end = b
    if record["done_reason"] not in {"terminal", "max_steps", "context_budget_exhausted", "length_limit", "no_actions"}:
        raise ValueError(f"Unknown outcome provenance: {record['done_reason']}")


class QwenReplay:
    def __init__(self, model_path, config):
        from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, __version__
        from packaging.version import Version
        if not torch.cuda.is_available():
            raise RuntimeError("Replay must run on the server GPU, not on the local machine")
        cfg = AutoConfig.from_pretrained(model_path, local_files_only=True)
        if (cfg.model_type, cfg.hidden_size, cfg.num_hidden_layers) != ("qwen3", 2048, 28):
            raise ValueError("Only the verified Qwen3-1.7B checkpoint is supported")
        if getattr(cfg, "sliding_window", None) and getattr(cfg, "use_sliding_window", False):
            raise ValueError("Selected-row diagnostics require full causal Qwen3 attention")
        self.config = config
        torch.manual_seed(int(config["seed"]))
        self.tokenizer = AutoTokenizer.from_pretrained(config.get("tokenizer_path", model_path), local_files_only=True)
        dtype_key = "dtype" if Version(__version__) >= Version("4.56.0") else "torch_dtype"
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path, local_files_only=True, **{dtype_key: getattr(torch, config["dtype"])},
            attn_implementation="sdpa",
        ).to(config["device"]).eval()
        self.model.requires_grad_(False)
        self.device = next(self.model.parameters()).device
        self.layers = self.model.model.layers[-int(config["last_layers"]):]
        if len(self.layers) != int(config["last_layers"]):
            raise ValueError("Requested attention layers unavailable")

    @contextmanager
    def collect_attention(self, turns, index, selected_targets, blocked, query_start):
        collected = {t: [] for t in selected_targets}

        def hook(module, args, kwargs):
            hidden = kwargs.get("hidden_states", args[0] if args else None)
            if hidden is None or "position_embeddings" not in kwargs:
                raise RuntimeError("Unsupported Qwen3 attention ABI: position_embeddings not exposed; do not upgrade")
            rotary = importlib.import_module(type(module).__module__).apply_rotary_pos_emb
            cos, sin = kwargs["position_embeddings"]
            length = hidden.shape[1]
            dim = module.head_dim
            q = module.q_norm(module.q_proj(hidden).view(1, length, -1, dim)).transpose(1, 2)
            k = module.k_norm(module.k_proj(hidden).view(1, length, -1, dim)).transpose(1, 2)
            q, k = rotary(q, k, cos, sin)
            k = k.repeat_interleave(q.shape[1] // k.shape[1], dim=1)
            keys = torch.arange(length, device=hidden.device)
            for t in selected_targets:
                turn = turns[t]
                positions = [index[m] - 1 for m in range(turn["action_token_start"], turn["action_token_end"])]
                assert min(positions) >= 0
                masses = torch.zeros(len(positions), t, dtype=torch.float32, device=hidden.device)
                for offset in range(0, len(positions), self.config["attention_query_chunk"]):
                    rows = torch.tensor(positions[offset:offset + self.config["attention_query_chunk"]], device=hidden.device)
                    scores = torch.matmul(q[:, :, rows].float(), k.float().transpose(-1, -2)) * module.scaling
                    allowed = keys[None, :] <= rows[:, None]
                    if blocked:
                        lo, hi = blocked
                        allowed &= ~((rows[:, None] >= query_start) & (keys[None, :] >= lo) & (keys[None, :] < hi))
                    scores.masked_fill_(~allowed[None, None], -torch.inf)
                    weights = scores.softmax(-1).mean(dim=1)[0]
                    for source in range(t):
                        span = turns[source]
                        visible = [index[j] for j in range(span["turn_token_start"], span["turn_token_end"]) if j in index]
                        if visible:
                            masses[offset:offset + len(rows), source] = weights[:, visible].sum(-1)
                collected[t].append(masses.cpu())

        handles = [layer.self_attn.register_forward_pre_hook(hook, with_kwargs=True) for layer in self.layers]
        try:
            yield collected
        finally:
            for handle in handles:
                handle.remove()

    @torch.inference_mode()
    def run(self, record, target=None, source=None, intervention="mask_span"):
        validate_record(record)
        turns = record["turns"]
        selected = list(range(len(turns))) if target is None else [target]
        if not selected:
            return {}
        # Causality permits dropping the suffix AFTER the last requested token.
        end = max(turns[t]["action_token_end"] for t in selected)
        original_ids = record["input_ids"][:end]
        retained = list(range(end))
        blocked, query_start = None, None
        if source is not None:
            assert target is not None and source < target
            lo, hi = turns[source]["turn_token_start"], turns[source]["turn_token_end"]
            if intervention == "drop_span":
                retained = [j for j in retained if not lo <= j < hi]
            elif intervention == "mask_span":
                if end > self.config["max_dense_mask_tokens"]:
                    raise ValueError(f"mask_span prefix {end} exceeds dense mask limit; explicitly choose drop_span or raise limit")
                blocked = (lo, hi)
                origin = self.config.get("mask_query_origin", "action")
                if origin not in {"action", "response"}:
                    raise ValueError("mask_query_origin must be action or response")
                query_start = turns[target]["action_token_start" if origin == "action" else "turn_token_start"] - 1
            else:
                raise ValueError(intervention)
        index = {old: new for new, old in enumerate(retained)}
        ids = torch.tensor([[original_ids[j] for j in retained]], device=self.device)
        positions = torch.tensor([retained], device=self.device)  # preserve RoPE positions even after dropping
        attention_mask = None
        if blocked:
            length = len(retained)
            # HF accepts a 4-D additive mask; target's FIRST predictive row is s-1.
            attention_mask = torch.full((length, length), -torch.inf,
                                        dtype=next(self.model.parameters()).dtype, device=self.device).triu_(1)
            attention_mask[query_start:, blocked[0]:blocked[1]] = -torch.inf
            attention_mask = attention_mask[None, None]
        with self.collect_attention(turns, index, selected, blocked, query_start) as attention:
            output = self.model.model(input_ids=ids, position_ids=positions, attention_mask=attention_mask,
                                      use_cache=False, output_attentions=False, return_dict=True)
        result = {}
        for t in selected:
            s, e = turns[t]["action_token_start"], turns[t]["action_token_end"]
            token_indices = [index[m] for m in range(s, e)]
            predictive = [m - 1 for m in token_indices]
            assert ids[0, token_indices].tolist() == turns[t]["action_token_ids"], "intervention changed target"
            logps = []
            for offset in range(0, len(predictive), 64):
                rows = predictive[offset:offset + 64]
                logits = self.model.lm_head(output.last_hidden_state[:, rows]).float()
                labels = ids[:, token_indices[offset:offset + 64]]
                values = logits.gather(-1, labels[..., None]).squeeze(-1) - logits.logsumexp(-1)
                logps.extend(values[0].cpu().tolist())
            if len(attention[t]) != len(self.layers):
                raise RuntimeError("Attention hooks did not run for all requested layers")
            per_token = torch.stack(attention[t]).mean(0)
            if source is not None and per_token[:, source].abs().max().item() > 1e-6:
                raise RuntimeError("Intervened source still receives direct attention")
            result[t] = {"logps": logps, "attention": per_token.mean(0).tolist(),
                         "attention_token_mass": per_token.tolist()}
            if not all(math.isfinite(v) for v in logps + result[t]["attention"]):
                raise ValueError("Nonfinite replay signal")
        return result
