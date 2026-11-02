"""CPU gradient and real actor-update-method regressions; no Ray/GPU/API.

The actor method is compiled from its AST to avoid importing CUDA-only VERL
dependencies. Its forward and optimizer are real small PyTorch components.
"""

import ast
import importlib.util
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
spec = importlib.util.spec_from_file_location(
    "rubric_generator_loss_tested", ROOT / "verl/trainer/ppo/rubric_generator_loss.py"
)
aux = importlib.util.module_from_spec(spec)
spec.loader.exec_module(aux)


class Config(dict):
    __getattr__ = dict.__getitem__


class Batch:
    def __init__(self, batch, non_tensor_batch=None, meta_info=None):
        self.batch = batch
        self.non_tensor_batch = non_tensor_batch or {}
        self.meta_info = meta_info or {}

    def __len__(self):
        return len(self.batch["responses"])

    def select(self, batch_keys, non_tensor_batch_keys):
        return Batch(
            {k: self.batch[k] for k in batch_keys},
            {k: self.non_tensor_batch[k] for k in non_tensor_batch_keys},
            self.meta_info,
        )

    def split(self, n):
        return [
            Batch(
                {k: v[i : i + n] for k, v in self.batch.items()},
                {k: v[i : i + n] for k, v in self.non_tensor_batch.items()},
                self.meta_info,
            )
            for i in range(0, len(self), n)
        ]

    def to(self, device):
        self.batch = {k: v.to(device) for k, v in self.batch.items()}
        return self


def record(response=None, advantage=1.0, context=None):
    response = [3, 4] if response is None else response
    return {
        "context_token_ids": [2] if context is None else context,
        "full_response_ids": response,
        "response_loss_mask": [1] * len(response),
        "temperature": 0.7,
        "advantage": advantage,
        "quality": 0.5,
    }


class CharTokenizer:
    all_special_ids = [0]

    def __call__(self, text, **kwargs):
        return {
            "input_ids": list(map(ord, text)),
            "offset_mapping": [(index, index + 1) for index in range(len(text))],
        }


class NonRoundTripTokenizer(CharTokenizer):
    def __call__(self, text, **kwargs):
        encoded = super().__call__(text, **kwargs)
        start = text.index("first")
        encoded["input_ids"][start] += 1000
        return encoded

    def decode(self, ids, **kwargs):
        return "".join(chr(token_id) for token_id in ids if token_id not in self.all_special_ids)


SETTINGS = {"coef": 0.01, "temperature": 0.7, "clip_ratio": 0.2, "micro_batch_size": 1, "has_signal": True}


def append_metrics(metrics, values):
    for key, value in values.items():
        metrics.setdefault(key, []).append(value)


def task_loss(**kwargs):
    p = kwargs
    ratio = (p["log_prob"] - p["old_log_prob"]).exp()
    objective = ratio * p["advantages"] * p["response_mask"]
    return -objective.mean(), {"actor/pg_objective_abs": objective.abs().mean().item()}


def load_actor_update():
    tree = ast.parse((ROOT / "verl/workers/actor/dp_actor.py").read_text())
    method = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "update_policy")
    method.decorator_list = []
    scope = {
        "DataProto": Batch,
        "torch": torch,
        "get_device_id": lambda: "cpu",
        "has_turn_replay": lambda data: "turn_replay" in data,
        "TURN_REPLAY_KEY": "turn_replay",
        "RUBRIC_GENERATOR_REPLAY_KEY": aux.REPLAY_KEY,
        "RUBRIC_GENERATOR_SETTINGS_KEY": aux.SETTINGS_KEY,
        "cache_old_logprobs": aux.cache_old_logprobs,
        "backward_generator_loss": aux.backward_generator_loss,
        "get_policy_loss_fn": lambda mode: task_loss,
        "append_to_dict": append_metrics,
    }
    exec(compile(ast.fix_missing_locations(ast.Module(body=[method], type_ignores=[])), "actor_update", "exec"), scope)
    return scope["update_policy"]


class ToyActor:
    update_policy = load_actor_update()

    def __init__(self, mini=2, micro=1, epochs=1):
        torch.manual_seed(31)
        self.actor_module = torch.nn.Embedding(8, 8)
        self.actor_optimizer = torch.optim.SGD(self.actor_module.parameters(), lr=0.1)
        self.scaler = None
        self.use_prefix_grouper = False
        self.config = Config(
            use_dynamic_bsz=False,
            use_kl_loss=False,
            ppo_mini_batch_size=mini,
            ppo_micro_batch_size_per_gpu=micro,
            ppo_epochs=epochs,
            entropy_coeff=0,
            loss_agg_mode="token-mean",
            calculate_entropy=False,
            policy_loss={"loss_mode": "vanilla"},
            use_rollout_log_probs=False,
        )
        self.steps = 0
        self.forwards = []

    def _forward_micro_batch(self, inputs, temperature, calculate_entropy):
        self.forwards.append(
            (
                self.steps,
                temperature,
                torch.is_grad_enabled(),
                inputs["input_ids"].clone(),
                self.actor_module.training,
            )
        )
        n = inputs["responses"].shape[-1]
        logits = self.actor_module(inputs["input_ids"])[:, -n - 1 : -1] / temperature
        return {"log_probs": logits.log_softmax(-1).gather(-1, inputs["responses"].unsqueeze(-1)).squeeze(-1)}

    def _forward_turn_replay_batch(self, inputs, temperature, calculate_entropy):
        return self._forward_micro_batch(inputs, temperature, calculate_entropy)

    def _optimizer_step(self):
        norm = torch.nn.utils.clip_grad_norm_(self.actor_module.parameters(), 10.0)
        self.actor_optimizer.step()
        self.steps += 1
        return norm


def task_batch(n=2, settings=None, records=None, zero_task=False, window=False):
    batch = {
        "input_ids": torch.tensor([[1, 5]] * n),
        "responses": torch.tensor([[5]] * n),
        "attention_mask": torch.ones(n, 2, dtype=torch.long),
        "position_ids": torch.tensor([[0, 1]] * n),
        "response_mask": torch.ones(n, 1),
        "old_log_probs": torch.zeros(n, 1),
        "advantages": torch.zeros(n, 1) if zero_task else torch.ones(n, 1),
    }
    replay = np.empty(n, dtype=object)
    replay[:] = records if records is not None else [record(response=[3], advantage=1) for _ in range(n)]
    meta = {"temperature": 1.0, "pad_token_id": 0}
    nt = {aux.REPLAY_KEY: replay}
    if settings is not None:
        meta[aux.SETTINGS_KEY] = settings.copy()
    if window:
        meta["turn_replay_enabled"] = True
        nt["turn_replay"] = np.array([{} for _ in range(n)], dtype=object)
    return Batch(batch, nt, meta)


class RubricLossTests(unittest.TestCase):
    def test_post_start_linear_decay_keeps_first_200_steps_flat(self):
        self.assertEqual(aux.post_start_linear_decay(1, 300, 200), (1.0, 0.0))
        self.assertEqual(aux.post_start_linear_decay(200, 300, 200), (1.0, 0.0))
        self.assertEqual(aux.post_start_linear_decay(250, 300, 200), (0.5, 0.5))
        self.assertEqual(aux.post_start_linear_decay(300, 300, 200), (0.0, 1.0))

    def test_alfworld_decay_is_flat_through_150_then_zero_at_300(self):
        self.assertEqual(aux.post_start_linear_decay(150, 300, 150), (1.0, 0.0))
        self.assertEqual(aux.post_start_linear_decay(225, 300, 150), (0.5, 0.5))
        self.assertEqual(aux.post_start_linear_decay(300, 300, 150), (0.0, 1.0))

    def test_linear_warmup_reaches_full_scale_then_stays_flat(self):
        self.assertEqual(aux.linear_warmup(0, 150), (0.0, 0.0))
        self.assertEqual(aux.linear_warmup(75, 150), (0.5, 0.5))
        self.assertEqual(aux.linear_warmup(150, 150), (1.0, 1.0))
        self.assertEqual(aux.linear_warmup(300, 150), (1.0, 1.0))

    def test_driver_double_gate_and_global_signal_before_dispatch(self):
        tree = ast.parse((ROOT / "verl/trainer/ppo/ray_trainer.py").read_text())
        method = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "_update_actor")
        scope = {"DataProto": Batch}
        exec(
            compile(ast.fix_missing_locations(ast.Module(body=[method], type_ignores=[])), "trainer_update", "exec"),
            scope,
        )
        update = scope["_update_actor"]
        for pool, enabled, signal in [
            (False, True, True),
            (True, False, True),
            (True, True, True),
            (True, True, False),
        ]:
            cfg = Config(enable=True, pool_enable=pool, pool_generator_loss_enable=enabled, temperature=0.7)
            captured = []
            trainer = types.SimpleNamespace(
                config=Config(
                    algorithm=Config(token_credit=Config(enable=True, rubric_pipeline=cfg)),
                    actor_rollout_ref=Config(
                        actor=Config(strategy="fsdp", use_dynamic_bsz=False),
                        rollout=Config(mode="async", temperature=1.0, multi_turn=Config(enable=True)),
                    ),
                ),
                use_legacy_worker_impl="enable",
                _configure_turn_replay=lambda batch: None,
                actor_rollout_wg=types.SimpleNamespace(update_actor=lambda batch, sink=captured: sink.append(batch)),
            )
            data = task_batch(settings=SETTINGS, records=[record([3], int(signal)), record([4], 0)])
            with patch.dict(sys.modules, {"verl.trainer.ppo.rubric_generator_loss": aux}):
                update(trainer, data)
            settings = captured[0].meta_info.get(aux.SETTINGS_KEY)
            if pool and enabled:
                self.assertEqual(settings["has_signal"], signal)
            else:
                self.assertIsNone(settings)  # Including stale metadata supplied by the caller.

    def test_driver_skips_generator_replay_when_decay_reaches_zero(self):
        tree = ast.parse((ROOT / "verl/trainer/ppo/ray_trainer.py").read_text())
        method = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "_update_actor")
        scope = {"DataProto": Batch}
        exec(
            compile(ast.fix_missing_locations(ast.Module(body=[method], type_ignores=[])), "trainer_update", "exec"),
            scope,
        )
        cfg = Config(
            enable=True,
            pool_enable=True,
            pool_generator_loss_enable=True,
            pool_generator_loss_coef=0.003,
            pool_generator_linear_decay_enable=True,
            pool_generator_linear_decay_start_step=150,
            pool_generator_linear_decay_final_scale=0.0,
            temperature=0.7,
        )
        captured = []
        trainer = types.SimpleNamespace(
            global_steps=300,
            total_training_steps=300,
            config=Config(
                algorithm=Config(token_credit=Config(enable=True, rubric_pipeline=cfg)),
                actor_rollout_ref=Config(
                    actor=Config(strategy="fsdp", use_dynamic_bsz=False),
                    rollout=Config(mode="async", temperature=1.0, multi_turn=Config(enable=True)),
                ),
            ),
            use_legacy_worker_impl="enable",
            _configure_turn_replay=lambda batch: None,
            actor_rollout_wg=types.SimpleNamespace(update_actor=lambda batch, sink=captured: sink.append(batch)),
        )
        data = task_batch(records=[record([3], 1), record([4], -1)])
        with patch.dict(sys.modules, {"verl.trainer.ppo.rubric_generator_loss": aux}):
            scope["_update_actor"](trainer, data)
        settings = captured[0].meta_info[aux.SETTINGS_KEY]
        self.assertEqual(settings["coef"], 0.0)
        self.assertFalse(settings["has_signal"])

    def test_driver_applies_generator_warmup_scale(self):
        tree = ast.parse((ROOT / "verl/trainer/ppo/ray_trainer.py").read_text())
        method = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "_update_actor")
        scope = {"DataProto": Batch}
        exec(
            compile(ast.fix_missing_locations(ast.Module(body=[method], type_ignores=[])), "trainer_update", "exec"),
            scope,
        )
        update = scope["_update_actor"]
        cfg = Config(
            enable=True,
            pool_enable=True,
            pool_generator_loss_enable=True,
            pool_generator_loss_coef=0.003,
            pool_generator_linear_warmup_enable=True,
            pool_generator_linear_warmup_steps=150,
            pool_generator_linear_warmup_initial_scale=0.0,
            temperature=0.7,
        )
        captured = []
        trainer = types.SimpleNamespace(
            global_steps=75,
            total_training_steps=300,
            config=Config(
                algorithm=Config(token_credit=Config(enable=True, rubric_pipeline=cfg)),
                actor_rollout_ref=Config(
                    actor=Config(strategy="fsdp", use_dynamic_bsz=False),
                    rollout=Config(mode="async", temperature=1.0, multi_turn=Config(enable=True)),
                ),
            ),
            use_legacy_worker_impl="enable",
            _configure_turn_replay=lambda batch: None,
            actor_rollout_wg=types.SimpleNamespace(update_actor=lambda batch, sink=captured: sink.append(batch)),
        )
        data = task_batch(records=[record([3], 1), record([4], -1)])
        with patch.dict(sys.modules, {"verl.trainer.ppo.rubric_generator_loss": aux}):
            update(trainer, data)
        settings = captured[0].meta_info[aux.SETTINGS_KEY]
        self.assertAlmostEqual(settings["coef"], 0.0015)
        self.assertEqual(settings["schedule_scale"], 0.5)
        self.assertEqual(settings["warmup_progress"], 0.5)

    def test_double_gate_and_zero_coef(self):
        for pool, enabled in [(False, True), (True, False), (False, False)]:
            self.assertIsNone(aux.loss_settings({"pool_enable": pool, "pool_generator_loss_enable": enabled}))
        cfg = {"pool_enable": True, "pool_generator_loss_enable": True}
        settings = aux.loss_settings(cfg)
        self.assertEqual(settings["coef"], 0.01)
        self.assertFalse(settings["item_credit"])
        self.assertFalse(settings["linear_decay_enable"])
        self.assertEqual(settings["linear_decay_final_scale"], 0.0)
        self.assertEqual(settings["linear_decay_start_step"], 0)
        self.assertFalse(settings["linear_warmup_enable"])
        self.assertEqual(settings["linear_warmup_steps"], 0)
        self.assertEqual(settings["linear_warmup_initial_scale"], 0.0)
        self.assertTrue(aux.loss_settings({**cfg, "pool_generator_item_credit_enable": True})["item_credit"])
        decayed = aux.loss_settings(
            {
                **cfg,
                "pool_generator_linear_decay_enable": True,
                "pool_generator_linear_decay_final_scale": 0.25,
                "pool_generator_linear_decay_start_step": 200,
            }
        )
        self.assertTrue(decayed["linear_decay_enable"])
        self.assertEqual(decayed["linear_decay_final_scale"], 0.25)
        self.assertEqual(decayed["linear_decay_start_step"], 200)
        warmed = aux.loss_settings(
            {
                **cfg,
                "pool_generator_linear_warmup_enable": True,
                "pool_generator_linear_warmup_steps": 150,
                "pool_generator_linear_warmup_initial_scale": 0.0,
            }
        )
        self.assertTrue(warmed["linear_warmup_enable"])
        self.assertEqual(warmed["linear_warmup_steps"], 150)
        self.assertIsNone(aux.loss_settings({**cfg, "pool_generator_loss_coef": 0}))
        with self.assertRaises(ValueError):
            aux.loss_settings({**cfg, "pool_generator_loss_coef": float("nan")})
        with self.assertRaises(ValueError):
            aux.loss_settings({**cfg, "pool_generator_linear_decay_final_scale": 1.1})
        with self.assertRaises(ValueError):
            aux.loss_settings({**cfg, "pool_generator_linear_decay_start_step": -1})
        with self.assertRaises(ValueError):
            aux.loss_settings({**cfg, "pool_generator_linear_warmup_enable": True})
        with self.assertRaises(ValueError):
            aux.loss_settings(
                {
                    **cfg,
                    "pool_generator_linear_decay_enable": True,
                    "pool_generator_linear_warmup_enable": True,
                    "pool_generator_linear_warmup_steps": 150,
                }
            )

    def test_execution_guards_only_when_enabled(self):
        aux.validate_execution(None, {"strategy": "megatron"}, {}, "disable")
        with self.assertRaises(ValueError):
            aux.validate_execution(SETTINGS, {"use_dynamic_bsz": True}, {}, "enable")
        with self.assertRaises(ValueError):
            aux.validate_execution(SETTINGS, {}, {}, "disable")

    def test_records_preserve_thinking_eos_and_padding(self):
        generated = Batch(
            {
                "prompts": torch.tensor([[0, 2, 6], [1, 2, 7]]),
                "responses": torch.tensor([[3, 4, 5, 7], [6, 7, 0, 0]]),
                "attention_mask": torch.tensor([[0, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 0, 0]]),
                "response_mask": torch.tensor([[1, 1, 1, 1], [1, 1, 0, 0]]),
            }
        )
        records = aux.extract_replay_records(generated, 0.7)
        self.assertEqual(records[0]["full_response_ids"], [3, 4, 5, 7])  # including pretend think/EOS
        self.assertEqual(records[1]["full_response_ids"], [6, 7])
        packed = aux.pack_records(records, 0, "cpu", 0.7)
        self.assertEqual(packed["input_ids"].tolist(), [[0, 2, 6, 3, 4, 5, 7], [1, 2, 7, 6, 7, 0, 0]])
        self.assertEqual(packed["position_ids"].tolist(), [[0, 0, 1, 2, 3, 4, 5], [0, 1, 2, 3, 4, 0, 0]])

    def test_bad_alignment_and_temperature_fail(self):
        r = record()
        with self.assertRaises(ValueError):
            aux.pack_records([r], 0, "cpu", 1.0)
        r["response_loss_mask"] = [1]
        with self.assertRaises(ValueError):
            aux.pack_records([r], 0, "cpu", 0.7)

    def test_gradient_signs_and_detached_targets(self):
        records = [record([3], 1), record([4], -1)]
        for r in records:
            r["old_log_probs"] = [-1.0]
        packed = aux.pack_records(records, 0, "cpu", 0.7, require_old=True)
        logp = torch.tensor([[-1.0], [-1.0]], requires_grad=True)
        packed["advantages"].requires_grad_()
        loss, _ = aux.clipped_generator_loss(logp, packed, 0.2)
        self.assertAlmostEqual(loss.item(), 0.0)  # zero scalar loss need not mean zero gradient
        loss.backward()
        self.assertLess(logp.grad[0].item(), 0)
        self.assertGreater(logp.grad[1].item(), 0)
        self.assertIsNone(packed["advantages"].grad)

    def test_item_spans_credit_only_their_sampled_tokens(self):
        raw = '{"rubrics":["first","second"]}'
        first = (raw.index("first"), raw.index("first") + len("first"))
        second = (raw.index("second"), raw.index("second") + len("second"))
        response = list(map(ord, raw)) + [0]
        loss_mask = [1] * len(response)
        masks = aux.align_item_spans_to_tokens(CharTokenizer(), response, loss_mask, raw, [first, second])
        token_advantages = aux.expand_item_advantages(masks, [1.0, -1.0], len(response))
        self.assertTrue(all(token_advantages[index] == 1 for index in range(*first)))
        self.assertTrue(all(token_advantages[index] == -1 for index in range(*second)))
        self.assertEqual(token_advantages[-1], 0)

        item_record = record(response=response, advantage=99)
        item_record["response_loss_mask"] = loss_mask
        item_record["token_advantages"] = token_advantages
        item_record["old_log_probs"] = [-1.0] * len(response)
        packed = aux.pack_records([item_record], 0, "cpu", 0.7, require_old=True)
        logp = torch.full((1, len(response)), -1.0, requires_grad=True)
        loss, details = aux.clipped_generator_loss(logp, packed, 0.2)
        loss.backward()
        self.assertLess(logp.grad[0, first[0]].item(), 0)
        self.assertGreater(logp.grad[0, second[0]].item(), 0)
        self.assertEqual(logp.grad[0, raw.index("rubrics")].item(), 0)
        self.assertEqual(logp.grad[0, -1].item(), 0)
        self.assertEqual(details["item_credit"], 1)
        self.assertTrue(aux.has_signal([item_record]))

    def test_item_span_alignment_tolerates_non_roundtrip_token_boundary(self):
        raw = '{"rubrics":["first","second"]}'
        spans = [
            (raw.index("first"), raw.index("first") + len("first")),
            (raw.index("second"), raw.index("second") + len("second")),
        ]
        response = list(map(ord, raw)) + [0]
        masks = aux.align_item_spans_to_tokens(
            NonRoundTripTokenizer(), response, [1] * len(response), raw, spans
        )
        self.assertTrue(all(masks[0][index] for index in range(*spans[0])))
        self.assertTrue(all(masks[1][index] for index in range(*spans[1])))
        self.assertFalse(any(left and right for left, right in zip(*masks, strict=True)))

    def test_absolute_objective_metric_does_not_cancel(self):
        records = [record([3], 1), record([4], -1)]
        for r in records:
            r["old_log_probs"] = [-1.0]
        packed = aux.pack_records(records, 0, "cpu", 0.7, require_old=True)
        logp = torch.tensor([[-1.0], [-1.0]], requires_grad=True)
        loss, details = aux.clipped_generator_loss(logp, packed, 0.2)
        self.assertAlmostEqual(loss.item(), 0.0)
        self.assertAlmostEqual(details["objective_abs"], 1.0)

    def test_generator_loss_weight_scales_only_rubric_update(self):
        weak, strong = ToyActor(), ToyActor()
        before = weak.actor_module.weight.detach().clone()
        weak.update_policy(task_batch(settings=SETTINGS))
        strong.update_policy(task_batch(settings={**SETTINGS, "coef": 0.02}))
        torch.testing.assert_close(weak.actor_module.weight[1], strong.actor_module.weight[1])
        torch.testing.assert_close(
            strong.actor_module.weight[2] - before[2],
            2 * (weak.actor_module.weight[2] - before[2]),
            atol=2e-7,
            rtol=1e-3,
        )

    def test_rubric_microbatch_size_does_not_change_weight(self):
        single, paired = ToyActor(micro=2), ToyActor(micro=2)
        records = [record([3], 0.8), record([3, 4, 5], -0.2)]
        single.update_policy(task_batch(settings=SETTINGS, records=records))
        paired.update_policy(task_batch(settings={**SETTINGS, "micro_batch_size": 2}, records=records))
        torch.testing.assert_close(single.actor_module.weight, paired.actor_module.weight)

    def test_clipping_and_masked_padding(self):
        records = [record([3, 4], 1), record([4], -1)]
        for r in records:
            r["old_log_probs"] = [-1.0] * len(r["full_response_ids"])
        packed = aux.pack_records(records, 0, "cpu", 0.7, require_old=True)
        logp = torch.tensor([[-0.1, -0.1], [-2.0, float("nan")]], requires_grad=True)
        loss, _ = aux.clipped_generator_loss(logp, packed, 0.2)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        torch.testing.assert_close(logp.grad, torch.zeros_like(logp))

    def test_shared_actor_gets_both_losses_one_step(self):
        actor = ToyActor()
        data = task_batch(settings=SETTINGS)
        original_adv = data.batch["advantages"].clone()
        before = actor.actor_module.weight.detach().clone()
        metrics = actor.update_policy(data)
        self.assertEqual(actor.steps, 1)
        self.assertFalse(torch.equal(before[1], actor.actor_module.weight[1]))  # task context
        self.assertFalse(torch.equal(before[2], actor.actor_module.weight[2]))  # rubric context
        torch.testing.assert_close(original_adv, data.batch["advantages"])
        self.assertEqual(metrics["rubric_pool/generator_loss_applied"], 1)
        self.assertAlmostEqual(metrics["rubric_generator/objective_abs"], 1.0)
        self.assertAlmostEqual(metrics["rubric_generator/objective_abs_contribution"], 0.01)
        self.assertAlmostEqual(metrics["rubric_generator/objective_abs_ratio_to_actor"], 0.01)
        self.assertTrue(actor.actor_module.training)

    def test_old_snapshot_before_all_minibatches_and_epochs(self):
        actor = ToyActor(mini=2, epochs=2)
        data = task_batch(n=4, settings=SETTINGS)
        actor.update_policy(data)
        anchors = [r["old_log_probs"][:] for r in data.non_tensor_batch[aux.REPLAY_KEY]]
        self.assertEqual(actor.steps, 4)
        self.assertEqual(len([x for x in actor.forwards if not x[2]]), 4)
        self.assertTrue(all(x[0] == 0 for x in actor.forwards if not x[2]))
        self.assertTrue(all(a == anchors[0] for a in anchors))
        self.assertTrue(all(x[1] == 0.7 for x in actor.forwards if not x[2]))
        self.assertTrue(all(not x[4] for x in actor.forwards if not x[2]))
        self.assertTrue(all(x[4] for x in actor.forwards if x[2]))

    def test_disabled_or_global_zero_signal_exact_baseline(self):
        baseline = ToyActor()
        baseline.update_policy(task_batch())
        inactive = ToyActor()
        settings = {**SETTINGS, "has_signal": False}
        metrics = inactive.update_policy(task_batch(settings=settings))
        torch.testing.assert_close(baseline.actor_module.weight, inactive.actor_module.weight, rtol=0, atol=0)
        self.assertEqual(len(baseline.forwards), len(inactive.forwards))
        self.assertEqual(metrics["rubric_pool/generator_loss_applied"], 0)

    def test_zero_advantage_rank_still_runs_collective_order(self):
        active, neutral = ToyActor(), ToyActor()
        active.update_policy(task_batch(settings=SETTINGS))
        neutral.update_policy(task_batch(settings=SETTINGS, records=[record([3], 0), record([], 0)]))
        self.assertEqual([(x[0], x[1], x[2]) for x in active.forwards], [(x[0], x[1], x[2]) for x in neutral.forwards])

    def test_auxiliary_updates_when_task_advantage_zero(self):
        actor = ToyActor()
        before = actor.actor_module.weight.detach().clone()
        actor.update_policy(task_batch(settings=SETTINGS, zero_task=True))
        torch.testing.assert_close(before[1], actor.actor_module.weight[1])
        self.assertFalse(torch.equal(before[2], actor.actor_module.weight[2]))

    def test_turn_replay_task_branch_still_receives_auxiliary_loss(self):
        actor = ToyActor()
        metrics = actor.update_policy(task_batch(settings=SETTINGS, window=True))
        self.assertEqual(actor.steps, 1)
        self.assertEqual(metrics["rubric_pool/generator_loss_applied"], 1)


if __name__ == "__main__":
    unittest.main()
