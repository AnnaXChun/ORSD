"""CPU adapter regression with synthetic tensors; no Ray, torch or paid API.

Run directly with a Python environment containing numpy. This checks grouping,
row alignment and fallback plumbing, not CUDA/vLLM numerical equivalence.
"""

import importlib.util
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[3]


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class Tensor(np.ndarray):
    def bool(self):
        return self.astype(bool)

    def detach(self):
        return self

    def cpu(self):
        return self


def tensor(values):
    return np.array(values).view(Tensor)


class Batch:
    def __init__(self, batch, non_tensor_batch):
        self.batch, self.non_tensor_batch = batch, non_tensor_batch
        self.meta_info = {}

    def __len__(self):
        return self.batch["responses"].shape[0]

    def select_idxs(self, indices):
        return Batch(
            {k: v[indices].copy() for k, v in self.batch.items()},
            {k: np.asarray(v, dtype=object)[indices].copy() for k, v in self.non_tensor_batch.items()},
        )


class AdapterTests(unittest.TestCase):
    def run_adapter(
        self,
        *,
        fail_api=False,
        generator_loss=False,
        unequal_quality=False,
        long_transcript=False,
        generator_budget=0,
        full_trajectory=True,
        generator_include_trajectory=True,
        generator_reward_source="valid",
        item_credit=False,
        generator_min_satisfied=2,
        generator_min_unsatisfied=2,
        template_mismatch=False,
        audit_mode="full",
    ):
        fake_helpers = types.ModuleType("verl.trainer.ppo.rubric_token_credit")
        fake_helpers.compact_online_trajectory = lambda text, max_chars: text[:max_chars]
        fake_helpers.extract_problem_text = lambda messages: messages[-1]["content"]
        with patch.dict(sys.modules):
            core = load_module("verl.trainer.ppo.rubric_pool", ROOT / "verl/trainer/ppo/rubric_pool.py")
            replay = load_module(
                "verl.trainer.ppo.rubric_generator_loss", ROOT / "verl/trainer/ppo/rubric_generator_loss.py"
            )
            sys.modules[fake_helpers.__name__] = fake_helpers
            adapter = load_module("online_rubric_pool_adapter_test", ROOT / "verl/trainer/ppo/online_rubric_pool.py")
            cfg = {
                "pool_enable": True,
                "pool_judge_model": "mock",
                "correctness_threshold": 0.6,
                "outcome_score_source": "training_reward",
                "max_rubric_chars": 1600,
                "pool_generator_loss_enable": generator_loss,
                "pool_generator_item_credit_enable": item_credit,
                "max_tokens": 32,
                "pool_generator_max_input_tokens": generator_budget,
                "pool_full_trajectory": full_trajectory,
                "pool_generator_include_trajectory": generator_include_trajectory,
                "pool_generator_reward_source": generator_reward_source,
                "pool_generator_min_satisfied": generator_min_satisfied,
                "pool_generator_min_unsatisfied": generator_min_unsatisfied,
                "pool_audit_mode": audit_mode,
            }
            outcomes = [1, 0, 1, 0, 1, 1, 1, 1]
            indices = list(range(8))
            raw_prompts = [[{"role": "user", "content": "task-a" if i < 4 else "task-b"}] for i in indices]
            metadata = {"uid": ["mixed"] * 4 + ["uniform"] * 4, "raw_prompt": raw_prompts}
            output = Batch(
                {
                    "responses": tensor([[i] for i in indices]),
                    "prompts": tensor([[1, 2]] * 8),
                    "attention_mask": tensor([[1, 1, 1]] * 8),
                },
                {
                    "interaction_trajectory_transcript": [
                        f"ACTION: inspect item {i}"
                        + (
                            "\nOBSERVATION: " + "visible product attributes " * 400 + "\nACTION: buy final item"
                            if long_transcript
                            else ""
                        )
                        for i in indices
                    ],
                    "reward_extra_info": [{"training_reward": x} for x in outcomes],
                },
            )
            original_response = output.batch["responses"].copy()

            class Judge:
                def score_groups(self, jobs):
                    self.jobs = jobs
                    outputs = {}
                    for job in jobs:
                        matrix = None
                        if not fail_api:
                            matrix = []
                            for row in range(4):
                                outcome = float(row % 2 == 0)
                                cells = []
                                for item in job["pool"]:
                                    source = item.sources[0]
                                    if not unequal_quality or source == 0:
                                        cells.append(outcome)
                                    elif source == 1:
                                        cells.append(1 - outcome)
                                    else:
                                        cells.append(0.5)
                                matrix.append(cells)
                        outputs[job["uid"]] = {
                            "matrix": matrix,
                            "requests": 4,
                            "error": "mock failure" if fail_api else None,
                        }
                    return outputs

            judge = Judge()
            generated_calls = []
            raw_generations = [
                json.dumps({"rubrics": [f"Check concrete constraint {i} item {k}." for k in range(5)]})
                for i in indices
            ]

            class ItemCreditTokenizer:
                all_special_ids = [0]

                @staticmethod
                def encode(text, **kwargs):
                    return list(map(ord, text))

                @staticmethod
                def apply_chat_template(messages, **kwargs):
                    return list(map(ord, "\n".join(m["content"] for m in messages) + "\nassistant:"))

                @staticmethod
                def decode(ids, **kwargs):
                    return "".join(map(chr, ids))

                def __call__(self, text, **kwargs):
                    return {
                        "input_ids": self.encode(text),
                        "offset_mapping": [(position, position + 1) for position in range(len(text))],
                    }

            tokenizer = (
                ItemCreditTokenizer()
                if item_credit
                else types.SimpleNamespace(
                    encode=lambda text, **kwargs: list(map(ord, text)),
                    apply_chat_template=lambda messages, **kwargs: list(
                        map(ord, "\n".join(m["content"] for m in messages) + "\nassistant:")
                    ),
                    decode=lambda ids, **kwargs: raw_generations[ids[0]],
                )
            )
            expected_contexts = []

            def generate(batch):
                generated_calls.append(batch)
                self.assertEqual(len(batch), 8)  # Includes uniform groups.
                self.assertEqual(batch.non_tensor_batch["raw_prompt"].shape, (8,))
                for i, prompt in enumerate(batch.non_tensor_batch["raw_prompt"]):
                    current = json.loads(prompt[-1]["content"])
                    if generator_include_trajectory:
                        self.assertIn(f"inspect item {i}", current["CURRENT_ATTEMPT"])
                    else:
                        self.assertEqual(current, {"CURRENT_TASK": "task-a" if i < 4 else "task-b"})
                    self.assertNotIn("training_reward", prompt[-1]["content"])
                expected_contexts[:] = [tokenizer.apply_chat_template(m) for m in batch.non_tensor_batch["raw_prompt"]]
                if template_mismatch:
                    expected_contexts[0].append(99)
                width = max(map(len, expected_contexts))
                response_rows = (
                    [list(map(ord, raw)) for raw in raw_generations]
                    if item_credit
                    else [[i] for i in indices]
                )
                response_width = max(map(len, response_rows))
                return Batch(
                    {
                        "responses": tensor([row + [0] * (response_width - len(row)) for row in response_rows]),
                        "prompts": tensor([[0] * (width - len(c)) + c for c in expected_contexts]),
                        "attention_mask": tensor(
                            [
                                [0] * (width - len(context))
                                + [1] * len(context)
                                + [1] * len(response)
                                + [0] * (response_width - len(response))
                                for context, response in zip(expected_contexts, response_rows, strict=True)
                            ]
                        ),
                        "response_mask": tensor(
                            [[1] * len(row) + [0] * (response_width - len(row)) for row in response_rows]
                        ),
                    },
                    {},
                ), 0

            with tempfile.TemporaryDirectory() as directory:
                trainer = types.SimpleNamespace(
                    async_rollout_mode=True,
                    global_steps=1,
                    _cloud_rubric_pool_judge=judge,
                    config=types.SimpleNamespace(
                        algorithm=types.SimpleNamespace(token_credit=types.SimpleNamespace(rubric_pipeline=cfg)),
                        trainer=types.SimpleNamespace(default_local_dir=directory),
                        actor_rollout_ref=types.SimpleNamespace(
                            rollout=types.SimpleNamespace(response_length=64, prompt_length=2048, max_model_len=40960),
                            model={},
                        ),
                        data={},
                    ),
                    _rubric_outcome_score=lambda m, i: m["reward_extra_info"][i]["training_reward"],
                    _auxiliary_actor_generation_meta_info=lambda x: x,
                    _generate_auxiliary_actor_sequences=generate,
                    tokenizer=tokenizer,
                )
                result, metrics = adapter.generate_online_rubric_pool(trainer, output, metadata)
                self.assertEqual(len(generated_calls), 1)
                self.assertEqual(len(judge.jobs), 1)  # Do not spend on uniform-group judging.
                np.testing.assert_array_equal(result.batch["responses"], original_response)
                self.assertEqual([r["training_reward"] for r in result.non_tensor_batch["reward_extra_info"]], outcomes)
                self.assertNotIn("raw_prompt", output.non_tensor_batch)  # Auxiliary prompt did not leak into task.
                if generator_loss:
                    self.assertNotIn("rubric_pool/generator_loss_applied", metrics)  # Actor reports after backward.
                    self.assertEqual(generated_calls[0].meta_info["top_p"], 1.0)
                    saved = result.non_tensor_batch[replay.REPLAY_KEY]
                    self.assertEqual(saved.shape, (8,))
                    for i, record in enumerate(saved):
                        self.assertEqual(record["context_token_ids"], expected_contexts[i])
                        self.assertEqual(
                            record["full_response_ids"],
                            list(map(ord, raw_generations[i])) if item_credit else [i],
                        )
                        self.assertEqual(
                            record["advantage"], result.non_tensor_batch["rubric_generator_quality_advantage"][i]
                        )
                        if item_credit:
                            self.assertEqual(len(record["token_advantages"]), len(record["full_response_ids"]))
                    if fail_api:
                        self.assertFalse(replay.has_signal(saved))
                    elif unequal_quality:
                        self.assertTrue(replay.has_signal(saved))
                        if generator_reward_source == "selected_spearman_sum":
                            expected_best_quality = 5.0
                        elif generator_reward_source == "selected_loo_spearman_relu":
                            expected_best_quality = 0.8
                        else:
                            expected_best_quality = 1.0
                        np.testing.assert_allclose(
                            result.non_tensor_batch["rubric_generator_quality"][:4],
                            [expected_best_quality, 0.0, 0.0, 0.0],
                        )
                        self.assertGreater(saved[0]["advantage"], 0)
                        self.assertLess(saved[1]["advantage"], 0)
                else:
                    self.assertEqual(metrics["rubric_pool/generator_loss_applied"], 0)
                    self.assertNotIn(replay.REPLAY_KEY, result.non_tensor_batch)
                    self.assertEqual(generated_calls[0].meta_info["top_p"], 0.9)
                self.assertEqual(
                    result.non_tensor_batch["online_rubric_valid"].tolist(), [not fail_api] * 4 + [False] * 4
                )
                self.assertEqual(result.non_tensor_batch["rubric_generator_quality_advantage"][4:].tolist(), [0.0] * 4)
                audits = list(Path(directory).glob("rubric_pool_audit/*.json"))
                self.assertEqual(len(audits), 1)
                audit = json.loads(audits[0].read_text())
                self.assertEqual(audit["version"], core.POOL_VERSION)
                self.assertEqual(len(audit["groups"]), 2)
                self.assertEqual(audit["audit_mode"], audit_mode)
                self.assertEqual(audit["generator_include_trajectory"], generator_include_trajectory)
                self.assertEqual(audit["generator_reward_source"], generator_reward_source)
                if audit_mode == "compact":
                    group = audit["groups"][0]
                    self.assertIn("bundles", group)
                    self.assertIn("generator_quality_advantage", group)
                    self.assertNotIn("transcripts", group)
                    self.assertNotIn("generator_transcripts", group)
                    self.assertNotIn("matrix", group)
                if long_transcript and full_trajectory:
                    for transcript in judge.jobs[0]["transcripts"]:
                        self.assertGreater(len(transcript), 6000)
                        self.assertIn("buy final item", transcript)
                        self.assertNotIn("evidence omitted", transcript)
                    self.assertEqual(metrics["rubric_pool/judge_transcript_truncated_frac"], 0)
                    self.assertEqual(
                        metrics["rubric_pool/generator_transcript_truncated_frac"], float(generator_budget > 0)
                    )
                    for group in audit["groups"]:
                        for a in group["generator_context_audit"]:
                            self.assertLessEqual(a["actual_input_prompt_tokens"], a["input_token_budget"])
                    if generator_budget:
                        self.assertIn("evidence omitted", audit["groups"][0]["generator_transcripts"][0])
                elif long_transcript:
                    self.assertEqual(metrics["rubric_pool/judge_transcript_truncated_frac"], 1.0)
                    self.assertTrue(all(len(t) <= 6000 for t in judge.jobs[0]["transcripts"]))
                    self.assertFalse(audit["full_trajectory"])
                if generator_loss:
                    self.assertEqual(audit["generator_update_status"], "pending")
                    self.assertNotIn("generator_loss_applied", audit)
                else:
                    self.assertFalse(audit["generator_loss_applied"])
                if fail_api:
                    self.assertTrue(all(q == 0 for q in result.non_tensor_batch["rubric_generator_quality"]))
                else:
                    first_text = result.non_tensor_batch["online_rubric"][0]
                    self.assertTrue(first_text)
                    self.assertTrue(all(x == first_text for x in result.non_tensor_batch["online_rubric"][:4]))

    def test_row_aligned_pool_generation_filtering_and_audit(self):
        self.run_adapter()

    def test_compact_audit_omits_raw_model_io(self):
        self.run_adapter(generator_loss=True, unequal_quality=True, audit_mode="compact")

    def test_cloud_failure_preserves_environment_reward_and_falls_back(self):
        self.run_adapter(fail_api=True)

    def test_joint_mode_saves_separate_token_aligned_records(self):
        self.run_adapter(generator_loss=True)

    def test_joint_mode_api_failure_has_no_quality_update(self):
        self.run_adapter(generator_loss=True, fail_api=True)

    def test_cloud_matrix_produces_signed_generator_advantages(self):
        self.run_adapter(generator_loss=True, unequal_quality=True)

    def test_selected_spearman_reward_uses_only_teacher_items(self):
        self.run_adapter(
            generator_loss=True,
            unequal_quality=True,
            generator_reward_source="selected_spearman_sum",
        )

    def test_item_credit_keeps_bundle_rollout_and_writes_token_advantages(self):
        self.run_adapter(
            generator_loss=True,
            unequal_quality=True,
            generator_reward_source="selected_spearman_sum",
            item_credit=True,
        )

    def test_loo_reward_is_source_held_out_and_reaches_item_credit_replay(self):
        self.run_adapter(
            generator_loss=True,
            unequal_quality=True,
            generator_reward_source="selected_loo_spearman_relu",
            item_credit=True,
            # The synthetic group has four trajectories, leaving three after
            # source holdout. Production N=8 uses the stricter 2/2 support.
            generator_min_satisfied=1,
            generator_min_unsatisfied=1,
        )

    def test_full_evidence_over_6000_chars_reaches_both_models(self):
        self.run_adapter(long_transcript=True, generator_loss=True)

    def test_generator_context_limit_never_shortens_cloud_evidence(self):
        self.run_adapter(long_transcript=True, generator_budget=5000, generator_loss=True)

    def test_task_only_generator_omits_trajectory_but_judge_keeps_it(self):
        self.run_adapter(long_transcript=True, generator_include_trajectory=False)

    def test_legacy_character_budget_can_be_restored(self):
        self.run_adapter(long_transcript=True, full_trajectory=False)

    def test_actual_generation_context_length_mismatch_is_not_silent(self):
        with self.assertRaisesRegex(ValueError, "differs from budgeted template"):
            self.run_adapter(template_mismatch=True)

    def test_disabled_pool_dispatch_preserves_original_code_path(self):
        import ast

        tree = ast.parse((ROOT / "verl/trainer/ppo/ray_trainer.py").read_text())
        method = next(
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef) and node.name == "_generate_online_actor_group_hints"
        )
        entry = method.body[1]  # docstring followed by opt-in dispatch
        self.assertIsInstance(entry, ast.If)
        self.assertEqual(entry.test.args[0].value, "pool_enable")
        self.assertIs(entry.test.args[1].value, False)
        self.assertIsInstance(entry.body[-1], ast.Return)


if __name__ == "__main__":
    unittest.main()
