"""Run directly with Python; no VERL, torch, Ray, GPU, or cloud API required."""

import importlib.util
import io
import json
import os
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[3]
SPEC = importlib.util.spec_from_file_location("rubric_pool_under_test", ROOT / "verl/trainer/ppo/rubric_pool.py")
pool_module = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = pool_module
SPEC.loader.exec_module(pool_module)
p = pool_module


def config(**kwargs):
    return {
        "pool_enable": True,
        "backend": "actor_rollout_group_hints",
        "pool_judge_model": "mock",
        "pool_judge_base_url": "https://example.invalid/v1",
        **kwargs,
    }


class RubricPoolTests(unittest.TestCase):
    def token_counter(self):
        # Multi-byte Unicode has a different cost here; never assume chars=tokens.
        return types.SimpleNamespace(
            encode=lambda text, **kwargs: list(text.encode()),
            apply_chat_template=lambda messages, **kwargs: list(
                ("\n".join(m["content"] for m in messages) + "\nassistant:").encode()
            ),
        )

    def test_complete_long_trajectory_kept_when_rendered_prompt_fits(self):
        transcript = "ACTION: open\nOBSERVATION: " + "visible " * 2000 + "\nACTION: buy"
        messages, used, audit = p.fit_generation_messages("task", transcript, 5, self.token_counter(), 20000)
        self.assertEqual(used, transcript)
        self.assertEqual(json.loads(messages[-1]["content"])["CURRENT_ATTEMPT"], transcript)
        self.assertFalse(audit["truncated"])
        self.assertGreater(audit["input_transcript_chars"], 6000)

    def test_generator_prompt_requests_five_to_eight_and_contains_fewshot(self):
        messages = p.build_generation_messages("task", "attempt", 8, min_count=5)
        system = messages[0]["content"]
        self.assertIn("between 5 and 8", system)
        self.assertIn("demonstration", system)
        self.assertEqual(len(messages), 4)
        self.assertEqual(json.loads(messages[2]["content"])["rubrics"], list(p.GENERATOR_FEWSHOT_RUBRICS))
        self.assertEqual(json.loads(messages[-1]["content"])["CURRENT_TASK"], "task")

    def test_simple_numbered_generator_prompt_has_no_fewshot(self):
        messages = p.build_generation_messages(
            "buy a red cup",
            "ACTION: search red cup",
            8,
            min_count=5,
            prompt_style="simple_numbered",
        )
        self.assertEqual([message["role"] for message in messages], ["system", "user"])
        self.assertIn("Write 5 to 8 short rubric criteria", messages[0]["content"])
        self.assertIn("one previous attempt", messages[0]["content"])
        self.assertNotIn("demonstration", " ".join(message["content"].lower() for message in messages))
        self.assertIn("ACTION: search red cup", messages[1]["content"])
        self.assertTrue(messages[1]["content"].endswith("OBSERVATION, or an explanation."))
        self.assertIn("only 5 to 8 numbered rubric criteria", messages[1]["content"])
        parsed, spans = p.parse_candidates_with_spans(
            "1. Search using the required attributes.\n"
            "2. Check the visible product details.\n"
            "3. Check the requested size.\n"
            "4. Select the requested color.\n"
            "5. Buy only after all options are selected.",
            max_items=8,
            min_items=5,
        )
        self.assertEqual(len(parsed), 5)
        self.assertEqual(len(spans), 5)

    def test_best_bundle_selection_keeps_one_complete_generated_group(self):
        bundles = [
            ["criterion a", "criterion b"],
            ["criterion c", "criterion d"],
        ]
        pool = p.build_pool(bundles)
        # Bundle 0 tracks the sibling outcomes; bundle 1 reverses them.  The
        # selector must return bundle 0 as a unit rather than mixing columns.
        matrix = [
            [0.1, 0.2, 0.9, 0.8],
            [0.2, 0.3, 0.8, 0.7],
            [0.8, 0.7, 0.2, 0.3],
            [0.9, 0.8, 0.1, 0.2],
        ]
        selected, diagnostics = p.select_best_bundle(
            bundles,
            pool,
            matrix,
            [0.0, 0.0, 1.0, 1.0],
        )
        self.assertEqual(selected, 0)
        self.assertTrue(diagnostics[0]["selected"])
        self.assertFalse(diagnostics[1]["selected"])
        self.assertGreater(
            diagnostics[0]["spearman_correlation"],
            diagnostics[1]["spearman_correlation"],
        )

    def test_lowest_bundle_uses_signed_correlation_and_stable_ties(self):
        bundles = [["a"], ["b"], ["c"], ["d"]]
        pool = p.build_pool(bundles)
        matrix = [[0, 1, 1, None], [0, 1, 1, None],
                  [1, 0, 0, None], [1, 0, 0, None]]
        rewards = [0, 0, 1, 1]
        selected, diagnostics = p.select_best_bundle(
            bundles, pool, matrix, rewards, select_lowest=True)
        self.assertEqual(selected, 1)
        self.assertEqual(diagnostics[1]["spearman_correlation"], -1)
        self.assertEqual(diagnostics[1]["selection_reason"], "lowest_correlation")
        self.assertEqual(diagnostics[3]["reason"], "insufficient_judge_coverage")
        self.assertEqual(p.select_best_bundle(bundles, pool, matrix, rewards)[0], 0)
        self.assertIsNone(p.select_best_bundle(
            bundles, pool, [[None] * 4 for _ in rewards], rewards, select_lowest=True)[0])
        self.assertIsNone(p.select_best_bundle(
            bundles, pool, [[0.5] * 4 for _ in rewards], rewards, select_lowest=True)[0])
        with self.assertRaises(ValueError):
            p.select_best_bundle(bundles, pool, matrix, rewards,
                                 select_lowest=True, baseline_index=0)

    def test_best_bundle_selection_falls_back_when_no_bundle_can_be_ranked(self):
        bundles = [["criterion a"], ["criterion b"]]
        pool = p.build_pool(bundles)
        selected, diagnostics = p.select_best_bundle(
            bundles,
            pool,
            [[0.5, 0.5], [0.5, 0.5], [0.5, 0.5], [0.5, 0.5]],
            [0.0, 0.0, 1.0, 1.0],
        )
        self.assertIsNone(selected)
        self.assertTrue(all(item["reason"] == "constant_bundle_scores" for item in diagnostics))

    def test_guarded_bundle_selection_replaces_baseline_only_for_clear_improvement(self):
        bundles = [["baseline criterion"], ["better criterion"]]
        pool = p.build_pool(bundles)
        selected, diagnostics = p.select_best_bundle(
            bundles,
            pool,
            [
                [0.2, 0.1],
                [0.8, 0.2],
                [0.3, 0.8],
                [0.7, 0.9],
            ],
            [0.0, 0.0, 1.0, 1.0],
            baseline_index=0,
            min_selected_correlation=0.0,
            min_correlation_improvement=0.05,
        )
        self.assertEqual(selected, 1)
        self.assertEqual(diagnostics[1]["selection_reason"], "guard_replaced_baseline")

    def test_guarded_bundle_selection_keeps_baseline_when_margin_is_not_met(self):
        bundles = [["baseline criterion"], ["candidate criterion"]]
        pool = p.build_pool(bundles)
        selected, diagnostics = p.select_best_bundle(
            bundles,
            pool,
            [
                [0.1, 0.1],
                [0.2, 0.2],
                [0.8, 0.8],
                [0.9, 0.9],
            ],
            [0.0, 0.0, 1.0, 1.0],
            baseline_index=0,
            min_selected_correlation=0.0,
            min_correlation_improvement=0.05,
        )
        self.assertEqual(selected, 0)
        self.assertEqual(diagnostics[0]["selection_reason"], "guard_kept_baseline")

    def test_guarded_bundle_selection_never_replaces_baseline_with_negative_correlation(self):
        bundles = [["baseline criterion"], ["less negative criterion"]]
        pool = p.build_pool(bundles)
        selected, diagnostics = p.select_best_bundle(
            bundles,
            pool,
            [
                [0.9, 0.8],
                [0.8, 0.4],
                [0.2, 0.7],
                [0.1, 0.3],
            ],
            [0.0, 0.0, 1.0, 1.0],
            baseline_index=0,
            min_selected_correlation=0.0,
            min_correlation_improvement=0.05,
        )
        self.assertLess(diagnostics[1]["spearman_correlation"], 0.0)
        self.assertEqual(selected, 0)
        self.assertEqual(diagnostics[0]["selection_reason"], "guard_kept_baseline")

    def test_task_only_generator_prompt_omits_current_attempt(self):
        messages = p.build_generation_messages(
            "find a red medium mug",
            "ACTION: inspect item",
            8,
            min_count=5,
            include_trajectory=False,
        )
        current = json.loads(messages[-1]["content"])
        self.assertEqual(current, {"CURRENT_TASK": "find a red medium mug"})
        self.assertIn("without seeing a previous attempt", messages[0]["content"])
        self.assertNotIn("ACTION: inspect item", json.dumps(messages))

        _, used, audit = p.fit_generation_messages(
            "find a red medium mug",
            "ACTION: inspect item",
            8,
            self.token_counter(),
            20000,
            min_count=5,
            include_trajectory=False,
        )
        self.assertEqual(used, "")
        self.assertFalse(audit["trajectory_conditioned"])
        self.assertEqual(audit["input_transcript_tokens"], 0)

    def test_fewshot_copy_rejects_entire_bundle_including_legacy_paraphrase(self):
        clean = [f"A task-grounded criterion {index}." for index in range(5)]
        accepted, audit = p.reject_fewshot_copy(clean)
        self.assertEqual(accepted, clean)
        self.assertTrue(audit["accepted"])

        copied = clean[:-1] + [p.GENERATOR_FEWSHOT_RUBRICS[0]]
        accepted, audit = p.reject_fewshot_copy(copied)
        self.assertEqual(accepted, [])
        self.assertEqual(audit["reason"], "fewshot_copy")

        legacy_paraphrase = clean[:-1] + [
            "The required brown color is visibly available and selected before purchase."
        ]
        accepted, audit = p.reject_fewshot_copy(legacy_paraphrase)
        self.assertEqual(accepted, [])
        self.assertGreaterEqual(audit["max_fewshot_similarity"], 0.82)

    def test_token_budget_counts_template_json_and_unicode(self):
        transcript = "INITIAL OBSERVATION: start\n" + '商品属性\\"\n' * 3000 + "\nACTION: click[Buy Now]"
        messages, used, audit = p.fit_generation_messages("task", transcript, 5, self.token_counter(), 4000)
        self.assertTrue(audit["truncated"])
        self.assertLessEqual(audit["input_prompt_tokens"], 4000)
        self.assertEqual(audit["input_prompt_tokens"], len(self.token_counter().apply_chat_template(messages)))
        self.assertTrue(used.startswith("INITIAL OBSERVATION: start"))
        self.assertTrue(used.endswith("ACTION: click[Buy Now]"))
        self.assertIn("evidence omitted", used)
        self.assertNotIn("\ufffd", used)

    def test_no_input_budget_fails_before_model_call(self):
        with self.assertRaises(ValueError):
            p.fit_generation_messages("task", "trajectory", 5, self.token_counter(), 0)
        with self.assertRaisesRegex(ValueError, "instructions exceed"):
            p.fit_generation_messages("task" * 1000, "trajectory", 5, self.token_counter(), 20)

    def test_parse_strips_thinking_and_supports_working_numbered_format(self):
        self.assertEqual(
            p.parse_candidates('<think>reasoning</think>{"rubrics":["Inspect selected size."]}', max_items=1),
            ["Inspect selected size."],
        )
        self.assertEqual(
            p.parse_candidates("reasoning</think>1. Inspect size.\n2. Inspect price.", max_items=2),
            ["Inspect size.", "Inspect price."],
        )
        self.assertEqual(p.parse_candidates('<think>{"rubrics":["bad"]}'), [])

    def test_parser_retains_json_and_numbered_item_spans(self):
        raw = '<think>hidden</think>{"rubrics":["Inspect size.", "Check price."]}'
        items, spans = p.parse_candidates_with_spans(raw, min_items=2, max_items=2)
        self.assertEqual(items, ["Inspect size.", "Check price."])
        self.assertEqual([raw[start:stop] for start, stop in spans], items)

        raw = "reasoning</think>1. Inspect size.\n2. Check price."
        items, spans = p.parse_candidates_with_spans(raw, min_items=2, max_items=2)
        self.assertEqual([raw[start:stop] for start, stop in spans], items)

    def test_oversized_and_malformed_candidates_not_truncated_into_false_rules(self):
        self.assertEqual(p.parse_candidates('{"rubrics":["abcdef"]}', max_items=1, max_item_chars=5), [])
        self.assertEqual(p.parse_candidates('{"rubrics":["a","b"]}', max_items=1), [])
        self.assertEqual(p.parse_candidates("Some commentary\n1. foo"), [])

    def test_parser_requires_between_five_and_eight_items(self):
        self.assertEqual(p.parse_candidates('{"rubrics":["one"]}', max_items=2), [])
        self.assertEqual(p.parse_candidates('{"rubrics":["one", 2]}', max_items=2), [])
        self.assertEqual(p.parse_candidates('{"rubrics":["one", "two"]}', max_items=2), ["one", "two"])
        five = [f"item-{i}" for i in range(5)]
        eight = [f"item-{i}" for i in range(8)]
        self.assertEqual(p.parse_candidates(json.dumps({"rubrics": five}), max_items=8), five)
        self.assertEqual(p.parse_candidates(json.dumps({"rubrics": eight}), max_items=8), eight)
        self.assertEqual(p.parse_candidates(json.dumps({"rubrics": five[:4]}), max_items=8), [])
        self.assertEqual(p.parse_candidates(json.dumps({"rubrics": eight + ["ninth"]}), max_items=8), [])

    def test_dedup_preserves_sources(self):
        items = p.build_pool([["Inspect size.", "Inspect size."], ["INSPECT size"], ["Inspect price."]])
        self.assertEqual(len(items), 2)
        self.assertEqual(items[0].sources, [0, 1])
        self.assertEqual(items[1].sources, [2])

    def test_positive_constant_and_inverse_correlation(self):
        items = p.build_pool([["positive", "constant", "inverse"], [], [], [], []])
        rewards = [1, 0, 1, 0, 1]
        matrix = [[r, 1, 1 - r] for r in rewards]
        text, stats = p.select_pool(items, matrix, rewards)
        self.assertEqual(text, "1. positive")
        self.assertEqual(stats[0]["evaluation_indices"], [0, 1, 2, 3, 4])
        self.assertEqual(stats[0]["correlation_method"], "spearman")
        self.assertAlmostEqual(stats[0]["correlation"], 1)
        self.assertAlmostEqual(stats[0]["spearman_correlation"], 1)
        self.assertAlmostEqual(stats[0]["pearson_correlation"], 1)
        self.assertTrue(stats[0]["valid"])
        self.assertGreater(stats[0]["score_std"], 0)
        self.assertEqual(stats[1]["reason"], "constant_scores")
        self.assertFalse(stats[1]["valid"])
        self.assertEqual(stats[2]["reason"], "weak_or_reversed_spearman")
        self.assertFalse(stats[2]["valid"])

    def test_validity_requires_strictly_greater_than_alpha(self):
        items = p.build_pool([["boundary"], [], [], [], []])
        rewards = [0, 1, 2, 3, 4]
        scores = [[0], [0.5], [0.25], [1], [0.75]]
        stats_correlation = p.spearman([row[0] for row in scores], rewards)
        _, stats = p.select_pool(items, scores, rewards, min_correlation=stats_correlation)
        self.assertAlmostEqual(stats[0]["spearman_correlation"], stats_correlation)
        self.assertFalse(stats[0]["valid"])

    def test_nearly_constant_scores_fail_even_with_perfect_rank_correlation(self):
        items = p.build_pool([["generic criterion"], [], [], [], []])
        rewards = [0, 1, 2, 3, 4]
        scores = [[0.90], [0.91], [0.92], [0.93], [0.94]]
        _, stats = p.select_pool(items, scores, rewards)
        self.assertAlmostEqual(stats[0]["spearman_correlation"], 1.0)
        self.assertEqual(stats[0]["reason"], "insufficient_score_std")
        self.assertFalse(stats[0]["valid"])

    def test_score_range_gate_rejects_small_effect_despite_sufficient_std(self):
        items = p.build_pool([["small effect"], [], [], [], []])
        rewards = [0, 1, 2, 3, 4]
        scores = [[0.0], [0.0], [0.19], [0.19], [0.19]]
        _, stats = p.select_pool(items, scores, rewards)
        self.assertGreaterEqual(stats[0]["score_std"], 0.08)
        self.assertEqual(stats[0]["reason"], "insufficient_score_range")
        self.assertFalse(stats[0]["valid"])

    def test_behavior_dedup_keeps_complementary_valid_criteria(self):
        items = p.build_pool([["best", "same-ranking", "complementary"], [], [], [], [], []])
        rewards = [0, 1, 2, 3, 4, 5]
        columns = [
            [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
            [0.0, 0.1, 0.3, 0.7, 0.9, 1.0],
            [0.0, 0.4, 0.2, 0.8, 1.0, 0.6],
        ]
        matrix = [list(row) for row in zip(*columns, strict=True)]
        text, stats = p.select_pool(
            items,
            matrix,
            rewards,
            behavior_dedup_enabled=True,
            max_behavior_similarity=0.9,
        )
        self.assertEqual(text, "1. best\n2. complementary")
        self.assertTrue(stats[1]["valid"])
        self.assertFalse(stats[1]["selected"])
        self.assertEqual(stats[1]["reason"], "redundant_behavior")
        self.assertEqual(stats[1]["redundant_with"], "r000")
        self.assertAlmostEqual(stats[1]["max_selected_similarity"], 1.0)
        self.assertTrue(stats[2]["selected"])

    def test_behavior_dedup_is_opt_in(self):
        items = p.build_pool([["first", "duplicate-behavior"], [], [], [], []])
        rewards = [0, 1, 0, 1, 0]
        matrix = [[reward, reward] for reward in rewards]
        text, stats = p.select_pool(items, matrix, rewards)
        self.assertEqual(text, "1. first\n2. duplicate-behavior")
        self.assertTrue(all(item["selected"] for item in stats))

    def test_source_trajectory_is_included_in_correlation(self):
        items = p.build_pool([["self-only"], [], [], [], []])
        rewards = [1, 0, 0, 1, 0]
        matrix = [[1], [0], [0], [0], [0]]
        text, stats = p.select_pool(items, matrix, rewards)
        self.assertEqual(text, "1. self-only")
        self.assertIn(0, stats[0]["evaluation_indices"])
        self.assertGreater(stats[0]["spearman_correlation"], 0.2)

    def test_uniform_groups_cannot_generate_credit(self):
        items = p.build_pool([["rule"], [], [], []])
        for rewards in ([1, 1, 1, 1], [0, 0, 0, 0]):
            text, stats = p.select_pool(items, [[1], [0], [1], [0]], rewards)
            self.assertEqual(text, "")
            self.assertEqual(stats[0]["reason"], "uniform_outcome_group")

    def test_continuous_reward_uses_spearman_for_selection(self):
        rewards = [1, 0.1, 0.4, 0.6, 0.9]
        items = p.build_pool([["graded"], [], [], [], []])
        nonlinear_scores = [[r**3] for r in rewards]
        text, stats = p.select_pool(items, nonlinear_scores, rewards)
        self.assertEqual(text, "1. graded")
        self.assertAlmostEqual(stats[0]["spearman_correlation"], 1)
        self.assertLess(stats[0]["pearson_correlation"], 1)

    def test_unknown_evidence_is_not_zero(self):
        items = p.build_pool([["rule"], [], [], [], []])
        text, stats = p.select_pool(items, [[1], [0], [None], [1], [0]], [1, 0, 0, 1, 0])
        self.assertEqual(text, "")
        self.assertEqual(stats[0]["reason"], "insufficient_judge_coverage")

    def test_partial_judge_matrix_uses_only_observed_trajectories(self):
        items = p.build_pool([["rule"], [], [], [], [], [], [], []])
        rewards = [1, 0, 1, 0, 1, 0, 1, 0]
        matrix = [[reward] for reward in rewards]
        matrix[1][0] = None
        matrix[6][0] = None
        text, stats = p.select_pool(items, matrix, rewards, min_judge_coverage=0.75)
        self.assertEqual(text, "1. rule")
        self.assertEqual(stats[0]["evaluation_indices"], [0, 2, 3, 4, 5, 7])
        self.assertTrue(stats[0]["valid"])

    def test_partial_judge_matrix_still_requires_outcome_contrast(self):
        items = p.build_pool([["rule"], [], [], [], [], [], [], []])
        rewards = [1, 0, 1, 0, 1, 0, 1, 0]
        matrix = [[1.0] if reward else [None] for reward in rewards]
        _, stats = p.select_pool(items, matrix, rewards, min_judge_coverage=0.5)
        self.assertEqual(stats[0]["reason"], "insufficient_outcome_contrast")
        self.assertFalse(stats[0]["valid"])

    def test_shared_sources_are_all_evaluated(self):
        items = p.build_pool([["same"], ["same"], ["same"], ["same"]])
        text, stats = p.select_pool(items, [[1], [0], [1], [0]], [1, 0, 1, 0])
        self.assertEqual(text, "1. same")
        self.assertEqual(stats[0]["evaluation_indices"], [0, 1, 2, 3])

    def test_teacher_budget_never_cuts_an_item_mid_sentence(self):
        items = p.build_pool([["A long complete criterion"], [], [], [], []])
        text, stats = p.select_pool(items, [[1], [0], [1], [0], [1]], [1, 0, 1, 0, 1], max_chars=5)
        self.assertEqual(text, "")
        self.assertEqual(stats[0]["reason"], "teacher_budget")

    def test_bad_matrix_raises(self):
        items = p.build_pool([["rule"], [], [], []])
        with self.assertRaises(ValueError):
            p.select_pool(items, [[1, 0, 1, 0]], [1, 0, 1, 0])
        for bad in (float("nan"), float("inf"), -1, 2, True):
            with self.assertRaises(ValueError):
                p.select_pool(items, [[bad], [0], [1], [0]], [1, 0, 1, 0])

    def test_generator_reward_is_valid_fraction_not_correlation_mean(self):
        bundles = [
            ["valid-a", "valid-a", "invalid-a", "invalid-b", "invalid-c"],
            ["valid-a", "valid-b", "invalid-a", "invalid-b", "invalid-c", "invalid-d", "invalid-e", "invalid-f"],
            [],
        ]
        items = p.build_pool(bundles)
        stats = [{"valid": item.text.startswith("valid")} for item in items]
        quality = p.generator_quality(bundles, items, stats)
        self.assertAlmostEqual(quality[0], 1 / 5)  # duplicate occupies K twice but counts valid once
        self.assertAlmostEqual(quality[1], 2 / 8)
        self.assertEqual(quality[2], 0)
        self.assertEqual(p.generator_quality(bundles, items, stats, valid_group=False), [0.0] * 3)

    def test_selected_reward_credits_only_teacher_context_items(self):
        bundles = [
            ["selected", "valid-only", "bad-a", "bad-b", "bad-c"],
            ["valid-only", "bad-d", "bad-e", "bad-f", "bad-g"],
        ]
        items = p.build_pool(bundles)
        stats = [
            {
                "valid": item.text in {"selected", "valid-only"},
                "selected": item.text == "selected",
                "spearman_correlation": 0.75 if item.text == "selected" else 0.5,
            }
            for item in items
        ]
        self.assertEqual(p.generator_quality(bundles, items, stats), [0.4, 0.2])
        self.assertEqual(
            p.generator_quality(bundles, items, stats, reward_source="selected"),
            [0.2, 0.0],
        )
        self.assertEqual(
            p.generator_quality(bundles, items, stats, reward_source="selected_spearman_sum"),
            [0.75, 0.0],
        )
        with self.assertRaisesRegex(ValueError, "valid, selected"):
            p.generator_quality(bundles, items, stats, reward_source="correlation")

    def test_continuous_loo_generator_quality_excludes_source_and_requires_support(self):
        bundles = [["strong", "strong"], [], [], [], []]
        pool = p.build_pool(bundles)
        diagnostics = [{"selected": True}]
        rewards = [999.0, 0.0, 0.0, 1.0, 1.0]
        matrix = [[999.0], [0.1], [0.2], [0.8], [0.9]]

        bundle_quality, item_quality, stats = p.leave_one_out_generator_quality(
            bundles,
            pool,
            matrix,
            rewards,
            diagnostics,
            min_correlation=0.4,
            support_threshold=0.5,
            min_satisfied=2,
            min_unsatisfied=2,
        )
        expected = max(0.0, p.spearman([0.1, 0.2, 0.8, 0.9], [0.0, 0.0, 1.0, 1.0]) - 0.4)
        self.assertAlmostEqual(item_quality[0][0], expected)
        self.assertEqual(item_quality[0][1], 0.0)
        self.assertAlmostEqual(bundle_quality[0], expected / 2)
        self.assertEqual(stats["support_pass_items"], 1)
        self.assertEqual(stats["positive_quality_items"], 1)

        # The source row is never used to grade its own rubric.
        changed_quality, _, _ = p.leave_one_out_generator_quality(
            bundles,
            pool,
            [[-999.0], *matrix[1:]],
            [-999.0, *rewards[1:]],
            diagnostics,
            min_correlation=0.4,
        )
        self.assertEqual(changed_quality, bundle_quality)

        # Three satisfied and one unsatisfied held-out trajectory fails 2/2 support.
        unsupported_quality, _, unsupported_stats = p.leave_one_out_generator_quality(
            bundles,
            pool,
            [[0.0], [1.0], [1.0], [1.0], [0.0]],
            rewards,
            diagnostics,
            min_correlation=0.4,
        )
        self.assertEqual(unsupported_quality, [0.0] * len(bundles))
        self.assertEqual(unsupported_stats["support_pass_items"], 0)

    def test_item_credit_is_selected_only_and_normalized_within_task(self):
        bundles = [["selected", "duplicate", "duplicate"], ["rejected", "second"]]
        items = p.build_pool(bundles)
        stats = [
            {
                "valid": item.text in {"selected", "duplicate", "second"},
                "selected": item.text in {"selected", "second"},
                "spearman_correlation": {"selected": 0.8, "second": 0.5}.get(item.text),
            }
            for item in items
        ]
        quality = p.generator_item_quality(
            bundles,
            items,
            stats,
            reward_source="selected_spearman_sum",
        )
        self.assertEqual(quality, [[0.8, 0.0, 0.0], [0.0, 0.5]])
        advantages, advantage_stats = p.reinforcepp_item_advantages(quality, [[0, 1]])
        self.assertGreater(advantages[0][0], 0)
        self.assertGreater(advantages[1][1], 0)
        self.assertTrue(all(advantages[row][local] < 0 for row, local in ((0, 1), (0, 2), (1, 0))))
        self.assertEqual(advantage_stats["active_group_count"], 1)
        self.assertEqual(advantage_stats["active_bundle_count"], 2)
        self.assertEqual(advantage_stats["active_item_count"], 5)

    def test_shared_criterion_gives_full_reward_to_each_source(self):
        bundles = [
            [
                "shared",
                f"invalid-{source}-a",
                f"invalid-{source}-b",
                f"invalid-{source}-c",
                f"invalid-{source}-d",
            ]
            for source in range(4)
        ]
        items = p.build_pool(bundles)
        stats = [{"valid": item.text == "shared"} for item in items]
        self.assertEqual(p.generator_quality(bundles, items, stats), [0.2] * 4)

    def test_reinforcepp_same_task_baseline_then_batch_normalization(self):
        quality = [0.2, -0.2, 0, 0, 0.9, 0.3, 0.6, 0.6]
        advantages, stats = p.reinforcepp_batch_advantages(quality, [[0, 1, 2, 3], [4, 5, 6, 7]])
        self.assertAlmostEqual(sum(advantages), 0)
        self.assertAlmostEqual(sum(value * value for value in advantages) / len(advantages), 1)
        self.assertEqual(stats["active_group_count"], 2)
        self.assertEqual(stats["active_bundle_count"], 8)
        self.assertAlmostEqual(sum(advantages[:4]), 0)
        self.assertAlmostEqual(sum(advantages[4:]), 0)

    def test_reinforcepp_skips_groups_without_relative_bundle_signal(self):
        advantages, stats = p.reinforcepp_batch_advantages([0.2] * 4 + [1, -1], [[0, 1, 2, 3], [4, 5]])
        self.assertEqual(advantages[:4], [0.0] * 4)
        self.assertGreater(advantages[4], 0)
        self.assertLess(advantages[5], 0)
        self.assertEqual(stats["active_group_count"], 1)

    def test_judge_ids_allow_reordering_but_never_duplicates(self):
        raw = '{"scores":[{"id":"r1","score":0},{"id":"r0","score":1}]}'
        self.assertEqual(p.parse_judge_response(raw, ["r0", "r1"]), [1, 0])
        with self.assertRaises(ValueError):
            p.parse_judge_response(raw.replace('"r1"', '"r0"'), ["r0", "r1"])

    def test_invalid_judge_cells_rejected(self):
        for score in (True, -1, 2, float("nan"), "1"):
            with self.assertRaises(ValueError):
                p.parse_judge_response(json.dumps({"scores": [{"id": "r0", "score": score}]}), ["r0"])
        with self.assertRaises(ValueError):
            p.parse_judge_response('{"scores":[{"id":"r0"}]}', ["r0"])
        self.assertIsNone(p.parse_judge_response('{"scores":[{"id":"r0","score":null}]}', ["r0"])[0])

    def test_score_labels_removed_from_judge_input(self):
        text = "ACTION: click[Buy Now]\nofficial_reward: 1.0\nYour score (min 0.0, max 1.0): 0.9"
        clean = p.clean_transcript(text)
        self.assertIn("click[Buy Now]", clean)
        self.assertNotIn("official_reward", clean)
        self.assertNotIn("0.9", clean)
        messages = p.build_generation_messages("task", text, 5)
        self.assertNotIn("official_reward", messages[1]["content"])

    def test_invalid_config_rejected(self):
        p.validate_pool_config({"pool_enable": False})
        p.validate_pool_config(config())
        for updates in (
            {"pool_judge_model": ""},
            {"pool_judge_base_url": "http://example.com/v1"},
            {"pool_min_correlation": 0},
            {"pool_min_selected": 9},
            {"pool_min_rubrics_per_bundle": 4},
            {"rubrics_per_prompt": 9},
            {"backend": "dataset"},
            {"pool_generator_max_input_tokens": -1},
            {"pool_generator_context_reserve_tokens": 0.5},
            {"pool_audit_mode": "verbose"},
            {"pool_audit_every_n_steps": 0},
            {"pool_min_score_std": 0},
            {"pool_min_score_range": 1.1},
            {"pool_fewshot_copy_similarity": 0},
            {"pool_max_behavior_similarity": 0},
            {"pool_min_judge_coverage": 0},
            {"pool_behavior_dedup_enable": "True"},
            {"pool_generator_include_trajectory": "False"},
            {"pool_generator_item_credit_enable": "True"},
            {"pool_generator_item_credit_enable": True, "pool_generator_reward_source": "valid"},
            {"pool_generator_reward_source": "all"},
            {"pool_generator_support_threshold": 1.0},
            {"pool_generator_min_satisfied": 0},
            {"pool_generator_min_unsatisfied": 1.5},
            {"pool_generator_linear_decay_enable": "True"},
            {"pool_generator_linear_decay_final_scale": 1.1},
            {"pool_generator_linear_decay_start_step": -1},
            {"pool_generator_linear_warmup_enable": "True"},
            {"pool_generator_linear_warmup_enable": True},
            {"pool_generator_linear_warmup_initial_scale": 1.1},
            {
                "pool_generator_linear_decay_enable": True,
                "pool_generator_linear_warmup_enable": True,
                "pool_generator_linear_warmup_steps": 150,
            },
        ):
            with self.assertRaises(ValueError):
                p.validate_pool_config(config(**updates))
        p.validate_pool_config(
            config(pool_generator_item_credit_enable=True, pool_generator_reward_source="selected_spearman_sum")
        )
        p.validate_pool_config(
            config(pool_generator_prompt_style="simple_numbered", pool_min_rubrics_per_bundle=5)
        )
        with self.assertRaises(ValueError):
            p.validate_pool_config(
                config(
                    pool_generator_prompt_style="simple_numbered",
                    pool_generator_include_trajectory=False,
                    pool_min_rubrics_per_bundle=5,
                )
            )
        p.validate_pool_config(
            config(
                pool_generator_item_credit_enable=True,
                pool_generator_reward_source="selected_loo_spearman_relu",
                pool_generator_support_threshold=0.5,
                pool_generator_min_satisfied=2,
                pool_generator_min_unsatisfied=2,
                pool_generator_linear_decay_enable=True,
                pool_generator_linear_decay_final_scale=0.0,
                pool_generator_linear_decay_start_step=200,
            )
        )
        p.validate_pool_config(
            config(
                pool_generator_item_credit_enable=True,
                pool_generator_reward_source="selected_loo_spearman_relu",
                pool_generator_linear_warmup_enable=True,
                pool_generator_linear_warmup_steps=150,
                pool_generator_linear_warmup_initial_scale=0.0,
            )
        )

    def test_missing_key_fails_before_cloud_request(self):
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(ValueError, "DASHSCOPE_API_KEY"):
                p.CloudRubricJudge(config())

    def test_http_payload_has_no_labels_or_sources(self):
        with patch.dict(os.environ, {"DASHSCOPE_API_KEY": "unit-test-only"}):
            judge = p.CloudRubricJudge(config())
        content = json.dumps({"scores": [{"id": "r000", "score": 1}]})
        reply = io.BytesIO(
            json.dumps({"choices": [{"message": {"content": content}, "finish_reason": "stop"}]}).encode()
        )
        with patch.object(p.request, "urlopen", return_value=reply) as call:
            cells = judge._request("task", "ACTION: open drawer\nofficial_reward: 1", p.build_pool([["rule"]]))
        self.assertEqual(cells[0], 1)
        req = call.call_args.args[0]
        body = json.loads(req.data)
        self.assertFalse(body["enable_thinking"])
        user_input = json.loads(body["messages"][1]["content"])
        self.assertEqual(set(user_input), {"task", "trajectory", "rubrics"})
        self.assertNotIn("official_reward", user_input["trajectory"])
        self.assertNotIn("sources", user_input["rubrics"][0])
        self.assertNotIn("unit-test-only", json.dumps(body))
        self.assertNotIn('"evidence"', body["messages"][0]["content"].lower())

    def test_http_unauthorized_not_retried_or_echoed(self):
        with patch.dict(os.environ, {"DASHSCOPE_API_KEY": "unit-test-only"}):
            judge = p.CloudRubricJudge(config())
        exc = p.error.HTTPError("https://example.invalid/v1", 401, "contains-sensitive-body", {}, None)
        with patch.object(p.request, "urlopen", side_effect=exc) as call:
            with self.assertRaisesRegex(RuntimeError, "HTTP 401") as raised:
                judge._request("task", "trajectory", p.build_pool([["rule"]]))
        self.assertEqual(call.call_count, 1)
        self.assertNotIn("sensitive", str(raised.exception))

    def test_cloud_payload_keeps_full_long_visible_trajectory(self):
        with patch.dict(os.environ, {"DASHSCOPE_API_KEY": "unit-test-only"}):
            judge = p.CloudRubricJudge(config())
        transcript = "ACTION: search\nOBSERVATION: " + "all product evidence " * 4000 + "\nACTION: click[Buy Now]"
        content = json.dumps({"scores": [{"id": "r000", "score": 1}]})
        reply = io.BytesIO(json.dumps({"choices": [{"message": {"content": content}}]}).encode())
        with patch.object(p.request, "urlopen", return_value=reply) as call:
            judge._request("task", transcript, p.build_pool([["rule"]]))
        user_input = json.loads(json.loads(call.call_args.args[0].data)["messages"][1]["content"])
        self.assertEqual(user_input["trajectory"], transcript)

    def test_cloud_context_overflow_falls_back_without_truncation_or_retry(self):
        with patch.dict(os.environ, {"DASHSCOPE_API_KEY": "unit-test-only"}):
            judge = p.CloudRubricJudge(config())
        exc = p.error.HTTPError("https://example.invalid/v1", 400, "context length exceeded", {}, None)
        with patch.object(p.request, "urlopen", side_effect=exc) as call:
            result = judge.score_groups(
                [
                    {
                        "uid": "overflow",
                        "problem": "task",
                        "transcripts": ["large trajectory"],
                        "pool": p.build_pool([["rule"]]),
                    }
                ]
            )
        self.assertEqual(call.call_count, 1)
        self.assertEqual(result["overflow"]["matrix"], [[None]])
        self.assertEqual(result["overflow"]["failed_requests"], 1)
        self.assertEqual(result["overflow"]["error"], "Cloud rubric judge HTTP 400")

    def test_http_transient_error_retried(self):
        with patch.dict(os.environ, {"DASHSCOPE_API_KEY": "unit-test-only"}):
            judge = p.CloudRubricJudge(config(pool_judge_retries=1))
        content = json.dumps({"scores": [{"id": "r000", "score": 0}]})
        reply = io.BytesIO(json.dumps({"choices": [{"message": {"content": content}}]}).encode())
        exc = p.error.HTTPError("https://example.invalid/v1", 429, "rate limit", {}, None)
        with patch.object(p.request, "urlopen", side_effect=[exc, reply]) as call, patch.object(p.time, "sleep"):
            self.assertEqual(judge._request("task", "trajectory", p.build_pool([["rule"]]))[0], 0)
        self.assertEqual(call.call_count, 2)

    def test_final_judge_failure_reason_distinguishes_response_and_transport(self):
        with patch.dict(os.environ, {"DASHSCOPE_API_KEY": "unit-test-only"}):
            judge = p.CloudRubricJudge(config(pool_judge_retries=0))
        invalid = io.BytesIO(b'{"choices": []}')
        with patch.object(p.request, "urlopen", return_value=invalid):
            with self.assertRaisesRegex(RuntimeError, "invalid_response"):
                judge._request("task", "trajectory", p.build_pool([["rule"]]))
        with patch.object(p.request, "urlopen", side_effect=TimeoutError()):
            with self.assertRaisesRegex(RuntimeError, "transport_or_timeout"):
                judge._request("task", "trajectory", p.build_pool([["rule"]]))

    def test_concurrent_matrix_scatter_and_group_failure(self):
        with patch.dict(os.environ, {"DASHSCOPE_API_KEY": "unit-test-only"}):
            judge = p.CloudRubricJudge(config(pool_judge_items_per_request=1))
        items = p.build_pool([["a", "b"]])

        def fake_request(problem, transcript, selected):
            if problem == "fail":
                raise RuntimeError("mock API failure")
            if problem == "partial" and transcript == "missing":
                if selected[0].text == "b":
                    raise RuntimeError("mock partial failure")
                return [0.0]
            return [float(transcript) for _ in selected]

        judge._request = fake_request
        result = judge.score_groups(
            [
                {"uid": "good", "problem": "ok", "transcripts": ["1", "0"], "pool": items},
                {"uid": "bad", "problem": "fail", "transcripts": ["1", "0"], "pool": items},
                {
                    "uid": "partial",
                    "problem": "partial",
                    "transcripts": ["1", "0", "1", "missing"],
                    "pool": items,
                },
            ]
        )
        self.assertEqual(result["good"]["matrix"], [[1.0, 1.0], [0.0, 0.0]])
        self.assertEqual(result["good"]["requests"], 4)
        self.assertEqual(result["bad"]["matrix"], [[None, None], [None, None]])
        self.assertEqual(result["bad"]["failed_requests"], 4)
        self.assertIn("failure", result["bad"]["error"])
        self.assertEqual(result["partial"]["matrix"], [[1.0, 1.0], [0.0, 0.0], [1.0, 1.0], [0.0, None]])
        self.assertEqual(result["partial"]["failed_requests"], 1)


if __name__ == "__main__":
    unittest.main()
