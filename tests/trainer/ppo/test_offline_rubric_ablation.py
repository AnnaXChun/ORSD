"""CPU checks for snapshot task identity and opt-in current-group eligibility."""
import ast
import importlib.util
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[3]
spec = importlib.util.spec_from_file_location('snapshot_check', ROOT / 'examples/agent_rubric2/check_offline_rubric_snapshot.py')
validator = importlib.util.module_from_spec(spec)
spec.loader.exec_module(validator)


class Mask(np.ndarray):
    def clone(self):
        return self.copy()

    def float(self):
        return self.astype(float)


def contrast_method():
    tree = ast.parse((ROOT / 'verl/trainer/ppo/ray_trainer.py').read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'RayPPOTrainer')
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == '_filter_dataset_rubric_contrast')
    scope = {}
    exec(compile(ast.Module(body=[method], type_ignores=[]), '<actual-trainer-method>', 'exec'), scope)
    return scope[method.name]


class OfflineRubricTests(unittest.TestCase):
    def test_current_group_gate_and_existing_invalid_mask(self):
        scores = [.1, .2, .8, .9, 0., 1., 0., 1.]
        batch = SimpleNamespace(non_tensor_batch={'uid': ['fail']*2+['success']*2+['mixed']*2+['invalid']*2})
        trainer = SimpleNamespace(_rubric_outcome_score=lambda metadata, i: scores[i])
        mask = np.array([True]*6+[False]*2).view(Mask)
        filtered, metrics = contrast_method()(trainer, batch, mask, {'correctness_threshold': .6})
        self.assertEqual(filtered.tolist(), [False]*4+[True]*2+[False]*2)
        self.assertEqual(mask.tolist(), [True]*6+[False]*2)
        self.assertEqual(metrics['rubric_pipeline/dataset_contrast_eligible_frac'], .25)

    def test_disabled_contrast_preserves_eligibility(self):
        mask = np.array([True, False]).view(Mask)
        filtered, _ = contrast_method()(None, None, mask, {'require_group_outcome_contrast': False})
        self.assertEqual(filtered.tolist(), mask.tolist())

    def test_gap_gate_opt_in_does_not_change_existing_backends(self):
        tree = ast.parse((ROOT / 'verl/trainer/ppo/ray_trainer.py').read_text())
        condition = next(n.test for n in ast.walk(tree) if isinstance(n, ast.If)
                         and 'dataset_match_online_gates' in ast.unparse(n.test)
                         and 'require_outcome_aligned_gap' in ast.unparse(n.test))
        code = compile(ast.Expression(condition), '<actual-gap-gate-condition>', 'eval')
        for backend, opt_in, gap_enabled, expected in [
            ('actor_rollout_group_hints', False, True, True),
            ('actor_rollout_group_hints', False, False, False),
            ('dataset', False, True, False),
            ('dataset', True, True, True),
            ('dataset', True, False, False),
        ]:
            actual = eval(code, {'backend': backend, 'dataset_match_online_gates': opt_in,
                                 'rubric_config': {'require_outcome_aligned_gap': gap_enabled}})
            self.assertEqual(actual, expected)

    def test_snapshot_identity_and_completeness(self):
        with tempfile.TemporaryDirectory() as tmp:
            original = pa.table({'prompt': ['task A', 'task B'], 'extra_info': ['{}', '{}']})
            snapshot = original.append_column('rubric', pa.array(['bundle A', 'bundle B'])).append_column('rubric_valid', pa.array([True, True])).append_column('rubric_selected_count', pa.array([5, 8]))
            a, b = str(Path(tmp)/'train.parquet'), str(Path(tmp)/'snapshot.parquet')
            pq.write_table(original, a)
            pq.write_table(snapshot, b)
            validator.validate_snapshot(a, b)
            pq.write_table(snapshot.take([1, 0]), b)
            with self.assertRaisesRegex(ValueError, 'tasks/order/prompts'):
                validator.validate_snapshot(a, b)
            bad = snapshot.set_column(snapshot.schema.get_field_index('rubric_valid'), 'rubric_valid', pa.array([True, False]))
            pq.write_table(bad, b)
            with self.assertRaisesRegex(ValueError, 'Row 1'):
                validator.validate_snapshot(a, b)

    def test_export_string_normalization_preserves_task_identity(self):
        with tempfile.TemporaryDirectory() as tmp:
            original = pa.table({
                'data_source': pa.array(['webshop', 'webshop'], type=pa.large_string()),
                'prompt': [['task A'], ['task B']],
                'extra_info': ['{}', '{}'],
            })
            normalized = original.set_column(0, 'data_source', pa.array(['webshop', 'webshop']))
            snapshot = normalized.append_column('rubric', pa.array(['bundle A', 'bundle B'])).append_column('rubric_valid', pa.array([True, True])).append_column('rubric_selected_count', pa.array([5, 8]))
            a, b = str(Path(tmp)/'train.parquet'), str(Path(tmp)/'snapshot.parquet')
            pq.write_table(original, a)
            pq.write_table(snapshot, b)
            self.assertFalse(original.equals(normalized, check_metadata=False))
            validator.validate_snapshot(a, b)
            bad = snapshot.set_column(1, 'prompt', pa.array([['changed task'], ['task B']]))
            pq.write_table(bad, b)
            with self.assertRaisesRegex(ValueError, r'row\[0\].prompt\[0\]'):
                validator.validate_snapshot(a, b)
            pq.write_table(snapshot.slice(0, 1), b)
            with self.assertRaisesRegex(ValueError, 'row counts original=2, snapshot=1'):
                validator.validate_snapshot(a, b)

    def test_snapshot_step_mismatch_and_missing_provenance(self):
        with tempfile.TemporaryDirectory() as tmp:
            original = pa.table({'prompt': ['task'], 'extra_info': ['{}']})
            snapshot = original.append_column('rubric', pa.array(['bundle'])).append_column('rubric_valid', pa.array([True])).append_column('rubric_selected_count', pa.array([5]))
            a, b = str(Path(tmp)/'train.parquet'), str(Path(tmp)/'snapshot.parquet')
            pq.write_table(original, a)
            for step in (None, 300, 1):
                table = snapshot if step is None else snapshot.append_column('rubric_snapshot_step', pa.array([step]))
                pq.write_table(table, b)
                if step == 1:
                    validator.validate_snapshot(a, b, expected_step=1)
                else:
                    with self.assertRaisesRegex(ValueError, 'Expected rubric snapshot step 1'):
                        validator.validate_snapshot(a, b, expected_step=1)

    def test_export_stop_is_opt_in_and_requires_completed_artifact(self):
        tree = ast.parse((ROOT / 'verl/trainer/ppo/ray_trainer.py').read_text())
        method = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)
                      and n.name == '_should_stop_after_rubric_snapshot')
        scope = {}
        exec(compile(ast.Module(body=[method], type_ignores=[]), '<actual-stop-method>', 'exec'), scope)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'snapshot.parquet'
            config = {'snapshot_enable': True, 'snapshot_step': 1, 'snapshot_stop_after_export': True}
            trainer = SimpleNamespace(config=SimpleNamespace(algorithm=SimpleNamespace(token_credit=SimpleNamespace(rubric_pipeline=config))), global_steps=1, _rubric_snapshot_output_path=lambda: path)
            check = scope[method.name]
            self.assertFalse(check(trainer))
            path.touch()
            self.assertTrue(check(trainer))
            trainer.global_steps = 0
            self.assertFalse(check(trainer))
            trainer.global_steps = 1
            config['snapshot_stop_after_export'] = False
            self.assertFalse(check(trainer))

    def test_actor_injection_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            original = pa.table({'prompt': ['task'], 'extra_info': ['{"interaction_kwargs":{"inject_rubric":true}}']})
            snapshot = original.append_column('rubric', pa.array(['bundle'])).append_column('rubric_valid', pa.array([True])).append_column('rubric_selected_count', pa.array([5]))
            a,b = str(Path(tmp)/'train.parquet'),str(Path(tmp)/'snapshot.parquet')
            pq.write_table(original,a); pq.write_table(snapshot,b)
            with self.assertRaisesRegex(ValueError, 'hidden'):
                validator.validate_snapshot(a,b)


if __name__ == '__main__':
    unittest.main()
