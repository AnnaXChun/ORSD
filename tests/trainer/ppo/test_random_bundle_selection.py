"""Exercise the actual no-judge selection branch without loading CUDA/Ray."""
import ast
from pathlib import Path
from types import SimpleNamespace
import unittest

ROOT = Path(__file__).resolve().parents[3]


def select(random_mode, force_first=False, malformed=False, fixed_candidate_index=-1, malformed_second=False):
    tree = ast.parse((ROOT / 'verl/trainer/ppo/ray_trainer.py').read_text())
    branch = next(node for node in ast.walk(tree) if isinstance(node, ast.If)
                  and ast.unparse(node.test) == 'select_by_correlation'
                  and any(isinstance(child, ast.Import) for child in node.orelse))
    groups = {str(i): [] for i in range(400)}
    groups['empty'] = []
    texts = {uid: ['a', 'b', 'c', 'd'] for uid in groups}
    items = {uid: [[], ['b'], ['c'], ['d']] if malformed else [['a'], ['b'], ['c'], ['d']] for uid in groups}
    if malformed_second:
        for uid in groups:
            items[uid][1] = []
    texts['empty'] = []
    items['empty'] = []
    scope = dict(self=SimpleNamespace(config=SimpleNamespace(data={'seed': 42}), global_steps=1),
                 groups=groups, candidate_texts_by_uid=texts, candidate_bundles_by_uid=items,
                 by_uid={uid: ('', False, 0) for uid in groups}, random_candidate=random_mode,
                 force_first_candidate=force_first, fixed_candidate_index=fixed_candidate_index)
    exec(compile(ast.Module(body=branch.orelse, type_ignores=[]), '<actual-selection>', 'exec'), scope)
    return scope


class RandomBundleTests(unittest.TestCase):
    def test_fixed_second_uses_generation_index_not_first_valid(self):
        scope = select(False, malformed=True, fixed_candidate_index=1)
        self.assertTrue(all(scope['by_uid'][str(i)][0] == 'b' for i in range(400)))
        self.assertFalse(scope['by_uid']['empty'][1])

    def test_fixed_second_invalid_does_not_use_other_valid_bundles(self):
        scope = select(False, fixed_candidate_index=1, malformed_second=True)
        self.assertTrue(all(not value[1] for value in scope['by_uid'].values()))

    def test_reproducible_and_all_four_positions_used(self):
        first = select(True)
        self.assertEqual(first['by_uid'], select(True)['by_uid'])
        self.assertEqual(set(first['random_indices']), {0, 1, 2, 3})
        self.assertEqual(len(first['random_indices']), 400)
        self.assertEqual(first['by_uid']['empty'], ('', False, 0))

    def test_invalid_selected_bundle_has_no_fallback(self):
        scope = select(True, malformed=True)
        for index, chosen in enumerate(scope['random_indices']):
            self.assertEqual(scope['by_uid'][str(index)][1], chosen != 0)

    def test_existing_first_and_first_valid_routes_unchanged(self):
        self.assertFalse(select(False, force_first=True, malformed=True)['by_uid']['0'][1])
        self.assertEqual(select(False, malformed=True)['by_uid']['0'][0], 'b')


if __name__ == '__main__':
    unittest.main()
