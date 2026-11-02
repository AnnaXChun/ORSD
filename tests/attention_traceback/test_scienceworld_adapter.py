"""Contract tests only; real simulator smoke is still required by run_all.sh."""
import asyncio
import sys
from types import SimpleNamespace

from verl.interactions import scienceworld_interaction as module


def test_adapter_preserves_real_step_fields_and_disables_gold(monkeypatch):
    calls = []

    class Env:
        def __init__(self, **kwargs):
            pass

        def load(self, name, variation, simplifications, generateGoldPath):
            calls.append((name, variation, simplifications, generateGoldPath))

        def reset(self):
            return 'water', {'score': 0, 'valid': ['heat water'], 'taskDesc': 'Boil water'}

        def get_possible_actions(self):
            return ['heat OBJ', 'look around']

        def step(self, action):
            assert action == 'heat water'
            return 'steam', 25, False, {'score': 25, 'valid': ['look around']}

        def close(self):
            calls.append('closed')

    monkeypatch.setattr(module, 'version', lambda _: '1.2.3')
    monkeypatch.setitem(sys.modules, 'scienceworld', SimpleNamespace(ScienceWorldEnv=Env))

    async def run():
        interaction = module.ScienceworldInteraction({})
        await interaction.start_interaction('a', task_name='boil', variation_idx=7, max_steps=1)
        initial = await interaction.get_initial_observation('a')
        assert 'heat OBJ' in initial and 'heat water' not in initial
        done, observation, reward, metrics = await interaction.generate_response(
            'a', [{'role': 'assistant', 'content': 'ACTION: heat water'}])
        assert done and reward == 25 and 'steam' in observation
        assert 'heat OBJ' not in observation
        assert metrics['observation_style'] == 'compact'
        assert metrics['task_name'] == 'boil' and metrics['variation_idx'] == 7
        assert metrics['score'] == 25 and metrics['done_reason'] == 'max_steps'
        assert metrics['valid_actions'] == ['look around']
        assert await interaction.calculate_score('a') == .25
        await interaction.finalize_interaction('a')
    asyncio.run(run())
    assert calls == [('boil', 7, '', False), 'closed']
