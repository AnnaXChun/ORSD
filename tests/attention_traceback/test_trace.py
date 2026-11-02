"""Run only on the user's server: pytest -q tests/attention_traceback."""
from types import SimpleNamespace

import pytest

from verl.experimental.attention_traceback.trace import reward_roots, trace_record, select_trace_roots
from verl.experimental.attention_traceback.report import bootstrap, unique_edges
from verl.experimental.attention_traceback.run import assign_grpo


def test_reward_roots():
    record = {'benchmark': 'webshop', 'turns': [{}] * 5, 'done_reason': 'terminal', 'step_rewards': [0] * 5}
    assert reward_roots(record) == [4]
    record['done_reason'] = 'max_steps'
    assert reward_roots(record) == [2, 3, 4]
    record.update(benchmark='scienceworld', step_rewards=[1, 0, -1, 0, 0])
    assert reward_roots(record) == [0, 2, 3, 4]
    record['done_reason'] = 'terminal'
    assert reward_roots(record) == [0, 2]
    record['benchmark'] = 'alfworld'
    record['done_reason'] = 'context_budget_exhausted'
    assert reward_roots(record) == []  # do not relabel token-limit truncation as outcome


class Backend:
    tokenizer = SimpleNamespace(decode=lambda ids, **kwargs: str(ids[0]))

    def run(self, record, target=None, source=None, intervention=None):
        if target is None:
            return {t: {'logps': [-1., -2.], 'attention': [.1] * max(0, t - 1) + ([.7] if t else []),
                        'attention_token_mass': []} for t in range(4)}
        return {target: {'logps': [-1.5, -1.], 'attention': [0.] * target, 'attention_token_mass': []}}


def test_null_threshold_keeps_signed_tokens_and_controls():
    record = {'benchmark': 'webshop', 'task_id': 'task', 'trajectory_id': 'trajectory',
              'trajectory_reward': 0, 'success': False, 'grpo_advantage': 0,
              'done_reason': 'terminal', 'step_rewards': [0] * 4,
              'turns': [{'action_text': f'a{i}', 'observation_text': f'o{i}',
                        'token_selection': 'executed_action', 'action_token_ids': [4, 5]} for i in range(4)]}
    config = {'timeout_root_window': 3, 'max_trace_depth': 6, 'seed': 1234,
              'intervention': 'mask_span', 'delta_threshold': None}
    traces = trace_record(record, Backend(), config)
    assert traces[0]['trace_depth'] == 3
    assert traces[0]['path_status'] == 'candidate_unthresholded'
    for edge in traces[0]['edges']:
        assert edge['token_delta'] == [.5, -1.]
        assert edge['dependency_valid'] is None
        assert edge['random_source'] != edge['source_turn']
        assert edge['previous_token_delta'] == edge['token_delta']
    assert traces[0]['edges'][-1]['random_token_delta'] is None
    assert len(unique_edges(traces + traces)) == 3
    config['delta_threshold'] = 1.
    rejected = trace_record(record, Backend(), config)[0]
    assert rejected['trace_depth'] == 1  # candidate kept, traversal stops
    assert rejected['edges'][0]['dependency_valid'] is False


def test_cluster_bootstrap_and_group_counts():
    assert bootstrap([('a', 1.), ('a', 0.)], 100, 3)['ci95'] is None
    estimate = bootstrap([('a', 1.), ('b', 0.)], 100, 3)
    assert estimate['estimate'] == .5 and estimate['ci95'] == [0., 1.]
    records = [{'task_id': 'a', 'trajectory_reward': reward} for reward in (0., 1.)]
    assign_grpo(records, 2)
    assert records[0]['grpo_advantage'] == pytest.approx(-.70710578)
    assert sum(r['grpo_advantage'] for r in records) == 0
    with pytest.raises(ValueError):
        assign_grpo(records, 4)


def test_truncation_roots_are_smoke_only_and_labelled():
    record = {'benchmark': 'alfworld', 'task_id': 'task', 'trajectory_id': 'truncated',
              'trajectory_reward': 0, 'success': False, 'grpo_advantage': 0,
              'done_reason': 'context_budget_exhausted', 'step_rewards': [0] * 4,
              'turns': [{'action_text': f'a{i}', 'observation_text': f'o{i}',
                        'token_selection': 'executed_action', 'action_token_ids': [4, 5]} for i in range(4)]}
    config = {'timeout_root_window': 3, 'max_trace_depth': 6, 'seed': 1234,
              'intervention': 'mask_span', 'delta_threshold': None}
    assert select_trace_roots(record, config) == ([], 'reward_protocol')
    assert trace_record(record, Backend(), config) == []
    config['diagnostic_truncation_roots'] = True
    assert select_trace_roots(record, config) == ([1, 2, 3], 'truncation_diagnostic')
    traces = trace_record(record, Backend(), config)
    assert len(traces) == 3
    assert all(t['root_origin'] == 'truncation_diagnostic' for t in traces)
    assert all(not t['eligible_for_reward_hypotheses'] for t in traces)
    assert all(t['done_reason'] == 'context_budget_exhausted' for t in traces)
    record['done_reason'] = 'length_limit'
    assert select_trace_roots(record, config)[0] == [1, 2, 3]
    record['turns'] = []
    assert select_trace_roots(record, config)[0] == []
