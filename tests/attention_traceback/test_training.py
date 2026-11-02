import pytest
import torch
from verl.experimental.attention_traceback.training import bounded_weights, record_credit, response_credit


def test_bounded_credit_preserves_mean_sign_and_zero_advantage():
    raw = torch.tensor([[0., 1., 9., 0., 0.], [0., 0., 0., 0., 0.], [2., 0., 0., 0., 0.]])
    mask = torch.tensor([[1,1,1,0,0], [1,1,0,0,0], [1,1,1,0,0]], dtype=torch.bool)
    weights = bounded_weights(raw, mask, .5)
    assert weights.min() >= .5 and weights.max() <= 1.5
    assert torch.equal(weights[~mask], torch.ones_like(weights[~mask]))
    advantages = torch.tensor([2., -1., 0.])[:,None].expand_as(raw)
    shaped = weights * advantages
    assert torch.allclose(shaped.mean(-1), advantages.mean(-1))
    assert torch.equal(shaped.sign(), advantages.sign())
    assert torch.equal(weights[1], torch.ones(5))
    assert torch.equal(bounded_weights(raw, mask, 0), torch.ones_like(raw))


def row():
    return {'benchmark':'webshop', 'done_reason':'terminal', 'trajectory_id':'x',
            'step_rewards':[0,1], 'input_ids':[1,2,3,4,5,6,7,8], 'turns':[
        dict(turn_idx=0, turn_token_start=2, action_token_start=3, action_token_end=4,
             observation_token_start=4, observation_token_end=5, turn_token_end=5,
             action_token_ids=[4], token_selection='executed_action'),
        dict(turn_idx=1, turn_token_start=5, action_token_start=5, action_token_end=7,
             observation_token_start=7, observation_token_end=8, turn_token_end=8,
             action_token_ids=[6,7], token_selection='executed_action')]}


class Backend:
    def run(self, record, target=None, source=None, **kw):
        return {target: {'attention':[1.], 'logps':[-2.,-3.] if source is None else [-4.,-2.]}}


def test_chain_delta_sign_audit_and_exact_response_alignment():
    record = row()
    cfg = dict(timeout_root_window=3, max_trace_depth=6, selection='attention',
               intervention='mask_span', delta_threshold=.001, depth_decay=.5)
    credit, audit = record_credit(record, Backend(), cfg)
    assert audit['edges'][0]['delta'] == [2., -1.]
    assert credit[3] == 1.5 and credit[5] == 2 and credit[6] == 1
    responses = torch.tensor([3,4,5,6,7,8,0,0])
    mask = torch.tensor([1,1,0,1,1,0,0,0])
    raw, action = response_credit(record, credit, responses, mask)
    assert raw.tolist() == [0,1.5,0,2,1,0,0,0]
    assert action.tolist() == [False,True,False,True,True,False,False,False]
    with pytest.raises(ValueError, match='IDs differ'):
        response_credit(record, credit, responses+1, mask)
    with pytest.raises(ValueError, match='loss mask'):
        response_credit(record, credit, responses, torch.zeros_like(mask))
    record['done_reason'] = 'length_limit'
    credit, audit = record_credit(record, Backend(), cfg)
    assert credit.sum() == 0 and audit['edges'] == []
    record['done_reason'] = 'terminal'
    record['turns'][1]['token_selection'] = 'full_response_unparseable'
    credit, audit = record_credit(record, Backend(), cfg)
    assert credit.sum() == 0 and audit['edges'] == []
