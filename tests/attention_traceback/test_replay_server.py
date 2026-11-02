"""Real Qwen3 attention ABI and shift tests using a tiny random server GPU model.

The tiny model checks mechanics only, never substitutes for benchmark smoke data.
"""
import pytest
import torch

from verl.experimental.attention_traceback.replay import QwenReplay, validate_record


@pytest.fixture
def backend():
    if not torch.cuda.is_available():
        pytest.fail('Run this test on the ORSD server GPU; local validation is not supported')
    from transformers import Qwen3Config, Qwen3ForCausalLM
    torch.manual_seed(17)
    cfg = Qwen3Config(vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=4,
                      num_attention_heads=4, num_key_value_heads=2, head_dim=8,
                      attention_dropout=0., max_position_embeddings=128)
    cfg._attn_implementation = 'sdpa'
    model = Qwen3ForCausalLM(cfg).to('cuda').eval()
    replay = QwenReplay.__new__(QwenReplay)
    replay.model, replay.device = model, torch.device('cuda')
    replay.layers = model.model.layers[-4:]
    replay.config = {'attention_query_chunk': 1, 'max_dense_mask_tokens': 128}
    return replay


def record():
    ids = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
    return {'input_ids': ids, 'done_reason': 'terminal', 'turns': [
        {'turn_idx': 0, 'turn_token_start': 3, 'action_token_start': 3, 'action_token_end': 5,
         'observation_token_start': 5, 'observation_token_end': 7, 'turn_token_end': 7, 'action_token_ids': ids[3:5]},
        {'turn_idx': 1, 'turn_token_start': 8, 'action_token_start': 8, 'action_token_end': 10,
         'observation_token_start': 10, 'observation_token_end': 11, 'turn_token_end': 11, 'action_token_ids': ids[8:10]},
        {'turn_idx': 2, 'turn_token_start': 12, 'action_token_start': 12, 'action_token_end': 13,
         'observation_token_start': 13, 'observation_token_end': 14, 'turn_token_end': 14, 'action_token_ids': ids[12:13]}]}


@torch.inference_mode()
def test_selected_rows_match_eager_and_predictive_shift(backend):
    row = record()
    result = backend.run(row)
    ids = torch.tensor([row['input_ids'][:13]], device='cuda')
    backend.model.config._attn_implementation = 'eager'
    expected = backend.model(input_ids=ids, use_cache=False, output_attentions=True)
    for t, turn in enumerate(row['turns']):
        s, e = turn['action_token_start'], turn['action_token_end']
        reference = expected.logits[:, s-1:e-1].log_softmax(-1).gather(-1, ids[:, s:e, None])[0, :, 0]
        assert result[t]['logps'] == pytest.approx(reference.cpu().tolist(), abs=1e-5)
        for k in range(t):
            lo, hi = row['turns'][k]['turn_token_start'], row['turns'][k]['turn_token_end']
            mass = torch.stack([a[0, :, s-1:e-1, lo:hi].sum(-1).mean() for a in expected.attentions[-4:]]).mean()
            assert result[t]['attention'][k] == pytest.approx(mass.item(), abs=1e-5)


@torch.inference_mode()
def test_mask_is_applied_to_first_predictive_row(backend):
    row = record()
    actual = backend.run(row, target=1, source=0)
    ids = torch.tensor([row['input_ids'][:10]], device='cuda')
    mask = torch.full((10, 10), -torch.inf, device='cuda').triu(1)
    mask[7:, 3:7] = -torch.inf
    backend.model.config._attn_implementation = 'eager'
    reference = backend.model(input_ids=ids, attention_mask=mask[None, None], output_attentions=True, use_cache=False)
    expected = reference.logits[:, 7:9].log_softmax(-1).gather(-1, ids[:, 8:10, None])[0, :, 0]
    assert actual[1]['logps'] == pytest.approx(expected.cpu().tolist(), abs=1e-5)
    assert actual[1]['attention'][0] == 0
    assert all(a[0, :, 7:9, 3:7].abs().max().item() == 0 for a in reference.attentions)
    # Earlier predictive rows retain ordinary causal attention.
    assert all(a[0, :, 6, 3:7].sum().item() > 0 for a in reference.attentions)


@torch.inference_mode()
def test_drop_preserves_original_target_and_rope_positions(backend):
    row = record()
    actual = backend.run(row, target=1, source=0, intervention='drop_span')[1]
    retained = [0, 1, 2, 7, 8, 9]
    ids = torch.tensor([[row['input_ids'][i] for i in retained]], device='cuda')
    reference = backend.model(input_ids=ids, position_ids=torch.tensor([retained], device='cuda'), use_cache=False)
    expected = reference.logits[:, 3:5].log_softmax(-1).gather(-1, ids[:, 4:6, None])[0, :, 0]
    assert actual['logps'] == pytest.approx(expected.cpu().tolist(), abs=1e-5)
    row['turns'][1]['action_token_start'] = 7
    with pytest.raises(AssertionError):
        validate_record(row)


@torch.inference_mode()
def test_response_origin_masks_thinking_predictive_rows(backend):
    row = record()
    row['turns'][1]['action_token_start'] = 9
    row['turns'][1]['action_token_ids'] = row['input_ids'][9:10]
    backend.config['mask_query_origin'] = 'response'
    actual = backend.run(row, target=1, source=0)[1]
    ids = torch.tensor([row['input_ids'][:10]], device='cuda')
    mask = torch.full((10, 10), -torch.inf, device='cuda').triu(1)
    mask[7:, 3:7] = -torch.inf
    backend.model.config._attn_implementation = 'eager'
    reference = backend.model(input_ids=ids, attention_mask=mask[None, None], use_cache=False)
    expected = reference.logits[:, 8:9].log_softmax(-1).gather(-1, ids[:, 9:10, None])[0, :, 0]
    assert actual['logps'] == pytest.approx(expected.cpu().tolist(), abs=1e-5)
