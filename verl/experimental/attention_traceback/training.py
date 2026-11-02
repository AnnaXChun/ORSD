"""Online, pre-update actor replay and bounded GRPO credit redistribution.

This is the training hypothesis built on the Phase A diagnostic, not an
assertion that attention dependencies are already validated reward credit.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import random
import subprocess
import sys
import time

import torch

from .replay import QwenReplay, validate_record
from .trace import reward_roots


def record_credit(record, backend, config):
    """Trace Top-1 historical interactions; retain signed deltas in the audit.

    Absolute target-token deltas weight the affected action, while their mean
    is propagated uniformly to the chosen source action. Decay limits deep
    chains. Shared edges contribute once, at the shallowest encountered depth.
    Unparsed responses and pure truncations never receive new credit.
    """
    validate_record(record)
    credit = torch.zeros(len(record['input_ids']), dtype=torch.float32)
    turns = record['turns']
    roots = reward_roots(record, config['timeout_root_window'])
    cache, edge_depth = {}, {}
    for root in roots:
        target = root
        for depth in range(config['max_trace_depth']):
            if target == 0 or turns[target]['token_selection'] != 'executed_action':
                break
            if target not in cache:
                normal = backend.run(record, target=target)[target]
                top = max(range(target), key=lambda k: normal['attention'][k])
                source = top
                if config['selection'] == 'previous':
                    source = target - 1
                elif config['selection'] == 'random':
                    candidates = [k for k in range(target) if k != top]
                    if not candidates:
                        break
                    key = f"{config['seed']}:{record['trajectory_id']}:{target}"
                    rng = random.Random(int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], 'big'))
                    source = rng.choice(candidates)
                if turns[source]['token_selection'] != 'executed_action':
                    break
                masked = backend.run(record, target=target, source=source,
                                     intervention=config['intervention'])[target]
                delta = [a - b for a, b in zip(normal['logps'], masked['logps'], strict=True)]
                if not delta or not torch.isfinite(torch.tensor(delta)).all():
                    raise ValueError('Empty/nonfinite action replay signal')
                strength = sum(abs(x) for x in delta) / len(delta)
                cache[target] = dict(source=source, target=target, delta=delta, strength=strength,
                                     source_action=turns[source].get('action_text', ''),
                                     target_action=turns[target].get('action_text', ''))
            edge = cache[target]
            if edge['strength'] <= config['delta_threshold']:
                break
            edge_depth[target] = min(depth, edge_depth.get(target, depth))
            target = edge['source']
    for target, depth in edge_depth.items():
        edge = cache[target]
        factor = config['depth_decay'] ** depth
        dst, src = turns[target], turns[edge['source']]
        credit[dst['action_token_start']:dst['action_token_end']] += factor * torch.tensor(edge['delta']).abs()
        credit[src['action_token_start']:src['action_token_end']] += factor * edge['strength']
    return credit, {'roots': roots, 'edges': [dict(cache[t], depth=d) for t, d in edge_depth.items()]}


def response_credit(record, credit, responses, response_mask):
    """Exact unpadded rollout -> padded PPO response alignment; never retokenize."""
    if not record['turns']:
        return torch.zeros_like(response_mask, dtype=torch.float32), torch.zeros_like(response_mask, dtype=torch.bool)
    offset = record['turns'][0]['turn_token_start']
    generated = record['input_ids'][offset:]
    # The final observation is clipped to the trajectory response budget by veRL.
    length = min(len(generated), len(responses))
    if responses[:length].tolist() != generated[:length]:
        raise ValueError('PPO response IDs differ from lossless capture; refusing misaligned credit')
    raw = torch.zeros_like(response_mask, dtype=torch.float32)
    action = torch.zeros_like(response_mask, dtype=torch.bool)
    for turn in record['turns']:
        if turn['token_selection'] != 'executed_action':
            continue
        start, end = turn['action_token_start'] - offset, turn['action_token_end'] - offset
        if not 0 <= start < end <= length:
            raise ValueError('Executed action was truncated from the training response')
        if not response_mask[start:end].bool().all():
            raise ValueError('Action tokens overlap an observation/padding loss mask')
        raw[start:end] = credit[offset + start:offset + end]
        action[start:end] = True
    return raw, action


def bounded_weights(raw, action_mask, alpha):
    """Action weights in [1-alpha,1+alpha], mean one; other tokens unchanged."""
    if not 0 <= alpha < 1 or not torch.isfinite(raw).all() or (raw < 0).any():
        raise ValueError('Invalid credit/alpha')
    weights = torch.ones_like(raw, dtype=torch.float32)
    for i in range(len(raw)):
        selected = action_mask[i].bool()
        if not selected.any():
            continue
        values = raw[i, selected]
        centered = values - values.mean()
        scale = centered.abs().max()
        if scale > 0:
            weights[i, selected] += alpha * centered / scale
    return weights


def replay_shard(worker, payload_path, snapshot_path, output_dir):
    payload = torch.load(payload_path, map_location='cpu', weights_only=False)
    config = dict(payload['config'], device=f'cuda:{worker}')
    snapshot = torch.load(snapshot_path, map_location='cpu', weights_only=True)
    if snapshot['step'] != payload['step']:
        raise ValueError('Replay actor snapshot is stale')
    backend = QwenReplay(payload['model_path'], config)
    # strict=True is essential: no partial update or silently retained base weights.
    backend.model.load_state_dict(snapshot['state_dict'], strict=True)
    del snapshot
    result = []
    for i in range(worker, len(payload['records']), config['workers']):
        record = payload['records'][i]
        credit, audit = record_credit(record, backend, config)
        raw, action = response_credit(record, credit, payload['responses'][i], payload['mask'][i])
        result.append({'index': i, 'trajectory_id': record['trajectory_id'],
                       'raw': raw, 'action': action, 'audit': audit})
        print(f"attention training replay step={payload['step']} worker={worker} row={i} edges={len(audit['edges'])}", flush=True)
    path = Path(output_dir) / f'credit_{worker}.pt'
    torch.save({'step': payload['step'], 'rows': result}, path)
    return str(path)


def apply_online_credit(batch, actor_group, full_config, step):
    """Called after GRPO advantages and before ANY actor optimizer update."""
    from omegaconf import OmegaConf
    config = OmegaConf.to_container(full_config.attention_traceback, resolve=True)
    if config['alpha'] == 0:
        return {'attention_credit/enabled': 1., 'attention_credit/alpha': 0.}
    records = list(batch.non_tensor_batch['attention_traceback_record'])
    root = Path(full_config.trainer.default_local_dir) / 'attention_credit'
    root.mkdir(parents=True, exist_ok=True)
    # TemporaryDirectory avoids reuse after a failed/resumed step; only summary/audit persist.
    import tempfile
    started = time.monotonic()
    with tempfile.TemporaryDirectory(prefix=f'step_{step}_', dir=root) as temporary:
        directory = Path(temporary)
        snapshot = directory / 'actor.pt'
        actor_group.export_attention_actor_snapshot(str(snapshot), int(step))
        payload_path = directory / 'batch.pt'
        torch.save({'step': int(step), 'config': config, 'records': records,
                    'model_path': str(full_config.actor_rollout_ref.model.path),
                    'responses': batch.batch['responses'].cpu(),
                    'mask': batch.batch['response_mask'].cpu()}, payload_path)
        environment = os.environ.copy()
        environment['CUDA_VISIBLE_DEVICES'] = str(config['replay_visible_devices'])
        # Child is standalone inference, never a member of the Ray/FSDP process group.
        for key in ('RANK', 'LOCAL_RANK', 'WORLD_SIZE', 'MASTER_ADDR', 'MASTER_PORT'):
            environment.pop(key, None)
        log_path = root / f'step_{step:06d}.log'
        with log_path.open('w') as log:
            subprocess.run([sys.executable, '-m', 'verl.experimental.attention_traceback.training',
                            '--payload', str(payload_path), '--snapshot', str(snapshot),
                            '--output', str(directory)], env=environment, stdout=log,
                           stderr=subprocess.STDOUT, check=True)
        raw = torch.zeros_like(batch.batch['advantages'], device='cpu', dtype=torch.float32)
        action = torch.zeros_like(raw, dtype=torch.bool)
        rows = []
        for worker in range(config['workers']):
            shard = torch.load(directory / f'credit_{worker}.pt', map_location='cpu', weights_only=False)
            if shard['step'] != step:
                raise ValueError('Stale replay result')
            rows.extend(shard['rows'])
        if sorted(row['index'] for row in rows) != list(range(len(records))):
            raise ValueError('Missing/duplicate replay results')
        for row in rows:
            i = row['index']
            if row['trajectory_id'] != records[i]['trajectory_id']:
                raise ValueError('Replay row order mismatch')
            raw[i], action[i] = row['raw'], row['action']
        weights = bounded_weights(raw, action, config['alpha']).to(batch.batch['advantages'].device)
        original = batch.batch['advantages']
        shaped = original * weights
        mask = batch.batch['response_mask'].to(original.dtype)
        # GRPO is trajectory-constant on generated tokens. Preserve its mean and sign.
        mean_error = (((shaped - original) * mask).sum(-1) / mask.sum(-1).clamp_min(1)).abs().max()
        if mean_error > 1e-5 or not torch.isfinite(shaped).all():
            raise ValueError('Credit redistribution violated the GRPO mean invariant')
        batch.batch['advantages'] = shaped
        audits = [{'trajectory_id': records[row['index']]['trajectory_id'],
                   'task_id': records[row['index']]['task_id'], 'step': step,
                   'done_reason': records[row['index']]['done_reason'], **row['audit']} for row in rows]
        (root / f'step_{step:06d}.json').write_text(json.dumps(audits, allow_nan=False))
        counts = [len(row['audit']['edges']) for row in rows]
        changed = weights.ne(1) & mask.bool()
        distinct_rollouts = len({tuple(record['input_ids']) for record in records})
        metrics = {'attention_credit/distinct_rollouts': float(distinct_rollouts),
                   'attention_credit/nonzero_advantage_fraction': (original.abs().sum(-1) > 0).float().mean().item(),
                   'attention_credit/enabled': 1., 'attention_credit/alpha': config['alpha'],
                   'attention_credit/edges': sum(counts),
                   'attention_credit/active_trajectory_fraction': sum(n > 0 for n in counts) / len(counts),
                   'attention_credit/changed_token_fraction': (changed.sum() / mask.sum().clamp_min(1)).item(),
                   'attention_credit/weight_min': weights.min().item(),
                   'attention_credit/weight_max': weights.max().item(),
                   'attention_credit/mean_advantage_error': mean_error.item(),
                   'attention_credit/seconds': time.monotonic() - started,
                   'attention_credit/actor_snapshot_step': float(step)}
        (root / f'step_{step:06d}_metrics.json').write_text(json.dumps(metrics, allow_nan=False))
        return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--payload', required=True)
    parser.add_argument('--snapshot', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    config = torch.load(args.payload, map_location='cpu', weights_only=False)['config']
    with ProcessPoolExecutor(max_workers=config['workers'], mp_context=multiprocessing.get_context('spawn')) as pool:
        futures = [pool.submit(replay_shard, worker, args.payload, args.snapshot, args.output)
                   for worker in range(config['workers'])]
        for future in futures:
            future.result()


if __name__ == '__main__':
    main()
