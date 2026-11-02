"""Float32 sensitivity of deterministically selected local and distant edges."""
import argparse
import hashlib
import json
from pathlib import Path

from omegaconf import OmegaConf
from verl.experimental.attention_traceback.replay import QwenReplay
from verl.experimental.attention_traceback.trace import delta_stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--directory', required=True)
    parser.add_argument('--limit', type=int, default=3, help='edges per distance stratum')
    args = parser.parse_args()
    directory = Path(args.directory)
    traces = [json.loads(line) for line in (directory / 'traces.jsonl').read_text().splitlines()]
    records = {r['trajectory_id']: r for r in (json.loads(line) for line in (directory / 'trajectories.jsonl').read_text().splitlines())}
    candidates = {}
    for trace in traces:
        for edge in trace['edges']:
            if edge['token_selection'] == 'executed_action':
                candidates[(trace['trajectory_id'], edge['target_turn'])] = edge
    selected = []
    for distant in (False, True):
        keys = [k for k,e in candidates.items() if (e['target_turn'] - e['source_turn'] >= 2) == distant]
        selected.extend(sorted(keys, key=lambda k: hashlib.sha256(str(k).encode()).hexdigest())[:args.limit])
    config = OmegaConf.to_container(OmegaConf.load('examples/attention_traceback/config.yaml').attention_traceback)
    config['dtype'] = 'float32'
    backend = QwenReplay('/workspace/checkpoints/Qwen3-1.7B', config)
    rows = []
    for key in selected:
        trajectory, target = key
        edge, record = candidates[key], records[trajectory]
        normal = backend.run(record, target=target)[target]
        source = edge['source_turn']
        result = {'trajectory_id': trajectory, 'target': target, 'source': source,
            'target_action': edge['target_action'], 'target_tokens': edge['target_tokens'],
            'bf16_mean_abs_delta': edge['mean_abs_delta'],
            'fp32_top_source': max(range(target), key=lambda i: normal['attention'][i]),
            'fp32_attention': normal['attention']}
        for label,k in [('top',source), ('previous',target-1), ('random',edge['random_source'])]:
            if k is None:
                continue
            masked = backend.run(record, target=target, source=k)[target]
            delta = [a-b for a,b in zip(normal['logps'],masked['logps'])]
            result[label] = {'token_delta': delta, **delta_stats(delta)}
        backend.config['mask_query_origin'] = 'response'
        masked = backend.run(record, target=target, source=source)[target]
        backend.config['mask_query_origin'] = 'action'
        delta = [a-b for a,b in zip(normal['logps'],masked['logps'])]
        result['response_origin'] = {'token_delta': delta, **delta_stats(delta)}
        rows.append(result)
        (directory / 'fp32_sensitivity.json').write_text(json.dumps(rows, indent=2))
        print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
