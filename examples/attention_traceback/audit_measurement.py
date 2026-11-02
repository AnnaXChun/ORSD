"""Server checkpoint audit, independent of benchmark hypothesis statistics."""
import argparse
import json
from pathlib import Path

import torch
from omegaconf import OmegaConf
from verl.experimental.attention_traceback.replay import QwenReplay


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--capture', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--dtype', choices=['bfloat16', 'float32'], default='bfloat16')
    args = parser.parse_args()
    config = OmegaConf.to_container(OmegaConf.load('examples/attention_traceback/config.yaml').attention_traceback)
    config['dtype'] = args.dtype
    backend = QwenReplay('/workspace/checkpoints/Qwen3-1.7B', config)
    findings = []
    for benchmark in ('webshop', 'alfworld', 'scienceworld'):
        record = json.loads(next((Path(args.capture) / benchmark).glob('*.json')).read_text())
        full = backend.run(record) if args.dtype == 'bfloat16' else None
        # A short prefix allows independent eager attention validation on the real checkpoint.
        target = 1
        row = record['turns'][target]
        prefix = backend.run(record, target=target)[target]
        source = max(range(target), key=lambda k: prefix['attention'][k])
        masked = backend.run(record, target=target, source=source)[target]
        backend.config['mask_query_origin'] = 'response'
        response_masked = backend.run(record, target=target, source=source)[target]
        backend.config['mask_query_origin'] = 'action'
        item = {'benchmark': benchmark, 'target': target, 'source': source,
                'action': row['action_text'],
                'thinking_tokens': row['action_token_start'] - row['turn_token_start'],
                'dtype': args.dtype,
                'original_logps': prefix['logps'], 'masked_logps': masked['logps'],
                'full_prefix_logp_max_difference': max(abs(a-b) for a,b in zip(full[target]['logps'], prefix['logps'])) if full else None,
                'action_mask_delta': [a-b for a,b in zip(prefix['logps'], masked['logps'])],
                'response_mask_delta': [a-b for a,b in zip(prefix['logps'], response_masked['logps'])]}
        if row['action_token_end'] <= 5000:
            backend.model.config._attn_implementation = 'eager'
            eager = backend.run(record, target=target, source=source)[target]
            backend.model.config._attn_implementation = 'sdpa'
            item['eager_sdpa_mask_logp_max_difference'] = max(abs(a-b) for a,b in zip(eager['logps'], masked['logps']))
            item['eager_sdpa_attention_max_difference'] = max(abs(a-b) for a,b in zip(eager['attention'], masked['attention']))
        # A repeated SDPA prefix forward estimates numerical repeatability.
        again = backend.run(record, target=target)[target]
        item['repeat_logp_max_difference'] = max(abs(a-b) for a,b in zip(again['logps'], prefix['logps']))
        findings.append(item)
        print(json.dumps(item), flush=True)
    Path(args.output).write_text(json.dumps(findings, indent=2))


if __name__ == '__main__':
    main()
