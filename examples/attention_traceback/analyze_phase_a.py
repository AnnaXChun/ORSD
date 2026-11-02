"""Additional coverage and lexical/format diagnostics; no new rollout or judge."""
import argparse
from collections import Counter
import json
from pathlib import Path
import statistics

from verl.experimental.attention_traceback.report import bootstrap, distribution


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True)
    args = parser.parse_args()
    root = Path(args.root)
    output = {}
    for benchmark in ('webshop', 'alfworld', 'scienceworld'):
        directory = root / benchmark
        if not (directory / 'traces.jsonl').exists():
            continue
        traces = [json.loads(line) for line in (directory / 'traces.jsonl').read_text().splitlines()]
        unique = {}
        for trace in traces:
            for edge in trace['edges']:
                unique[(trace['trajectory_id'], edge['target_turn'])] = (trace, edge)
        lexical_pairs, lexical, formatting, far_edges = [], [], [], []
        parsed_pairs = {'random': [], 'previous': []}
        peak_roles = Counter()
        for (trajectory_id, _), (trace, edge) in unique.items():
            if edge['token_selection'] != 'executed_action':
                continue
            for label in parsed_pairs:
                values = edge[f'{label}_token_delta']
                if values is not None:
                    parsed_pairs[label].append((trace['task_id'], edge['mean_abs_delta'] - statistics.mean(map(abs, values))))
            lex = [abs(v) for token, v in zip(edge['target_tokens'], edge['token_delta']) if any(c.isalnum() for c in token)]
            fmt = [abs(v) for token, v in zip(edge['target_tokens'], edge['token_delta']) if not any(c.isalnum() for c in token)]
            lexical.extend(lex)
            formatting.extend(fmt)
            if lex and fmt:
                lexical_pairs.append((trace['task_id'], statistics.mean(lex) - statistics.mean(fmt)))
            if any(edge['token_delta']):
                peak = max(range(len(edge['token_delta'])), key=lambda i: abs(edge['token_delta'][i]))
                peak_roles['lexical_or_mixed' if any(c.isalnum() for c in edge['target_tokens'][peak]) else 'formatting_only'] += 1
            if edge['target_turn'] - edge['source_turn'] >= 2:
                far_edges.append({'trajectory_id': trajectory_id, 'task_id': trace['task_id'],
                    'reward': trace['reward'], 'success': trace['success'],
                    **{k: edge[k] for k in ('source_turn', 'target_turn', 'source_action', 'target_action',
                        'mean_abs_delta', 'target_tokens', 'token_delta', 'previous_token_delta', 'random_token_delta')}})
        output[benchmark] = {'lexical_or_mixed_abs_delta': distribution(lexical),
            'formatting_only_abs_delta': distribution(formatting),
            'paired_lexical_minus_formatting': bootstrap(lexical_pairs, 2000, 1234),
            'max_effect_token_role': dict(peak_roles), 'cross_turn_edges': far_edges,
            'executed_action_only_comparisons': {label: bootstrap(values, 2000, 1234) for label, values in parsed_pairs.items()},
            'interpretation': 'Lexical/format is a mechanical proxy, not a decision-bearing semantic annotation.'}
        summary = json.loads((directory / 'summary.json').read_text())
        joint = []
        for sweep in summary['threshold_sweep']:
            threshold = sweep['threshold']
            qualifying = []
            for trace in traces:
                accepted = []
                for edge in trace['edges']:
                    if edge['mean_abs_delta'] <= threshold:
                        break
                    accepted.append(edge)
                if len(accepted) >= 2 and any(e['target_turn'] - e['source_turn'] >= 2 for e in accepted):
                    qualifying.append({'trajectory_id': trace['trajectory_id'], 'task_id': trace['task_id'],
                        'root': trace['root_turn'], 'depth': len(accepted), 'success': trace['success']})
            joint.append({'threshold': threshold, 'trace_count': len(qualifying),
                'trajectory_count': len({t['trajectory_id'] for t in qualifying}),
                'task_count': len({t['task_id'] for t in qualifying}),
                'qualifying': qualifying})
        output[benchmark]['joint_depth_ge2_and_distance_ge2'] = joint
    (root / 'supplementary_analysis.json').write_text(json.dumps(output, indent=2, ensure_ascii=False))
    print(json.dumps({b: {k:v for k,v in row.items() if k != 'cross_turn_edges'} for b,row in output.items()}, indent=2))


if __name__ == '__main__':
    main()
