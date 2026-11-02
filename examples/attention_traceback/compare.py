"""Joint Phase A evidence table. Human review remains mandatory for C/D."""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', default='results/attention_traceback')
    args = parser.parse_args()
    root = Path(args.root)
    counts = {'A': 0, 'B': 0}
    lines = ['| Benchmark | Top mean abs delta | Random | Previous | Top − Random 95% CI | Top − Previous 95% CI |',
             '|---|---:|---:|---:|---|---|']
    reports = {}
    complete = True
    for name in ('webshop', 'alfworld', 'scienceworld'):
        path = root / name / 'summary.json'
        if not path.exists():
            reports[name] = {'status': 'missing'}
            complete = False
            continue
        report = json.loads(path.read_text())
        reports[name] = report
        complete &= 50 <= report['number_of_tasks'] <= 100 and report['number_of_trajectories'] == 4 * report['number_of_tasks']
        means = [report['interventions'][key]['edge_mean_abs_delta']['mean'] for key in ('TopAttention', 'Random', 'PreviousTurn')]
        cis = [report['comparisons'][key]['paired_difference']['ci95'] for key in ('Random', 'PreviousTurn')]
        for label, ci in zip(('A', 'B'), cis):
            if ci and ci[0] > 0:
                counts[label] += 1
        lines.append('| ' + ' | '.join(map(str, [name, *means, *cis])) + ' |')
    status = 'PENDING_DATA' if not complete else ('NO_GO_STATISTICAL' if min(counts.values()) < 2 else 'PENDING_C_D_HUMAN_REVIEW')
    result = {'status': status, 'benchmark_pass_counts': counts,
              'rule_AB': 'paired task-bootstrap 95% CI lower bound > 0 in at least 2 benchmarks each',
              'C': 'Review threshold-sweep depth>=2 and distance>=2 stability; unthresholded depth alone is not evidence',
              'D': 'Inspect heatmaps and label decision-bearing vs formatting tokens; concentration alone is not semantic evidence',
              'rl_training_authorized': False}
    root.mkdir(parents=True, exist_ok=True)
    (root / 'comparison.md').write_text('\n'.join(lines) + '\n\n' + json.dumps(result, indent=2) + '\n')
    (root / 'go_no_go.json').write_text(json.dumps(result, indent=2))
    print('\n'.join(lines))
    print(status)


if __name__ == '__main__':
    main()
