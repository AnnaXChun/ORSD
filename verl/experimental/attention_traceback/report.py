"""Task-cluster bootstrap and standalone HTML/SVG diagnostics (no plotting deps)."""
from collections import Counter, defaultdict
import html
import json
import math
import random
import statistics
from pathlib import Path


def quantile(values, q):
    if not values:
        return None
    values = sorted(values)
    point = (len(values) - 1) * q
    low = int(point)
    high = min(low + 1, len(values) - 1)
    return values[low] + (values[high] - values[low]) * (point - low)


def distribution(values):
    return {"n": len(values), "mean": statistics.mean(values) if values else None,
            "median": quantile(values, .5), "p25": quantile(values, .25), "p75": quantile(values, .75)}


def bootstrap(rows, samples, seed):
    """rows=(task_id, value); resample tasks, retain sibling trajectories/edges.

    Point estimate is edge-weighted. All siblings and multiple roots remain in
    the same cluster. One-task smoke tests deliberately have no CI.
    """
    groups = defaultdict(list)
    for task, value in rows:
        groups[task].append(value)
    values = [value for _, value in rows]
    result = {"estimate": statistics.mean(values) if values else None, "n": len(rows),
              "task_clusters": len(groups), "ci95": None}
    if len(groups) < 2:
        return result
    keys = sorted(groups)
    sums = {key: sum(groups[key]) for key in keys}
    counts = {key: len(groups[key]) for key in keys}
    rng = random.Random(seed)
    estimates = []
    for _ in range(samples):
        chosen = rng.choices(keys, k=len(keys))
        estimates.append(sum(sums[k] for k in chosen) / sum(counts[k] for k in chosen))
    result["ci95"] = [quantile(estimates, .025), quantile(estimates, .975)]
    return result


def unique_edges(traces):
    found = {}
    for trace in traces:
        for edge in trace["edges"]:
            key = (trace["trajectory_id"], edge["target_turn"])
            found[key] = (trace["task_id"], edge)
    return list(found.values())


def summarize(records, traces, config):
    edges = unique_edges(traces)
    depths = [trace["trace_depth"] for trace in traces]
    distances = [edge["target_turn"] - edge["source_turn"] for _, edge in edges]
    summary = {"number_of_trajectories": len(records), "number_of_tasks": len({r['task_id'] for r in records}),
        "number_of_reward_roots": sum(t.get("eligible_for_reward_hypotheses", True) for t in traces),
        "number_of_truncation_diagnostic_roots": sum(not t.get("eligible_for_reward_hypotheses", True) for t in traces),
        "statistics_scope": "smoke_mechanics_only" if any(not t.get("eligible_for_reward_hypotheses", True) for t in traces) else "reward_protocol",
        "number_of_traces": len(traces),
        "number_of_unique_candidate_edges": len(edges), "depth": distribution(depths),
        "distance": distribution(distances), "depth_histogram": dict(Counter(depths)),
        "distance_histogram": dict(Counter(distances)),
        "fraction_depth_ge_2": sum(d >= 2 for d in depths) / len(depths) if depths else None,
        "fraction_depth_ge_3": sum(d >= 3 for d in depths) / len(depths) if depths else None,
        "fraction_distance_ge_2": sum(d >= 2 for d in distances) / len(distances) if distances else None,
        "fraction_distance_ge_3": sum(d >= 3 for d in distances) / len(distances) if distances else None,
        "intervention": config['intervention'], "path_status": "candidate_unthresholded" if config['delta_threshold'] is None else "thresholded",
        "outcomes": dict(Counter(r['done_reason'] for r in records)),
        "successes": sum(bool(r['success']) for r in records),
        "statistics_unit": "unique target edge; bootstrap resamples entire task clusters",
        "interventions": {}, "comparisons": {}}
    summary['structural_ci'] = {
        'mean_depth': bootstrap([(t['task_id'], t['trace_depth']) for t in traces], config['bootstrap_samples'], config['seed']),
        'mean_distance': bootstrap([(task, e['target_turn'] - e['source_turn']) for task, e in edges], config['bootstrap_samples'], config['seed']),
    }
    root_trajectories = {t['trajectory_id'] for t in traces if t.get('eligible_for_reward_hypotheses', True)}
    by_task = defaultdict(list)
    for record in records:
        by_task[record['task_id']].append(record)
    summary['coverage'] = {
        'trajectories_with_reward_roots': len(root_trajectories),
        'fraction_with_reward_roots': len(root_trajectories) / len(records) if records else None,
        'tasks_with_candidate_edges': len({task for task, _ in edges}),
        'top_equals_previous_count': sum(e['source_turn'] == e['target_turn'] - 1 for _, e in edges),
        'distinct_token_trajectories_per_task': {
            task: len({tuple(r['input_ids']) for r in siblings}) for task, siblings in by_task.items()},
        'distinct_sampling_seeds_per_task': {
            task: len({r.get('sampling_params', {}).get('seed') for r in siblings}) for task, siblings in by_task.items()},
    }
    for limit in (2, 3):
        summary['structural_ci'][f'fraction_depth_ge_{limit}'] = bootstrap(
            [(t['task_id'], float(t['trace_depth'] >= limit)) for t in traces], config['bootstrap_samples'], config['seed'])
        summary['structural_ci'][f'fraction_distance_ge_{limit}'] = bootstrap(
            [(task, float(e['target_turn'] - e['source_turn'] >= limit)) for task, e in edges], config['bootstrap_samples'], config['seed'])
    for label, key in (("TopAttention", "token_delta"), ("Random", "random_token_delta"), ("PreviousTurn", "previous_token_delta")):
        rows = [(task, statistics.mean(map(abs, edge[key]))) for task, edge in edges if edge[key] is not None]
        tokens = [abs(v) for _, edge in edges if edge[key] is not None for v in edge[key]]
        summary['interventions'][label] = {"edge_mean_abs_delta": distribution([v for _, v in rows]),
            "token_abs_delta": distribution(tokens), "mean_ci": bootstrap(rows, config['bootstrap_samples'], config['seed'])}
    for label, key in (("Random", "random_token_delta"), ("PreviousTurn", "previous_token_delta")):
        paired = [(task, edge['mean_abs_delta'] - statistics.mean(map(abs, edge[key])))
                  for task, edge in edges if edge[key] is not None]
        summary['comparisons'][label] = {
            "paired_difference": bootstrap(paired, config['bootstrap_samples'], config['seed']),
            "fraction_top_greater": bootstrap([(task, float(v > 0)) for task, v in paired], config['bootstrap_samples'], config['seed']),
            "ties": sum(v == 0 for _, v in paired),
        }
        task_values = defaultdict(list)
        for task, difference in paired:
            task_values[task].append(difference)
        summary['comparisons'][label]['task_equal_weight_difference'] = bootstrap(
            [(task, statistics.mean(values)) for task, values in task_values.items()],
            config['bootstrap_samples'], config['seed'])
    effects = [edge['mean_abs_delta'] for _, edge in edges]
    thresholds = sorted({0.0, *[quantile(effects, q) for q in config['threshold_quantiles']]}) if effects else []
    sweep = []
    for threshold in thresholds:
        accepted_depths, accepted_distances = [], []
        unique = set()
        for trace in traces:
            depth = 0
            for edge in trace['edges']:
                if edge['mean_abs_delta'] <= threshold:
                    break
                depth += 1
                key = (trace['trajectory_id'], edge['target_turn'])
                if key not in unique:
                    accepted_distances.append(edge['target_turn'] - edge['source_turn'])
                    unique.add(key)
            accepted_depths.append(depth)
        sweep.append({"threshold": threshold, "depth": distribution(accepted_depths),
            "edge_count": len(unique), "distance": distribution(accepted_distances),
            "fraction_depth_ge_2": sum(d >= 2 for d in accepted_depths) / len(accepted_depths) if accepted_depths else None,
            "fraction_distance_ge_2": sum(d >= 2 for d in accepted_distances) / len(accepted_distances) if accepted_distances else None})
    summary['threshold_sweep'] = sweep
    # Concentration is descriptive. It does not label a token's semantic role.
    concentration = []
    cvs = []
    for _, edge in edges:
        values = [abs(v) for v in edge['token_delta']]
        if sum(values) > 0:
            concentration.append(sum(sorted(values, reverse=True)[:max(1, math.ceil(len(values) * .2))]) / sum(values))
            cvs.append(statistics.pstdev(values) / statistics.mean(values))
    summary['token_selection_counts'] = dict(Counter(e['token_selection'] for _, e in edges))
    summary['token_structure'] = {"top20pct_token_effect_share": distribution(concentration),
        "coefficient_of_variation": distribution(cvs), "semantic_decision_token_test": "requires human labels; see examples"}
    summary['go_no_go'] = "PENDING: joint benchmark report and human semantic review required; no RL authorization"
    return summary


def esc(value):
    return html.escape(str(value))


def edge_html(edge):
    scale = max(map(abs, edge['token_delta'])) or 1.0
    cells = []
    for token, value in zip(edge['target_tokens'], edge['token_delta'], strict=True):
        intensity = abs(value) / scale
        color = f"rgba({('220,55,45' if value >= 0 else '45,100,220')},{intensity:.3f})"
        cells.append(f'<span class="token" style="background:{color}">{esc(token)}<small>{value:+.4f}</small></span>')
    ranks = sorted(range(len(edge['token_delta'])), key=lambda i: abs(edge['token_delta'][i]), reverse=True)[:5]
    changed = ', '.join(f"{edge['target_tokens'][i]!r}: {edge['token_delta'][i]:+.4f}" for i in ranks)
    rows = ''.join(f'<tr><td>{i}</td><td>{a:.5f}</td><td>{b:.5f}</td></tr>'
                   for i, (a, b) in enumerate(zip(edge['attention_before'], edge['attention_after'], strict=True)))
    return f'''<section><h3>Turn {edge['source_turn']} → Turn {edge['target_turn']}</h3>
<p>Attention {edge['attention_score']:.5f}; mean |Δlogp| {edge['mean_abs_delta']:.5f}; status {esc(edge['dependency_valid'])}</p>
<details><summary>Source action and observation</summary><pre>{esc(edge['source_action'])}\n{esc(edge['source_observation'])}</pre></details>
<pre>{esc(edge['target_action'])}</pre><div>{''.join(cells)}</div><p>Largest changes: {esc(changed)}</p>
<p>Source redistribution: {edge['original_top_source']} → {edge['new_top_source']}</p>
<table><tr><th>History turn</th><th>Before</th><th>After</th></tr>{rows}</table></section>'''


STYLE = '<style>body{font:16px system-ui;max-width:1100px;margin:32px auto;padding:16px}pre{white-space:pre-wrap}section{border-top:1px solid #ddd;padding:16px 0}.token{display:inline-block;padding:8px;margin:3px;border:1px solid #ccc;white-space:pre-wrap}small{display:block;font-family:monospace}td,th{padding:6px;text-align:right}table{border-collapse:collapse}</style>'


def histogram_svg(values, title):
    counts = sorted(Counter(values).items())
    height = 70 + 28 * len(counts)
    peak = max((v for _, v in counts), default=1)
    bars = ''.join(f'<text x="10" y="{65+i*28}">{key}</text><rect x="60" y="{48+i*28}" width="{450*count/peak}" height="20" fill="#487ec9"/><text x="{70+450*count/peak}" y="{65+i*28}">{count}</text>' for i, (key, count) in enumerate(counts))
    return f'<svg xmlns="http://www.w3.org/2000/svg" width="640" height="{height}"><rect width="100%" height="100%" fill="white"/><text x="10" y="25">{esc(title)}</text>{bars}</svg>'


def write_report(directory, records, traces, config):
    directory = Path(directory)
    examples = directory / 'examples'
    plots = directory / 'plots'
    examples.mkdir(parents=True, exist_ok=True)
    plots.mkdir(exist_ok=True)
    summary = summarize(records, traces, config)
    selected = {}
    for success in (True, False):
        candidates = sorted((t for t in traces if t['success'] == success),
            key=lambda t: (t['trace_depth'] >= 2, max((e['mean_abs_delta'] for e in t['edges']), default=0),
                           max((e['target_turn'] - e['source_turn'] for e in t['edges']), default=0)), reverse=True)
        diverse, remaining, tasks_seen = [], [], set()
        for candidate in candidates:
            if candidate['task_id'] not in tasks_seen:
                diverse.append(candidate)
                tasks_seen.add(candidate['task_id'])
            else:
                remaining.append(candidate)
        candidates = diverse + remaining
        seen = set()
        selected['success' if success else 'failure'] = []
        for trace in candidates:
            if trace['trajectory_id'] in seen:
                continue
            seen.add(trace['trajectory_id'])
            label = f"{'success' if success else 'failure'}_{len(seen)}"
            path = examples / f'{label}.html'
            chain = ' → '.join(str(e['source_turn']) for e in reversed(trace['edges']))
            body = ''.join(edge_html(edge) for edge in trace['edges'])
            for edge_index, edge in enumerate(trace['edges']):
                (plots / f'{label}_edge{edge_index}_heatmap.svg').write_text(heatmap_svg(edge), encoding='utf-8')
            path.write_text(f'<!doctype html><meta charset="utf-8">{STYLE}<h1>{label}</h1><p>Task {esc(trace["task_id"])}; reward {trace["reward"]}; root {trace["root_turn"]}; {esc(trace["path_status"])}</p><p>{chain} → {trace["root_turn"]} → Reward</p>{body}', encoding='utf-8')
            (examples / f'{label}.json').write_text(json.dumps(trace, ensure_ascii=False, indent=2), encoding='utf-8')
            selected['success' if success else 'failure'].append(str(path.name))
            if len(seen) == 5:
                break
    summary['examples'] = selected
    summary['example_shortfall'] = {k: max(0, 5 - len(v)) for k, v in selected.items()}
    (directory / 'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False), encoding='utf-8')
    edges = unique_edges(traces)
    (plots / 'depth.svg').write_text(histogram_svg([t['trace_depth'] for t in traces], 'Candidate trace depth'), encoding='utf-8')
    (plots / 'distance.svg').write_text(histogram_svg([e['target_turn'] - e['source_turn'] for _, e in edges], 'Source-target distance'), encoding='utf-8')
    links = ''.join(f'<li><a href="examples/{f}">{f}</a></li>' for files in selected.values() for f in files)
    (directory / 'report.html').write_text(f'<!doctype html><meta charset="utf-8">{STYLE}<h1>Policy dependency diagnostics</h1><p>Red: support. Blue: suppression. Attention proposes; masking measures policy dependence. No environment causal-effect claim.</p><ul>{links}</ul><img src="plots/depth.svg"><img src="plots/distance.svg"><pre>{esc(json.dumps(summary, indent=2))}</pre>', encoding='utf-8')
    return summary


def heatmap_svg(edge):
    """Fixed-width wrapped token cells; preserve every token in a shareable SVG."""
    tokens, values = edge['target_tokens'], edge['token_delta']
    scale = max(map(abs, values)) or 1.
    width, columns = 1320, 10
    height = 70 + 74 * math.ceil(len(values) / columns)
    cells = []
    for i, (token, value) in enumerate(zip(tokens, values, strict=True)):
        x, y = 12 + (i % columns) * 130, 48 + (i // columns) * 74
        rgb = (220, 55, 45) if value >= 0 else (45, 100, 220)
        strength = abs(value) / scale
        color = '#%02x%02x%02x' % tuple(round(255 + (c - 255) * strength) for c in rgb)
        # Full tokens are also in <title> and HTML; SVG lengthAdjust keeps long tokens in their cells.
        cells.append(f'<g><title>{esc(token)}: {value:+.6f}</title><rect x="{x}" y="{y}" width="124" height="64" rx="4" fill="{color}" stroke="#ddd"/><text x="{x+5}" y="{y+23}" font-size="12" textLength="112" lengthAdjust="spacingAndGlyphs">{esc(repr(token))}</text><text x="{x+5}" y="{y+48}" font-size="13">{value:+.5f}</text></g>')
    return f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}"><rect width="100%" height="100%" fill="white"/><text x="12" y="28" font-size="18">Turn {edge["source_turn"]} → {edge["target_turn"]}; red=support, blue=suppression; scale ±{scale:.4f}</text>{"".join(cells)}</svg>'
