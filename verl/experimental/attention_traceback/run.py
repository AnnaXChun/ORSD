"""Server-only replay CLI. Output is atomically resumable per completed trajectory."""
import argparse
from collections import defaultdict
import hashlib
from importlib.metadata import version, PackageNotFoundError
import json
from pathlib import Path
import platform
import statistics


def replay_shard(records, model, config, trace_config, shards, worker):
    """One independent server GPU per process; atomic per-trajectory outputs."""
    from .replay import QwenReplay
    from .trace import trace_record, TRACE_IMPLEMENTATION_VERSION
    config = dict(config, device=f'cuda:{worker}')
    backend = QwenReplay(model, config)
    for i, record in enumerate(records):
        path = Path(shards) / f"{record['trajectory_id']}.json"
        complete = json.loads(path.read_text()) if path.exists() else {}
        if (complete.get('trace_implementation_version') == TRACE_IMPLEMENTATION_VERSION
                and complete.get('diagnostic_truncation_roots') == trace_config['diagnostic_truncation_roots']):
            continue
        traces = trace_record(record, backend, trace_config)
        temp = path.with_suffix('.tmp')
        temp.write_text(json.dumps({'record': record, 'traces': traces,
            'trace_implementation_version': TRACE_IMPLEMENTATION_VERSION,
            'diagnostic_truncation_roots': trace_config['diagnostic_truncation_roots']}, ensure_ascii=False,
            allow_nan=False), encoding='utf-8')
        temp.replace(path)
        print(f"GPU {worker}: {i+1}/{len(records)} trajectories, {len(traces)} roots", flush=True)


def load_records(path):
    path = Path(path)
    if path.is_dir():
        files = sorted(path.glob('*.json'))
        records = [json.loads(file.read_text()) for file in files]
    elif path.suffix == '.jsonl':
        records = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    else:
        records = [json.loads(path.read_text())]
    if not records:
        raise ValueError('No captured trajectories; text-only logs are not lossless rollout caches')
    if len({r['trajectory_id'] for r in records}) != len(records):
        raise ValueError('Duplicate trajectory IDs in cache')
    return records


def assign_grpo(records, expected_siblings):
    # Same sample standard deviation and 1e-6 epsilon as veRL GRPO.
    groups = defaultdict(list)
    for record in records:
        groups[record['task_id']].append(record)
    for task, siblings in groups.items():
        if len(siblings) != expected_siblings:
            raise ValueError(f'{task}: expected {expected_siblings} siblings, found {len(siblings)}; do not mix cache runs')
        rewards = [r['trajectory_reward'] for r in siblings]
        mean, std = statistics.mean(rewards), statistics.stdev(rewards) if len(rewards) > 1 else 1.0
        for record in siblings:
            record['grpo_advantage'] = (record['trajectory_reward'] - mean) / (std + 1e-6)
    return groups


def main():
    from omegaconf import OmegaConf
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--model', required=True, help='same existing HF Qwen3-1.7B checkpoint as rollout')
    parser.add_argument('--config', default='examples/attention_traceback/config.yaml')
    parser.add_argument('--stage', choices=['smoke', 'phase-a'], default='smoke')
    parser.add_argument('--smoke-dir', help='parent directory containing three smoke_pass.json receipts')
    parser.add_argument('--workers', type=int, default=1, help='independent replay GPU processes')
    parser.add_argument('overrides', nargs='*')
    args = parser.parse_args()
    if args.workers < 1:
        raise ValueError('workers must be positive')
    cfg = OmegaConf.merge(OmegaConf.load(args.config), OmegaConf.from_dotlist(args.overrides))
    if cfg.credit_assignment.type != 'attention_traceback':
        raise ValueError('Replay CLI requires credit_assignment.type=attention_traceback')
    config = OmegaConf.to_container(cfg.attention_traceback, resolve=True)
    if config['intervention'] not in {'mask_span', 'drop_span'}:
        raise ValueError('intervention must be mask_span or drop_span')
    if config['max_trace_depth'] < 1 or config['timeout_root_window'] < 1:
        raise ValueError('Trace limits must be positive')
    if config['delta_threshold'] is not None and config['delta_threshold'] < 0:
        raise ValueError('delta_threshold must be null or nonnegative')
    records = load_records(args.input)
    benchmark = records[0]['benchmark']
    if benchmark not in {'webshop', 'alfworld', 'scienceworld'} or any(r['benchmark'] != benchmark for r in records):
        raise ValueError('One supported benchmark per replay run')
    count = 2 if args.stage == 'smoke' else 4
    groups = assign_grpo(records, count)
    if args.stage == 'smoke' and len(groups) != 1:
        raise ValueError('Smoke requires exactly 1 task × 2 trajectories')
    smoke_model_fingerprints = []
    if args.stage == 'phase-a':
        if not 50 <= len(groups) <= 100:
            raise ValueError('Phase A requires 50–100 tasks × 4 trajectories')
        if not args.smoke_dir:
            raise ValueError('Run all three server smokes first, then provide --smoke-dir')
        for name in ('webshop', 'alfworld', 'scienceworld'):
            receipt = json.loads((Path(args.smoke_dir) / name / 'smoke_pass.json').read_text())
            smoke_model_fingerprints.append(receipt.get('model_fingerprint'))
            if not receipt['passed'] or receipt['intervention'] != config['intervention']:
                raise ValueError('Smoke receipt missing or intervention differs')
            if Path(receipt['model']).resolve() != Path(args.model).resolve():
                raise ValueError('Smoke model differs from Phase A model')
    from .capture import tokenizer_fingerprint, checkpoint_fingerprint
    from .replay import QwenReplay, validate_record
    from .trace import trace_record, select_trace_roots, TRACE_IMPLEMENTATION_VERSION
    from .report import write_report
    trace_config = dict(config, diagnostic_truncation_roots=args.stage == 'smoke')
    root_diagnostics = []
    for record in records:
        roots, origin = select_trace_roots(record, trace_config)
        diagnostic = {'trajectory_id': record['trajectory_id'], 'done_reason': record['done_reason'],
                      'turn_count': len(record['turns']), 'token_count': len(record['input_ids']),
                      'root_turns': roots, 'root_origin': origin}
        root_diagnostics.append(diagnostic)
        print('Root selection: ' + json.dumps(diagnostic), flush=True)
    model_fingerprint = checkpoint_fingerprint(args.model)
    if any(value != model_fingerprint for value in smoke_model_fingerprints):
        raise ValueError('Smoke was run with different checkpoint weights')
    config["tokenizer_path"] = records[0].get("tokenizer_path", args.model)
    backend = QwenReplay(args.model, config)
    if records[0].get("tokenizer_chat_template") is not None:
        backend.tokenizer.chat_template = records[0]["tokenizer_chat_template"]
    fingerprint = tokenizer_fingerprint(backend.tokenizer)
    for record in records:
        validate_record(record)
        if record.get('model_fingerprint') != model_fingerprint:
            raise ValueError('Checkpoint weights differ from captured policy')
        if record['tokenizer_fingerprint'] != fingerprint:
            raise ValueError('Tokenizer vocabulary/chat template differs from captured rollout')
        if Path(record['model_path']).resolve() != Path(args.model).resolve():
            raise ValueError('Replay checkpoint path differs from captured policy; use the same checkpoint')
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    packages = {}
    for name in ('torch', 'transformers', 'verl', 'vllm', 'sglang', 'scienceworld'):
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = None
    signature = hashlib.sha256(json.dumps({'config': config, 'records': records}, sort_keys=True).encode()).hexdigest()
    manifest = {'signature': signature, 'model': str(Path(args.model).resolve()), 'config': config,
                'tokenizer_fingerprint': fingerprint, 'model_fingerprint': model_fingerprint, 'packages': packages, 'python': platform.python_version(),
                'stage': args.stage, 'benchmark': benchmark, 'intervention': config['intervention']}
    manifest['observation_styles'] = sorted({
        turn.get('environment_metrics', {}).get('observation_style', 'unspecified')
        for record in records for turn in record['turns']})
    manifest_path = output / 'manifest.json'
    if manifest_path.exists() and json.loads(manifest_path.read_text())['signature'] != signature:
        raise ValueError('Output belongs to different data/config; choose a new output directory')
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    (output / 'root_diagnostics.json').write_text(json.dumps(root_diagnostics, indent=2), encoding='utf-8')
    if args.stage == 'smoke':
        (output / 'smoke_pass.json').write_text(json.dumps({**manifest, 'passed': False,
            'scope': 'replay_mechanics_only', 'status': 'replay_in_progress'}, indent=2), encoding='utf-8')
    shards = output / 'completed'
    shards.mkdir(exist_ok=True)
    if args.workers > 1:
        import torch
        import multiprocessing
        from concurrent.futures import ProcessPoolExecutor
        if args.workers > torch.cuda.device_count():
            raise ValueError('Not enough visible GPUs for replay workers')
        del backend
        torch.cuda.empty_cache()
        with ProcessPoolExecutor(max_workers=args.workers,
                mp_context=multiprocessing.get_context('spawn')) as pool:
            futures = [pool.submit(replay_shard, records[worker::args.workers], args.model, config,
                trace_config, str(shards), worker) for worker in range(args.workers)]
            for future in futures:
                future.result()
    all_traces, replayed = [], []
    for i, record in enumerate(records):
        path = shards / f"{record['trajectory_id']}.json"
        complete = json.loads(path.read_text()) if path.exists() else {}
        if (complete.get('trace_implementation_version') == TRACE_IMPLEMENTATION_VERSION
                and complete.get('diagnostic_truncation_roots') == trace_config['diagnostic_truncation_roots']):
            traces, record = complete['traces'], complete['record']
        else:
            traces = trace_record(record, backend, trace_config)
            temp = path.with_suffix('.tmp')
            temp.write_text(json.dumps({'record': record, 'traces': traces,
                'trace_implementation_version': TRACE_IMPLEMENTATION_VERSION,
                'diagnostic_truncation_roots': trace_config['diagnostic_truncation_roots']}, ensure_ascii=False, allow_nan=False), encoding='utf-8')
            temp.replace(path)
        replayed.append(record)
        all_traces.extend(traces)
        print(f"{benchmark}: {i + 1}/{len(records)} trajectories, {len(traces)} roots", flush=True)
    for name, rows in (('traces.jsonl', all_traces), ('trajectories.jsonl', replayed)):
        path = output / name
        temp = path.with_suffix('.tmp')
        temp.write_text(''.join(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n' for row in rows), encoding='utf-8')
        temp.replace(path)
    summary = write_report(output, replayed, all_traces, config)
    if args.stage == 'smoke':
        # Each trajectory must exercise at least one candidate and all controls
        # wherever a second eligible history exists. Do not pass an empty trace.
        ids_with_edges = {t['trajectory_id'] for t in all_traces if t['edges']}
        if len(ids_with_edges) != 2:
            raise RuntimeError('Smoke needs at least one history edge per trajectory. '
                               f'Inspect {output / "root_diagnostics.json"}: '
                               + json.dumps(root_diagnostics))
        receipt = {**manifest, 'passed': True, 'trajectories': 2,
                   'candidate_edges': summary['number_of_unique_candidate_edges'],
                   'trace_implementation_version': TRACE_IMPLEMENTATION_VERSION,
                   'scope': 'replay_mechanics_only',
                   'truncation_diagnostic_traces': sum(t.get('root_origin') == 'truncation_diagnostic' for t in all_traces)}
        (output / 'smoke_pass.json').write_text(json.dumps(receipt, indent=2), encoding='utf-8')
    print(f"Report: {output / 'report.html'}", flush=True)


if __name__ == '__main__':
    main()
