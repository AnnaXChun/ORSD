"""Subset existing benchmark parquet without editing prompts; build ScienceWorld tasks."""
import argparse
import hashlib
import json
import random
from pathlib import Path


def main():
    import pyarrow as pa
    import pyarrow.parquet as pq
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", choices=["webshop", "alfworld", "scienceworld"], required=True)
    parser.add_argument("--input", help="existing ORSD parquet; required for WebShop/ALFWorld")
    parser.add_argument("--output", required=True)
    parser.add_argument("--tasks", type=int, default=1)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--max-steps", type=int, default=100)
    args = parser.parse_args()
    if args.tasks < 1:
        raise ValueError("tasks must be positive")
    rng = random.Random(args.seed)
    if args.benchmark != "scienceworld":
        if not args.input:
            parser.error("--input is required; existing ORSD prompts must be reused")
        table = pq.read_table(args.input)
        if len(table) < args.tasks:
            raise ValueError("Not enough tasks in source parquet")
        indices = sorted(rng.sample(range(len(table)), args.tasks))
        table = table.take(pa.array(indices))
        rows = table.to_pylist()
        for row in rows:
            extra = row["extra_info"]
            if isinstance(extra, str):
                extra = json.loads(extra)
            interaction = extra["interaction_kwargs"]
            if interaction["name"] != args.benchmark:
                raise ValueError("Mixed benchmark parquet")
            if interaction.get("inject_rubric"):
                raise ValueError("Input has rubric injection; use the existing plain-policy parquet")
            if args.benchmark == "webshop" and not (interaction.get("use_small") and
                    interaction.get("num_products") == 1000 and interaction.get("max_steps") == 15):
                raise ValueError("Use ORSD WebShop-small parquet with 1000 products and max_steps=15")
    else:
        from importlib.metadata import version
        from scienceworld import ScienceWorldEnv
        if version("scienceworld") != "1.2.3":
            raise ValueError("ScienceWorld 1.2.3 required")
        names = ["boil", "melt", "freeze", "use-thermometer", "test-conductivity", "power-component", "grow-plant", "grow-fruit"]
        # Interleave task types, choosing fixed official test variations only.
        env = ScienceWorldEnv(envStepLimit=1000000)
        tasks = []
        try:
            pools = {}
            for name in names:
                env.load(name, 0, "", generateGoldPath=False)
                variants = sorted(env.get_variations_test())
                rng.shuffle(variants)
                pools[name] = variants
            while len(tasks) < args.tasks:
                progressed = False
                for name in names:
                    if pools[name] and len(tasks) < args.tasks:
                        tasks.append((name, pools[name].pop()))
                        progressed = True
                if not progressed:
                    raise ValueError("Requested more official test variations than available")
        finally:
            env.close()
        rows = []
        for i, (name, variation) in enumerate(tasks):
            rows.append({"data_source": "scienceworld", "agent_name": "tool_agent",
                "prompt": [{"role": "system", "content": "You are a science environment agent. Follow the task and current observation. Reply with exactly one line: ACTION: <one currently valid action>"},
                           {"role": "user", "content": "Complete the task given by the environment."}],
                "ability": "agent", "reward_model": {"style": "rule", "ground_truth": ""},
                "extra_info": {"index": i, "split": "test", "interaction_kwargs": {
                    "name": "scienceworld", "task_name": name, "variation_idx": variation,
                    "task_id": f"{name}:{variation}", "max_steps": args.max_steps,
                    "observation_style": "compact",
                    "seed": args.seed, "inject_rubric": False, "use_interaction_reward": True}}})
        table = pa.Table.from_pylist(rows)
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, output)
    manifest = {"benchmark": args.benchmark, "tasks": args.tasks, "seed": args.seed,
                "source": args.input, "source_sha256": hashlib.sha256(Path(args.input).read_bytes()).hexdigest() if args.input else None,
                "extra_info": [r["extra_info"] for r in rows]}
    output.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
