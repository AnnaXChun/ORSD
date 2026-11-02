#!/usr/bin/env python3
"""Audit WebShop-small data/index/runtime alignment without using an LLM.

This script distinguishes an environment/index bug from a weak policy by
checking the exact goal mapping, Lucene retrieval, reward self-consistency,
and an oracle purchase trajectory for a small set of held-out tasks.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--webshop-root",
        type=Path,
        default=Path(os.environ.get("WEBSHOP_ROOT", root / "WebShop")),
    )
    parser.add_argument(
        "--parquet",
        type=Path,
        default=root / "Data/agent_rubric/webshop_small/webshop_test.parquet",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=8,
        help="Number of parquet rows for retrieval and oracle-purchase checks.",
    )
    parser.add_argument(
        "--row-offset",
        type=int,
        default=0,
        help="Start row in the parquet; use this to inspect another slice.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1234,
        help="Must match WebshopInteraction.seed and preprocessing.",
    )
    parser.add_argument("--max-search-results", type=int, default=50)
    parser.add_argument("--max-steps", type=int, default=15)
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional JSON report path.",
    )
    return parser.parse_args()


def load_parquet_rows(path: Path) -> list[dict[str, Any]]:
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError("pyarrow is required to read the VERL parquet") from exc
    return pq.read_table(path).to_pylist()


def normalize_extra_info(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, str):
        value = json.loads(value)
    if not isinstance(value, dict):
        raise TypeError(f"extra_info must be dict or JSON string, got {type(value).__name__}")
    return value


def extract_row_goal_idx(row: dict[str, Any]) -> int:
    extra_info = normalize_extra_info(row.get("extra_info"))
    interaction = extra_info.get("interaction_kwargs") or {}
    candidates = [interaction.get("goal_idx"), row.get("goal_idx")]
    for value in candidates:
        if value is not None:
            return int(value)
    raise KeyError("row has no goal_idx in extra_info.interaction_kwargs or top-level goal_idx")


def extract_row_instruction(row: dict[str, Any]) -> str:
    if row.get("instruction"):
        return str(row["instruction"])
    if row.get("task"):
        return str(row["task"])
    prompt = row.get("prompt") or []
    if isinstance(prompt, str):
        prompt = json.loads(prompt)
    for message in reversed(prompt):
        if message.get("role") == "user":
            return str(message.get("content", ""))
    raise KeyError("row has no instruction/task/user prompt")


def load_resource_ids(resource_path: Path) -> tuple[set[str], list[str]]:
    ids: set[str] = set()
    errors: list[str] = []
    if not resource_path.is_file():
        return ids, [f"missing index resource JSONL: {resource_path}"]
    with resource_path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
                ids.add(str(item["id"]))
            except Exception as exc:  # diagnostic path: retain all malformed lines
                errors.append(f"resource line {line_no}: {exc}")
    return ids, errors


def search_ranks(searcher: Any, queries: dict[str, str], target_asin: str, k: int) -> dict[str, Any]:
    result: dict[str, Any] = {}
    target = target_asin.casefold()
    for label, query in queries.items():
        try:
            hits = searcher.search(query, k=k)
            ids = []
            for hit in hits:
                document = searcher.doc(hit.docid)
                raw = json.loads(document.raw())
                ids.append(str(raw["id"]))
            rank = next((i for i, asin in enumerate(ids, 1) if asin.casefold() == target), None)
            result[label] = {
                "query": query,
                "target_rank": rank,
                "num_hits": len(ids),
                "top5_asins": ids[:5],
            }
        except Exception as exc:
            result[label] = {"query": query, "error": f"{type(exc).__name__}: {exc}"}
    return result


def direct_target_reward(server: Any, goal: dict[str, Any]) -> float:
    from web_agent_site.engine.goal import get_reward

    asin = goal["asin"]
    product = server.product_item_dict[asin]
    price = server.product_prices[asin]
    options = dict(goal.get("goal_options") or {})
    return float(get_reward(product, goal, price=price, options=options))


def choose_oracle_query(retrieval: dict[str, Any], max_results: int) -> tuple[str | None, int | None, str | None]:
    # Prefer the public goal query; title is a diagnostic fallback proving the
    # index/environment path even when the benchmark query has weak recall.
    for label in ("goal_query", "target_title"):
        item = retrieval.get(label) or {}
        rank = item.get("target_rank")
        if isinstance(rank, int) and 1 <= rank <= max_results:
            return str(item["query"]), rank, label
    return None, None, None


def run_oracle_purchase(
    env: Any,
    goal_idx: int,
    goal: dict[str, Any],
    retrieval: dict[str, Any],
    max_search_results: int,
    max_steps: int,
) -> dict[str, Any]:
    query, rank, query_source = choose_oracle_query(retrieval, max_search_results)
    result: dict[str, Any] = {
        "query_source": query_source,
        "query": query,
        "target_rank": rank,
        "actions": [],
        "official_reward": None,
        "done": False,
        "within_step_budget": False,
        "error": None,
    }
    if query is None or rank is None:
        result["error"] = (
            f"target ASIN was not retrieved in top {max_search_results} "
            "by goal query or exact title"
        )
        return result

    try:
        env.reset(goal_idx)

        def act(action: str) -> tuple[str, float, bool, Any]:
            result["actions"].append(action)
            return env.step(action)

        _, reward, done, _ = act(f"search[{query}]")
        target_page = (rank - 1) // 10 + 1
        for _ in range(1, target_page):
            available = {str(x).casefold() for x in env.get_available_actions()["clickables"]}
            if "next >" not in available:
                raise RuntimeError(f"target rank={rank}, but Next > is unavailable")
            _, reward, done, _ = act("click[Next >]")

        target_asin = str(goal["asin"])
        available = {str(x).casefold() for x in env.get_available_actions()["clickables"]}
        if target_asin.casefold() not in available:
            raise RuntimeError(
                f"target ASIN {target_asin} not clickable on expected result page {target_page}"
            )
        _, reward, done, _ = act(f"click[{target_asin}]")

        for option_name, option_value in (goal.get("goal_options") or {}).items():
            available = {str(x).casefold() for x in env.get_available_actions()["clickables"]}
            if str(option_value).casefold() not in available:
                raise RuntimeError(
                    f"goal option {option_name}={option_value!r} is not clickable on target item page"
                )
            _, reward, done, _ = act(f"click[{option_value}]")

        available = {str(x).casefold() for x in env.get_available_actions()["clickables"]}
        if "buy now" not in available:
            raise RuntimeError("Buy Now is unavailable after selecting target options")
        _, reward, done, _ = act("click[Buy Now]")
        result["official_reward"] = float(reward)
        result["done"] = bool(done)
        result["within_step_budget"] = len(result["actions"]) <= max_steps
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
    return result


def main() -> None:
    args = parse_args()
    webshop_root = args.webshop_root.expanduser().resolve()
    parquet_path = args.parquet.expanduser().resolve()
    products_path = webshop_root / "data/items_shuffle_1000.json"
    attributes_path = webshop_root / "data/items_ins_v2_1000.json"
    human_attributes_path = webshop_root / "data/items_human_ins.json"
    resource_path = webshop_root / "search_engine/resources_1k/documents.jsonl"
    index_path = webshop_root / "search_engine/indexes_1k"

    required = [
        webshop_root / "web_agent_site",
        products_path,
        attributes_path,
        human_attributes_path,
        index_path,
        parquet_path,
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise SystemExit("Missing required WebShop-small paths:\n  " + "\n  ".join(missing))
    if args.num_samples <= 0:
        raise SystemExit("--num-samples must be positive")

    rows = load_parquet_rows(parquet_path)
    selected_rows = rows[args.row_offset : args.row_offset + args.num_samples]
    if not selected_rows:
        raise SystemExit(
            f"No rows selected: parquet_rows={len(rows)}, row_offset={args.row_offset}"
        )

    # Patch upstream module globals before importing engine.py. This mirrors
    # preprocessing and WebshopInteraction._configure_small_catalog_defaults.
    if str(webshop_root) not in sys.path:
        sys.path.insert(0, str(webshop_root))
    import web_agent_site.utils as webshop_utils

    webshop_utils.DEFAULT_FILE_PATH = str(products_path)
    webshop_utils.DEFAULT_ATTR_PATH = str(attributes_path)
    from web_agent_site.envs.web_agent_text_env import WebAgentTextEnv

    old_random_state = random.getstate()
    random.seed(args.seed)
    try:
        env = WebAgentTextEnv(
            observation_mode="text_rich",
            file_path=str(products_path),
            human_goals=False,
            num_products=1000,
            filter_goals=None,
            limit_goals=-1,
        )
    finally:
        random.setstate(old_random_state)

    server = env.server
    catalog_ids = set(server.product_item_dict)
    resource_ids, resource_errors = load_resource_ids(resource_path)
    num_docs_value = server.search_engine.num_docs
    index_num_docs = int(num_docs_value() if callable(num_docs_value) else num_docs_value)

    # Mapping is cheap, so audit every row even when only a few rows run the
    # more expensive retrieval and oracle-purchase checks.
    mapping_mismatches = []
    for row_number, row in enumerate(rows):
        try:
            goal_idx = extract_row_goal_idx(row)
            row_instruction = extract_row_instruction(row)
            runtime_instruction = str(server.goals[goal_idx]["instruction_text"])
            if row_instruction != runtime_instruction:
                mapping_mismatches.append(
                    {
                        "row": row_number,
                        "goal_idx": goal_idx,
                        "parquet_instruction": row_instruction,
                        "runtime_instruction": runtime_instruction,
                    }
                )
        except Exception as exc:
            mapping_mismatches.append(
                {"row": row_number, "error": f"{type(exc).__name__}: {exc}"}
            )

    samples = []
    for relative_row, row in enumerate(selected_rows):
        parquet_row = args.row_offset + relative_row
        goal_idx = extract_row_goal_idx(row)
        goal = server.goals[goal_idx]
        target_asin = str(goal["asin"])
        retrieval = search_ranks(
            server.search_engine,
            {
                "goal_query": str(goal.get("query") or ""),
                "target_title": str(goal.get("name") or ""),
                "full_instruction": str(goal["instruction_text"]),
            },
            target_asin,
            args.max_search_results,
        )
        try:
            target_reward = direct_target_reward(server, goal)
            target_reward_error = None
        except Exception as exc:
            target_reward = None
            target_reward_error = f"{type(exc).__name__}: {exc}"
        oracle = run_oracle_purchase(
            env,
            goal_idx,
            goal,
            retrieval,
            args.max_search_results,
            args.max_steps,
        )
        sample = {
            "parquet_row": parquet_row,
            "goal_idx": goal_idx,
            "instruction": str(goal["instruction_text"]),
            "target_asin": target_asin,
            "target_in_catalog": target_asin in catalog_ids,
            "target_in_resource_jsonl": target_asin in resource_ids,
            "goal_options": goal.get("goal_options") or {},
            "target_price": server.product_prices.get(target_asin),
            "price_upper": goal.get("price_upper"),
            "direct_target_reward": target_reward,
            "direct_target_reward_error": target_reward_error,
            "retrieval": retrieval,
            "oracle_purchase": oracle,
        }
        samples.append(sample)
        print(
            f"row={parquet_row} goal={goal_idx} asin={target_asin} "
            f"query_rank={retrieval['goal_query'].get('target_rank')} "
            f"title_rank={retrieval['target_title'].get('target_rank')} "
            f"direct_reward={target_reward} "
            f"oracle_reward={oracle.get('official_reward')} "
            f"oracle_steps={len(oracle['actions'])} error={oracle.get('error')}"
        )

    hard_failures = []
    if len(server.goals) != 6910:
        hard_failures.append(f"expected 6910 synthetic goals, got {len(server.goals)}")
    if len(catalog_ids) != 1000:
        hard_failures.append(f"expected 1000 runtime catalog products, got {len(catalog_ids)}")
    if index_num_docs != 1000:
        hard_failures.append(f"expected 1000 Lucene documents, got {index_num_docs}")
    if resource_errors:
        hard_failures.extend(resource_errors)
    if resource_ids and resource_ids != catalog_ids:
        hard_failures.append(
            "resource/catalog ASIN mismatch: "
            f"missing_from_resource={len(catalog_ids - resource_ids)}, "
            f"extra_in_resource={len(resource_ids - catalog_ids)}"
        )
    if mapping_mismatches:
        hard_failures.append(
            f"{len(mapping_mismatches)}/{len(rows)} parquet rows do not match runtime goal ordering"
        )
    for sample in samples:
        prefix = f"row={sample['parquet_row']} goal={sample['goal_idx']}"
        if not sample["target_in_catalog"]:
            hard_failures.append(f"{prefix}: target ASIN absent from runtime catalog")
        if sample["direct_target_reward"] is None or not math.isclose(
            float(sample["direct_target_reward"]), 1.0, abs_tol=1e-8
        ):
            hard_failures.append(
                f"{prefix}: target product/options do not produce reward=1 "
                f"(got {sample['direct_target_reward']})"
            )
        oracle = sample["oracle_purchase"]
        if oracle["error"] is not None:
            hard_failures.append(f"{prefix}: oracle path failed: {oracle['error']}")
        elif not math.isclose(float(oracle["official_reward"]), 1.0, abs_tol=1e-8):
            hard_failures.append(
                f"{prefix}: oracle purchase reward is {oracle['official_reward']}, expected 1"
            )
        if not oracle["within_step_budget"]:
            hard_failures.append(
                f"{prefix}: oracle needs {len(oracle['actions'])} actions, over budget {args.max_steps}"
            )

    query_recall = sum(
        sample["retrieval"]["goal_query"].get("target_rank") is not None for sample in samples
    ) / len(samples)
    title_recall = sum(
        sample["retrieval"]["target_title"].get("target_rank") is not None for sample in samples
    ) / len(samples)
    oracle_success = sum(
        sample["oracle_purchase"].get("official_reward") is not None
        and math.isclose(float(sample["oracle_purchase"]["official_reward"]), 1.0, abs_tol=1e-8)
        for sample in samples
    ) / len(samples)

    report = {
        "verdict": "PASS" if not hard_failures else "FAIL",
        "interpretation": (
            "Data, index, runtime goal mapping, and official reward are aligned. "
            "A low model success rate is therefore a policy/search problem, not an index bug."
            if not hard_failures
            else "At least one environment/data/index invariant failed; inspect hard_failures and samples."
        ),
        "paths": {
            "webshop_root": str(webshop_root),
            "parquet": str(parquet_path),
            "products": str(products_path),
            "attributes": str(attributes_path),
            "index": str(index_path),
            "resource": str(resource_path),
        },
        "counts": {
            "parquet_rows": len(rows),
            "runtime_goals": len(server.goals),
            "catalog_products": len(catalog_ids),
            "resource_documents": len(resource_ids),
            "lucene_documents": index_num_docs,
            "mapping_mismatches": len(mapping_mismatches),
        },
        "sample_metrics": {
            "num_samples": len(samples),
            "goal_query_recall_at_50": query_recall,
            "target_title_recall_at_50": title_recall,
            "oracle_success_rate": oracle_success,
        },
        "hard_failures": hard_failures,
        "mapping_mismatch_examples": mapping_mismatches[:10],
        "samples": samples,
    }

    output_path = args.output
    if output_path is None:
        output_path = parquet_path.with_name("webshop_small_diagnostic.json")
    output_path = output_path.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print("\n" + "=" * 78)
    print(f"VERDICT: {report['verdict']}")
    print(f"Parquet/runtime mapping mismatches: {len(mapping_mismatches)}/{len(rows)}")
    print(f"Goal-query target recall@50: {query_recall:.1%}")
    print(f"Exact-title target recall@50: {title_recall:.1%}")
    print(f"Oracle official success: {oracle_success:.1%}")
    if hard_failures:
        print("Hard failures:")
        for failure in hard_failures:
            print(f"  - {failure}")
    print(f"Report: {output_path}")
    print("=" * 78)
    env.close()

    if hard_failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
