"""Validate a fixed-rubric ablation without dropping or reordering training tasks."""
import argparse
import json
import math

import pyarrow.parquet as pq


def first_content_difference(left, right, path):
    """Compare decoded values, not Arrow offset widths or struct field order."""
    if type(left) is not type(right):
        return path, left, right
    if isinstance(left, dict):
        if left.keys() != right.keys():
            return path + ".keys", sorted(left), sorted(right)
        for key in left:
            mismatch = first_content_difference(left[key], right[key], f"{path}.{key}")
            if mismatch:
                return mismatch
    elif isinstance(left, (list, tuple)):
        if len(left) != len(right):
            return path + ".length", len(left), len(right)
        for index, (a, b) in enumerate(zip(left, right)):
            mismatch = first_content_difference(a, b, f"{path}[{index}]")
            if mismatch:
                return mismatch
    elif isinstance(left, float) and math.isnan(left) and math.isnan(right):
        return None
    elif left != right:
        return path, left, right
    return None


def validate_snapshot(original_path, snapshot_path, expected_step=None):
    original = pq.read_table(original_path)
    snapshot = pq.read_table(snapshot_path)
    rubric_columns = {"rubric", "rubric_valid", "rubric_selected_count", "rubric_snapshot_step", "rubric_snapshot_row_index"}
    required = {"rubric", "rubric_valid", "rubric_selected_count"}
    if not required.issubset(snapshot.column_names) or not snapshot.num_rows:
        raise ValueError("Snapshot must be nonempty and contain rubric, rubric_valid and rubric_selected_count")
    original_columns = [name for name in original.column_names if name not in rubric_columns]
    if not set(original_columns).issubset(snapshot.column_names):
        missing = sorted(set(original_columns).difference(snapshot.column_names))
        raise ValueError(f"Snapshot is missing original training columns: {missing}")
    if original.num_rows != snapshot.num_rows:
        raise ValueError(
            f"Snapshot tasks/order/prompts differ: row counts original={original.num_rows}, "
            f"snapshot={snapshot.num_rows}. Snapshot export uses the loaded training dataset, "
            "which may have been filtered or subsampled; check the source run and dataset paths. "
            f"original={original_path}; snapshot={snapshot_path}"
        )
    left = original.select(original_columns)
    right = snapshot.select(original_columns)
    if not left.equals(right, check_metadata=False):
        # Training normalizes large_string to string before snapshot export.
        # Arrow schema equality is stronger than the task-identity requirement.
        for index, (a, b) in enumerate(zip(left.to_pylist(), right.to_pylist())):
            mismatch = first_content_difference(a, b, f"row[{index}]")
            if mismatch:
                field, before, after = mismatch
                raise ValueError(
                    f"Snapshot tasks/order/prompts differ at {field}: "
                    f"original={repr(before)[:200]}; snapshot={repr(after)[:200]}. "
                    f"original_file={original_path}; snapshot_file={snapshot_path}"
                )
        changed_types = [
            f"{name}: {left.schema.field(name).type} -> {right.schema.field(name).type}"
            for name in original_columns
            if left.schema.field(name).type != right.schema.field(name).type
        ]
        print("Snapshot contents and row order match; accepted Arrow representation differences: "
              + "; ".join(changed_types)[:1000])
    for index, row in enumerate(snapshot.to_pylist()):
        extra = row.get("extra_info") or {}
        if isinstance(extra, str):
            extra = json.loads(extra)
        if (extra.get("interaction_kwargs") or {}).get("inject_rubric", False):
            raise ValueError(f"Row {index}: rubric must remain hidden from actor rollout")
        if not row["rubric_valid"] or not str(row["rubric"] or "").strip() or not 5 <= int(row["rubric_selected_count"] or 0) <= 8:
            raise ValueError(f"Row {index}: expected one valid complete 5-8 criterion bundle")
    steps = sorted(set(snapshot["rubric_snapshot_step"].to_pylist())) if "rubric_snapshot_step" in snapshot.column_names else []
    if expected_step is not None and steps != [expected_step]:
        raise ValueError(f"Expected rubric snapshot step {expected_step}, got {steps}; select the matching export")
    print(f"Offline rubric preflight: {snapshot.num_rows} unchanged train rows; hidden complete bundles; source steps={steps}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("original")
    parser.add_argument("snapshot")
    parser.add_argument("--expected-step", type=int)
    args = parser.parse_args()
    validate_snapshot(args.original, args.snapshot, args.expected_step)
