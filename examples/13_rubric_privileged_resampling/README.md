# Experiment 13: selected-rubric privileged resampling

This is an inference-only causal test. It does not train or modify the model.
The evaluator is standalone and does not import the VERL runtime, so
`tensordict` and other training-only dependencies are not required.
The default `opsd_filtered` run has three separate stages:

1. The base model produces an independent set of **selection rollouts** from
   the original problem.
2. The original OPSD judge and reliability formula select candidate rubric
   items whose satisfaction separates correct and incorrect rollouts. It also
   removes answer-leaking and never-satisfied criteria.
3. Using a fresh random seed, the same base checkpoint samples two paired
   evaluation conditions:
   - `baseline`: the original problem;
   - `rubric privilege`: the original problem plus the selected rubric.

Selection rollouts are never counted in the final comparison. The script
reports solution accuracy (Average@N), Pass@N, majority-vote accuracy,
selection success rate, paired problem wins/ties/losses, and a problem-level
bootstrap confidence interval for the accuracy difference.

The primary result is computed over the **entire evaluation set**. When no
rubric passes selection for a problem, the rubric policy falls back to the
exact baseline samples, so that problem contributes zero paired change. A
separate eligible-subset result is retained only as a mechanism diagnostic;
it is not a held-out headline metric.

The input parquet provides problem-only **candidate** rubrics in `rubric`.
`rubric_valid` only means that generation was parseable; reliability filtering
is performed online in stage 2.

## AIME evaluation

`run_dataset.sh` first generates problem-only candidate rubrics for the local
evaluation parquet, then runs selection and paired resampling. The default is
AIME 2024 with all 30 problems, 12 selection rollouts, and 12 fresh samples per
evaluation condition:

```bash
cd /workspace/verl_wsm
export RUBRIC_API_KEY='your-key'
DATASET=aime24 \
bash examples/13_rubric_privileged_resampling/run_dataset.sh
```

Supported local evaluation datasets:

```text
aime24  -> Data/aime_2024.parquet
aime25  -> Data/aime25_test.parquet
math500 -> Data/math500_test.parquet
amc23   -> Data/amc23_test.parquet
```

Run AIME24 and AIME25 sequentially:

```bash
DATASETS='aime24 aime25' \
bash examples/13_rubric_privileged_resampling/run_eval_suite.sh
```

## DashScope Qwen3-8B capability check

This variant keeps the same candidate rubrics (when the local run has already
created them), OPSD filtering, sampling parameters, thinking mode, N, scoring,
and full-set fallback statistics. It replaces the local Qwen3-1.7B solver with
DashScope `qwen3-8b` for selection rollouts, baseline samples, and
rubric-privileged samples. The OPSD reliability judge remains unchanged. If no
candidate parquet exists yet, the remote model creates it once and later runs
reuse that exact file.

```bash
cd /workspace/verl_wsm
export RUBRIC_API_KEY='your-dashscope-key'
DATASET=aime24 \
bash examples/13_rubric_privileged_resampling/run_remote_qwen3_8b.sh
```

Remote outputs are isolated under:

```text
experiment_outputs/13_rubric_privileged_resampling/remote_qwen3_8b/
```

The full N=12 remote protocol can require roughly 700--1,000 independent
long-thinking API calls on AIME24. Use the N=4 capability diagnostic first:

```bash
DATASET=aime24 \
bash examples/13_rubric_privileged_resampling/run_remote_qwen3_8b_quick.sh
```

It still evaluates all 30 AIME24 problems, uses the same 8192-token cap and
OPSD filter, but uses four selection rollouts and four evaluation samples.
Results are isolated under
`experiment_outputs/13_rubric_privileged_resampling/remote_qwen3_8b_quick_n4/`.

Run the full N=12 DashScope Qwen3-14B comparison with exactly the same
candidate rubrics and protocol as the Qwen3-8B run:

```bash
DATASET=aime24 \
bash examples/13_rubric_privileged_resampling/run_remote_qwen3_14b.sh
```

Its independent result directory is
`experiment_outputs/13_rubric_privileged_resampling/remote_qwen3_14b/`.

Run all four evaluation datasets:

```bash
bash examples/13_rubric_privileged_resampling/run_eval_suite.sh
```

Candidate-rubric parquets are generated once under
`experiment_outputs/13_rubric_privileged_resampling/candidate_rubrics/` and
reused on later runs. Each dataset has an independent output/progress directory,
so interrupted runs resume safely.

If the selected rubric has already been written to the parquet and no online
judge should be used:

```bash
SELECTION_MODE=preselected \
RUBRIC_COLUMN=selected_rubric \
RUBRIC_VALID_COLUMN=selected_rubric_valid \
bash examples/13_rubric_privileged_resampling/run.sh
```

For a cheaper smoke test:

```bash
DATASET=aime24 LIMIT=10 SELECTION_NUM_SAMPLES=4 NUM_SAMPLES=4 \
bash examples/13_rubric_privileged_resampling/run_dataset.sh
```

The summary is written to
`experiment_outputs/13_rubric_privileged_resampling/<dataset>/summary.json`.
Per-problem selection and paired scores are appended to `paired_results.jsonl`,
so the same command resumes after interruption. Pass `--overwrite` to
deliberately restart.

## Training-set scale test

Scale the original 64-problem local Qwen3-1.7B smoke test to 1,000 unique
training problems with the same N=4 selection/evaluation protocol:

```bash
bash examples/13_rubric_privileged_resampling/run_train_1000.sh
```

It reads `Data/train_with_rubrics.parquet` and writes resumable results to
`experiment_outputs/13_rubric_privileged_resampling/train_1000_qwen3_1_7b/`.
The primary comparison is computed over all 1,000 attempted problems; the
eligible subset remains a separate diagnostic.

Run the same N=4 training-set protocol on 200 problems with remote DashScope
Qwen3-8B:

```bash
bash examples/13_rubric_privileged_resampling/run_train_200_remote_qwen3_8b.sh
```

It reuses `Data/train_with_rubrics.parquet` and stores independent resumable
results under
`experiment_outputs/13_rubric_privileged_resampling/train_200_qwen3_8b_remote/`.

If there is no validity column because every non-empty candidate block is
usable, set:

```bash
RUBRIC_VALID_COLUMN='' \
bash examples/13_rubric_privileged_resampling/run.sh
```
