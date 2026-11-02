# ORSD

Online rubric-guided training for interactive agents, implemented on veRL.
This checkout is a source-only export of the current development tree. It includes historical ablations and experimental modules; these are not all part of the paper method.

## Setup

Use a CUDA environment compatible with the dependencies in `pyproject.toml` and the relevant `requirements*.txt`, then run `pip install -e .`.
Download Qwen model weights and the official WebShop/ALFWorld dependencies separately. Dataset preprocessing and setup scripts are in `examples/agent_rubric2/` and `examples/agent_rubric/`.

Copy `.env.example` to `.env`, fill in your paths and an authorized OpenAI-compatible judge endpoint, and export the variables (`set -a; source .env; set +a`). The example endpoint is a placeholder. No credentials are bundled.

## Experiments

Launchers live in `examples/agent_rubric2/`:

- `run_webshop_small_qwen3_4b_bestof4_judge_4gpu.sh`: four complete candidate bundles, highest Spearman selection.
- `run_webshop_small_qwen3_4b_opd_only_no_pool_4gpu.sh`: single bundle without judging.
- `run_webshop_small_qwen3_1_7b_bestof4_weak_verifier_4gpu.sh` and `run_webshop_small_qwen3_4b_bestof4_weak_verifier_4gpu.sh`: Flash-verifier ablations.
- `run_webshop_small_qwen3_1_7b_offline_rubric_opd_4gpu.sh`: fixed rubric snapshot ablation. Snapshot source step must match the exported parquet metadata.

For example, after preparing data and environment variables:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 WEBSHOP_FIRST_VLLM_GPU_UTIL=0.5 WEBSHOP_WEAK_VERIFIER_SNAPSHOT_ENABLE=False bash examples/agent_rubric2/run_webshop_small_qwen3_4b_bestof4_weak_verifier_4gpu.sh
```

Check each launcher's printed configuration. Some historical scripts retain server-style default paths; override them for your environment. Training budgets and gates differ across historical experiments.

## Scope and attribution

Model weights, datasets, logs, local configuration, Git history and checkpoints are excluded. WebShop and OPSD external repositories are not redistributed here. The separate attention-traceback experiments are not the ORSD paper method.

Upstream veRL copyright and license notices are retained in `LICENSE`, `Notice.txt` and source headers. This export has not been GPU smoke-tested as a standalone installation.
