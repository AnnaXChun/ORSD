# ORSD

Online rubric-guided reinforcement learning for interactive agents, built on veRL. ORSD generates process rubrics from agent trajectories and uses rubric-conditioned token likelihoods for credit assignment.

## Setup

```bash
pip install -e .
cp .env.example .env
```

Set model paths and judge API credentials in `.env`, then load them:

```bash
set -a
source .env
set +a
```

Prepare the official WebShop or ALFWorld environment and datasets using the setup and preprocessing scripts in `examples/agent_rubric2/` and `examples/agent_rubric/`.

## Experiments

Launchers are in `examples/agent_rubric2/`:

| Script | Experiment |
| --- | --- |
| `run_webshop_small_qwen3_4b_bestof4_judge_4gpu.sh` | Generate four rubric bundles and select by Spearman correlation. |
| `run_webshop_small_qwen3_4b_opd_only_no_pool_4gpu.sh` | Generate one bundle without verifier selection. |
| `run_webshop_small_qwen3_1_7b_bestof4_weak_verifier_4gpu.sh` | Qwen3-1.7B with a Flash verifier. |
| `run_webshop_small_qwen3_4b_bestof4_weak_verifier_4gpu.sh` | Qwen3-4B with a Flash verifier. |
| `run_webshop_small_qwen3_1_7b_offline_rubric_opd_4gpu.sh` | Train with fixed offline rubrics. |

Example:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
WEBSHOP_FIRST_VLLM_GPU_UTIL=0.5 \
WEBSHOP_BESTOF4_SNAPSHOT_ENABLE=False \
bash examples/agent_rubric2/run_webshop_small_qwen3_4b_bestof4_judge_4gpu.sh
```

## License

See [LICENSE](LICENSE) and [Notice.txt](Notice.txt) for licensing and upstream attribution.
