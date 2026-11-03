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

Paper experiments use Qwen3-1.7B and Qwen3-4B on ALFWorld and WebShop. Launchers are in `examples/agent_rubric2/`.

### Main experiments

- `run_alfworld_first_version_4gpu.sh`
- `run_alfworld_qwen3_4b_first_version.sh`
- `run_webshop_small_first_version_4gpu.sh`
- `run_webshop_small_qwen3_4b_bestof4_judge_4gpu.sh`

### WebShop ablations

| Ablation | Launcher |
| --- | --- |
| Offline rubric | `run_webshop_small_qwen3_1_7b_offline_rubric_opd_4gpu.sh`<br>`run_webshop_small_qwen3_4b_offline_rubric_opd_4gpu.sh` |
| Single bundle, no selection | `run_webshop_small_qwen3_1_7b_opd_only_no_pool_4gpu.sh`<br>`run_webshop_small_qwen3_4b_opd_only_no_pool_4gpu.sh` |
| Random selection from four bundles | `run_webshop_small_qwen3_1_7b_random_bundle_of4_4gpu.sh`<br>`run_webshop_small_qwen3_4b_random_bundle_of4_4gpu.sh` |
| Coefficient sensitivity (0.001 and 0.1) | `run_webshop_small_qwen3_1_7b_orsd_opd0p001_4gpu.sh`<br>`run_webshop_small_qwen3_1_7b_orsd_opd0p1_4gpu.sh` |

The main experiment supplies the default coefficient 0.01. Offline runs require `OFFLINE_RUBRIC_PARQUET` and matching `OFFLINE_RUBRIC_EXPECTED_STEP` metadata.

### GRPO baselines

- `run_alfworld_matched_grpo_4gpu_b8_n8_s300.sh`
- `run_alfworld_qwen3_4b_matched_grpo.sh`
- `run_webshop_small_matched_grpo.sh`
- `run_webshop_small_qwen3_4b_matched_grpo_hybrid.sh`

Other retained shell files provide shared training, data setup or snapshot-export support.

## License

See [LICENSE](LICENSE) and [Notice.txt](Notice.txt) for licensing and upstream attribution.
