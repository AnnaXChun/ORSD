# Attention Traceback MVP（仅服务器诊断）

已在服务器 `verl0.8_wsm_attn` 完成三个 benchmark 的 smoke，服务器测试 11 项通过；详见 [服务器验证记录](SERVER_VALIDATION_20260928.md)。未在本机执行测试、加载模型或跑环境。ALFWorld / ScienceWorld 使用截断诊断 roots，smoke 只证明 replay 链路可执行；H1–H4 和 Go/No-Go 仍待正式数据。没有修改 GRPO loss 或实现 advantage shaping；仅补装 ScienceWorld 1.2.3 和 py4j 0.10.9.7，原 ML 栈保持不变。

## 备份

项目根目录 `backups/orsd_before_attention_traceback_20260928_204517.tar.gz` 保存本次修改前的工作树（包含 `.git`、未提交/未跟踪代码、WebShop 和 OPSD），旁边有 `.sha256`。排除了 `backups/`、`.venv/`、Python/pytest/ruff 缓存、根目录 checkpoints 和 logs。解包到单独目录，不要覆盖当前工作树。

## 实现与语义

- `credit_assignment.type=orsd` 是兼容默认值，沿用原配置；`grpo` 显式关闭 rubric/token credit；`attention_traceback` 强制 validation-only、无 rubric、无 KL、无 resume，并拒绝进入训练循环。
- 通过原 veRL `ToolAgentLoop` 和 WebShop/ALFWorld interaction 采集。正常 rollout 无 intervention。使用现有数据 parquet 的原始 prompt，WebShop-small 保持 1000 商品、15 actions；ALFWorld 保持 parquet 中的官方 split/step limit。
- 完成真实环境轨迹后保存原始 token IDs、assistant response 和 observation、真实 reward、各 turn span。模板的 next-assistant generation header 保留，不混入被删除的前一轮 observation。目标 action 只选择实际执行的 action 字符范围所覆盖的原始 token；解析失败时标记 `full_response_unparseable`，不能把这些 token 当作已知的 decision-bearing tokens。
- `turn_token_start` 从完整生成 response 开始，所以 historical turn masking 包括其 reasoning、action、EOS 和 resulting observation（不含下一轮 assistant header）。这是原 rollout 的完整历史交互。
- 使用同一 Qwen3-1.7B checkpoint；模型权重 SHA-256、tokenizer 词表/特殊 token/chat template hash 必须一致。只读加载现有 Transformers 的 SDPA；不修改 vLLM、SGLang 或训练环境。
- 正常 replay 一次获取全部 action logps 和最后 4 层、所有 heads 平均的历史 attention mass。通过 Qwen3 的 normalized/rotary Q/K 仅重建所需 predictive rows，避免保存每层完整 L×L attention。
- 目标 token `m` 使用 hidden/attention row `m−1`。`mask_span` 在每层将 source keys 对 target 首个 predictive row `s−1` 及之后所有 query 屏蔽。target 前的中间 hidden states 保持原 forward 依赖，因此测量的是该 mask 定义下的 policy dependency，仍可有中间表征传递的间接信息。
- `drop_span` 是显式备选：删除整个 source token span，其他 token 不变，保留原始 RoPE position IDs。它会改变 source 后所有 hidden states，与 query-selective `mask_span` 不同；两种方法分开运行、单独比较，不混合统计。
- 默认 `delta_threshold=null`：保留所有 visited target 的 Top-1 proposal，继续采样候选链，`dependency_valid=null` / `candidate_unthresholded`。不能把候选链长度当作 H3 已成立。后处理用 empirical quantiles 的 threshold sweep 重算可通过的链深；显式非空 threshold 才据阈值停止。
- Random 在历史 turn 中均匀选择且排除 Top-1。只有一个历史 turn 时 Random 为 null（不造样本）。PreviousTurn 允许与 Top-1 相同并复用同一 replay，这类比较计作 tie。所有原始有符号 token delta 保留。
- ScienceWorld 每个 reward-changing turn 为独立 root；max-step 同时加入最后 3 decisions。WebShop/ALFWorld terminal 以最终 action 为 root；纯 token/context budget 截断没有明确 reward root，单独统计，不冒充 max-step。
- 原始 `environment_reward` 与现有 ORSD `trajectory_reward` 分开保存。WebShop 默认保留原 hybrid reward 定义供 GRPO advantage 复现；trace 展示的 `reward` 使用 official environment reward。ALFWorld 使用官方 binary outcome。数据按 task 分组补写 GRPO advantage（sample std + 1e-6），仅元数据，不用于训练。

## ScienceWorld

固定 `scienceworld==1.2.3`，官方 tag 对应 commit `5d569f9e4ff43ab6242a3622f7e25f9bc367ccd1`：
https://github.com/allenai/ScienceWorld/tree/1.2.3

`ScienceworldInteraction` 复用 BaseInteraction 合约，保存 task_name、variation_idx、observation、action、reward、score、done、valid_actions。始终 `generateGoldPath=False`；不用 oracle、expert 或 external judge。`reward` 保存官方 score 增量（0–100 单位），trajectory reward 保存 final score/100。使用固定 official test variations；模拟器由 variation 决定初始化，额外 seed 用于固定任务挑选和 controls，不调用不存在的环境 seed API。无 simplifications。

服务器需已有 Java 和 ScienceWorld 1.2.3。若缺少，仅补该独立环境及缺失依赖；**不要执行升级 ML 栈的安装命令**。脚本不自动安装任何内容。官方该版本仅声明 `py4j` 为 Python 依赖。若尚未安装 ScienceWorld，可先检查已有 `py4j` 和 Java，再只安装环境包：

```bash
java -version
python3 -m pip show py4j
python3 -m pip install --no-deps scienceworld==1.2.3
```

缺少 `py4j` 时单独补齐；不要升级已有 torch/transformers/veRL/vLLM/SGLang/flash-attn。

## 服务器运行

先进入已有 ORSD conda/venv，然后 `cd` 到同步后的仓库，直接运行：

```bash
bash examples/attention_traceback/run_all.sh smoke
```

`run_all.sh` 和 `collect.sh` 共用 `orsd_paths.sh`，自动沿用之前 ORSD 的路径约定：

| 变量 | 默认路径 |
|---|---|
| `MODEL_PATH` | 依次查找 `/workspace/checkpoints/Qwen3-1.7B`、`/workspace/checkpoints/qwen/Qwen3-1.7B`、仓库 `checkpoints/Qwen3-1.7B` |
| `WEBSHOP_ROOT` | 仓库 `WebShop` |
| `WEBSHOP_PARQUET` | 仓库 `Data/agent_rubric/webshop_small/webshop_test.parquet` |
| `ALFWORLD_CONFIG` | 仓库 `alfworld/configs/base_config.yaml` |
| `ALFWORLD_DATA` | 仓库 `alfworld` |
| `ALFWORLD_PARQUET` | 仓库 `Data/agent_rubric/alfworld/alfworld_valid_unseen.parquet` |

显式设置的变量优先，同时兼容 ORSD 的 `QWEN3_1_7B_MODEL_PATH`、`WEBSHOP_VAL_PARQUET`、`WEBSHOP_SMALL_DATA_DIR`、`ALFWORLD_VAL_UNSEEN_PARQUET`。ALFWorld 此诊断入口默认取 unseen split，不在路径查找时切换 split。沿用当前可见 GPU，默认 4 卡、TP=1。启动时打印实际解析的路径；仅复用路径默认值，不执行旧训练 launcher。

手动执行单独 replay/compare 命令时，可先把同样的默认值载入当前 shell：

```bash
source examples/attention_traceback/orsd_paths.sh
```

smoke 先记录 Python、nvidia-smi、torch/verl 版本及 CUDA 可用性；运行 server-only 对齐测试；三个环境各 1 task × 2 trajectories；每条都必须真正执行到至少一条 dependency edge 才写 `smoke_pass.json`。如果选中任务过早结束、没有历史 root，脚本明确失败，需要选择不同的固定任务（保留失败记录）。这不是无条件的通过标记。

三环境通过后：

```bash
TASKS=50 bash examples/attention_traceback/run_all.sh phase-a
python3 examples/attention_traceback/compare.py --root "$RESULT_ROOT"
```

Phase A 要求每 benchmark 50–100 tasks、每 task 恰好 4 trajectories。先检查三个 smoke receipts 后才启动正式采集。收集/重放分为独立进程，rollout 完毕才加载 replay 模型。Replay 默认只用第一张可见 GPU；它不需要训练时的 optimizer/FSDP。

现有完整 cache 优先复用；`run_all.sh` 看到 capture 目录后不会重做 environment rollout。中断时 per-trajectory replay shards 可续跑；若采集不完整则报告 sibling 数错误，不偷偷重新采样。不同 checkpoint、tokenizer、config 或数据不得复用同一结果目录。

单独重放（无环境交互）：

```bash
python3 -m verl.experimental.attention_traceback.run \
  --input /path/to/lossless/capture \
  --output "$RESULT_ROOT/smoke/webshop" \
  --model "$MODEL_PATH" --stage smoke
```

Cache 输入可为每条 trajectory 一个 JSON 的目录，或完整 schema 的 JSONL；schema 参考 `capture.finish_record`。普通旧版 `rollout_data_dir` 的 input/output 文本不是无损 cache：缺少 checkpoint/结束原因/实际 turn token 信息时不能直接声称与原 rollout 严格一致。不要通过拼接 `ACTION/OBSERVATION` 日志猜原始 reasoning/token 边界。已有完整 token 和消息的缓存可用 `import_flat_cache.py` 转换，无需环境重跑。

### 大 context 与 fallback

`mask_span` 的 4-D additive mask 有 L² 显存开销；默认上限 40960 tokens（bf16 mask 约 3.1 GiB，与现有 ORSD context budget 对齐），超限明确停止，不截断轨迹，也不暗中混用干预。优先根据服务器可用显存提高 `attention_traceback.max_dense_mask_tokens`，或完整重跑单独的 `drop_span` 实验：

```bash
RESULT_ROOT="$PWD/results/attention_traceback_drop" INTERVENTION=drop_span \
  bash examples/attention_traceback/run_all.sh smoke
# 复用已有 capture 请使用独立 replay 命令，而非重新 rollout。
python3 -m verl.experimental.attention_traceback.run \
  --input /path/to/existing/capture --output /path/to/drop/smoke/webshop \
  --model "$MODEL_PATH" --stage smoke attention_traceback.intervention=drop_span
```

模型 ABI 不兼容会报出具体错误。先运行 tiny Qwen3 mechanics tests 验证 SDPA/4-D mask/selected rows 与 eager 一致，不通过时不要升级 transformers 来掩盖不兼容。

## 输出和判断

每 benchmark 输出：

- `trajectories.jsonl`：完整 turn/token 信息、replay logps、reward、group advantage。
- `traces.jsonl`：所有 roots 和 visited candidate edges，包括三组有符号 token delta、mask 前后 attention。
- `summary.json`：轨迹/root 数、depth/distance 分布、三组 token/edge 指标、paired difference、胜率及 task-cluster bootstrap 95% CI、阈值扫描。
- `examples/`：最多 5 成功 + 5 失败的独立轨迹，含 token 热图、最大 delta tokens、source observation、redistribution 和 JSON；不足时记录缺额，不虚构。
- `plots/`、`report.html`：可离线查看/分享的 SVG 图和 HTML 报告。
- `manifest.json`：完整配置、checkpoint/tokenizer fingerprint、包版本；`completed/` 用于 replay 续跑。

同一 trajectory 的多个 root 共享 target edge；统计按 unique target 去重。CI 以 task 为 resampling cluster，保留 sibling 相关性。一任务 smoke 不出伪置信区间。主对照以 edge mean |delta| 作摘要，同时保留 token-weighted分布和全部原始 tokens。

`comparison.md` / `go_no_go.json` 汇总三环境。A/B 采用 paired task-bootstrap CI 下界 > 0；至少两个环境分别满足。C 需查看 threshold sweep 后 depth>=2、distance>=2 是否稳定；D 的 token concentration 只是描述量，需要人工检查语义。统计不通过标记 NO_GO；通过 A/B 仍等待 C/D review，不自动宣布 Go，不授权 RL。

## 文件改动范围

新增：`verl/experimental/attention_traceback/`（config/capture/replay/trace/report/run）；`verl/interactions/scienceworld_interaction.py`；`verl/trainer/config/attention_traceback.yaml`；`examples/attention_traceback/`；`tests/attention_traceback/`。

接入：`tool_agent_loop.py` 保存原始 token/spans；`main_ppo.py` 路由；`ray_trainer.py` 在 validation unpadding 后导出、阻止进入训练；WebShop/ALFWorld interaction 补 outcome metadata；4 个 PPO YAML 加独立开关。原有 ORSD 实现保留，已有未提交修改保留。


## Smoke 的零 root 与截断诊断

`context_budget_exhausted` / `length_limit` 的轨迹没有正常 terminal reward root。
smoke 现在允许使用最后 3 个已完成 decision 作为 `truncation_diagnostic` roots，
只验证 attention → masking → delta → traceback 链路；原始 done_reason 保留，
`eligible_for_reward_hypotheses=false`，不将这些 root 计入 number_of_reward_roots。
Phase A 不启用这个 fallback，仍使用原 reward-root 规则。smoke 通过仅代表 replay mechanics 通过。

启动时打印每条轨迹的 done_reason、turn_count、token_count 和 root_turns，并写入
`root_diagnostics.json`。只有 0/1 个 decision、没有合法历史 edge 的轨迹仍不能通过 smoke。
代码版本更新后会自动重算旧 `completed/` 中的 replay 结果，直接复用 capture，不需要删除缓存或重跑环境。

## 用户授权后的训练扩展（2026-09-30）

新增独立路由 `credit_assignment.type=attention_traceback_train`，原 `attention_traceback`
仍是 diagnostic-only。训练公式、ORSD 预算对照、GRPO/Previous/Random 对照命令和审计输出见
[TRAINING.md](TRAINING.md)。训练入口为 `bash examples/attention_traceback/train.sh webshop train`
或 `alfworld train`；固定使用物理 GPU 4–7。训练授权来自用户后续明确要求，不由 Phase A
Go/No-Go 文件自动触发。ScienceWorld 暂未接入训练入口。
