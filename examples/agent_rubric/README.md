# ALFWorld / WebShop：轨迹感知 Rubric Token-Credit GRPO

## 结论

当前 Verl 没有原生的 ALFWorld 或 WebShop adapter。本目录补齐了：

1. 官方环境多轮交互；
2. 官方 benchmark reward 回传到 GRPO；
3. 本地 vLLM 离线采集完整 rollout，或由训练中的当前 actor 在线采集；
4. `task + self-generated complete rollouts + official outcomes` 条件下，由同一 actor 生成 rubric；
5. 独立 judge 逐条验证、失败重试、每题至少 4 条和全数据覆盖率 60% 硬门槛。

Rubric 只作为 teacher scoring context。actor rollout 看不到 rubric；当前更新前 actor 分别在原始环境上下文
与 rubric 增强上下文下，对同一条多轮 response 计算 token log-probability，并以系数 `0.01` 将差值加到
GRPO advantage。训练 reward 始终来自官方环境，不能由 rubric 生成器或 judge 改写。

Qwen3 agent rollout、当前 actor 在线 rubric generation，以及 rubric-conditioned scoring 默认全部启用
thinking。三个开关分别是 `ACTOR_THINKING`、`RUBRIC_GENERATOR_THINKING` 和 `TEACHER_THINKING`；rubric
解析和轨迹压缩都会移除 `<think>...</think>` 私有推理，只保留可执行 rubric 与可观察环境证据。

```text
A_it = A_i^GRPO + alpha * [log p_old(y_it | task, rubric, interaction_history)
                           - log p_old(y_it | task, interaction_history)]
alpha = 0.01
```

两项都由同一个更新前 actor 计算，固定 reference 只承担标准 KL regularization。多轮 teacher forward 会保留
初始 observation 和所有环境返回作为 causal context；这些环境 token 的 loss mask 仍为 0，只对 actor action
token 应用 credit。

服务器环境必须能直接导入官方 `alfworld`，并准备好官方 WebShop checkout、products 数据和检索索引。
先执行以下预检，任一失败都不要开始大规模 rollout：

```bash
python3 -c 'from alfworld.agents.environment import get_environment; print("ALFWorld OK")'
PYTHONPATH=/workspace/WebShop python3 -c \
  'from web_agent_site.envs.web_agent_text_env import WebAgentTextEnv; print("WebShop OK")'
```

## 官方 Reward

| Benchmark | 训练分数 | 代码来源 |
|---|---:|---|
| ALFWorld | `1.0` 当且仅当 `infos["won"]`，否则 `0.0` | 官方 TextWorld 环境 |
| WebShop | 官方购买终局 reward，保持原始 `[0,1]` | 官方 `WebAgentTextEnv.step` |

WebShop 官方 baseline wrapper 会为了显示把 reward 乘以 10；本实现没有使用该缩放。
同一 goal 的并行 rollout 会使用不同 session key，避免共享 product server 时轨迹状态互相覆盖。

## 1. 准备 Parquet

ALFWorld：

```bash
python3 examples/agent_rubric/preprocess_alfworld.py \
  --data-dir /workspace/data/alfworld \
  --output-dir /workspace/data/agent_rubric
```

ALFWorld 的 validator-only reference 自带官方成功 demonstration/high-level plan；但 rubric 生成仍强制要求
先补采当前基模的完整 rollout，不能只用静态 plan 代替 actor trajectory。

WebShop：

```bash
python3 examples/agent_rubric/preprocess_webshop.py \
  --webshop-root /workspace/WebShop \
  --output-dir /workspace/data/agent_rubric
```

划分与官方 baseline 一致：test `0:500`、eval `500:1500`、train `1500:`。

## 2. 本地 vLLM 采集完整 Rollout

ALFWorld：

```bash
export ALFWORLD_CONFIG=/workspace/alfworld/configs/base_config.yaml
export ALFWORLD_DATA=/workspace/data/alfworld
export INPUT_PARQUET=/workspace/data/agent_rubric/alfworld_train.parquet
export OUTPUT_PARQUET=/workspace/data/agent_rubric/alfworld_train_rollouts.parquet
export ROLLOUT_MODEL=/workspace/checkpoints/Qwen3-1.7B
NUM_ROLLOUTS=4 ROLLOUT_TP=2 bash examples/agent_rubric/collect_alfworld_rollouts.sh
```

WebShop：

```bash
export WEBSHOP_ROOT=/workspace/WebShop
export INPUT_PARQUET=/workspace/data/agent_rubric/webshop_train.parquet
export OUTPUT_PARQUET=/workspace/data/agent_rubric/webshop_train_rollouts.parquet
export ROLLOUT_MODEL=/workspace/checkpoints/Qwen3-1.7B
NUM_ROLLOUTS=4 ROLLOUT_TP=2 bash examples/agent_rubric/collect_webshop_rollouts.sh
```

每条 trajectory 保存初始 observation，以及每一步的 model response、解析 action、环境返回、官方分数和
终止状态。脚本不会静默截断 trajectory 后再生成 rubric。

## 3. 生成并验证 Rubric

默认配置使用本地 vLLM 生成 rubric，使用独立 DashScope 模型验证：

```bash
export RUBRIC_API_KEY=YOUR_DASHSCOPE_KEY
export INPUT_PARQUET=/workspace/data/agent_rubric/alfworld_train_rollouts.parquet
export OUTPUT_PARQUET=/workspace/data/agent_rubric/alfworld_train_rubric_valid.parquet
export RUBRIC_GENERATOR_MODEL=/workspace/checkpoints/Qwen3-1.7B

TARGET_VALID_RATE=0.60 \
MIN_RUBRICS_PER_TASK=4 \
RUBRIC_VALIDATION_ROUNDS=5 \
bash examples/agent_rubric/generate_validated_rubrics.sh
```

Rubric generator 使用与 actor 相同的模型检查点，且看不到
`reference_answer_json` 中的 oracle milestones、隐藏目标或答案；完整 reference 只提供给独立 judge 做
accept/reject 验证，不参与 rubric 内容生成。Judge 的 free-text rejection reason 也不会回传给 generator，
从而避免在重试轮次形成隐藏答案侧信道。一条 rubric 必须同时通过：

- 与 task/reference answer constraints 一致；
- 有 rollout transition 或官方 demonstration 支撑；
- 与 official reward 方向一致；
- 能改变当前 state 下的 action 选择；
- actor 可以从可见 observation 检查；
- 不泄漏 target ASIN、内部 game path、隐藏位置或完整 oracle action sequence；
- 与已选 rubric 不重复。

一题至少有 4 条通过才算有效。默认保留全量训练题：有效题启用 rubric token credit，无效题的 rubric
置空并自动退化为纯 GRPO，避免筛掉困难题造成 selection bias。若有效题覆盖率低于 60%，脚本退出非零并
拒绝写出训练 parquet。`.rubric-progress.jsonl` 支持断点续跑，`.rubric-summary.json` 记录覆盖率和配置。

这里保证的是“独立验证协议下至少 60% 的 task coverage、且输出项逐条通过检查”。它不是对训练后
reward 提升幅度的预先承诺；最终效果需要与不注入 rubric 的标准 GRPO 做同 seed A/B。

正式训练前可先在 20 个任务上运行配对 prompt 诊断。同一个 1.7B 先生成无 rubric trajectory，再从这些
trajectory 生成 rubric，并在相同任务和采样参数下重新执行。最终只比较官方环境 reward：

```bash
export ALFWORLD_CONFIG=/workspace/verl_wsm/alfworld/configs/base_config.yaml
export ALFWORLD_DATA=/workspace/verl_wsm/alfworld
export MODEL_PATH=/workspace/checkpoints/Qwen3-1.7B
bash examples/agent_rubric/debug_alfworld_rubrics_20.sh
```

结果写入 `experiment_outputs/agent_rubric/alfworld_prompt_debug_20/summary.json`，包含成功率、官方平均
reward、invalid-action rate、动作多样性以及配对 wins/ties/losses。该诊断使用和在线训练完全相同的
one-shot actor prompt 与解析门槛，不调用外部 judge；逐题去除 thinking 后的生成和 rubric 保存在
`validated_rubrics.prompt-audit.json`，用于人工检查 action consistency。

## 4. Rubric Token-Credit GRPO 训练

ALFWorld：

```bash
export BENCHMARK=alfworld
export ALFWORLD_CONFIG=/workspace/alfworld/configs/base_config.yaml
export ALFWORLD_DATA=/workspace/data/alfworld
export TRAIN_PARQUET=/workspace/data/agent_rubric/alfworld_train.parquet
export VAL_PARQUET=/workspace/data/agent_rubric/alfworld_valid_unseen.parquet
export MODEL_PATH=/workspace/checkpoints/Qwen3-1.7B
HISTORY_WINDOW=8 RUBRIC_BACKEND=actor_rollout bash examples/agent_rubric/run_agent_grpo.sh
```

WebShop：

```bash
export BENCHMARK=webshop
export WEBSHOP_ROOT=/workspace/WebShop
export TRAIN_PARQUET=/workspace/data/agent_rubric/webshop_train.parquet
export VAL_PARQUET=/workspace/data/agent_rubric/webshop_test.parquet
export MODEL_PATH=/workspace/checkpoints/Qwen3-1.7B
HISTORY_WINDOW=8 RUBRIC_BACKEND=actor_rollout bash examples/agent_rubric/run_agent_grpo.sh
```

默认 `actor_rollout` 后端在每个训练 step 聚合本批次 on-policy trajectories，由当前 actor 在线生成 rubric，
不跨 step 缓存。rubric 不直接暴露给用于取得官方 reward 的 rollout actor，而是作为同一个 pre-update actor
重新计算相同 action token logprob 时的特权上下文。无法解析出至少 4 条 rubric 的组自动退化为纯 GRPO。

离线对照仍可设置 `RUBRIC_BACKEND=dataset` 并传入验证后的 rubric parquet。离线输出 parquet 已将
`extra_info.interaction_kwargs.inject_rubric=false`：rubric 不直接暴露给 rollout actor。
直接把 `algorithm.token_credit.enable=False` 传给训练脚本即可运行纯 GRPO 对照；reward、模型、rollout 数、
训练 batch 和验证集均保持不变。若要做“rubric 直接进入 actor prompt”的独立消融，生成 parquet 时显式传
`--inject-rubric`，且不要把它与 token-credit 主实验混在一起。

`HISTORY_WINDOW=8` 开启 action-only history 和逐 turn logprob replay：完整 thinking response 仍参与 PPO/GRPO
loss，但下一轮只看到 `ACTION: ...`，并固定保留 system、task、initial observation 与最近 8 个完整交互 turn。
不设置或设为 `0` 时完全使用 VERL 原始增长 prompt/flatten replay 路径，便于运行不受改造影响的纯 GRPO。

## 关键文件

- `verl/interactions/alfworld_interaction.py`：固定 ALFWorld game、多轮动作和官方 success reward。
- `verl/interactions/webshop_interaction.py`：固定 WebShop goal、共享产品 server 和官方连续 reward。
- `collect_rollout_trajectories.py`：本地 vLLM batched long-horizon rollout。
- `precompute_trajectory_rubrics.py`：trajectory-aware generator、answer-aware validator、重试与门槛。
- `run_agent_grpo.sh`：官方环境 reward 的全参 GRPO，加当前 actor 的 rubric/plain token-logprob 差值。

官方实现：[ALFWorld](https://github.com/alfworld/alfworld)、
[WebShop environment](https://github.com/princeton-nlp/WebShop/blob/master/web_agent_site/envs/web_agent_text_env.py)。
