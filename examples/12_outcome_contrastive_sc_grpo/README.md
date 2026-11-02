# 实验 12：Outcome-Contrastive SC-GRPO

本实验保留标准 GRPO 的组内 advantage 方向，不使用 rubric、自蒸馏或固定 reference model 产生 credit。

对同一问题的 rollout，验证器先划分正确轨迹与错误轨迹。仅当组内至少有 2 条正确轨迹且存在错误轨迹时，为目标轨迹选择：

- 一条非自身的验证正确参考；
- 一条非自身的验证错误参考。

当前 actor 在两种特权上下文下，对同一目标 response 做 teacher-forcing，并计算每个 token 的全词表 Jensen–Shannon divergence：

```text
C_t = JS(
  pi_theta(. | problem, verified-correct reference, target prefix),
  pi_theta(. | problem, verified-incorrect reference, target prefix)
)
```

以当前 batch 的有效 `C_t` 的 75% 分位数作为尺度 `c`：

```text
w_t = C_t / (C_t + c)
A_token = A_GRPO * w_t
```

权重完全 detach，不做均值归一化。无法构造正负非自身参考的轨迹严格回退普通 GRPO。

## 默认设置

- 模型：Qwen3-1.7B，全参数训练；
- 训练集：MATH-DAPO-17K；
- 每步 8 个 prompt，每个 prompt 8 个 rollout；
- student、正参考条件前向和负参考条件前向均关闭 thinking；
- rollout 最大长度 8192；
- 正负参考条件 prompt 上限 10240；
- PPO/GRPO 与 KL 正则参数保持现有纯 GRPO 配置；
- 每 25 步在 AIME 2024、AIME 2025、MATH-500、AMC 2023 上验证。

## 运行

```bash
cd /workspace/verl_wsm
bash examples/12_outcome_contrastive_sc_grpo/run.sh
```

先做 25 步快速验证：

```bash
MAX_STEPS=25 SAVE_FREQ=25 TEST_FREQ=25 \
bash examples/12_outcome_contrastive_sc_grpo/run.sh
```

同参数纯 GRPO 对照：

```bash
bash examples/12_outcome_contrastive_sc_grpo/run_grpo_control.sh
```

显存不足时，优先减小 `OC_JS_CHUNK_SIZE`；仍不足再缩短
`OC_TEACHER_MAX_PROMPT_LENGTH`。不要增大 `OC_MICRO_BATCH_SIZE_PER_GPU`。

## 关键监控项

- `outcome_contrastive/eligible_group_frac`
- `outcome_contrastive/active_trajectory_frac`
- `token_credit/oc_js_mean`
- `token_credit/oc_scale`
- `token_credit/weight_mean`
- `token_credit/weight_min`
- `token_credit/weight_max`

若 `eligible_group_frac` 长期接近 0，说明 rollout 组缺少可用的正负结果对，此时训练行为将接近纯 GRPO。
