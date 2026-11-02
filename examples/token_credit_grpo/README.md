# Additive log-probability-gap GRPO in VERL

This implementation keeps VERL's ordinary full-parameter GRPO objective but
adds a small detached comparison-context signal to each sampled response token:

\[
A_{i,t}=A_i+\alpha\left[
\log\pi_C(y_{i,t})-\log\pi_{\mathrm{old}}(y_{i,t})
\right],
\qquad \alpha=0.01.
\]

There is no centering, quantile scaling, tanh, reward-polarity multiplication,
or trajectory weight normalization. The coefficient is deliberately small.
The plain signal is the recomputed old-policy log probability, and the gap is
detached before PPO epochs start.

Two teacher conditions are implemented:

1. **No rubric:** the frozen reference teacher sees the same ordinary problem
   as the student. This is the first smoke-test condition.
2. **Online actor rubric contrast:** problem-only rubrics are generated offline
   and stored in the training parquet. Before each PPO update, the same current
   old actor teacher-forces the exact rollout response twice: once under the
   ordinary problem and once under the rubric-augmented prompt. Subtracting
   these aligned response-token log probabilities isolates the rubric context
   effect without adding actor/reference drift. The fixed reference model is
   used only by VERL's standard KL loss.

Important invariants:

- `alpha=0` is exactly ordinary GRPO.
- padding, non-finite gaps, and rubric-ineligible tokens keep the original
  GRPO advantage.
- the default `alpha=0.01` makes the auxiliary token signal small.
- unlike the old multiplicative rule, a context gap can provide a
  small signal when a GRPO group has zero outcome advantage.
- `actor_rollout_ref.model.lora_rank=0` makes the supplied run script a
  full-parameter FSDP run.

The previous centered multiplicative implementation remains available as
`algorithm.token_credit.mode=polarity_gap` for reproducibility.

Rubric generation is problem-only: the generator never receives the sampled
trajectory or the ground-truth answer. Rubrics are generated once per unique
problem, cached, and expanded to its rollout group. A generation failure sets
`rubric_teacher_eligible=0`; that trajectory uses exact ordinary GRPO rather
than an unvalidated rubric signal.

## Run

Run the strictly matched ordinary-GRPO baseline first:

```bash
MAX_STEPS=250 \
bash examples/token_credit_grpo/run_qwen3_1_7b_math_grpo_baseline.sh
```

This wrapper reuses every training and evaluation setting below but forces
`algorithm.token_credit.enable=False`. It therefore differs from the additive
experiment only in the proposed token-credit mechanism.

```bash
MAX_STEPS=25 \
bash examples/token_credit_grpo/run_qwen3_1_7b_math_no_rubric.sh
```

This first run uses no rubric and requires only the eight VERL GPUs.

For the online actor-rubric condition, first generate candidates and validate
every unique training problem with the answer-aware semantic judge:

```bash
export RUBRIC_API_KEY=YOUR_DASHSCOPE_KEY
bash examples/token_credit_grpo/precompute_validated_rubrics.sh
```

The generator receives only the problem. The validator receives the problem,
training ground truth, and every candidate criterion, and checks mathematical
correctness, problem relevance, actionability, self-containment, and final-
answer leakage. A problem is valid only when at least four criteria pass all
checks. Problems with fewer than four accepted criteria are
regenerated for up to four rounds. The command writes
`Data/train_with_validated_rubrics.parquet`, records validation metadata in
separate columns, resumes from a JSONL progress file, and exits non-zero if the
unique-problem validation rate remains below 70%. Rerun it to retry only the
invalid problems, or increase `MAX_VALIDATION_ROUNDS`/`CANDIDATE_COUNT`.

This semantic validation rate is not the same as OPSD's mixed-rollout dynamic
selection rate. The latter should still be reported separately as a diagnostic
of outcome-discriminative signal. After vLLM exits and releases the GPUs, start
training:

```bash
MAX_STEPS=50 \
bash examples/token_credit_grpo/run_qwen3_1_7b_math_rubric.sh
```

The training process reads the `rubric` and `rubric_valid` parquet columns and
does not launch or contact a separate rubric-vLLM server.

Run the compute-matched ordinary-GRPO control with the same reference forward:

```bash
bash examples/token_credit_grpo/run_qwen3_1_7b_math_control.sh
```

The control sets `alpha=0`, which is algebraically identical to ordinary GRPO,
while preserving the otherwise unused reference forward pass so wall-clock and
memory overhead are matched.

Useful overrides:

```bash
MAX_STEPS=50 TOKEN_CREDIT_ALPHA=0.01 \
DATA_ROOT=/workspace/verl_wsm/Data \
MODEL_PATH=/workspace/checkpoints/Qwen3-1.7b \
bash examples/token_credit_grpo/run_qwen3_1_7b_math.sh
```

The script trains on the local MATH-DAPO-17K parquet and validates on local
AIME24, AIME25, MATH500, and AMC23 parquet files. It requests eight GPUs,
eight rollouts per training prompt, two validation samples at temperature 0.6,
8192 response tokens, low-variance KL regularization, and full-parameter FSDP.

Monitor these metrics:

- `token_credit/additive_coefficient`
- `token_credit/active_frac`
- `token_credit/active_trajectory_frac`
- `token_credit/eligible_frac`
- `token_credit/eligible_trajectory_frac`
- `token_credit/finite_gap_frac`
- `token_credit/nonzero_gap_frac`
- `token_credit/nonzero_adjustment_frac`
- `token_credit/zero_advantage_activated_frac`
- `token_credit/raw_gap_mean`
- `token_credit/raw_gap_abs_mean`
- `token_credit/gap_abs_mean`
- `token_credit/active_gap_mean`
- `token_credit/active_gap_abs_mean`
- `token_credit/effective_gap_mean`
- `token_credit/effective_gap_abs_mean`
- `token_credit/teacher_preferred_frac`
- `token_credit/student_preferred_frac`
- `token_credit/grpo_advantage_mean`
- `token_credit/grpo_advantage_abs_mean`
- `token_credit/grpo_advantage_std`
- `token_credit/active_grpo_advantage_mean`
- `token_credit/active_grpo_advantage_abs_mean`
- `token_credit/adjusted_advantage_mean`
- `token_credit/adjusted_advantage_abs_mean`
- `token_credit/adjusted_advantage_std`
- `token_credit/advantage_delta_mean`
- `token_credit/advantage_delta_abs_mean`
- `token_credit/adjustment_to_grpo_abs_ratio`
- `token_credit/adjustment_to_grpo_abs_pct`
- `token_credit/adjustment_to_active_grpo_abs_ratio`
- `token_credit/adjustment_to_active_grpo_abs_pct`
- `token_credit/adjustment_mean`
- `token_credit/adjustment_abs_mean`
- `token_credit/adjustment_std`
- `token_credit/adjustment_min`
- `token_credit/adjustment_max`
- `rubric_pipeline/success_rate`
- `rubric_pipeline/cache_hit_rate`
- `rubric_pipeline/latency_s`
- `rubric_pipeline/rubric_chars_mean`
- `rubric_pipeline/scorer_is_current_actor`
- `token_credit/rubric_gap_correct_mean`
- `token_credit/rubric_gap_wrong_mean`
- `token_credit/rubric_gap_outcome_margin`
- `token_credit/rubric_gap_reward_correlation`
- `token_credit/rubric_gap_scored_trajectory_frac`
- `token_credit/reference_drift_abs_mean`
- `token_credit/rubric_to_reference_drift_abs_ratio`

If `active_frac=0` or `adjustment_abs_mean=0`, the update for that batch is
ordinary GRPO. Use `eligible_trajectory_frac` to detect rubric generation or
validation failures and `zero_advantage_activated_frac` to measure how often
the context gap adds signal to tokens whose GRPO outcome advantage is zero.
The outcome margin and reward correlation diagnose whether the rubric-induced
likelihood change actually favors correct rollouts; a non-positive margin is a
warning that the gap should not be trusted as credit assignment.
