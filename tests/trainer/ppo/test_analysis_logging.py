import numpy as np
import torch

from verl.trainer.ppo.analysis_logging import TURN_REPLAY_KEY, build_training_analysis_record


class _Batch:
    def __init__(self):
        self.batch = {
            "response_mask": torch.tensor([[1, 1, 0]], dtype=torch.long),
            "token_level_rewards": torch.tensor([[0.0, 1.0, 0.0]]),
            "advantages": torch.tensor([[0.5, 1.5, 0.0]]),
            "grpo_advantages": torch.tensor([[1.0, 1.0, 0.0]]),
            "token_credit_weights": torch.tensor([[0.5, 1.5, 1.0]]),
            "token_credit_raw_gap": torch.tensor([[-0.2, 0.4, 0.0]]),
        }
        self.non_tensor_batch = {
            "uid": np.asarray(["task-1"], dtype=object),
            "task_id": np.asarray(["game-1"], dtype=object),
            "online_rubric_valid": np.asarray([True]),
            "interaction_turn_metrics": np.asarray(
                [[{"invalid_action": 0.0}, {"repeat_no_progress_action": 1.0}]], dtype=object
            ),
            TURN_REPLAY_KEY: np.asarray(
                [
                    {
                        "turns": [
                            {"flat_response_start": 0, "flat_response_end": 1, "history_response": "ACTION: look"},
                            {"flat_response_start": 1, "flat_response_end": 2, "history_response": "ACTION: look"},
                        ]
                    }
                ],
                dtype=object,
            ),
        }

    def __len__(self):
        return 1


def test_analysis_record_preserves_mean_and_environment_action_labels():
    record = build_training_analysis_record(_Batch(), {"token_credit/group_gate_active_frac": 1.0}, 7, 42)
    trajectory = record["trajectories"][0]
    assert trajectory["training_return"] == 1.0
    assert trajectory["trajectory_average_correction"] == 0.0
    assert trajectory["token_sign_flip_rate"] == 0.0
    assert [turn["action_type"] for turn in trajectory["turns"]] == ["normal", "no_progress"]
    assert trajectory["turns"][0]["advantage_correction"] == -0.5
    assert trajectory["turns"][1]["advantage_correction"] == 0.5
