from verl.trainer.ppo.ray_trainer import RayPPOTrainer


def test_alfworld_rubric_outcome_uses_binary_success_not_shaped_reward():
    non_tensor_batch = {
        "data_source": ["alfworld", "alfworld"],
        "reward_extra_info": [
            {"training_reward": 9.8, "score": 9.8, "official_reward": 1.0, "success": 1.0},
            {"training_reward": -0.2, "score": -0.2, "official_reward": 0.0, "success": 0.0},
        ],
    }

    assert RayPPOTrainer._online_rollout_score(non_tensor_batch, 0) == 1.0
    assert RayPPOTrainer._online_rollout_score(non_tensor_batch, 1) == 0.0


def test_webshop_rubric_outcome_uses_strict_success_not_shaped_reward():
    non_tensor_batch = {
        "data_source": ["webshop", "webshop"],
        "reward_extra_info": [
            {"training_reward": 0.65, "score": 0.65, "official_reward": 0.7, "success": 0.0},
            {"training_reward": 0.90, "score": 0.90, "official_reward": 1.0, "success": 1.0},
        ],
    }

    assert RayPPOTrainer._online_rollout_score(non_tensor_batch, 0) == 0.0
    assert RayPPOTrainer._online_rollout_score(non_tensor_batch, 1) == 1.0


def test_webshop_r2_rubric_outcome_can_use_continuous_training_reward():
    non_tensor_batch = {
        "data_source": ["webshop", "webshop"],
        "reward_extra_info": [
            {"training_reward": 0.47, "official_reward": 0.5, "success": 0.0},
            {"training_reward": 0.72, "official_reward": 0.75, "success": 0.0},
        ],
    }

    assert RayPPOTrainer._online_rollout_score(
        non_tensor_batch, 0, score_source="training_reward"
    ) == 0.47
    assert RayPPOTrainer._online_rollout_score(
        non_tensor_batch, 1, score_source="training_reward"
    ) == 0.72


def test_webshop_rubric_outcome_rejects_unknown_score_source():
    non_tensor_batch = {"data_source": ["webshop"]}

    try:
        RayPPOTrainer._online_rollout_score(non_tensor_batch, 0, score_source="unknown")
    except ValueError as error:
        assert "outcome_score_source" in str(error)
    else:
        raise AssertionError("unknown rubric outcome source must raise ValueError")


def test_webshop_rubric_outcome_derives_strict_success_when_explicit_bit_is_missing():
    non_tensor_batch = {
        "data_source": ["webshop", "webshop"],
        "reward_extra_info": [
            {"training_reward": 0.75, "official_reward": 0.75},
            {"training_reward": 0.95, "official_reward": 1.0},
        ],
    }

    assert RayPPOTrainer._online_rollout_score(non_tensor_batch, 0) == 0.0
    assert RayPPOTrainer._online_rollout_score(non_tensor_batch, 1) == 1.0


def test_explicit_success_survives_missing_generated_batch_data_source():
    non_tensor_batch = {
        "reward_extra_info": [
            {"training_reward": 0.65, "official_reward": 0.7, "success": 0.0},
            {"training_reward": 0.90, "official_reward": 1.0, "success": 1.0},
        ],
    }

    assert RayPPOTrainer._online_rollout_score(non_tensor_batch, 0) == 0.0
    assert RayPPOTrainer._online_rollout_score(non_tensor_batch, 1) == 1.0
