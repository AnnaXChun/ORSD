from omegaconf import OmegaConf
import pytest

from verl.experimental.attention_traceback.config import configure_credit_assignment


def config(route):
    return OmegaConf.create({'credit_assignment': {'type': route}, 'algorithm': {
        'adv_estimator': 'grpo', 'token_credit': {'enable': True, 'rubric_pipeline': {
            'enable': True, 'pool_enable': True, 'pool_generator_loss_enable': True,
            'snapshot_enable': True}}}})


def test_existing_orsd_route_is_unchanged():
    cfg = config('orsd')
    before = OmegaConf.to_container(cfg)
    configure_credit_assignment(cfg)
    assert OmegaConf.to_container(cfg) == before


def test_grpo_disables_all_rubric_branches():
    cfg = config('grpo')
    configure_credit_assignment(cfg)
    assert not cfg.algorithm.token_credit.enable
    assert not any(cfg.algorithm.token_credit.rubric_pipeline.values())


def test_unknown_route_fails_before_ray_or_model_load():
    with pytest.raises(ValueError):
        configure_credit_assignment(config('typo'))


def test_diagnostic_siblings_have_distinct_reproducible_seeds():
    from verl.experimental.attention_traceback.capture import sampling_seed
    seeds = [sampling_seed(1234, task, sibling) for task in ('a', 'b') for sibling in range(4)]
    assert len(set(seeds)) == 8
    assert seeds == [sampling_seed(1234, task, sibling) for task in ('a', 'b') for sibling in range(4)]
