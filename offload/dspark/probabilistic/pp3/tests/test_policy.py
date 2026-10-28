import pytest

from offload.dspark.worker.sample_policy import enabled, request_key


def test_default_off_and_explicit_switch() -> None:
    assert not enabled({})
    assert not enabled({'DSPARK_SAMPLED_TOP32': '0'})
    assert enabled({'DSPARK_SAMPLED_TOP32': '1'})
    with pytest.raises(ValueError):
        enabled({'DSPARK_SAMPLED_TOP32': 'yes'})


def test_rng_owners_are_stable_and_isolated() -> None:
    keys = {request_key(session, slot, generation, seed)
            for session in (1, 2) for slot in range(4) for generation in (0, 1) for seed in (0, 42)}
    assert len(keys) == 2 * 4 * 2 * 2
    assert request_key(1, 0, 1, 42) == request_key(1, 0, 1, 42)
    with pytest.raises(ValueError):
        request_key(1, 4, 0, 0)
