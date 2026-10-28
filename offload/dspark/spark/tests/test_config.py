# SPDX-License-Identifier: Apache-2.0
import hashlib

import pytest

from remote_dspark.config import (RemoteConfig, apply_cost_overrides, n_out_table,
                                 remote_requested)

ID = "ab" * 32


def env(**kw):
    e = {"VLLM_DSPARK_REMOTE": "tcp://10.202.28.1:18620?bind=10.202.28.2",
         "VLLM_DSPARK_REMOTE_IDENTITY": ID}
    e.update(kw)
    return e


def test_requested_and_defaults():
    assert not remote_requested({})
    assert not remote_requested({"VLLM_DSPARK_REMOTE": "  "})
    c = RemoteConfig.from_env(env())
    assert c.policy == "shadow" and c.keeps_local_drafter and c.want_margin
    assert c.endpoint.host == "10.202.28.1" and c.endpoint.port == 18620
    assert c.endpoint.options["bind"] == "10.202.28.2"
    assert c.timeout_ms == 50 and c.on_fail == "target_only"
    assert c.markov_identity == bytes(32)


def test_capacity_not_local():
    c = RemoteConfig.from_env(env(VLLM_DSPARK_REMOTE_POLICY="capacity"))
    assert not c.keeps_local_drafter and not c.want_margin


@pytest.mark.parametrize("kw,match", [
    ({"VLLM_DSPARK_REMOTE_IDENTITY": ""}, "IDENTITY is required"),
    ({"VLLM_DSPARK_REMOTE_IDENTITY": "12"}, "64 hex"),
    ({"VLLM_DSPARK_REMOTE_POLICY": "fast"}, "POLICY"),
    ({"VLLM_DSPARK_REMOTE": "udp://x:1"}, "scheme"),
    ({"VLLM_DSPARK_REMOTE_TIMEOUT_MS": "0"}, ">= 1"),
    ({"VLLM_DSPARK_REMOTE_DRAFT_MS": "1,2"}, "one number"),
    ({"VLLM_DSPARK_REMOTE_STRICT_START": "maybe"}, "0 or 1"),
    ({"VLLM_DSPARK_REMOTE_BACKOFF_MS": "100", "VLLM_DSPARK_REMOTE_BACKOFF_MAX_MS": "10"},
     "BACKOFF"),
])
def test_bad_values_raise(kw, match):
    with pytest.raises(ValueError, match=match):
        RemoteConfig.from_env(env(**kw))


def test_markov_identity_from_cache_file(tmp_path):
    f = tmp_path / "markov_top_tokens.json"
    f.write_bytes(b"[1,2,3]")
    c = RemoteConfig.from_env(env(VLLM_DSPARK_MARKOV_CACHE=str(f)))
    assert c.markov_identity == hashlib.sha256(b"[1,2,3]").digest()
    with pytest.raises(ValueError, match="unreadable"):
        RemoteConfig.from_env(env(VLLM_DSPARK_MARKOV_CACHE=str(tmp_path / "missing")))


def test_cost_overrides():
    e = env(VLLM_DSPARK_REMOTE_POLICY="capacity")
    c = RemoteConfig.from_env(e)
    with pytest.raises(ValueError, match="REMOTE_DRAFT_MS"):
        apply_cost_overrides(c, e, depth_policy_on=True)
    assert apply_cost_overrides(c, e, depth_policy_on=False) == {}
    e2 = env(VLLM_DSPARK_REMOTE_POLICY="capacity", VLLM_DSPARK_REMOTE_DRAFT_MS="3.2",
             VLLM_DSPARK_REMOTE_ROUND_MS="30,33,36,39,42,45")
    c2 = RemoteConfig.from_env(e2)
    ch = apply_cost_overrides(c2, e2, depth_policy_on=True)
    assert e2["VLLM_DSPARK_DRAFT_MS"] == "3.2"
    assert e2["VLLM_DSPARK_ROUND_MS"] == "30,33,36,39,42,45"
    assert set(ch) == {"VLLM_DSPARK_DRAFT_MS", "VLLM_DSPARK_ROUND_MS"}
    e3 = env(VLLM_DSPARK_REMOTE_DRAFT_MS="3.2")          # shadow: no change
    assert apply_cost_overrides(RemoteConfig.from_env(e3), e3, depth_policy_on=True) == {}


def test_bind_env():
    c = RemoteConfig.from_env({"VLLM_DSPARK_REMOTE": "o1rdma://10.202.28.1:18620",
                               "VLLM_DSPARK_REMOTE_BIND": "10.202.28.2",
                               "VLLM_DSPARK_REMOTE_IDENTITY": ID})
    assert c.endpoint.scheme == "o1rdma" and c.endpoint.options["bind"] == "10.202.28.2"
    with pytest.raises(ValueError, match="conflicts"):
        RemoteConfig.from_env(env(VLLM_DSPARK_REMOTE_BIND="10.202.29.2"))


def test_async_validation():
    c = RemoteConfig.from_env(env(VLLM_DSPARK_REMOTE_POLICY="capacity",
                                  VLLM_DSPARK_REMOTE_ASYNC="1"))
    assert c.async_mode and c.switch_interval_us == 500
    with pytest.raises(ValueError, match="POLICY=capacity"):
        RemoteConfig.from_env(env(VLLM_DSPARK_REMOTE_POLICY="resilience",
                                  VLLM_DSPARK_REMOTE_ASYNC="1"))
    with pytest.raises(ValueError, match="cannot raise"):
        RemoteConfig.from_env(env(VLLM_DSPARK_REMOTE_POLICY="capacity",
                                  VLLM_DSPARK_REMOTE_ASYNC="1",
                                  VLLM_DSPARK_REMOTE_ON_FAIL="raise"))


def test_n_out_table():
    dense = [0, 5, 5, 4, 4]                  # [[1,2,5],[3,4,4]] at max_num_seqs 4
    cap = RemoteConfig.from_env(env(VLLM_DSPARK_REMOTE_POLICY="capacity"))
    res = RemoteConfig.from_env(env(VLLM_DSPARK_REMOTE_POLICY="resilience"))
    sh = RemoteConfig.from_env(env())
    off = RemoteConfig.from_env(env(VLLM_DSPARK_REMOTE_POLICY="capacity",
                                    VLLM_DSPARK_REMOTE_N_OUT="0"))
    for c in (cap, res):
        for mode in ("conf", "confb", "confr"):
            assert n_out_table(c, 5, mode, None, dense)[0] == [0, 5, 5, 4, 4]
    assert n_out_table(cap, 5, "conf", None, [0, 7, 3])[0] == [0, 5, 3]   # clamped to N
    assert n_out_table(sh, 5, "conf", None, dense)[0] is None            # parity: all N
    assert n_out_table(off, 5, "conf", None, dense)[0] is None
    for mode in ("off", "fixed"):                # no cap masking: drafts past cap can be used
        assert n_out_table(cap, 5, mode, None, dense)[0] is None
    assert n_out_table(cap, 5, "conf", "/tmp/depth", dense)[0] is None   # mode can change
    assert n_out_table(cap, 5, "conf", None, None)[0] is None
    assert n_out_table(cap, 5, "conf", None, [0, 5, 5])[0] is None       # never below N
    with pytest.raises(ValueError, match="N_OUT"):
        RemoteConfig.from_env(env(VLLM_DSPARK_REMOTE_N_OUT="4"))



# ------------------------------------------------------------------- S5 #4: cost inputs

LIVE_ROUND = "33.26,36.33,38.91,41.97,45.31,48.03"
REMOTE_ROUND = "29.66,32.73,35.31,38.63,42.43,44.55"


def s5_cfg():
    return RemoteConfig.from_env({"VLLM_DSPARK_REMOTE": "tcp://127.0.0.1:1",
                                  "VLLM_DSPARK_REMOTE_IDENTITY": bytes(32).hex(),
                                  "VLLM_DSPARK_REMOTE_POLICY": "capacity",
                                  "VLLM_DSPARK_REMOTE_DRAFT_MS": "4"})


def test_s5_cost_override_requires_verify_only_table():
    """S5 test_cost_override_requires_verify_only_table, verbatim, plus: nothing is written."""
    env = {"VLLM_DSPARK_ROUND_MS": "33.26,36.33,38.91"}
    with pytest.raises(ValueError, match="REMOTE_ROUND_MS"):
        apply_cost_overrides(s5_cfg(), env, depth_policy_on=True)
    assert env == {"VLLM_DSPARK_ROUND_MS": "33.26,36.33,38.91"}


@pytest.mark.parametrize("policy", ["capacity", "resilience"])
@pytest.mark.parametrize("given,missing", [
    ({}, "VLLM_DSPARK_REMOTE_DRAFT_MS and VLLM_DSPARK_REMOTE_ROUND_MS"),
    ({"VLLM_DSPARK_REMOTE_DRAFT_MS": "4"}, "missing VLLM_DSPARK_REMOTE_ROUND_MS"),
    ({"VLLM_DSPARK_REMOTE_ROUND_MS": REMOTE_ROUND}, "missing VLLM_DSPARK_REMOTE_DRAFT_MS"),
])
def test_depth_policy_on_needs_both_remote_costs(policy, given, missing):
    e = env(VLLM_DSPARK_REMOTE_POLICY=policy, VLLM_DSPARK_ROUND_MS=LIVE_ROUND,
            VLLM_DSPARK_DRAFT_MS="0", **given)
    before = dict(e)
    with pytest.raises(ValueError, match=missing):
        apply_cost_overrides(RemoteConfig.from_env(e), e, depth_policy_on=True)
    assert e == before                       # no partial override: the live table stays


@pytest.mark.parametrize("policy", ["capacity", "resilience"])
def test_both_remote_costs_replace_the_local_table(policy):
    e = env(VLLM_DSPARK_REMOTE_POLICY=policy, VLLM_DSPARK_ROUND_MS=LIVE_ROUND,
            VLLM_DSPARK_DRAFT_MS="0", VLLM_DSPARK_REMOTE_DRAFT_MS="4.25",
            VLLM_DSPARK_REMOTE_ROUND_MS=REMOTE_ROUND)
    ch = apply_cost_overrides(RemoteConfig.from_env(e), e, depth_policy_on=True,
                              depth_mode="conf", num_spec=5)
    assert ch == {"VLLM_DSPARK_DRAFT_MS": "4.25", "VLLM_DSPARK_ROUND_MS": REMOTE_ROUND}
    assert e["VLLM_DSPARK_ROUND_MS"] == REMOTE_ROUND and e["VLLM_DSPARK_DRAFT_MS"] == "4.25"


def test_shadow_keeps_local_measurements():
    e = env(VLLM_DSPARK_ROUND_MS=LIVE_ROUND, VLLM_DSPARK_DRAFT_MS="0",
            VLLM_DSPARK_REMOTE_DRAFT_MS="4", VLLM_DSPARK_REMOTE_ROUND_MS=REMOTE_ROUND)
    before = dict(e)
    for on in (True, False):
        assert apply_cost_overrides(RemoteConfig.from_env(e), e, depth_policy_on=on,
                                    depth_mode="confr" if on else "off") == {}
    assert e == before
    e = env(VLLM_DSPARK_REMOTE_DRAFT_MS="4")                  # shadow: incomplete is fine too
    assert apply_cost_overrides(RemoteConfig.from_env(e), e, depth_policy_on=True) == {}


def test_depth_policy_off_overrides_are_inert_but_validated():
    e = env(VLLM_DSPARK_REMOTE_POLICY="capacity", VLLM_DSPARK_REMOTE_DRAFT_MS="3.5")
    assert apply_cost_overrides(RemoteConfig.from_env(e), e, depth_policy_on=False) == {
        "VLLM_DSPARK_DRAFT_MS": "3.5"}
    e = env(VLLM_DSPARK_REMOTE_POLICY="capacity", VLLM_DSPARK_REMOTE_ROUND_MS="1,2,3")
    with pytest.raises(ValueError, match="expected 6"):
        apply_cost_overrides(RemoteConfig.from_env(e), e, depth_policy_on=False)


@pytest.mark.parametrize("kw,match", [
    ({"VLLM_DSPARK_REMOTE_DRAFT_MS": "nan"}, "finite"),
    ({"VLLM_DSPARK_REMOTE_DRAFT_MS": "inf"}, "finite"),
    ({"VLLM_DSPARK_REMOTE_DRAFT_MS": "-1"}, "non-negative"),
    ({"VLLM_DSPARK_REMOTE_ROUND_MS": "29,nan,35,38,42,44"}, "finite"),
    ({"VLLM_DSPARK_REMOTE_ROUND_MS": "29,32,35,38,42,inf"}, "finite"),
    ({"VLLM_DSPARK_REMOTE_ROUND_MS": "29,32,x,38,42,44"}, "numbers"),
])
def test_non_finite_costs_refused_at_parse(kw, match):
    with pytest.raises(ValueError, match=match):
        RemoteConfig.from_env(env(VLLM_DSPARK_REMOTE_POLICY="capacity", **kw))


@pytest.mark.parametrize("table,match", [
    ("0,32.73,35.31,38.63,42.43,44.55", "> 0"),
    ("29.66,32.73,35.31,38.63,42.43", "expected 6"),             # k + 1 = 6 depths
    ("29.66,32.73,35.31,38.63,42.43,44.55,47", "expected 6"),
    ("29.66,32.73,31.0,38.63,42.43,44.55", "non-decreasing"),
])
def test_remote_round_table_must_be_a_verify_table(table, match):
    e = env(VLLM_DSPARK_REMOTE_POLICY="capacity", VLLM_DSPARK_REMOTE_DRAFT_MS="4",
            VLLM_DSPARK_REMOTE_ROUND_MS=table, VLLM_DSPARK_ROUND_MS=LIVE_ROUND)
    with pytest.raises(ValueError, match=match):
        apply_cost_overrides(RemoteConfig.from_env(e), e, depth_policy_on=True,
                             depth_mode="conf", num_spec=5)
    assert e["VLLM_DSPARK_ROUND_MS"] == LIVE_ROUND


def test_confr_table_cannot_take_remote_costs():
    """confr:<rows ms> carries its own step table (measured with the local drafter); the
    remote cost inputs never reach it, so capacity/resilience refuse it."""
    e = env(VLLM_DSPARK_REMOTE_POLICY="capacity", VLLM_DSPARK_REMOTE_DRAFT_MS="4",
            VLLM_DSPARK_REMOTE_ROUND_MS=REMOTE_ROUND)
    with pytest.raises(ValueError, match="confr"):
        apply_cost_overrides(RemoteConfig.from_env(e), e, depth_policy_on=True,
                             depth_mode="confr", num_spec=5)
    for mode in ("conf", "confb", "fixed"):
        apply_cost_overrides(RemoteConfig.from_env(e), e, depth_policy_on=True,
                             depth_mode=mode, num_spec=5)
