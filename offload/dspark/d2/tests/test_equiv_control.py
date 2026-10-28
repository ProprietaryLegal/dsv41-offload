"""The equivalence gate's BF16-noise rule calibrated to a same-run noise control (coordinator amendment 2026-10-23
14:59Z, documents/plans/2026-10-23-d2-plan.md): control arm N = decode taps + Gaussian relative noise sigma = 2^-8
(seed 20261023); met when pos1 >= control pos1 - 0.05 (amendment 2, 16:30Z: a gross-error backstop), every
per-position median KL(D || R) <= max(1e-3, 2 x control) and <= 0.05 nats."""

from __future__ import annotations

import json
import types

import pytest
import torch

from d2 import equiv as E
from d2.tests.test_s14_fixes import _eq_rec, _report


def _recs(n, kl, ctl_kl, r_pos1_miss=0, ctl_pos1_miss=0, equal_from=None):
    """n inputs-differ blocks; the first r_pos1_miss / ctl_pos1_miss differ at position 1, the rest past it."""
    equal_from = n if equal_from is None else equal_from
    out = []
    for j in range(n):
        rec = _eq_rec(j, j >= equal_from, 128, kl=kl, pos1_same=j >= r_pos1_miss, ctl_kl=ctl_kl)
        if j < ctl_pos1_miss:
            rec["N"]["drafts"] = [6, 2, 3, 4, 5]
        out.append(rec)
    return out


def test_the_04_percent_noise_experiment_now_passes(tmp_path):
    """D2-DATA flag 9's measurement: 0.4 % noise (~ the control's 2^-8), median KL 0.008-0.020 nats, half the
    differing blocks unequal. With the control at the same level the rule is met."""
    recs = _recs(100, kl=None, ctl_kl=None, equal_from=50)
    kls = [0.008, 0.011, 0.014, 0.017, 0.020]
    ctl = [0.007, 0.010, 0.012, 0.016, 0.018]
    for r in recs:
        r["kl_D_R"], r["kl_D_N"] = list(kls), list(ctl)
        r["N"] = {"drafts": r["D"]["drafts"] if r["key"] >= "r000060" else [1, 2, 3, 9, 5], "margin": [1.0] * 5}
        r["control"] = {"sigma": E.CONTROL_SIGMA, "seed": E.CONTROL_SEED}
    g = _report(tmp_path, recs)["gate"]
    assert g["block_equal_frac"] == 0.5 and g["verdict"] == "pass_bf16_noise" and g["pass"]
    r2 = g["rule2_bf16_noise"]
    assert r2["checks"] == {"pos1": True, "kl_within_control_bound": True, "kl_within_ceiling": True}
    assert r2["kl_bound_per_position"] == [max(1e-3, 2 * c) for c in ctl]
    # the old fixed bound (1e-3) would have failed this
    assert max(r2["kl_median_per_position"]) > 1e-3


def test_real_divergence_fails(tmp_path):
    # 10x the control's KL (as 10x noise or shuffled taps give): over the control bound
    g = _report(tmp_path, _recs(100, kl=0.10, ctl_kl=0.01, equal_from=50))["gate"]
    assert g["verdict"] == "fail" and not g["pass"]
    assert g["rule2_bf16_noise"]["checks"]["kl_within_control_bound"] is False
    # shuffled taps: position-1 drafts disagree on a third of the blocks while the control agrees
    g = _report(tmp_path, _recs(90, kl=0.012, ctl_kl=0.01, r_pos1_miss=30, equal_from=45))["gate"]
    assert g["verdict"] == "fail" and g["rule2_bf16_noise"]["checks"]["pos1"] is False


def test_ceiling_binds_even_when_the_control_is_noisy(tmp_path):
    # control KL 0.04 -> control bound 0.08, but R's 0.06 is over the 0.05 ceiling
    g = _report(tmp_path, _recs(100, kl=0.06, ctl_kl=0.04, equal_from=50))["gate"]
    c = g["rule2_bf16_noise"]["checks"]
    assert c == {"pos1": True, "kl_within_control_bound": True, "kl_within_ceiling": False} and g["verdict"] == "fail"


def test_kl_floor_when_the_control_is_quiet(tmp_path):
    # control KL 1e-5 -> bound max(1e-3, 2e-5) = 1e-3: R at 8e-4 passes, at 2e-3 fails
    assert _report(tmp_path, _recs(100, kl=8e-4, ctl_kl=1e-5, equal_from=50))["gate"]["verdict"] == "pass_bf16_noise"
    assert _report(tmp_path, _recs(100, kl=2e-3, ctl_kl=1e-5, equal_from=50))["gate"]["verdict"] == "fail"


def test_pos1_threshold_follows_the_control():
    # amendment 2: threshold = control pos1 - 0.05 (absolute). Control 0.95 -> 0.90: R 0.90 passes, 0.89 fails
    r = E.bf16_noise_rule(_recs(100, kl=0.01, ctl_kl=0.01, r_pos1_miss=10, ctl_pos1_miss=5, equal_from=50))
    assert r["control"]["pos1_equal_frac"] == 0.95 and r["pos1_threshold"] == 0.9 and r["checks"]["pos1"]
    assert r["pos1_slack"] == 0.05
    r = E.bf16_noise_rule(_recs(100, kl=0.01, ctl_kl=0.01, r_pos1_miss=11, ctl_pos1_miss=5, equal_from=50))
    assert r["pos1_equal_frac"] == 0.89 and not r["checks"]["pos1"] and not r["met"]
    # control pos1 1.0 -> threshold 0.95 (no 0.99 cap any more)
    r = E.bf16_noise_rule(_recs(100, kl=0.01, ctl_kl=0.01, r_pos1_miss=5, equal_from=50))
    assert r["pos1_threshold"] == 0.95 and r["checks"]["pos1"]
    r = E.bf16_noise_rule(_recs(100, kl=0.01, ctl_kl=0.01, r_pos1_miss=6, equal_from=50))
    assert not r["checks"]["pos1"]


# ------------------------------------------------------------------------------------- amendment 2 (16:30Z)
def _flips(n, r_flips, ctl_flips, kl, ctl_kl):
    """n differing blocks; R flips position 1 on the first r_flips, the control on the first ctl_flips; equal past
    position 1 on half the blocks (so the 99 % block gate is not met and the noise rule decides)."""
    recs = _recs(n, kl=None, ctl_kl=None, r_pos1_miss=r_flips, ctl_pos1_miss=0, equal_from=n // 2)
    for j, r in enumerate(recs):
        r["kl_D_R"], r["kl_D_N"] = list(kl), list(ctl_kl)
        r["N"] = {"drafts": [6, 2, 3, 4, 5] if j < ctl_flips else r["D"]["drafts"], "margin": [1.0] * 5}
        r["control"] = {"sigma": E.CONTROL_SIGMA, "seed": E.CONTROL_SEED}
    return recs


S16_R_KL = (0.0014, 0.0035, 0.0042, 0.0091, 0.0058)          # S16.report.md 7.4, the 7x-quieter 1-ULP arm
S16_CTL_KL = (0.0038, 0.0090, 0.0053, 0.0094, 0.0103)


def test_amendment2_s16_quieter_case_at_its_26_blocks():
    """S16 7.4 as measured: 26 differing blocks, R flips 3, the control 1, R's KL under the control's everywhere. With
    an absolute slack of 0.05 the threshold is 25/26 - 0.05 = 0.9115 and R's 23/26 = 0.8846 is still under it: two
    extra flips in 26 blocks are 7.7 points. The KL checks pass."""
    r = E.bf16_noise_rule(_flips(26, 3, 1, S16_R_KL, S16_CTL_KL))
    assert r["pos1_equal_frac"] == round(23 / 26, 4) and r["control"]["pos1_equal_frac"] == round(25 / 26, 4)
    assert r["checks"] == {"pos1": False, "kl_within_control_bound": True, "kl_within_ceiling": True}


def test_amendment2_s16_quieter_flip_counts_pass_at_every_1():
    """The same flip counts (R 3, control 1) and KL in an --every 1 population (530 differing blocks): passes. So does
    a replay that flips at the control's own rate (8 % both, R 5 more blocks than the control)."""
    r = E.bf16_noise_rule(_flips(530, 3, 1, S16_R_KL, S16_CTL_KL))
    assert r["checks"] == {"pos1": True, "kl_within_control_bound": True, "kl_within_ceiling": True} and r["met"]
    r = E.bf16_noise_rule(_flips(530, 47, 42, S16_R_KL, S16_CTL_KL))
    assert r["met"]


def test_amendment2_row_keying_error_still_fails_all_three():
    """S16 7.4 / D2-DATA round 3: row p given p-1's taps: pos1 16/26 = 0.615 against the control's 25/26, KL 30-100x."""
    r = E.bf16_noise_rule(_flips(26, 10, 1, (0.75, 0.81, 0.30, 0.33, 0.23), S16_CTL_KL))
    assert r["checks"] == {"pos1": False, "kl_within_control_bound": False, "kl_within_ceiling": False}
    assert not r["met"]


def test_amendment2_exactly_control_minus_005_passes_one_block_worse_fails():
    # 100 blocks: control 98/100 -> threshold exactly 93/100
    r = E.bf16_noise_rule(_flips(100, 7, 2, S16_R_KL, S16_CTL_KL))
    assert r["pos1_threshold"] == 0.93 and r["pos1_equal_frac"] == 0.93 and r["checks"]["pos1"] and r["met"]
    r = E.bf16_noise_rule(_flips(100, 8, 2, S16_R_KL, S16_CTL_KL))
    assert not r["checks"]["pos1"] and not r["met"]
    # 530 blocks (--every 1): control 520/530 -> threshold 493.5/530: R 494/530 passes, 493/530 fails
    r = E.bf16_noise_rule(_flips(530, 36, 10, S16_R_KL, S16_CTL_KL))
    assert r["checks"]["pos1"]
    r = E.bf16_noise_rule(_flips(530, 37, 10, S16_R_KL, S16_CTL_KL))
    assert not r["checks"]["pos1"]


def test_mixed_controls_are_refused():
    recs = _recs(10, kl=0.01, ctl_kl=0.01)
    recs[3]["control"] = {"sigma": 2.0 ** -7, "seed": E.CONTROL_SEED}
    with pytest.raises(SystemExit, match="different noise controls"):
        E.bf16_noise_rule(recs)


def test_report_keys_s14_probes_read_still_exist(tmp_path):
    rep = _report(tmp_path, _recs(100, kl=0.01, ctl_kl=0.01, equal_from=50))
    for k in ("block_equal_frac", "block_equal_frac_with_decode_rows", "block_equal_frac_all_blocks",
              "blocks_inputs_differ", "blocks_inputs_identical", "block_equal_frac_inputs_identical",
              "decode_rows_in_window_bins", "committed_trajectory_proxy"):
        assert k in rep, k
    assert {"pass", "verdict", "rule2_bf16_noise", "block_equal_frac"} <= set(rep["gate"])
    assert rep["gate"]["rule2_bf16_noise"]["kl_recorded"] is True
    assert set(rep["committed_trajectory_proxy"]) == {"D", "R", "N", "engine"}     # recorded, not gating


class _Row:
    def __init__(self, tok, taps):
        self.tok, self.round, self.mx, self.taps = tok, 0, None, taps


def _cs():
    g = torch.Generator().manual_seed(1)
    rows = {}
    pre = {}
    for r in ("reqA", "reqB"):
        rows[r] = {p: _Row(p, torch.randn(E.ROW, generator=g).to(torch.bfloat16) if p % 2 else None)
                   for p in range(8)}
        pre[r] = torch.randn(8, E.ROW, generator=g).to(torch.bfloat16)
    return types.SimpleNamespace(rows=rows), pre


def test_control_arm_is_deterministic_and_only_perturbs_decode_rows():
    cs, pre = _cs()
    reqs = ["reqA", "reqB"]
    n1 = E.arm_rows(cs, pre, "N", reqs)
    n2 = E.arm_rows(cs, pre, "N", reqs)
    d = E.arm_rows(cs, pre, "D")
    for r in reqs:
        for p in range(8):
            assert torch.equal(n1[r][p].taps, n2[r][p].taps)                # same seed -> same control
            if cs.rows[r][p].taps is None:
                assert torch.equal(n1[r][p].taps, pre[r][p]) and torch.equal(d[r][p].taps, pre[r][p])
            else:
                x, y = cs.rows[r][p].taps.float(), n1[r][p].taps.float()
                rel = float(((y - x) / x.abs().clamp_min(1e-6)).std())
                assert 0.5 * E.CONTROL_SIGMA < rel < 2 * E.CONTROL_SIGMA, rel
                assert not torch.equal(n1[r][p].taps, cs.rows[r][p].taps)
    # a row's noise depends on (seed, request, position) only: the same row drafted alone gets the same noise
    alone = E.control_noise(cs.rows["reqB"][3].taps, 1, 3)
    assert torch.equal(alone, n1["reqB"][3].taps)
    assert not torch.equal(E.control_noise(cs.rows["reqB"][3].taps, 1, 5), alone)
    assert not torch.equal(E.control_noise(cs.rows["reqB"][3].taps, 1, 3, seed=1), alone)
    with pytest.raises(ValueError, match="request order"):
        E.arm_rows(cs, pre, "N")


def test_control_constants_are_the_amendment_s():
    assert E.CONTROL_SIGMA == 2.0 ** -8 and E.CONTROL_SEED == 20261023
    assert (E.POS1_SLACK, E.KL_FLOOR, E.KL_CONTROL_FACTOR, E.KL_CEILING) == (0.05, 1e-3, 2.0, 0.05)
    assert not hasattr(E, "POS1_MIN")                          # amendment 2: no 0.99 cap on the pos1 threshold
    assert json.dumps({"sigma": E.CONTROL_SIGMA, "seed": E.CONTROL_SEED})
