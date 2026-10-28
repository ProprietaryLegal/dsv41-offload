from __future__ import annotations

import json
from pathlib import Path

import pytest

from offload.kvpack import kv_arith as k
from offload.pf2_lowkv import arith as a
from offload.pf2_lowkv import baselines as b


def test_kv_arith_transcription_still_validates() -> None:
    for kv, cap, pages, swa, want, _ in k.CHECKS:
        assert k.capacity(kv, cap, pages, swa)[0] == want


def test_h_arm_matches_the_measured_pf1_engine_pool() -> None:
    # PF1 W1/W2/W3/CW/MATCH rank-0 logs at 524,288 with 6 GiB: "GPU KV cache size: 2,820,756 tokens"
    assert a.expected_tokens(a.H_KV) == 2_820_756 == a.KV_TOKENS[a.H_KV]
    assert a.arm(a.H_KV).blocks == 46_603


def test_pick_is_the_low_constant_and_the_smallest_passing_candidate() -> None:
    assert a.pick() == a.LOW_KV == 3 * a.GIB
    low, two = a.arm(a.LOW_KV), a.arm(2 * a.GIB)
    assert not two.fits_two and two.blocks == 15_534 < two.two_long_need
    assert low.fits_two and low.fits_seats and low.meets_margin
    assert low.blocks == 23_301 and low.two_long_need == 17_329 and low.seats_need == 21_853
    assert a.expected_tokens(a.LOW_KV) == a.KV_TOKENS[a.LOW_KV] == 1_410_348


def test_admission_cap_request_and_lookahead() -> None:
    assert k.per_request(a.CAP, a.PAGES, a.SWA_GROUPS) == 6_827 + 1 + 14 * 131 == 8_662
    assert a.step_blocks() == 2


def test_gain_is_whole_blocks_and_about_three_gib() -> None:
    low = a.arm(a.LOW_KV)
    assert low.gain_vs_h_bytes == (46_603 - 23_301) * k.STRIDE
    assert abs(low.gain_vs_h_bytes / a.GIB - 3.0) < .001
    assert low.allocated_bytes + low.unused_bytes == a.LOW_KV and 0 <= low.unused_bytes < k.STRIDE


@pytest.mark.parametrize('bad', [2 * a.GIB, a.H_KV, 7 * a.GIB, 3 * a.GIB + 1, 0])
def test_check_low_refuses(bad: int) -> None:
    with pytest.raises(ValueError):
        a.check_low(bad)


def test_check_low_accepts_the_pick_and_larger_mib_values() -> None:
    assert a.check_low(a.LOW_KV).blocks == 23_301
    assert a.check_low(4 * a.GIB).fits_seats


def test_cli_prints_the_pick(capsys: pytest.CaptureFixture[str]) -> None:
    assert a.main(['--json']) == 0
    out = json.loads(capsys.readouterr().out)
    assert out['pick'] == a.LOW_KV and len(out['rows']) == 4


def _window(root: Path, arms: list[tuple[float, list[int]]]) -> Path:
    d = root / 'W'
    d.mkdir()
    ledger: list[dict[str, object]] = []
    tele: dict[str, list[dict[str, object]]] = {h: [] for h in b.HOSTS}
    for i, (t0, mems) in enumerate(arms):
        ledger.append({'event': 'send_intent', 'id': f'w{i}', 'kind': 'warmup', 'target': 32000, 'time': t0})
        ledger.append({'event': 'completed', 'id': f'w{i}', 'time': t0 + 10})
        for j, m in enumerate(mems):
            for h in b.HOSTS:
                tele[h].append({'kind': 'memory', 'utc': t0 + j, 'kb': {'MemAvailable': m, 'SwapFree': 1}})
                tele[h].append({'kind': 'gpu', 'utc': t0 + j, 'csv': 'x'})
    (d / 'requests-ledger.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in ledger))
    for h, rows in tele.items():
        (d / f'telemetry-{h}.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in rows))
    return d


def test_baseline_arms_split_on_warmups(tmp_path: Path) -> None:
    d = _window(tmp_path, [(100.0, [6 * 2**20] * 5), (200.0, [7 * 2**20, 5 * 2**20, 7 * 2**20])])
    rows = b.window(d, (('h', 90.0), ('l', None)))
    got = {(r.arm, r.host): r for r in rows}
    assert got[('h', 'sparkhead')].samples == 5 and got[('h', 'sparkhead')].median_gib == 6.0
    assert got[('l', 'sparkworker-wired')].min_gib == 5.0 and got[('l', 'sparkworker-wired')].position == 2


def test_baseline_refuses_arm_count_mismatch(tmp_path: Path) -> None:
    d = _window(tmp_path, [(100.0, [6 * 2**20])])
    with pytest.raises(ValueError, match='warm-ups'):
        b.window(d, (('h', None), ('l', None)))


def test_fw1_inventory_parses_and_refuses(tmp_path: Path) -> None:
    good = tmp_path / 'inventory-post.txt'
    good.write_text('7.0.0-1004-nvidia\nDGX_OTA_VERSION="7.6.0"\n580.178.02\nSystem Firmware:\n  Current version: 0x03000d07\n'
                    'MemTotal:       125000000 kB\nMemAvailable:   110000000 kB\nSwapTotal:      16777212 kB\n')
    assert b.inventory(good) == {'kernel': '7.0.0-1004-nvidia', 'driver': '580.178.02', 'MemTotal': 125000000,
                                 'MemAvailable': 110000000, 'SwapTotal': 16777212}
    bad = tmp_path / 'inventory-pre.txt'
    bad.write_text('MemTotal: 1 kB\n')
    with pytest.raises(ValueError, match='not an FW1 inventory'):
        b.inventory(bad)
