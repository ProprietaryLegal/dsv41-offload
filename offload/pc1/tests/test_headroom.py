"""Round3 source/observed memory admission and unchanged shared telemetry flow."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
from typing import cast

import pytest

from offload.pc1 import build, headroom, window as w
from offload.pc1.tests.test_window import Remote, rehearsal as rehearsal
from offload.prefill_maxlen.evidence import Json, number, record

CAMPAIGN = Path('/opt/dsv41/campaigns/DSV41-OFFLOAD')


def rows(path: Path) -> list[dict[str, Json]]:
    return [record(cast(Json, json.loads(line))) for line in path.read_text().splitlines()]


def test_measured_inputs_and_memory_peaks_reproduce() -> None:
    inputs = cast(dict[str, str], json.loads((build.HERE / 'headroom-inputs.json').read_text()))
    for name, sha in inputs.items():
        assert hashlib.sha256(Path(name).read_bytes()).hexdigest() == sha, name
    minima: dict[str, int] = {}
    for folder in ('PC1-chunk2048-W-20261027T100908Z', 'SP1-W-20261026T144153Z',
                   'SP1-W-20261026T160012Z', 'SP1-W-20261026T191247Z'):
        root = CAMPAIGN / folder
        ledger = rows(root / 'requests-ledger.jsonl')
        cell = '150k-b1' if folder.startswith('PC1') else 'ctl-150k-b1'
        times = [number(r['time']) for r in ledger if r['cell'] == cell]
        for host in headroom.MEASURED_MIN_KB:
            samples = rows(root / f'telemetry-{host}.jsonl')
            available = [int(number(record(r['kb'])['MemAvailable'])) for r in samples
                         if r['kind'] == 'memory' and min(times) <= number(r['utc']) <= max(times)]
            assert available
            minima[host] = min(minima.get(host, max(available)), min(available))
    assert minima == headroom.MEASURED_MIN_KB
    for host in headroom.MEASURED_MIN_KB:
        name = next(name for name in inputs if '/memprobe/' + host + '/' in name)
        phases = [r for r in rows(Path(name)) if r['ev'] == 'phase' and number(r['toks']) >= 4092]
        assert max(number(r['peak_over']) for r in phases) == headroom.PEAK_MIB
        assert max(number(r['peak_over']) for r in phases if r['toks'] == 4092 and r['tag'] == 'ffn') == 875


@pytest.mark.parametrize('arm,extra,min_sparkhead,admitted', [
    ('chunk8192', 1168384, 5580896, True), ('chunk16384', 2980864, 3768416, True)])
def test_registered_headroom_estimate(arm: str, extra: int, min_sparkhead: int, admitted: bool) -> None:
    proof = headroom.estimate(arm)
    assert proof['extra_activation_reserve_kb'] == extra
    assert record(proof['predicted_min_kb'])['sparkhead'] == min_sparkhead
    assert proof['admissible'] is admitted
    if admitted:
        headroom.admit(arm)
    else:
        with pytest.raises(ValueError, match='activation peak breaks 3 GiB'):
            headroom.admit(arm)


def test_fresh_control_reserve_boundary() -> None:
    required = headroom.FLOOR_KB + 1168384
    headroom.control_reserve('chunk8192', dict.fromkeys(headroom.MEASURED_MIN_KB, required))
    for host in headroom.MEASURED_MIN_KB:
        minima = dict.fromkeys(headroom.MEASURED_MIN_KB, required)
        minima[host] -= 1
        with pytest.raises(ValueError, match='fresh control lacks'):
            headroom.control_reserve('chunk8192', minima)
    with pytest.raises(ValueError, match='both ranks'):
        headroom.control_reserve('chunk8192', {'sparkhead': required})


def test_floor_is_ndm_3gib_and_refuses_one_kb_below() -> None:
    assert headroom.FLOOR_KB == 3 * 1024**2
    proof = headroom.estimate('chunk16384')
    assert min(int(number(v)) for v in record(proof['predicted_min_kb']).values()) >= headroom.FLOOR_KB
    required = headroom.FLOOR_KB + 2980864
    headroom.control_reserve('chunk16384', dict.fromkeys(headroom.MEASURED_MIN_KB, required))
    with pytest.raises(ValueError, match='fresh control lacks'):
        headroom.control_reserve('chunk16384', {**dict.fromkeys(headroom.MEASURED_MIN_KB, required), 'sparkhead': required - 1})


def test_8192_missing_fresh_reserve_restores_before_candidate_start(rehearsal: tuple[w.Live, Remote],
        monkeypatch: pytest.MonkeyPatch) -> None:
    live, fake = rehearsal
    live.group = 'chunk8192'
    original = live.finish
    def finish(arm: w.Arm) -> dict[str, Json]:
        result = original(arm)
        live.sup.arm_memory['s1-control']['sparkhead'] = headroom.FLOOR_KB
        return result
    monkeypatch.setattr(live, 'finish', finish)
    result = w.window(live, live.group, lambda row: None)
    assert 'fresh control lacks' in str(result['stop'])
    assert 's2-chunk8192-start' not in fake.commands
    assert fake.restores == ['start', 'check', 'close'] and not fake.hold
    assert record(result['decision'])['status'] == 'NOT A CANDIDATE'


def test_live_supervisor_enforces_3gib_without_changing_shared_floor(tmp_path: Path) -> None:
    # Real shared poll over inert selector/process/demand plumbing; no RPC.
    from offload.launch.windows import pf1_window as pf
    sup = w.LargeChunkSupervisor(tmp_path, demand=lambda: 0.)
    sup.arm_minima = dict.fromkeys(headroom.MEASURED_MIN_KB, headroom.FLOOR_KB)
    try:
        sup.poll()
        for host in headroom.MEASURED_MIN_KB:
            sup.arm_minima = dict.fromkeys(headroom.MEASURED_MIN_KB, headroom.FLOOR_KB)
            sup.arm_minima[host] -= 1
            with pytest.raises(ValueError, match='below 3 GiB'):
                sup.poll()
        assert pf.MEM_FLOOR_KB == 4 * 1024**2
    finally:
        sup.close()


def test_larger_chunk_budget_uses_measured_start_basis_and_explicit_allowance() -> None:
    data = record(record(cast(Json, json.loads((build.HERE / 'budget.json').read_text())))['round3'])
    assert data['pc1_ready_observations_s'] == [353, 337]
    for name, observed in zip(cast(list[str], data['pc1_ready_sources']), (353, 337)):
        assert f'ready after {observed} s' in (CAMPAIGN / name).read_text()
    win2 = record(record(record(cast(Json, json.loads((build.HERE / 'budget.json').read_text())))['win2'])['groups'])
    for arm in build.LARGE_ARMS:
        # Round3 (pre-WIN2, every cell x2) recorded 1424 s / 78.2 min; WIN2 plans keep the same +180 s allowance.
        assert data['candidate_block_s'] == 1214 + w.QUALIFY_S + headroom.STARTUP_EXTRA_S == 1424
        assert record(data['hold_estimate_minutes_by_arm'])[arm] == 78.2
        assert w.arm_seconds(arm, arm) == record(win2[arm])['candidate_block_s'] == 1394
        assert w.prepare_seconds(arm) == w.PREPARE_S + headroom.STARTUP_EXTRA_S
        assert w.budget_minutes(arm) == record(win2[arm])['hold_estimate_minutes'] == 76.7
    assert w.SUBMIT_S == w.ABORT_S == 73 * 60 and w.RESTORE_S == 85 * 60
