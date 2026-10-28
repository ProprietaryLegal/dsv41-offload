"""WIN2 chain mode in the UP3 runner (NDM 2026-10-27 "go window after window"): group-agnostic, CPU fakes only."""
from __future__ import annotations

import inspect
import json
import os
import types
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import cast

import pytest

from offload.launch.windows import chain, pf1_window as pf, up3_window as w, nbw_cells as greedy_driver
from offload.launch.windows import post_close_watch as watchdog, production_health
from offload.launch.windows.tests import test_pf1_window as base
from offload.prefill_maxlen.evidence import Json, record
from offload.up3.tests.test_up3_window import Remote

Rig = Callable[..., tuple[w.Live, Remote, list[str]]]


@pytest.fixture
def rig(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Rig]:
    import offload.prefill_trace.window_requests as wr
    monkeypatch.setattr(greedy_driver, 'greedy_inputs', lambda: (['public']*6, [{'i': i, 'text': 'public'} for i in range(6)]))
    monkeypatch.setattr(wr, 'corpus', lambda root: ('public fixture', {}))
    monkeypatch.setattr(pf, 'CAMPAIGN', tmp_path)
    (tmp_path / 'D2-GEN.PAUSE').touch()
    monitor = tmp_path / 'monitor.py'
    monitor.write_text(base.MONITOR)
    sups: list[pf.Supervisor] = []

    def make(name: str, *, fail: str = '', down: bool = False, identity: Path | None = None) -> tuple[w.Live, Remote, list[str]]:
        clock = base.Clock()
        out = tmp_path / name
        out.mkdir()
        sup = base.NoTunnel(out, clock=clock, demand=lambda: 0., monitor=lambda host: [
            os.environ.get('UP3_TEST_PYTHON', '/home/operator/miniconda3/bin/python3'), str(monitor), 'low', str(tmp_path / 'trip')])
        sups.append(sup)
        remote = Remote(out, sup, clock, fail, trip=tmp_path / 'trip')
        remote.down = down
        live = w.Live(out, sup, group='k5', remote='dsv41-deploy-offload', expected='a' * 40, data_dir=tmp_path, tag='tag',
                      call=remote.call, plain=remote.plain, clock=clock, production_identity=identity)
        remote.live = live
        labels: list[str] = []
        original = remote.remote
        def recorded(label: str, host: str, command: str, stdin: str | None) -> str:
            labels.append(label)
            return original(label, host, command, stdin)
        remote.remote = recorded  # type: ignore[method-assign]
        return live, remote, labels
    try:
        yield make
    finally:
        for sup in sups:
            sup.close()


def run(live: w.Live) -> dict[str, Json]:
    return w.window(live, 'k5', lambda row: None)


def status(result: dict[str, Json]) -> str:
    return str(record(result['restore'])['status'])


def watched(result: dict[str, Json]) -> None:
    """main()'s tail after a strict restore: the post-close watch passed, then end_chain (WIN2.md: clear only now)."""
    result['post_close_watch'] = {'status': 'PASS'}
    w.end_chain(result)


def first_window(rig: Rig, monkeypatch: pytest.MonkeyPatch) -> w.Live:
    monkeypatch.setenv('WINDOW_CHAIN', '1')
    live, fake, _ = rig('W1')
    result = run(live)
    assert 'stop' not in result, result.get('stop')
    assert status(result).startswith('CHAINED') and not fake.hold and fake.restores == []
    return live


def test_first_chained_window_leaves_production_down(rig: Rig, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv('WINDOW_CHAIN', '1')
    live, fake, labels = rig('W1')
    result = run(live)
    assert 'stop' not in result, result.get('stop')
    assert 'chained_end_error' not in result, result['chained_end_error']
    assert result['status'] == 'COMPLETE' and record(result['decision'])['status'] == 'CANDIDATE FOR REVIEW'
    assert status(result).startswith('CHAINED: production left down'), result['restore']
    assert result['chain'] == 'CHAINED' and result['end_action'] == 'chained'  # chain_prev_ok.sh gate field
    assert live.chain_mode == 'production' and 'prod-models' in labels
    assert fake.restores == [] and not fake.hold  # no W8 start/check/close; own HOLD removed
    order = [fake.commands.index(label) for label in ('placeholder-keep-sparkworker-wired', 'experiment-stop', 'jit-wait-sparkworker-wired',
                                                         'placeholder-probe-sparkworker-wired', 'chained-close', 'commit-close',
                                                         'cancel-close')]
    assert order == sorted(order) and 'retain-close' not in fake.commands
    assert not [c for c in fake.commands if c.startswith('placeholder-release')]   # kept up for the next window
    marker = chain.read_marker()
    assert marker is not None and marker['placeholders'] == chain.PLACEHOLDERS_KEPT
    rollback = live.out / 'rollback.json'
    assert marker['production_snapshot'] == chain.snapshot_ref(rollback) and chain.production_snapshot()[0] == rollback
    assert record(record(marker['quiet_proof'])['sparks']) == {host: 'UP3 EXPERIMENT ABSENT' for host in pf.HOSTS}


def test_chained_window_runs_from_the_marker(rig: Rig, monkeypatch: pytest.MonkeyPatch) -> None:
    first = first_window(rig, monkeypatch)
    live, fake, labels = rig('W2', down=True)
    result = run(live)
    assert 'stop' not in result, result.get('stop')
    assert result['status'] == 'COMPLETE' and status(result).startswith('CHAINED')
    assert live.chain_mode == 'chained' and fake.restores == []
    # Production is down: no production health/model/identity/idle probe; placeholders and quiet proved before the HOLD.
    assert not [x for x in labels if x.startswith(('prod-health', 'prod-models', 'prod-identity', 'prod-idle', 'prod-fullenv'))]
    assert labels.index('chain-placeholders-sparkworker-wired') < labels.index('chain-quiet-sparkworker-wired') < labels.index('hold-open')
    rollback = json.loads((live.out / 'rollback.json').read_text())
    assert rollback['chained_from'] == str(first.out / 'rollback.json')
    assert rollback['ranks'] == json.loads((first.out / 'rollback.json').read_text())['ranks']
    marker = chain.read_marker()
    assert marker is not None and record(marker['production_snapshot'])['path'] == str(live.out / 'rollback.json')


def test_last_window_restores_strictly_and_clears(rig: Rig, monkeypatch: pytest.MonkeyPatch) -> None:
    first_window(rig, monkeypatch)
    monkeypatch.setenv('WINDOW_CHAIN_LAST', '1')
    live, fake, _ = rig('W2', down=True)
    result = run(live)
    assert 'stop' not in result, result.get('stop')
    assert status(result).startswith('RESTORED') and fake.restores == ['start', 'check', 'close']
    assert record(result['placeholders']) == {h: 'RELEASED' for h in pf.HOSTS}   # window.sh close cannot (private socket)
    assert chain.MARKER.exists()   # kept until the post-close watch passes
    watched(result)
    assert not chain.MARKER.exists()


@pytest.mark.parametrize('watch', ['raises', 'fails'])
def test_failed_post_close_watch_keeps_the_marker(rig: Rig, monkeypatch: pytest.MonkeyPatch, watch: str) -> None:
    """UP2-ADOPT b1483bc regression: chain.clear() ran inside window(), before main()'s post-close watch. WIN2.md: the
    marker is cleared only after the strict restore AND the watch PASS; a failed watch leaves it for the coordinator."""
    first_window(rig, monkeypatch)
    monkeypatch.setenv('WINDOW_CHAIN_LAST', '1')
    live, fake, _ = rig('W2', down=True)
    result = run(live)
    assert status(result).startswith('RESTORED') and fake.restores == ['start', 'check', 'close']
    assert chain.MARKER.exists(), 'window() cleared the chain marker before the post-close watch'
    def failed(*args: object, **kwargs: object) -> object:
        if watch == 'raises':
            raise RuntimeError('production unhealthy after close')
        return types.SimpleNamespace(status='FAILED', reason='production unhealthy after close')
    monkeypatch.setattr(production_health, 'ProductionHealth', lambda: object())
    monkeypatch.setattr(watchdog, 'ScopedRecovery', lambda tag, out: object())
    monkeypatch.setattr(watchdog, 'watch', failed)
    w.post_close(result, live.out, 'tag')
    w.end_chain(result)
    assert record(result['post_close_watch'])['status'] == 'FAILED' and chain.MARKER.exists()
    assert record(result['decision'])['status'] == 'NOT A CANDIDATE'


def test_main_clears_the_marker_only_after_the_watch() -> None:
    assert 'chain.clear' not in inspect.getsource(w.window)
    tail = inspect.getsource(w.main)
    assert tail.index('post_close(result, out, tag)') < tail.index('end_chain(result)')


@pytest.mark.parametrize('fail', ['start', 'STOP'])
def test_any_stop_in_a_chain_restores_strictly(rig: Rig, monkeypatch: pytest.MonkeyPatch, fail: str) -> None:
    first_window(rig, monkeypatch)
    live, fake, _ = rig('W2', fail=fail, down=True)
    result = run(live)
    assert 'stop' in result and record(result['decision'])['status'] == 'NOT A CANDIDATE'
    assert status(result).startswith('RESTORED') and fake.restores == ['start', 'check', 'close']
    assert chain.MARKER.exists()
    watched(result)
    assert not chain.MARKER.exists()


def test_chained_window_without_any_start_still_restarts_production(rig: Rig, monkeypatch: pytest.MonkeyPatch) -> None:
    first_window(rig, monkeypatch)
    live, fake, _ = rig('W2', down=True)
    calls = iter((False, True))
    original = w.chain_identity
    def identity(explicit: Path | None) -> tuple[Path, dict[str, Json]]:
        if next(calls):
            raise ValueError('identity changed after preflight')
        return original(explicit)
    monkeypatch.setattr(w, 'chain_identity', identity)
    result = run(live)
    assert 'identity changed after preflight' in str(result['stop']) and 'starts' in result and not record(result['starts'])
    assert fake.restores == ['start', 'check', 'close'] and status(result).startswith('RESTORED')


def test_failed_chained_close_keeps_the_hold_after_restoring(rig: Rig, monkeypatch: pytest.MonkeyPatch) -> None:
    # The window's one -restore-close scope (the rm shim needs HOLD's own tag) is spent by the failed chained close: the
    # fallback strict restore brings production back and verifies it, but leaves the HOLD to the coordinator.
    monkeypatch.setenv('WINDOW_CHAIN', '1')
    live, fake, _ = rig('W1', fail='chained-close')
    result = run(live)
    assert 'chained-close' in fake.commands and 'chained_end_error' in result
    assert fake.restores == ['start', 'check'] and status(result) == 'FAILED' and fake.hold
    assert 'close scope already spent' in str(record(result['restore'])['error'])
    assert not chain.MARKER.exists() and result['status'] == 'COMPLETE'


def test_failed_chained_end_before_its_close_restores_and_closes(rig: Rig, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv('WINDOW_CHAIN', '1')
    live, fake, _ = rig('W1')
    def missing(label: str, command: str, marker: str) -> dict[str, Json]:
        raise ValueError(f'{label}: hold placeholder unconfirmed')
    monkeypatch.setattr(live, 'placeholders', missing)
    result = run(live)
    assert 'placeholder-keep' in str(result['chained_end_error'])
    assert status(result).startswith('RESTORED') and fake.restores == ['start', 'check', 'close']


def write_marker(proof: dict[str, Json], snapshot: Json = None) -> None:
    chain.MARKER.parent.mkdir(parents=True, exist_ok=True)
    chain.MARKER.write_text(json.dumps({'schema': chain.SCHEMA, 'window': 'x', 'time': 0, 'utc': 'x', 'quiet_proof': proof,
                                        'production_snapshot': snapshot}))


@pytest.mark.parametrize('case,message', [
    ('marker-without-chain', 'run with WINDOW_CHAIN=1'),
    ('down-without-marker', 'production not serving and no chain marker'),
    ('marker-but-serving', 'production is serving'),
    ('placeholders-missing', 'hold placeholders missing'),
    ('no-snapshot-marker', 'carries no production snapshot'),
    ('changed-identity', 'changed since the marker was written'),
    ('filtered-env', 'env is filtered'),
    ('identity-in-production-mode', 'for chained windows only'),
])
def test_chain_preflight_refuses_before_hold(rig: Rig, monkeypatch: pytest.MonkeyPatch, case: str, message: str) -> None:
    first = first_window(rig, monkeypatch)
    rollback = first.out / 'rollback.json'
    explicit: Path | None = None
    down, fail = True, ''
    if case == 'marker-without-chain':
        monkeypatch.delenv('WINDOW_CHAIN')
    elif case == 'down-without-marker':
        chain.clear()
    elif case == 'marker-but-serving':
        down = False
    elif case == 'placeholders-missing':
        fail = 'placeholders'
    elif case == 'no-snapshot-marker':
        write_marker({'runner': 'older'})
    elif case == 'changed-identity':
        rollback.write_text(rollback.read_text() + ' ')
    elif case == 'filtered-env':
        # Another runner's R2 marker whose snapshot is a filtered (pf1_window) capture.
        data = json.loads(rollback.read_text())
        for rank in data['ranks']:
            rank['env'] = {k: v for k, v in rank['env'].items() if k.startswith('VLLM_')}
        nbw = rollback.parent.parent / 'NBW' / 'rollback.json'
        nbw.parent.mkdir()
        nbw.write_text(json.dumps(data))
        write_marker({'runner': 'nbw2'}, cast(Json, chain.snapshot_ref(nbw)))
    else:
        chain.clear()
        explicit, down = rollback, False
    live, fake, labels = rig('W2', down=down, identity=explicit, fail=fail)
    marker_before = chain.MARKER.exists()
    result = run(live)
    assert message in str(result.get('stop')), result.get('stop')
    assert 'hold-open' not in labels and fake.restores == []
    # S92 FIX 1: with the marker present production is down by the chain, so a pre-HOLD stop runs the chain abort.
    assert (result['end_action'] == 'abort') == marker_before and (live.out / 'chain-abort').is_dir() == marker_before
    expected = chain.ABORT_RESTORED if marker_before else 'NOT NEEDED: preflight only'
    assert status(result) == expected


def test_explicit_identity_admits_a_marker_without_a_full_capture(rig: Rig, monkeypatch: pytest.MonkeyPatch) -> None:
    first = first_window(rig, monkeypatch)
    write_marker({'runner': 'nbw2'})
    live, fake, _ = rig('W2', down=True, identity=first.out / 'rollback.json')
    result = run(live)
    assert 'stop' not in result, result.get('stop')
    assert live.chain_mode == 'chained' and status(result).startswith('CHAINED')
    assert json.loads((live.out / 'rollback.json').read_text())['chained_from'] == str(first.out / 'rollback.json')


def test_chained_windows_lift_the_hold_ceilings(rig: Rig, monkeypatch: pytest.MonkeyPatch) -> None:
    # Coordinator 2026-10-27 (R-UPX): production is down for the whole chain; a chained window may submit until
    # HOLD+120 and must restore by HOLD+132 (same 12-min recovery reserve). Unchained windows keep 73/73/85.
    assert w.hold_bounds() == (w.SUBMIT_S, w.ABORT_S, w.RESTORE_S) == (73 * 60, 73 * 60, 85 * 60)
    live = first_window(rig, monkeypatch)
    assert w.hold_bounds() == (120 * 60, 120 * 60, 132 * 60) and live.bounds == w.hold_bounds()
    deadlines = json.loads((live.out / 'deadlines.json').read_text())
    assert deadlines['submit'] - deadlines['opened'] == 120 * 60 and deadlines['restore_by'] - deadlines['opened'] == 132 * 60
    monkeypatch.delenv('WINDOW_CHAIN')
    assert w.hold_bounds()[2] == 85 * 60 and w.budget_minutes('k5') <= 85
