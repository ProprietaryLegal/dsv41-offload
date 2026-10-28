"""WIN2 chain mode in the UP3 runner (NDM 2026-10-27 "go window after window"): group-agnostic, CPU fakes only."""
from __future__ import annotations

import json
import os
from collections.abc import Callable, Iterator
from pathlib import Path

import pytest

from offload.launch.windows import chain, pf1_window as pf, up3_window as w, nbw_cells as greedy_driver
from offload.launch.windows import post_close_watch as watchdog
from offload.launch.windows.tests import test_pf1_window as base
from offload.prefill_maxlen.evidence import Json, record
from offload.up3.tests.conftest import AbortRig
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


def watched(result: dict[str, Json], watch: str = 'PASS') -> None:
    """main(): the post-close watch result, then close_chain (the marker clears only on own RESTORED + watch PASS)."""
    result['post_close_watch'] = {'status': watch}
    w.close_chain(result)


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
    assert result['status'] == 'COMPLETE' and record(result['decision'])['status'] == 'CANDIDATE FOR REVIEW'
    assert status(result).startswith('CHAINED: production left down'), result['restore']
    assert result['chain'] == 'CHAINED'  # chain_prev_ok.sh gate field
    assert live.chain_mode == 'production' and 'prod-models' in labels
    assert fake.restores == [] and not fake.hold  # no W8 start/check/close; own HOLD removed
    # WIN2-R2: chain.CHAINED_CLOSE (placeholders proved up, only the HOLD file) records the unlink in a numbered close
    # transaction of this window's HOLD; the fenced commit-close removes it.
    order = [fake.commands.index(c) for c in ('experiment-stop', 'jit-wait-sparkworker-wired', 'chained-close', 'commit-close')]
    assert order == sorted(order) and 'cancel-close' in fake.commands and 'retain-close' not in fake.commands
    assert result['end_action'] == 'chained' and not [x for x in labels if x.startswith('chain-placeholders-')]
    marker = chain.read_marker()
    assert marker is not None
    proof = record(marker['quiet_proof'])
    rollback = live.out / 'rollback.json'
    assert proof['production_identity'] == {'path': str(rollback), 'sha256': pf.digest(rollback)}
    assert record(proof['sparks']) == {host: 'UP3 EXPERIMENT ABSENT' for host in pf.HOSTS}
    assert record(marker['production_snapshot'])['path'] == str(rollback) and marker['placeholders'] == chain.PLACEHOLDERS_KEPT


def test_chained_window_runs_from_the_marker(rig: Rig, monkeypatch: pytest.MonkeyPatch) -> None:
    first = first_window(rig, monkeypatch)
    live, fake, labels = rig('W2', down=True)
    result = run(live)
    assert 'stop' not in result, result.get('stop')
    assert result['status'] == 'COMPLETE' and status(result).startswith('CHAINED')
    assert live.chain_mode == 'chained' and fake.restores == []
    # Production is down: no production health/model/identity/idle probe, but both Sparks proved quiet first.
    assert not [x for x in labels if x.startswith(('prod-health', 'prod-models', 'prod-identity', 'prod-idle', 'prod-fullenv'))]
    assert labels.index('chain-quiet-sparkworker-wired') < labels.index('hold-open')
    # WIN2-R2: the placeholders are proved on both Sparks before the chained preflight admits.
    assert labels.index('chain-placeholders-sparkhead') < labels.index('chain-placeholders-sparkworker-wired') < labels.index('hold-open')
    rollback = json.loads((live.out / 'rollback.json').read_text())
    assert rollback['chained_from'] == str(first.out / 'rollback.json')
    assert rollback['ranks'] == json.loads((first.out / 'rollback.json').read_text())['ranks']
    marker = chain.read_marker()
    assert marker is not None
    assert record(record(marker['quiet_proof'])['production_identity'])['path'] == str(live.out / 'rollback.json')
    # The chain's production snapshot is inherited from the window that started from serving production.
    assert record(marker['production_snapshot'])['path'] == str(first.out / 'rollback.json')


def test_last_window_restores_strictly_and_clears(rig: Rig, monkeypatch: pytest.MonkeyPatch) -> None:
    first_window(rig, monkeypatch)
    monkeypatch.setenv('WINDOW_CHAIN_LAST', '1')
    live, fake, _ = rig('W2', down=True)
    result = run(live)
    assert 'stop' not in result, result.get('stop')
    assert status(result).startswith('RESTORED') and fake.restores == ['start', 'check', 'close']
    assert result['end_action'] == 'restore' and chain.MARKER.exists()  # cleared only after the post-close watch
    watched(result, 'FAILED')
    assert chain.MARKER.exists()
    watched(result)
    assert not chain.MARKER.exists()


@pytest.mark.parametrize('fail', ['start', 'STOP'])
def test_any_stop_in_a_chain_restores_strictly(rig: Rig, monkeypatch: pytest.MonkeyPatch, fail: str) -> None:
    first_window(rig, monkeypatch)
    live, fake, _ = rig('W2', fail=fail, down=True)
    result = run(live)
    assert 'stop' in result and record(result['decision'])['status'] == 'NOT A CANDIDATE'
    assert status(result).startswith('RESTORED') and fake.restores == ['start', 'check', 'close']
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


def test_failed_chained_close_falls_back_to_strict_restore(rig: Rig, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv('WINDOW_CHAIN', '1')
    live, fake, _ = rig('W1', fail='chained-close')
    result = run(live)
    assert 'chained-close' in fake.commands and 'chained_end_error' in result
    assert status(result).startswith('RESTORED') and fake.restores == ['start', 'check', 'close']
    assert not chain.MARKER.exists() and result['status'] == 'COMPLETE' and result['end_action'] == 'restore'
    # The fallback W8 close runs in the window's own unfenced -restore-close scope (the chained close used close txn 1).
    assert fake.commands.count('commit-close') == 1 and not fake.hold


def write_marker(proof: dict[str, Json]) -> None:
    chain.MARKER.parent.mkdir(parents=True, exist_ok=True)
    chain.MARKER.write_text(json.dumps({'schema': chain.SCHEMA, 'window': 'x', 'time': 0, 'utc': 'x', 'quiet_proof': proof}))


@pytest.mark.parametrize('case,message', [
    ('marker-without-chain', 'run with WINDOW_CHAIN=1'),
    ('down-without-marker', 'production not serving and no chain marker'),
    ('marker-but-serving', 'production is serving'),
    ('foreign-marker', 'lacks a UP3 production_identity'),
    ('changed-identity', 'changed since the marker'),
    ('filtered-env', 'env is filtered'),
    ('identity-in-production-mode', 'for chained windows only'),
])
def test_chain_preflight_refuses_before_hold(rig: Rig, monkeypatch: pytest.MonkeyPatch, chain_abort: AbortRig,
                                            case: str, message: str) -> None:
    first = first_window(rig, monkeypatch)
    rollback = first.out / 'rollback.json'
    explicit: Path | None = None
    down = True
    if case == 'marker-without-chain':
        monkeypatch.delenv('WINDOW_CHAIN')
    elif case == 'down-without-marker':
        chain.clear()
    elif case == 'marker-but-serving':
        down = False
    elif case == 'foreign-marker':
        write_marker({'runner': 'nbw2', 'nbw2_production_snapshot': str(rollback)})
    elif case == 'changed-identity':
        rollback.write_text(rollback.read_text() + ' ')
    elif case == 'filtered-env':
        data = json.loads(rollback.read_text())
        for rank in data['ranks']:
            rank['env'] = {k: v for k, v in rank['env'].items() if k.startswith('VLLM_')}
        rollback.write_text(json.dumps(data))
        explicit = rollback
    else:
        chain.clear()
        explicit, down = rollback, False
    live, fake, labels = rig('W2', down=down, identity=explicit)
    marker = chain.marker_present()
    result = run(live)
    assert message in str(result.get('stop')), result.get('stop')
    assert 'hold-open' not in labels and fake.restores == []
    if marker:
        # WIN2-R2 (S92 FIX 1): production is down from the previous chained window, so a refusal before HOLD aborts the
        # chain (strict W8 under the abort's own scoped HOLD + watch), never "NOT NEEDED".
        assert result['end_action'] == 'abort' and chain_abort.calls == [live.out / 'chain-abort']
        assert status(result).startswith('RESTORED: chain ended by abort') and not chain.marker_present()
        assert record(result['post_close_watch'])['status'] == 'PASS'
    else:
        assert result['end_action'] == 'none' and status(result) == 'NOT NEEDED: preflight only' and not chain_abort.calls


def test_explicit_identity_admits_a_foreign_marker(rig: Rig, monkeypatch: pytest.MonkeyPatch) -> None:
    first = first_window(rig, monkeypatch)
    write_marker({'runner': 'nbw2', 'nbw2_production_snapshot': 'elsewhere'})
    live, fake, _ = rig('W2', down=True, identity=first.out / 'rollback.json')
    result = run(live)
    assert 'stop' not in result, result.get('stop')
    assert live.chain_mode == 'chained' and status(result).startswith('CHAINED')
    assert json.loads((live.out / 'rollback.json').read_text())['chained_from'] == str(first.out / 'rollback.json')


def test_chained_start_refuses_without_the_hold_placeholders(rig: Rig, monkeypatch: pytest.MonkeyPatch,
                                                            chain_abort: AbortRig) -> None:
    first_window(rig, monkeypatch)
    live, fake, labels = rig('W2', down=True)
    fake.placeholders = False
    result = run(live)
    assert 'hold placeholders missing between chained windows' in str(result['stop'])
    assert 'hold-open' not in labels and result['end_action'] == 'abort' and chain_abort.calls == [live.out / 'chain-abort']


def test_pre_hold_failure_in_a_chain_takes_the_abort(rig: Rig, monkeypatch: pytest.MonkeyPatch, chain_abort: AbortRig) -> None:
    first_window(rig, monkeypatch)
    live, fake, labels = rig('W2', down=True, fail='hold-open-refused')
    result = run(live)
    assert 'admission refused' in str(result['stop']) and 'hold-open' in labels and not fake.hold
    # The window's own restore fenced every phase, re-read HOLD, found none; the chain marker makes it an abort.
    assert str(record(result['own_restore'])['status']).startswith('NOT NEEDED: no owned HOLD')
    assert result['end_action'] == 'abort' and chain_abort.calls == [live.out / 'chain-abort'] and fake.restores == []
    assert status(result).startswith('RESTORED: chain ended by abort') and not chain.marker_present()
    summary = json.loads((live.out / 'chain-abort' / 'summary.json').read_text())
    assert summary['chain'].startswith('ABORTED') and record(result['post_close_watch'])['status'] == 'PASS'


def test_abort_with_a_hold_present_only_alerts(rig: Rig, monkeypatch: pytest.MonkeyPatch, chain_abort: AbortRig) -> None:
    first_window(rig, monkeypatch)
    live, fake, _ = rig('W2', down=True, fail='hold-open-refused')
    chain_abort.hold = True
    result = run(live)
    assert result['end_action'] == 'abort' and status(result).startswith('ALERT: a HOLD exists')
    assert chain.marker_present() and record(result['post_close_watch'])['status'] == 'NOT RUN'


def test_abort_watch_failure_keeps_the_marker(rig: Rig, monkeypatch: pytest.MonkeyPatch, chain_abort: AbortRig) -> None:
    first_window(rig, monkeypatch)
    live, fake, _ = rig('W2', down=True, fail='hold-open-refused')
    chain_abort.watch = lambda out, log: {'status': 'FAILED'}
    result = run(live)
    assert status(result).startswith('RESTORED BUT POST-CLOSE WATCH FAILED') and chain.marker_present()
    assert record(result['post_close_watch'])['status'] == 'FAILED'
    w.close_chain(result)   # only the window's OWN restore + watch PASS clears; an abort keeps its own verdict
    assert chain.marker_present()
    # main() runs no second watch after an abort (chain.abort ran its own).
    monkeypatch.setattr(watchdog, 'watch', lambda *a, **k: pytest.fail('second post-close watch after a chain abort'))
    w.post_close(result, live.out, 'tag')
    assert record(result['post_close_watch'])['status'] == 'FAILED'
