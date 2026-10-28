"""UP3 integration with the reviewed shared owner; no private ownership implementation."""
from __future__ import annotations

from pathlib import Path

import pytest

from offload.launch.windows import up3_window as w, scoped_remote as owner, post_close_watch as watchdog, production_health
from offload.prefill_maxlen.evidence import Json, record
from offload.up3.tests.test_up3_window import Remote, rehearsal as rehearsal


@pytest.mark.parametrize('mode', ['fence', 'reap', 'absent'])
@pytest.mark.parametrize('host', [*owner.NODES, 'localhost'])
def test_each_shared_cleanup_failure_keeps_hold(mode: str, host: str, rehearsal: tuple[w.Live, Remote]) -> None:
    live, fake = rehearsal
    fake.fail = f'owned-{mode}-tag-hold-{host}'
    live.open_hold()
    with watchdog.masked_signals(), pytest.raises(ValueError, match='HOLD stays open'):
        live.restore(production_stopped=False)
    assert fake.hold
    assert 'commit-close' not in fake.commands
    # All receiving hosts are attempted at a stage before moving to kills/proof.
    for node in (*owner.NODES, 'localhost'):
        assert f'owned-{mode}-tag-hold-{node}' in fake.commands


def test_scope_deadline_cannot_renew_and_no_private_registry(rehearsal: tuple[w.Live, Remote]) -> None:
    live, _ = rehearsal
    live.remember('tag-engine', 100)
    with pytest.raises(ValueError, match='immutable'):
        live.remember('tag-engine', 101)
    assert not (Path(w.__file__).parents[2] / 'up3/remote.py').exists()


@pytest.mark.parametrize('status', ['PASS', 'FAILED: recurrence'])
def test_shared_post_close_watch_gates_candidate(status: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    result: dict[str, Json] = {'restore': {'status': 'RESTORED'}, 'decision': {'status': 'CANDIDATE FOR REVIEW'}}
    seen: list[object] = []
    def watch(health: watchdog.Health, recovery: watchdog.Recovery, **kwargs: object) -> watchdog.Result:
        seen.extend((health, recovery))
        return watchdog.Result(status, 1, 0, 0)
    monkeypatch.setattr(watchdog, 'watch', watch)
    w.post_close(result, tmp_path, 'tag')
    assert isinstance(seen[0], production_health.ProductionHealth)
    assert isinstance(seen[1], watchdog.ScopedRecovery)
    assert record(result['post_close_watch'])['status'] == status
    assert record(result['decision'])['status'] == ('CANDIDATE FOR REVIEW' if status == 'PASS' else 'NOT A CANDIDATE')
    result['restore'] = {'status': 'FAILED'}
    seen.clear()
    w.post_close(result, tmp_path, 'tag')
    assert not seen
