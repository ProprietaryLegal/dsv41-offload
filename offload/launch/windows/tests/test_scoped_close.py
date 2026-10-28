"""Shared control-plane HOLD transactions, using private files and fake listings."""
from __future__ import annotations

from pathlib import Path
import time

import pytest

from offload.launch.windows import scoped_remote as s


def setup_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path, str]:
    monkeypatch.setenv('HOME', str(tmp_path))
    monkeypatch.setenv('SCOPED_REMOTE_DEADLINE', str(time.time() + 30))
    hold = tmp_path/'dsv41-deploy/HOLD'
    hold.parent.mkdir()
    tag = 'own1test-close-restore-close'
    hold.write_text('2026-10-26T23:55:00Z NBW-WINDOW own1test-close: original timestamp\n')
    assert s.main(['rm', tag, '-f', str(hold)]) == 0
    root = s.root_for(tag)
    assert hold.exists() and (root/'CLOSE_HOLD').read_text() == hold.read_text()
    return hold, root, tag


def test_defer_prove_commit_cancel_and_lost_reply_retention(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    hold, root, tag = setup_home(tmp_path, monkeypatch)
    original = hold.read_text()
    monkeypatch.setattr(s, 'units', lambda tag: [])
    with pytest.raises(ValueError, match='persistent fence'):
        s.main(['commit-close', tag])
    assert s.main(['fence', tag]) == 0
    assert s.main(['commit-close', tag]) == 0
    assert not hold.exists()
    assert s.main(['cancel-close', tag]) == 0
    assert s.main(['retain-close', tag]) == 0
    assert hold.read_text() == original
    with pytest.raises(ValueError, match='committed/cancelled'):
        s.main(['commit-close', tag])
    assert (root/'COMMITTED').exists()


@pytest.mark.parametrize('bad', ['changed', 'active', 'expired', 'other-run'])
def test_failed_close_preserves_hold(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, bad: str) -> None:
    hold, root, tag = setup_home(tmp_path, monkeypatch)
    monkeypatch.setattr(s, 'units', lambda tag: ['active.scope'] if bad == 'active' else [])
    assert s.main(['fence', tag]) == 0
    if bad == 'changed':
        hold.write_text('new HOLD')
    elif bad == 'expired':
        monkeypatch.setenv('SCOPED_REMOTE_DEADLINE', '1')
    elif bad == 'other-run':
        hold.write_text('NBW-WINDOW another-run: do not overwrite')
    with pytest.raises(ValueError):
        s.main(['commit-close', tag])
    assert hold.exists()
    if bad == 'other-run':
        with pytest.raises(ValueError, match='another HOLD'):
            s.main(['retain-close', tag])
        assert 'another-run' in hold.read_text()


@pytest.mark.parametrize('other', ['own1test-close-extra', 'other-own1test-close'])
def test_exact_run_marker_not_substring(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, other: str) -> None:
    monkeypatch.setenv('HOME', str(tmp_path))
    monkeypatch.setenv('SCOPED_REMOTE_DEADLINE', str(time.time()+30))
    hold = tmp_path/'dsv41-deploy/HOLD'
    hold.parent.mkdir()
    hold.write_text(f'NBW-WINDOW {other}: someone else')
    with pytest.raises(ValueError, match='another window'):
        s.main(['rm', 'own1test-close-restore-close', '-f', str(hold)])


def test_prefix_phase_units_do_not_cross_fences(monkeypatch: pytest.MonkeyPatch) -> None:
    exact = 'dsv41w-own1test-phase-' + 'a'*32 + '.scope'
    other = 'dsv41w-own1test-phase-engine-' + 'b'*32 + '.scope'
    monkeypatch.setattr(s, 'checked', lambda argv: f'{exact} loaded active running\n{other} loaded active running\n')
    assert s.units('own1test-phase') == [exact]


def test_failed_transport_still_fences_other_destinations() -> None:
    called: list[str] = []
    def invoke(host: str, command: str) -> tuple[int, str]:
        called.append(host)
        if host == 'sparkhead':
            raise OSError('lost transport')
        return 0, s.FENCED
    with pytest.raises(ValueError, match='lost transport'):
        s.fence_all('own1test-transports', invoke)
    assert called == list(s.NODES)


def test_absence_checks_transitive_cgroup_populated(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = tmp_path/'control'
    root.mkdir()
    (root/'ABORT').touch()
    cgroup = tmp_path/'cgroup'
    cgroup.mkdir()
    (cgroup/'cgroup.events').write_text('populated 1\nfrozen 0\n')
    monkeypatch.setattr(s, 'units', lambda tag: [])
    monkeypatch.setattr(s, 'paths', lambda root, tag: [cgroup])
    with pytest.raises(ValueError, match='cgroups remain'):
        s.prove(root, 'own1test-cgroup')
    (cgroup/'cgroup.events').write_text('populated 0\nfrozen 0\n')
    s.prove(root, 'own1test-cgroup')


def test_fenced_in_scope_first_action_exits_75_without_creating_helpers(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv('HOME', str(tmp_path))
    tag = 'own1test-late'
    root = s.root_for(tag)
    root.mkdir(parents=True)
    (root/'ABORT').touch()
    mutation = tmp_path/'mutation'
    assert s.main(['guard', tag, 'a'*32, 'touch', str(mutation)]) == 75
    assert not mutation.exists()
    assert list(root.iterdir()) == [root/'ABORT']


def test_guardian_error_log_never_yields_absence(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (tmp_path/'ABORT').touch()
    (tmp_path/'failed.watch.log').write_text('guardian failed loudly')
    monkeypatch.setattr(s, 'units', lambda tag: [])
    with pytest.raises(ValueError, match='guardian failed'):
        s.prove(tmp_path, 'own1test-guardian')
