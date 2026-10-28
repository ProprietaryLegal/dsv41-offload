"""WIN2 API: back-to-back (chain) mode helpers (CPU only; marker redirected to tmp)."""
from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pytest

from offload.launch.windows import chain
from offload.prefill_maxlen.evidence import record

OWNER = 'chain-20261027T220000Z-0123abcd'


@pytest.fixture(autouse=True)
def marker(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    path = tmp_path / 'CHAINED'
    monkeypatch.setattr(chain, 'MARKER', path)
    monkeypatch.delenv('WINDOW_CHAIN', raising=False)
    monkeypatch.delenv('WINDOW_CHAIN_LAST', raising=False)
    return path


def test_flags(monkeypatch: pytest.MonkeyPatch) -> None:
    assert not chain.enabled() and not chain.last() and not chain.skip_restore(False)
    monkeypatch.setenv('WINDOW_CHAIN', '1')
    assert chain.enabled() and chain.skip_restore(False) and not chain.skip_restore(True)
    monkeypatch.setenv('WINDOW_CHAIN_LAST', '1')
    assert chain.last() and not chain.skip_restore(False)


def test_finish_and_chained_preflight(monkeypatch: pytest.MonkeyPatch, marker: Path, tmp_path: Path) -> None:
    monkeypatch.setenv('WINDOW_CHAIN', '1')
    chain.finish_chained(tmp_path / 'W1', {'sparkhead': 'QUIET', 'sparkworker-wired': 'QUIET'}, placeholder_owner=OWNER)
    data = chain.read_marker()
    assert data is not None and data['window'] == str(tmp_path / 'W1') and marker.exists()
    assert chain.preflight(production_serving=False, hold_exists=False, placeholders_present=True) == 'chained'
    with pytest.raises(ValueError, match='serving'):
        chain.preflight(production_serving=True, hold_exists=False, placeholders_present=True)
    with pytest.raises(ValueError, match='HOLD'):
        chain.preflight(production_serving=False, hold_exists=True, placeholders_present=True)
    with pytest.raises(ValueError, match='end_chain'):
        chain.clear()                                   # the marker records placeholders: only end_chain() clears it
    chain.clear(placeholders_released=True)
    assert not marker.exists()
    assert chain.preflight(production_serving=True, hold_exists=False) == 'production'


def test_marker_without_chain_mode_refuses(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv('WINDOW_CHAIN', '1')
    chain.finish_chained(tmp_path / 'W1', {'sparkhead': 'QUIET'}, placeholder_owner=OWNER)
    monkeypatch.delenv('WINDOW_CHAIN')
    with pytest.raises(ValueError, match='WINDOW_CHAIN=1'):
        chain.preflight(production_serving=False, hold_exists=False)


def test_production_down_without_marker_refuses() -> None:
    with pytest.raises(ValueError, match='no chain marker'):
        chain.preflight(production_serving=False, hold_exists=False)


@pytest.mark.parametrize('env', [{}, {'WINDOW_CHAIN': '1', 'WINDOW_CHAIN_LAST': '1'}])
def test_finish_refused_outside_non_last_chain(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, env: dict[str, str]) -> None:
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    with pytest.raises(ValueError, match='non-last'):
        chain.finish_chained(tmp_path / 'W', {'sparkhead': 'QUIET'}, placeholder_owner=OWNER)


def test_finish_requires_quiet_proof_and_schema_guard(monkeypatch: pytest.MonkeyPatch, marker: Path, tmp_path: Path) -> None:
    monkeypatch.setenv('WINDOW_CHAIN', '1')
    with pytest.raises(ValueError, match='quiet proof'):
        chain.finish_chained(tmp_path / 'W', {}, placeholder_owner=OWNER)
    marker.write_text('{"schema": "other"}\n')
    with pytest.raises(ValueError, match='schema'):
        chain.read_marker()


def test_gate_uses_injected_probes(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    assert chain.gate(lambda: True, lambda: False) == 'production'
    monkeypatch.setenv('WINDOW_CHAIN', '1')
    chain.finish_chained(tmp_path / 'W1', {'sparkhead': 'QUIET'}, placeholder_owner=OWNER)
    assert chain.gate(lambda: False, lambda: False, lambda: True) == 'chained'
    with pytest.raises(ValueError, match='HOLD'):
        chain.gate(lambda: False, lambda: True, lambda: True)
    with pytest.raises(ValueError, match='placeholders missing'):
        chain.gate(lambda: False, lambda: False, lambda: False)


def test_production_probe_refuses_unreachable() -> None:
    assert chain.production_serving('http://127.0.0.1:9/v1/models', timeout=0.5) is False


def test_cli_status_and_clear(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.setenv('WINDOW_CHAIN', '1')
    chain.finish_chained(tmp_path / 'W1', {'sparkhead': 'QUIET'}, placeholder_owner=OWNER)
    assert chain.main(['status']) == 0 and '"enabled": true' in capsys.readouterr().out
    assert chain.main(['clear']) == 2 and chain.read_marker() is not None      # placeholders recorded: refused
    assert chain.main(['clear', '--placeholders-released']) == 0 and chain.read_marker() is None



# ---------------------------------------------------------------- WIN2-R2: hold placeholders kept across chained windows

def test_chained_preflight_requires_placeholders(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv('WINDOW_CHAIN', '1')
    chain.finish_chained(tmp_path / 'W1', {'sparkhead': 'QUIET'}, placeholder_owner=OWNER)
    marker = record(chain.read_marker() or {})
    assert marker['placeholder_owner'] == OWNER and 'only end_chain() takes them down' in str(marker['placeholders'])
    with pytest.raises(ValueError, match='must prove the hold placeholders'):
        chain.preflight(production_serving=False, hold_exists=False)          # older two-argument call refuses
    with pytest.raises(ValueError, match='placeholders missing'):
        chain.preflight(production_serving=False, hold_exists=False, placeholders_present=False)
    chain.clear(placeholders_released=True)
    probed: list[int] = []

    def probe() -> bool:
        probed.append(1)
        return True
    assert chain.gate(lambda: True, lambda: False, probe) == 'production'
    assert probed == []                                         # no marker: the placeholders are not probed


def _fake_ssh(tmp_path: Path, rc: int, state: str = f'UP owner={OWNER} pid=4242') -> Path:
    """ssh fake: logs the host to ssh.log and the command to ssh.cmds (NUL-separated); a placeholder step (it calls the
    absolute tmux) answers `state`."""
    bin_dir = tmp_path / 'bin'
    bin_dir.mkdir(exist_ok=True)
    ssh = bin_dir / 'ssh'
    ssh.write_text(f'#!/bin/bash\nprintf "%s\\n" "${{@: -2:1}}" >> {tmp_path}/ssh.log\n'
                   f'printf "%s\\0" "${{@: -1}}" >> {tmp_path}/ssh.cmds\n'
                   f'case "$*" in *"{chain.PLACEHOLDER_TMUX}"*) [ {rc} = 0 ] && echo "PLACEHOLDER {state}" ;; esac\n'
                   f'exit {rc}\n')
    ssh.chmod(0o755)
    return bin_dir


@pytest.mark.parametrize('rc,state,present', [(0, f'UP owner={OWNER} pid=4242', True), (0, 'ABSENT', False),
                                              (0, 'FOREIGN owner=none pid=17', False), (1, '', False), (255, '', False)])
def test_placeholder_probe_needs_this_chains_placeholder_on_both_sparks(monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
                                                                        rc: int, state: str, present: bool) -> None:
    monkeypatch.setenv('PATH', f'{_fake_ssh(tmp_path, rc, state)}:{os.environ["PATH"]}')
    assert chain.placeholders_present(OWNER) is present
    assert (tmp_path / 'ssh.log').read_text().splitlines() == list(chain.PLACEHOLDER_HOSTS)
    commands = (tmp_path / 'ssh.cmds').read_text().split('\0')[:-1]
    assert len(commands) == 2 and all(chain.PLACEHOLDER_TMUX in c and f'owner={OWNER}' in c and 'new-session' not in c
                                      for c in commands)


def test_placeholder_probe_owner_comes_from_the_marker(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv('PATH', f'{_fake_ssh(tmp_path, 0)}:{os.environ["PATH"]}')
    assert chain.placeholders_present() is False and not (tmp_path / 'ssh.log').exists()     # no marker, no owner
    monkeypatch.setenv('WINDOW_CHAIN', '1')
    chain.finish_chained(tmp_path / 'W1', {'s': 'QUIET'}, placeholder_owner=OWNER)
    assert chain.chain_owner() == OWNER and chain.placeholders_present() is True
    with pytest.raises(ValueError, match='differs'):
        chain.finish_chained(tmp_path / 'W2', {'s': 'QUIET'}, placeholder_owner='chain-other')


def test_marker_without_placeholder_owner_stops_the_chain(monkeypatch: pytest.MonkeyPatch, marker: Path) -> None:
    """A marker written before the WIN2-R3 fix has no owner: its placeholders cannot be proved, so the gate stops."""
    monkeypatch.setenv('WINDOW_CHAIN', '1')
    marker.write_text(json.dumps({'schema': chain.SCHEMA, 'window': 'W0', 'production_snapshot': None}) + '\n')
    with pytest.raises(ValueError, match='no placeholder owner'):
        chain.gate(lambda: False, lambda: False, lambda: True)
    with pytest.raises(ValueError, match='no placeholder owner'):
        chain.chain_owner()
    assert chain.placeholders_present() is False
    chain.clear()                                                # no owner recorded: a plain clear is allowed


def test_new_owner_and_invalid_owner() -> None:
    a, b = chain.new_owner(), chain.new_owner()
    assert a != b and all(chain.check_owner(o) == o for o in (a, b))
    for bad in ('', 'x' * 65, 'a b', "a'b", 'a;rm -rf ~'):
        with pytest.raises(ValueError, match='invalid chain placeholder owner'):
            chain.placeholder_script('probe', bad)
    with pytest.raises(ValueError, match='unknown placeholder action'):
        chain.placeholder_script('kill', OWNER)


def test_placeholder_scripts_use_the_default_socket_and_carry_only_their_own_branch() -> None:
    probe, up, down = (chain.placeholder_script(a, OWNER) for a in ('probe', 'up', 'down'))
    for script in (probe, up, down):
        assert chain.PLACEHOLDER_TMUX in script and f'owner={OWNER}' in script
        # every tmux call goes through T=(/usr/bin/tmux -L default); a bare `tmux` would hit a scope's shim
        assert ' tmux ' not in script.replace(chain.PLACEHOLDER_TMUX, '')
    assert 'new-session' not in probe and 'kill-session' not in probe and 'dsv41w-' not in probe
    assert 'new-session' in up and 'kill-session' not in up and 'dsv41w-' in up
    assert 'kill-session' in down and 'new-session' not in down and 'dsv41w-' in down


@pytest.mark.parametrize('ssh_rc,state', [(0, f'UP owner={OWNER} pid=4242'), (0, 'FOREIGN owner=none pid=9'),
                                          (0, 'ABSENT'), (255, '')])
def test_chained_close_proves_this_chains_placeholders_and_removes_only_hold(tmp_path: Path, ssh_rc: int,
                                                                            state: str) -> None:
    """The shared close-own-HOLD-without-production script: never touches a placeholder; refuses unless both are ours."""
    close = chain.chained_close(OWNER)
    assert 'kill-session' not in close and 'new-session' not in close and '8000' not in close
    deploy = tmp_path / 'dsv41-deploy'
    deploy.mkdir()
    (deploy / 'dsv41.env').write_text('HEAD=spark\nWORKER=sparkworker\n')
    (deploy / 'HOLD').write_text('2026-10-27T20:00:00Z DSV41-OFFLOAD (session coordinator): test\n')
    env = {**os.environ, 'HOME': str(tmp_path), 'PATH': f'{_fake_ssh(tmp_path, ssh_rc, state)}:{os.environ["PATH"]}'}
    r = subprocess.run(['bash', '-c', close], capture_output=True, text=True, env=env, timeout=30)
    calls = (tmp_path / 'ssh.log').read_text().splitlines()
    if ssh_rc == 0 and state.startswith('UP '):
        assert r.returncode == 0 and chain.CHAINED_CLOSE_OK in r.stdout.splitlines() and not (deploy / 'HOLD').exists()
        assert calls == ['spark', 'sparkworker']
    else:
        assert r.returncode != 0 and chain.CHAINED_CLOSE_OK not in r.stdout and (deploy / 'HOLD').exists()
        assert ('missing on spark' if ssh_rc == 0 else 'probe failed on spark') in r.stdout


class Down:
    """Fake transport for end_chain: answers every `down` with the given per-host state."""

    def __init__(self, states: dict[str, str]) -> None:
        self.states = states
        self.calls: list[tuple[str, str]] = []

    def __call__(self, host: str, command: str) -> tuple[int, str]:
        self.calls.append((host, command))
        return 0, 'PLACEHOLDER ' + self.states[host] + '\n'


@pytest.mark.parametrize('states,ok', [({'sparkhead': 'DOWN', 'sparkworker-wired': 'DOWN'}, True),
                                       ({'sparkhead': 'DOWN', 'sparkworker-wired': 'ABSENT'}, True),
                                       ({'sparkhead': 'DOWN', 'sparkworker-wired': 'FOREIGN owner=none pid=9'}, False)])
def test_end_chain_takes_placeholders_down_then_clears(monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
                                                       states: dict[str, str], ok: bool) -> None:
    monkeypatch.setenv('WINDOW_CHAIN', '1')
    chain.finish_chained(tmp_path / 'W1', {'s': 'QUIET'}, placeholder_owner=OWNER)
    down = Down(states)
    result = chain.end_chain(run=down)
    assert [h for h, _ in down.calls] == list(chain.PLACEHOLDER_HOSTS)
    assert all('kill-session' in c and f'owner={OWNER}' in c for _, c in down.calls)
    assert result['placeholder_owner'] == OWNER and result['marker'] == 'cleared' and chain.read_marker() is None
    assert str(result['status']).startswith('ENDED:' if ok else 'ENDED WITH ALERT')


def test_end_chain_with_an_explicit_owner_and_without_any(tmp_path: Path) -> None:
    """A window whose chained end failed before finish_chained passes its own owner; with none there is nothing to do."""
    down = Down({'sparkhead': 'DOWN', 'sparkworker-wired': 'DOWN'})
    assert chain.end_chain('chain-own-start', run=down)['status'] == 'ENDED: chain placeholders down on both Sparks'
    assert all('owner=chain-own-start' in c for _, c in down.calls)
    nothing = Down({})
    assert chain.end_chain(run=nothing)['status'] == 'ENDED: no chain placeholders to remove' and nothing.calls == []


def test_placeholders_up_raises_unless_both_up() -> None:
    assert chain.placeholders_up(OWNER, run=Down({'sparkhead': f'UP owner={OWNER} pid=1',
                                                  'sparkworker-wired': f'UP owner={OWNER} pid=2'}))['sparkworker-wired'].startswith('UP')
    for bad in ('FOREIGN owner=chain-x pid=3', 'REFUSED: up inside a scoped_remote window scope', 'BROKEN pane runs: sh'):
        with pytest.raises(ValueError, match='not up on both Sparks'):
            chain.placeholders_up(OWNER, run=Down({'sparkhead': f'UP owner={OWNER} pid=1', 'sparkworker-wired': bad}))


# ---------------------------------------------------------------- WIN2-R2: production snapshot carried by the marker

def _rollback(window: Path, commit: str = '5' * 40) -> Path:
    window.mkdir(parents=True, exist_ok=True)
    path = window / 'rollback.json'
    path.write_text(json.dumps({'production_commit': commit, 'ranks': [{'id': 'a'}, {'id': 'b'}]}) + '\n')
    return path


def test_marker_carries_and_inherits_the_production_snapshot(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv('WINDOW_CHAIN', '1')
    first = _rollback(tmp_path / 'PC1-W-1')                     # started from serving production: its own capture
    chain.finish_chained(tmp_path / 'PC1-W-1', {'s': 'QUIET'}, placeholder_owner=OWNER)
    path, data = chain.production_snapshot()
    assert path == first and data['production_commit'] == '5' * 40
    (tmp_path / 'NBW2-W-2').mkdir()                              # chained window: nothing captured, inherits
    chain.finish_chained(tmp_path / 'NBW2-W-2', {'s': 'QUIET'}, placeholder_owner=OWNER)
    marker = record(chain.read_marker() or {})
    assert marker['window'] == str(tmp_path / 'NBW2-W-2') and record(marker['production_snapshot'])['path'] == str(first)
    assert chain.production_snapshot()[0] == first
    first.write_text(first.read_text().replace('5', '6'))        # the capture changed after it was recorded
    with pytest.raises(ValueError, match='changed'):
        chain.production_snapshot()
    with pytest.raises(ValueError, match='changed'):
        chain.finish_chained(tmp_path / 'UP3-W-3', {'s': 'QUIET'}, placeholder_owner=OWNER)


def test_snapshot_explicit_missing_and_invalid(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv('WINDOW_CHAIN', '1')
    chain.finish_chained(tmp_path / 'W-none', {'s': 'QUIET'}, placeholder_owner=OWNER)   # no capture anywhere: recorded null, refused on use
    with pytest.raises(ValueError, match='no production snapshot'):
        chain.production_snapshot()
    chain.clear(placeholders_released=True)
    explicit = _rollback(tmp_path / 'elsewhere')
    chain.finish_chained(tmp_path / 'W-x', {'s': 'QUIET'}, placeholder_owner=OWNER, production_snapshot=explicit)
    assert chain.production_snapshot()[0] == explicit
    with pytest.raises(ValueError, match='40-hex'):
        chain.snapshot_ref(_rollback(tmp_path / 'short', '596ba1c'))
    with pytest.raises(ValueError, match='absolute'):
        chain.snapshot_ref(Path('rollback.json'))
