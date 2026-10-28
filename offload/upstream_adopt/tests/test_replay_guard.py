from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from offload.upstream_adopt import replay_guard as guard
from offload.upstream_adopt.make_configs import CFG
from offload.upstream_adopt.build_overlay import ROOT

PAIR = guard.SOURCE + '=' + guard.DESTINATION


def test_exact_scheduler_and_manifest_pass() -> None:
    guard.verify(PAIR)
    guard.verify(PAIR, '-v $HOME/dsv41-offload-run/ofl:/opt/dsv41-ofl:ro')


@pytest.mark.parametrize('overlays', [
    '', 'offload/so2/overlay/scheduler.py=' + guard.DESTINATION,
    PAIR + ' ' + PAIR,
    PAIR + ' other=/usr/local/lib/python3.12/dist-packages/vllm',
    'malformed',
])
def test_missing_replaced_duplicate_and_shadowed_mounts_refuse(overlays: str) -> None:
    with pytest.raises(ValueError, match='UP2 replay scheduler guard'):
        guard.verify(overlays)


@pytest.mark.parametrize('extra', ['-v other:/scheduler', '--volume=other:/scheduler',
                                   '--mount type=bind,src=other,dst=/scheduler'])
def test_raw_docker_mounts_refuse(extra: str) -> None:
    with pytest.raises(ValueError, match='raw Docker mounts'):
        guard.verify(PAIR, extra)


def test_scheduler_tampering_refuses(tmp_path: Path) -> None:
    source = tmp_path/guard.SOURCE
    source.parent.mkdir(parents=True)
    source.write_text('# image or SO2 scheduler\n')
    (tmp_path/'offload/upstream_adopt/MANIFEST.json').write_text(json.dumps({
        'files': [{'path': guard.SCHEDULER, 'sha256': '0' * 64}]}))
    with pytest.raises(ValueError, match='hash differs'):
        guard.verify(PAIR, root=tmp_path)


@pytest.mark.parametrize('replay,overlays,ok', [('0', '', True), ('1', PAIR, True),
                                               ('1', '', False), ('1', 'so2=' + guard.DESTINATION, False)])
def test_actual_startup_preflight_before_remote_actions(replay: str, overlays: str, ok: bool) -> None:
    launcher = (ROOT/'offload/launch/start_exp.sh').read_text()
    # Execute just the actual preflight, no HOLD changes or remote commands.
    start = launcher.index('# Fail before any SSH')
    end = launcher.index('SERVE_DIR=')
    assert start < end < launcher.index('for n in $HEAD $WORKER') < launcher.index('./stop.sh\n')
    environment = dict(os.environ, EXP_DOCKER_ENV='VLLM_DSV41_REPLAY_SKIP=' + replay,
                       OVERLAYS=overlays, EXTRA_DOCKER_ARGS='')
    # Use this CPU interpreter for the launcher's python3 via a shell function.
    script = 'set -eu\npython3() { "$UP2_TEST_PYTHON" "$@"; }\n' + launcher[start:end]
    environment['UP2_TEST_PYTHON'] = sys.executable
    result = subprocess.run(['bash', '-c', script], env=environment, text=True, capture_output=True)
    assert (result.returncode == 0) == ok, result.stderr
    if not ok:
        assert 'UP2 replay scheduler guard' in result.stderr


@pytest.mark.parametrize('replay', ['0', '1'])
def test_up2_wrapper_passes_preflight(replay: str) -> None:
    environment = dict(os.environ, UP2_REPLAY=replay)
    script = 'set -eu; source "$1"; "$2" -m offload.upstream_adopt.replay_guard'
    # Config variables need export just as start_exp.sh's set -a does.
    exported = subprocess.run(['bash', '-ac', script, 'guard-test', str(CFG/'UP2-0075.env'), sys.executable],
                              env=environment, text=True, capture_output=True)
    assert exported.returncode == 0, exported.stderr


def up3_replay_config(tmp_path: Path) -> tuple[str, str]:
    """OVERLAYS / EXTRA_DOCKER_ARGS exactly as offload/up3/window.env builds them on current production."""
    from offload.up3 import build as b
    prod = tmp_path / 'dsv41.env'
    prod.write_bytes(b.production_file('dsv41.env'))
    rows = subprocess.check_output(['bash', '-eu', '-c', 'UP3_ARM=replay UP3_PROD_ENV=$1; source offload/up3/window.env; '
                                    'printf "%s\\n" "$OVERLAYS" "$EXTRA_DOCKER_ARGS"', 'up3', str(prod)],
                                   cwd=ROOT, text=True).splitlines()
    return rows[0], rows[1]


def test_up3_replay_config_passes(tmp_path: Path) -> None:
    overlays, extra = up3_replay_config(tmp_path)
    guard.verify_up3(overlays, extra)
    # The UP2 (UP1-composition) guard never accepts the production-based arm.
    with pytest.raises(ValueError, match='UP2 replay scheduler guard'):
        guard.verify(overlays, extra)


def test_up3_replay_refusals(tmp_path: Path) -> None:
    overlays, extra = up3_replay_config(tmp_path)
    remote = '$HOME/dsv41-offload-prod/dspark/remote_dspark:' + guard.UP3_REMOTE + ':ro'
    for bad_extra in (extra + ' -v ' + remote, extra + ' -v other:' + guard.DESTINATION,
                      extra + ' -v other:/usr/local/lib/python3.12/dist-packages/vllm/v1:ro',
                      extra + ' -v other:' + guard.UP3_REMOTE + '/suffix_provenance.py:ro',
                      extra + ' -v other:/usr/local/lib/python3.12/dist-packages/vllm/models/deepseek_v41/replay_skip.py:ro',
                      extra + ' --volume=other:/x', extra + ' -vother:/x', extra + ' -v a:b:rw'):
        with pytest.raises(ValueError, match='UP3 replay guard'):
            guard.verify_up3(overlays, bad_extra)
    pairs = overlays.split()
    no_remote = ' '.join(p for p in pairs if not p.endswith('/remote_dspark'))
    up2_scheduler = ' '.join(guard.SOURCE + '=' + guard.DESTINATION if p.endswith('/scheduler.py') else p for p in pairs)
    for bad in (no_remote, up2_scheduler, overlays + ' ' + pairs[0], overlays + ' x=/usr/local/lib/python3.12/dist-packages/vllm',
                overlays + ' malformed'):
        with pytest.raises(ValueError, match='UP3 replay guard'):
            guard.verify_up3(bad, extra)


def test_up3_replay_scheduler_tampering_refuses(tmp_path: Path) -> None:
    overlays, extra = up3_replay_config(tmp_path)
    root = tmp_path / 'root'
    for name in ('offload/up3/manifests/replay.json', 'offload/upstream_adopt/MANIFEST.json', guard.UP3_SOURCE):
        (root / name).parent.mkdir(parents=True, exist_ok=True)
        (root / name).write_bytes((ROOT / name).read_bytes())
    guard.verify_up3(overlays, extra, root=root)
    (root / guard.UP3_SOURCE).write_text('# image scheduler\n')
    with pytest.raises(ValueError, match='hash differs'):
        guard.verify_up3(overlays, extra, root=root)
    (root / guard.UP3_SOURCE).write_bytes((ROOT / guard.UP3_SOURCE).read_bytes())
    manifest = json.loads((root / 'offload/up3/manifests/replay.json').read_text())
    manifest['files'][guard.SCHEDULER] = '0' * 64
    (root / 'offload/up3/manifests/replay.json').write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="not UP2's reviewed scheduler"):
        guard.verify_up3(overlays, extra, root=root)


@pytest.mark.parametrize('arm,ok', [('replay', True), ('control', False), ('', False)])
def test_start_exp_preflight_dispatches_on_up3_arm(tmp_path: Path, arm: str, ok: bool) -> None:
    overlays, extra = up3_replay_config(tmp_path)
    launcher = (ROOT/'offload/launch/start_exp.sh').read_text()
    start, end = launcher.index('# Fail before any SSH'), launcher.index('SERVE_DIR=')
    environment = dict(os.environ, EXP_DOCKER_ENV='VLLM_DSV41_REPLAY_SKIP=1', OVERLAYS=overlays, EXTRA_DOCKER_ARGS=extra,
                       UP2_TEST_PYTHON=sys.executable, UP3_ARM=arm)
    script = 'set -eu\npython3() { "$UP2_TEST_PYTHON" "$@"; }\n' + launcher[start:end]
    result = subprocess.run(['bash', '-c', script], env=environment, cwd=ROOT, text=True, capture_output=True)
    assert (result.returncode == 0) == ok, result.stderr


def production_0075(tmp_path: Path) -> tuple[str, str, Path]:
    """OVERLAYS='' and EXTRA_DOCKER_ARGS exactly as start_exp derives them from a switched (0075) production dsv41.env,
    plus a fake home holding the staged package's scheduler."""
    from offload.promote.up2 import switch
    from offload.up3 import build as b
    manifest = switch.package('qualified', ROOT / 'offload/promote/up2')
    env = tmp_path / 'dsv41.env'
    env.write_text(switch.edit_env(b.production_file('dsv41.env').decode(), 'ops1', manifest))
    extra = subprocess.check_output(['bash', '-c', 'set -a; . "$1" >/dev/null; printf %s "$OFFLOAD_DOCKER_ARGS"', '_',
                                     str(env)], text=True)
    home = tmp_path / 'home'
    staged = home / f"dsv41-offload-prod/up2/{manifest['id']}" / guard.SCHEDULER
    staged.parent.mkdir(parents=True)
    staged.write_bytes((ROOT / guard.UP3_SOURCE).read_bytes())
    return extra, str(manifest['id']), home


def test_production_0075_mode_accepts_exactly_the_staged_package(tmp_path: Path) -> None:
    extra, ident, home = production_0075(tmp_path)
    guard.verify_production('', extra, ident, home)
    for overlays in (guard.SOURCE + '=' + guard.DESTINATION, 'x=' + guard.UP3_REMOTE + '/round.py',
                     'x=/usr/local/lib/python3.12/dist-packages/vllm/v1/worker/gpu/input_batch.py',
                     'x=/usr/local/lib/python3.12/dist-packages/vllm', 'malformed'):
        with pytest.raises(ValueError, match='UP2 production guard'):
            guard.verify_production(overlays, extra, ident, home)
    guard.verify_production('offload/up3/overlay/split/vllm_exl3_moe/smoe.py='
                            '/usr/local/lib/python3.12/dist-packages/vllm_exl3_moe/smoe.py', extra, ident, home)
    for bad_extra in (extra + ' -v other:' + guard.DESTINATION, extra.replace(ident, '0' * 12),
                      extra + ' -v other:/usr/local/lib/python3.12/dist-packages/vllm/v1/core:ro', extra + ' --volume=a:b',
                      extra + ' -v other:' + guard.UP3_REMOTE + '/round.py:ro'):
        with pytest.raises(ValueError, match='UP2 production guard'):
            guard.verify_production('', bad_extra, ident, home)
    with pytest.raises(ValueError, match='invalid PROD_UP2'):
        guard.verify_production('', extra, 'x', home)
    (home / f'dsv41-offload-prod/up2/{ident}' / guard.SCHEDULER).write_text('# image scheduler\n')
    with pytest.raises(ValueError, match='differs from UP2 manifest'):
        guard.verify_production('', extra, ident, home)


@pytest.mark.parametrize('prod_up2,ok', [(True, True), (False, False)])
def test_start_exp_preflight_uses_production_mode_when_dsv41_env_selects_0075(tmp_path: Path, prod_up2: bool,
                                                                             ok: bool) -> None:
    extra, ident, home = production_0075(tmp_path)
    launcher = (ROOT/'offload/launch/start_exp.sh').read_text()
    start, end = launcher.index('# Fail before any SSH'), launcher.index('SERVE_DIR=')
    environment = dict(os.environ, EXP_DOCKER_ENV='VLLM_DSV41_REPLAY_SKIP=1', OVERLAYS='', EXTRA_DOCKER_ARGS=extra,
                       UP2_TEST_PYTHON=sys.executable, HOME=str(home), **({'PROD_UP2': ident} if prod_up2 else {}))
    environment.pop('UP3_ARM', None)
    script = 'set -eu\npython3() { "$UP2_TEST_PYTHON" "$@"; }\n' + launcher[start:end]
    result = subprocess.run(['bash', '-c', script], env=environment, cwd=ROOT, text=True, capture_output=True)
    assert (result.returncode == 0) == ok, result.stderr
