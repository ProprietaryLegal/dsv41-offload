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
