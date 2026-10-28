"""CPU fixtures execute the shell config against git's frozen OPS1 production bytes."""
from __future__ import annotations

import dataclasses
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import cast

import pytest

from offload.launch.windows import pf1_window as pf
from offload.launch.windows import sp1_window as sp
from offload.prefill_maxlen.evidence import Json, record
from offload.tv1 import build_overlay as build
from offload.tv1 import production
from offload.tv1.window_config import FLAGS


@dataclasses.dataclass(frozen=True)
class Arm(pf.Arm):
    role: str = 'off'


def ops1_env() -> str:
    return subprocess.check_output(['git', 'show', production.PROD_COMMIT + ':dsv41.env'],
                                   cwd=build.ROOT, text=True)


def shell_config(root: Path, role: str, *, extra: str = '', source: str | None = None) -> subprocess.CompletedProcess[str]:
    prod = root / 'dsv41.env'
    prod.write_text(ops1_env() if source is None else source)
    env = {key: value for key, value in os.environ.items() if not key.startswith(('TV1_', 'PF1_', 'AK_'))}
    env.update(TV1_PROD_ENV=str(prod), TV1_ROLE=role, TV1_TREE='offload/tv1/generated-fixture/' + role,
               PATH=str(Path(sys.executable).parent) + os.pathsep + env['PATH'])
    keys = ('EXP_DOCKER_ENV', 'EXTRA_DOCKER_ARGS', 'OVERLAYS', 'EXP_EXTRA_ARGS',
            'MAX_LEN_OVERRIDE', 'KV_BYTES_OVERRIDE', 'LABEL', 'OFFLOAD_DOCKER_ENV',
            'OFFLOAD_DSPARK_ENV', 'OFFLOAD_DOCKER_ARGS')
    # Reproduce start_exp.sh's source order without executing any launch/SSH command.
    code = ('set -euo pipefail\nset -a\n. ' + shlex.quote(str(prod)) + '\nPORT=8001\n' + extra + '\n. ' +
            shlex.quote(str(build.ROOT / 'offload/tv1/window.env')) + '\n' +
            shlex.quote(sys.executable) + ' -c ' +
            shlex.quote('import os,json; print(json.dumps({k:os.environ.get(k) for k in ' + repr(keys) + '}))'))
    return subprocess.run(['bash', '-c', code], cwd=root, env=env, capture_output=True, text=True, check=False)


def make_overlay(root: Path, role: str) -> build.Manifest:
    tree = root / 'offload/tv1/generated-fixture' / role
    tree.parent.mkdir(parents=True, exist_ok=True)
    result = build.build(tree, () if role == 'off' else (role,))
    tree.with_suffix('.manifest.json').write_text(json.dumps(result))
    return result


def parse_env(value: str) -> dict[str, str]:
    return dict(token.split('=', 1) for token in shlex.split(value))


@pytest.mark.parametrize('role', tuple(FLAGS))
def test_actual_ops1_shell_config_preserves_environment_mounts_and_profiler(tmp_path: Path, role: str) -> None:
    manifest = make_overlay(tmp_path, role)
    completed = shell_config(tmp_path, role)
    assert completed.returncode == 0, completed.stderr
    result = cast(dict[str, str], json.loads(completed.stdout))
    production_env = parse_env(result['OFFLOAD_DOCKER_ENV'] + ' ' + result['OFFLOAD_DSPARK_ENV'])
    assert production_env['VLLM_ENGRAM_REMOTE_REQUIRED'] == '0'
    assert parse_env(result['EXP_DOCKER_ENV']) == {**production_env, **FLAGS[role]}
    assert result['MAX_LEN_OVERRIDE'] == '327680' and result['KV_BYTES_OVERRIDE'] == '6442450944'
    assert result['LABEL'] == 'TV1-' + role
    assert shlex.split(result['EXP_EXTRA_ARGS']) == ['--profiler-config', sp.PROFILER_JSON]
    old, new = shlex.split(result['OFFLOAD_DOCKER_ARGS']), shlex.split(result['EXTRA_DOCKER_ARGS'])
    replaced = {build.PREFIX + entry['path'] for entry in manifest['files']}
    expected = [token for i in range(0, len(old), 2) if old[i + 1].split(':')[1] not in replaced
                for token in old[i:i + 2]]
    assert new == [*expected, '-v', '$HOME/dsv41-offload-run/sp1-profile:/sp1-profile:rw']
    assert result['OVERLAYS'].split() == [
        f'offload/tv1/generated-fixture/{role}/{entry["path"]}={build.PREFIX}{entry["path"]}'
        for entry in manifest['files']]


@pytest.mark.parametrize('extra', ['AK_MIN=3', 'PF1_TRACE=1', 'EXP_DOCKER_ENV=x', 'SPEC_OVERRIDE=x',
                                   'SP1_PROFILER_CONFIG=x', 'K1_KV_PAIR=1'])
def test_shell_config_rejects_inherited_features_before_launch(tmp_path: Path, extra: str) -> None:
    make_overlay(tmp_path, 'off')
    completed = shell_config(tmp_path, 'off', extra=extra)
    assert completed.returncode != 0 and 'TV1 STOP' in completed.stderr


def test_shell_config_refuses_required_substitution_and_duplicate_env(tmp_path: Path) -> None:
    make_overlay(tmp_path, 'off')
    original = ops1_env()
    for changed in (original.replace('VLLM_ENGRAM_REMOTE_REQUIRED=0', 'VLLM_ENGRAM_REMOTE_REQUIRED=1'),
                    original + '\nOFFLOAD_DOCKER_ENV="$OFFLOAD_DOCKER_ENV VLLM_ENGRAM_REMOTE_REQUIRED=1"\n'):
        result = shell_config(tmp_path, 'off', source=changed)
        assert result.returncode != 0


def identities(manifest: build.Manifest) -> tuple[dict[str, Json], dict[str, Json]]:
    prod: dict[str, Json] = {'id': 'prod', 'state': 'running', 'image': pf.IMAGE, 'engine': '0.31.1',
        'files': {'/srv/dsv41/legal-v2a/config.json': 'f' * 64},
        'mounted': {build.PREFIX + 'vllm/v1/worker/gpu/model_runner.py': 'a' * 64, '/serve.sh': 'b' * 64},
        'env': {'VLLM_ENGRAM_REMOTE_REQUIRED': '0', 'NCCL_PROTO': '^LL128', 'PATH': '/usr/bin', 'HOSTNAME': 'fixed'},
        'engram_identity': {'shim': 'c' * 64},
        'argv': 'vllm serve /srv/dsv41/legal-v2a --port 8000 --max-model-len 327680 --max-num-seqs 4'}
    live: dict[str, Json] = json.loads(json.dumps(prod))
    record(live['env']).update(cast(dict[str, Json], FLAGS['off']))
    record(live['mounted']).update({build.PREFIX + entry['path']: entry['sha256'] for entry in manifest['files']})
    live['argv'] = str(prod['argv']).replace('--port 8000', '--port 8001') + ' --profiler-config ' + shlex.quote(sp.PROFILER_JSON)
    return prod, live


def test_ops1_runtime_passes_required_one_and_all_other_drift_fail(tmp_path: Path) -> None:
    manifest = make_overlay(tmp_path, 'off')
    prod, live = identities(manifest)
    completed = shell_config(tmp_path, 'off')
    assert completed.returncode == 0, completed.stderr
    composed = cast(dict[str, str], json.loads(completed.stdout))
    production_env = parse_env(composed['OFFLOAD_DOCKER_ENV'] + ' ' + composed['OFFLOAD_DSPARK_ENV'])
    record(prod['env']).update(cast(dict[str, Json], production_env))
    record(live['env']).update(cast(dict[str, Json], parse_env(composed['EXP_DOCKER_ENV'])))
    production.runtime(prod, live, Arm('off', 327680, 0), manifest, 'r')
    for key in ('VLLM_ENGRAM_REMOTE_REQUIRED', 'NCCL_PROTO', 'PATH', 'HOSTNAME', 'UNEXPECTED_SETTING'):
        broken = cast(dict[str, Json], json.loads(json.dumps(live)))
        record(broken['env'])[key] = '1'
        with pytest.raises(ValueError, match='environment'):
            production.runtime(prod, broken, Arm('off', 327680, 0), manifest, 'r')
    for key, value in (('image', 'other'), ('files', {}), ('engram_identity', {}), ('mounted', {}),
                       ('argv', str(live['argv']).replace('327680', '524288'))):
        with pytest.raises(ValueError):
            production.runtime(prod, {**live, key: value}, Arm('off', 327680, 0), manifest, 'r')


def test_ledger_executes_exact_ops1_head_tag_and_rejects_obsolete_head(tmp_path: Path) -> None:
    home = tmp_path / 'home'
    (home / 'dsv41-deploy').mkdir(parents=True)
    binary = tmp_path / 'bin'
    binary.mkdir()
    fake_git = binary / 'git'
    fake_git.write_text('#!/bin/bash\ncase "$*" in\n"rev-parse HEAD") echo "$FIXTURE_HEAD";;\n'
        f'"rev-list -n1 {production.PROD_TAG}") echo "$FIXTURE_TAG";;\n'
        '"diff --exit-code HEAD -- dsv41.env serve.sh start.sh stop.sh offload/launch/window.sh offload/launch/warmup.py offload/launch/warmup_serve.sh") exit "$FIXTURE_DIRTY";;\n'
        '*) exit 81;;\nesac\n')
    fake_git.chmod(0o700)
    env = {**os.environ, 'HOME': str(home), 'PATH': str(binary) + os.pathsep + os.environ['PATH'],
           'FIXTURE_HEAD': production.PROD_COMMIT, 'FIXTURE_TAG': production.PROD_COMMIT, 'FIXTURE_DIRTY': '0'}
    command = ['bash', '-c', production.ledger_script()]
    assert subprocess.run(command, env=env, check=False).returncode == 0
    for key, value in (('FIXTURE_HEAD', '21bed31'), ('FIXTURE_TAG', '21bed31'), ('FIXTURE_DIRTY', '1')):
        assert subprocess.run(command, env={**env, key: value}, check=False).returncode != 0
    (home / 'dsv41-deploy/HOLD').touch()
    assert subprocess.run(command, env=env, check=False).returncode != 0
    metadata = production.rollback_metadata()
    assert metadata['production_commit'] == production.PROD_COMMIT
    assert metadata['tag'] == 'prod-k1-327-ops1-20261026'
