"""SP1-profile.env against production's real 21bed31 dsv41.env; CPU only, no service or remote call."""
from __future__ import annotations

import ast
import json
import os
import shlex
import subprocess
from pathlib import Path

import pytest

from offload.launch.windows import sp1_window as s

ROOT = Path(__file__).resolve().parents[4]
PROD = '21bed31'
SRC_IMAGE = Path('/opt/dsv41/data/src-image')
FIELDS = ('LABEL', 'MAX_LEN_OVERRIDE', 'KV_BYTES_OVERRIDE', 'EXP_DOCKER_ENV', 'EXTRA_DOCKER_ARGS', 'OVERLAYS', 'EXP_EXTRA_ARGS')
RUNNER_ENV = {'PF1_CAP': '327680', 'PF1_TRACE': '0', 'PF1_EAGER': '0'}    # what Live.start passes


def production() -> str:
    try:
        return subprocess.run(['git', 'show', f'{PROD}:dsv41.env'], cwd=ROOT, text=True, capture_output=True,
                              check=True).stdout
    except subprocess.CalledProcessError:
        pytest.skip('production commit 21bed31 not in this clone')


def source(home: Path, config: str, **env: str) -> dict[str, str]:
    (home / 'dsv41-deploy').mkdir(parents=True, exist_ok=True)
    (home / 'dsv41-deploy/dsv41.env').write_text(production())
    printer = ' '.join(f'"${{{f}-}}"' for f in FIELDS)
    script = (f'set -euo pipefail; set -a; . ./dsv41.env; PORT=8001; . {config}; set +a; '
              f'test "$PORT" = 8001; test -z "$(compgen -v | grep -E "^(pf1_|sp1_|SP1_)" || true)"; printf "%s\\0" {printer}')
    result = subprocess.run(['bash', '-c', script], cwd=ROOT, text=True, capture_output=True,
                            env={'HOME': str(home), 'PATH': os.environ['PATH'], **env})
    if result.returncode:
        raise RuntimeError(result.stderr.strip())
    return dict(zip(FIELDS, result.stdout.split('\0')))


def test_profile_arm_is_pf1_off_327_plus_exactly_two_additions(tmp_path: Path) -> None:
    off = source(tmp_path, 'offload/launch/configs/PF1-window.env', PF1_CAP='327680')
    got = source(tmp_path, s.CONFIG, **RUNNER_ENV)
    for key in ('MAX_LEN_OVERRIDE', 'KV_BYTES_OVERRIDE', 'EXP_DOCKER_ENV', 'OVERLAYS'):
        assert got[key] == off[key]
    assert got['MAX_LEN_OVERRIDE'] == '327680' and got['KV_BYTES_OVERRIDE'] == '6442450944' and got['OVERLAYS'] == ''
    assert off['EXP_EXTRA_ARGS'] == ''
    assert got['EXTRA_DOCKER_ARGS'] == off['EXTRA_DOCKER_ARGS'] + ' -v $HOME/dsv41-offload-run/sp1-profile:/sp1-profile:rw'
    assert ':rw' not in off['EXTRA_DOCKER_ARGS']
    assert shlex.split(got['EXP_EXTRA_ARGS']) == ['--profiler-config', s.PROFILER_JSON]
    assert got['LABEL'] == 'SP1-profile-327680'


def test_config_json_is_the_runner_constant_and_the_runner_env_is_accepted_without_it(tmp_path: Path) -> None:
    text = (ROOT / s.CONFIG).read_text()
    assert f"SP1_PROFILER_CONFIG='{s.PROFILER_JSON}'" in text
    assert source(tmp_path, s.CONFIG)['LABEL'] == 'SP1-profile-327680'     # defaults equal the runner's values


@pytest.mark.parametrize('env,message', [
    ({'PF1_CAP': '524288'}, '327,680 OFF arm only'),
    ({'PF1_TRACE': '1'}, '327,680 OFF arm only'),
    ({'PF1_EAGER': '1'}, '327,680 OFF arm only'),
    ({'PF1_TRACE_RUN': 'x'}, 'inherited PF1 trace'),
    ({'SP1_PROFILER_CONFIG': '{}'}, 'inherited PF1 trace or SP1 profiler'),
    ({'EXP_EXTRA_ARGS': '--enforce-eager'}, 'inherited EXP_EXTRA_ARGS'),
    ({'EXTRA_DOCKER_ARGS': '-v a:/b:rw'}, 'inherited EXTRA_DOCKER_ARGS'),
])
def test_config_refuses_anything_but_the_reviewed_arm(tmp_path: Path, env: dict[str, str], message: str) -> None:
    with pytest.raises(RuntimeError, match=message):
        source(tmp_path, s.CONFIG, **{**RUNNER_ENV, **env})


def test_every_profiler_field_exists_in_the_image_profiler_config() -> None:
    path = SRC_IMAGE / 'vllm/config/profiler.py'
    if not path.is_file():
        pytest.skip('reference image source not present')
    tree = ast.parse(path.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'ProfilerConfig')
    fields = {n.target.id for n in cls.body if isinstance(n, ast.AnnAssign) and isinstance(n.target, ast.Name)}
    assert set(s.PROFILER_CONFIG) <= fields
    assert json.loads(s.PROFILER_JSON) == s.PROFILER_CONFIG
    # defaults that would hurt this window are overridden explicitly
    assert s.PROFILER_CONFIG['torch_profiler_with_stack'] is False and s.PROFILER_CONFIG['ignore_frontend'] is True
    assert s.PROFILER_CONFIG['torch_profiler_dump_cuda_time_total'] is False and s.PROFILER_CONFIG['max_iterations'] == 16


def test_serve_script_line_keeps_the_json_one_argument(tmp_path: Path) -> None:
    """EXP_EXTRA_ARGS reaches /serve.sh through serve_exp.sh EXTRA and the launcher's `echo "  $EXTRA"`."""
    got = source(tmp_path, s.CONFIG, **RUNNER_ENV)
    line = subprocess.run(['bash', '-c', 'EXTRA="--a b${EXP_EXTRA_ARGS:+ $EXP_EXTRA_ARGS}"; echo "  $EXTRA"'],
                          env={'EXP_EXTRA_ARGS': got['EXP_EXTRA_ARGS'], 'PATH': os.environ['PATH']},
                          capture_output=True, text=True, check=True).stdout
    argv = shlex.split(line)
    assert argv[:2] == ['--a', 'b'] and argv[2:] == ['--profiler-config', s.PROFILER_JSON]
    assert s.strip_profiler('vllm serve x --port 8001 \\\n ' + line) == 'vllm serve x --port 8001 --a b'
