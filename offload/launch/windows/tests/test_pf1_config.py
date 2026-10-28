"""PF1-window.env against production's real 21bed31 dsv41.env; CPU only, no service or remote call."""
from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[4]
CONFIG = 'offload/launch/configs/PF1-window.env'
PROD = '21bed31'
SRC_IMAGE = Path('/opt/dsv41/data/src-image')
FIELDS = ('LABEL', 'MAX_LEN_OVERRIDE', 'KV_BYTES_OVERRIDE', 'EXP_DOCKER_ENV', 'EXTRA_DOCKER_ARGS', 'OVERLAYS', 'EXP_EXTRA_ARGS')


def production() -> str:
    try:
        return subprocess.run(['git', 'show', f'{PROD}:dsv41.env'], cwd=ROOT, text=True, capture_output=True,
                              check=True).stdout
    except subprocess.CalledProcessError:
        pytest.skip('production commit 21bed31 not in this clone')


def prod_value(text: str, key: str, home: Path) -> str:
    script = f'set +u; . /dev/stdin >/dev/null; printf %s "${{{key}}}"'
    return subprocess.run(['bash', '-c', script], input=text, text=True, capture_output=True, check=True,
                          env={'HOME': str(home), 'PATH': os.environ['PATH']}).stdout


def source(home: Path, prod_text: str, **env: str) -> dict[str, str]:
    (home / 'dsv41-deploy').mkdir(parents=True, exist_ok=True)
    (home / 'dsv41-deploy/dsv41.env').write_text(prod_text)
    printer = ' '.join(f'"${{{f}-}}"' for f in FIELDS)
    script = (f'set -euo pipefail; set -a; . ./dsv41.env; PORT=8001; . {CONFIG}; set +a; '
              f'test "$PORT" = 8001; test -z "$(compgen -v | grep ^pf1_ || true)"; printf "%s\\0" {printer}')
    result = subprocess.run(['bash', '-c', script], cwd=ROOT, text=True, capture_output=True,
                            env={'HOME': str(home), 'PATH': os.environ['PATH'], **env})
    if result.returncode:
        raise RuntimeError(result.stderr.strip())
    return dict(zip(FIELDS, result.stdout.split('\0')))


def trace_root(tmp_path: Path) -> Path:
    if not SRC_IMAGE.is_dir():
        pytest.skip('reference image source not present')
    out = tmp_path / 'generated'
    subprocess.run(['python3', '-m', 'offload.prefill_trace.build_overlay', '--enabled', '--output-root', str(out)],
                   cwd=ROOT, check=True, capture_output=True, env={**os.environ, 'CUDA_VISIBLE_DEVICES': ''})
    return out


@pytest.mark.parametrize('cap', ['327680', '524288'])
def test_off_arm_is_production_launch_with_required_engram(tmp_path: Path, cap: str) -> None:
    text = production()
    got = source(tmp_path, text, PF1_CAP=cap)
    prod_env = (prod_value(text, 'OFFLOAD_DOCKER_ENV', tmp_path) + ' ' +
                prod_value(text, 'OFFLOAD_DSPARK_ENV', tmp_path)).split()
    expected = [e.replace('VLLM_ENGRAM_REMOTE_REQUIRED=0', 'VLLM_ENGRAM_REMOTE_REQUIRED=1') for e in prod_env]
    assert got['EXP_DOCKER_ENV'].split() == expected
    assert sum(e != g for e, g in zip(prod_env, expected)) == 1
    assert got['EXTRA_DOCKER_ARGS'] == prod_value(text, 'OFFLOAD_DOCKER_ARGS', tmp_path)
    assert '$HOME/dsv41-kvpack/559c5867ba42/' in got['EXTRA_DOCKER_ARGS']
    assert 'dsv41-offload-prod/dspark/remote_dspark' in got['EXTRA_DOCKER_ARGS']
    assert got['OVERLAYS'] == '' and got['EXP_EXTRA_ARGS'] == ''
    assert got['MAX_LEN_OVERRIDE'] == cap and got['KV_BYTES_OVERRIDE'] == '6442450944'
    assert got['LABEL'] == f'PF1-window-{cap}-trace0-eager0'


def test_on_arm_replaces_exactly_two_targets(tmp_path: Path) -> None:
    text = production()
    root = trace_root(tmp_path)
    got = source(tmp_path / 'home', text, PF1_CAP='524288', PF1_TRACE='1', PF1_TRACE_RUN='on-524',
                 PF1_TRACE_ROOT=str(root))
    off = source(tmp_path / 'home', text, PF1_CAP='524288')
    pins = [line.split()[1] for line in (ROOT / 'offload/prefill_trace/OVERLAY.sha256').read_text().splitlines()]
    assert got['OVERLAYS'].split() == [f'{root}/{p}=/usr/local/lib/python3.12/dist-packages/{p}' for p in pins]
    kept = got['EXTRA_DOCKER_ARGS'].split()
    removed = [x for x in off['EXTRA_DOCKER_ARGS'].split() if x not in kept]
    assert sorted(r.split(':')[1].rsplit('/', 1)[1] for r in removed) == ['flashinfer_sparse.py', 'model_runner.py']
    assert kept[-2:] == ['-v', '$HOME/dsv41-offload-run/pf1-trace/on-524:/pf1-trace:rw']
    assert got['EXP_DOCKER_ENV'].split()[-3:] == ['VLLM_DSV41_PREFILL_TRACE=1', 'VLLM_DSV41_PREFILL_TRACE_DIR=/pf1-trace',
                                                  'VLLM_DSV41_PREFILL_TRACE_RUN=on-524']
    assert got['EXP_DOCKER_ENV'].split()[:-3] == off['EXP_DOCKER_ENV'].split()
    eager = source(tmp_path / 'home', text, PF1_CAP='327680', PF1_EAGER='1')
    assert eager['EXP_EXTRA_ARGS'] == '--enforce-eager' and eager['LABEL'].endswith('eager1')


def test_tampered_trace_source_and_missing_run_refuse(tmp_path: Path) -> None:
    text = production()
    root = trace_root(tmp_path)
    with pytest.raises(RuntimeError, match='PF1_TRACE_RUN'):
        source(tmp_path / 'home', text, PF1_TRACE='1', PF1_TRACE_ROOT=str(root))
    with pytest.raises(RuntimeError, match='invalid PF1_TRACE_RUN'):
        source(tmp_path / 'home', text, PF1_TRACE='1', PF1_TRACE_RUN='a b', PF1_TRACE_ROOT=str(root))
    target = root / 'vllm/v1/worker/gpu/model_runner.py'
    target.write_text(target.read_text() + '\n')
    with pytest.raises(RuntimeError, match='OVERLAY.sha256'):
        source(tmp_path / 'home', text, PF1_TRACE='1', PF1_TRACE_RUN='on-327', PF1_TRACE_ROOT=str(root))


@pytest.mark.parametrize('env,message', [({'PF1_CAP': '393216'}, 'PF1_CAP'), ({'PF1_TRACE': '2'}, 'PF1_TRACE'),
    ({'PF1_EAGER': 'yes'}, 'PF1_EAGER'), ({'K1_MEMPROBE': '/root/.cache/x'}, 'inherited K1_MEMPROBE'),
    ({'SPEC_OVERRIDE': '{}'}, 'inherited SPEC_OVERRIDE'), ({'OVERLAYS': 'a=b'}, 'inherited OVERLAYS')])
def test_bad_or_inherited_settings_refuse(tmp_path: Path, env: dict[str, str], message: str) -> None:
    with pytest.raises(RuntimeError, match=message):
        source(tmp_path, production(), **env)


@pytest.mark.parametrize('edit,message', [
    (('PROD_MAX_LEN=327680', 'PROD_MAX_LEN=524288'), 'reconciled ledger'),
    (('PROD_K1_PROFILE=327680', 'PROD_K1_PROFILE=524288'), 'reconciled ledger'),
    (('VLLM_ENGRAM_REMOTE_REQUIRED=0', 'VLLM_ENGRAM_REMOTE_REQUIRED=1'), 'lacks VLLM_ENGRAM_REMOTE_REQUIRED=0'),
    (('VLLM_INDEXER_PREFILL_BUFFER_FACTOR=4', 'VLLM_INDEXER_PREFILL_BUFFER_FACTOR=8'), 'FACTOR=4'),
    (('VLLM_DSPARK_REMOTE_ASYNC_PROXY=process', 'VLLM_DSPARK_REMOTE_ASYNC_PROXY=thread'), 'PROXY=process'),
    (('tf_experts.py:ro"', 'tf_experts.py:rw"'), 'not read-only'),
    (('VLLM_ENGRAM_REMOTE_LOG_S=60', 'VLLM_ENGRAM_REMOTE_LOG_S=60 VLLM_ENGRAM_REMOTE_REQUIRED=1'), 'duplicate Engram'),
])
def test_production_drift_refuses(tmp_path: Path, edit: tuple[str, str], message: str) -> None:
    text = production()
    old, new = edit
    assert old in text
    with pytest.raises(RuntimeError, match=message):
        source(tmp_path, text.replace(old, new, 1))


def test_missing_production_config_refuses(tmp_path: Path) -> None:
    production()
    with pytest.raises(RuntimeError, match='missing'):
        source(tmp_path, '', PF1_PROD_ENV=str(tmp_path / 'absent.env'))
