"""Production-derived NBW arms; shell runs only on CPU fixtures."""
from __future__ import annotations

import json
import os
import shlex
import subprocess
from pathlib import Path

import pytest

from offload.launch.windows import pf1_window as pf

CONFIG = pf.ROOT / 'offload/launch/configs/NBW-native.env'
ENV = ('VLLM_DSV41_KV_PAIR=3 VLLM_DSV41_DROP_MTP_BUFFER=1 VLLM_EXL3_TF_PROMPT_TRANSIENT=1 '
       'VLLM_EXL3_TF_PROMPT_ALIAS=1 VLLM_INDEXER_PREFILL_BUFFER_FACTOR=4 VLLM_ENGRAM_REMOTE_REQUIRED=0')
DRAFT = ('VLLM_DSPARK_REMOTE_POLICY=capacity VLLM_DSPARK_REMOTE_ASYNC=1 VLLM_DSPARK_REMOTE_ASYNC_PROXY=process '
         f'VLLM_DSPARK_REMOTE_IDENTITY={"a" * 64} VLLM_DSPARK_REMOTE_MARKOV_IDENTITY={"b" * 64}')


def source(tmp_path: Path, role: str, *, draft: str = DRAFT, extra: dict[str, str] | None = None,
           kv: int = 6442450944) -> subprocess.CompletedProcess[str]:
    prod = tmp_path / 'production.env'
    prod.write_text(f'OFFLOAD_DOCKER_ENV={shlex.quote(ENV)}\nOFFLOAD_DSPARK_ENV={shlex.quote(draft)}\n'
                    'OFFLOAD_DOCKER_ARGS="-v /production/code:/engine:ro"\n'
                    f'PROD_KV_BYTES={kv}\nPROD_MAX_LEN=327680\nPROD_K1=1\nPROD_K1_PROFILE=327680\nPORT=8000\n')
    env = {k: v for k, v in os.environ.items() if not k.startswith(('NBW_', 'PF1_', 'K1_'))}
    env.update({'NBW_PROD_ENV': str(prod), 'NBW_ROLE': role, 'NBW_NATIVE_IDENTITY': 'c' * 64,
                'NBW_NATIVE_MARKOV_IDENTITY': 'b' * 64, **(extra or {})})
    script = f'source {shlex.quote(str(CONFIG))} || exit\n' + '\n'.join(
        f'printf "%s\\n" "${{{key}-}}"' for key in
        ('EXP_DOCKER_ENV', 'EXTRA_DOCKER_ARGS', 'MAX_LEN_OVERRIDE', 'KV_BYTES_OVERRIDE', 'PORT', 'OVERLAYS'))
    return subprocess.run(['bash', '-c', script], env=env, capture_output=True, text=True, check=False)


def test_k_and_i_differ_only_by_reviewed_identity(tmp_path: Path) -> None:
    k, i = source(tmp_path, 'K'), source(tmp_path, 'I')
    assert k.returncode == i.returncode == 0, (k.stderr, i.stderr)
    ka, ia = k.stdout.splitlines(), i.stdout.splitlines()
    ke, ie = (dict(token.split('=', 1) for token in row.split()) for row in (ka[0], ia[0]))
    assert {key for key in ke if ke[key] != ie[key]} == {'VLLM_DSPARK_REMOTE_IDENTITY'}
    assert ie['VLLM_DSPARK_REMOTE_IDENTITY'] == 'c' * 64
    assert ka[0] == ENV + ' ' + DRAFT
    assert ka[1:] == ia[1:] == ['-v /production/code:/engine:ro', '327680', '6442450944', '', '']


@pytest.mark.parametrize('role,draft,extra,kv', [
    ('X', DRAFT, {}, 6442450944), ('I', DRAFT, {'NBW_NATIVE_IDENTITY': ''}, 6442450944),
    ('I', DRAFT + ' VLLM_DSPARK_REMOTE_IDENTITY=' + 'a' * 64, {}, 6442450944),
    ('I', DRAFT.replace('VLLM_DSPARK_REMOTE_MARKOV_IDENTITY', 'OTHER'), {}, 6442450944),
    ('K', DRAFT.replace('ASYNC_PROXY=process', 'ASYNC_PROXY=thread'), {}, 6442450944),
    ('K', DRAFT, {'EXP_EXTRA_ARGS': '--enforce-eager'}, 6442450944), ('K', DRAFT, {}, 3221225472)])
def test_config_refuses_drift(tmp_path: Path, role: str, draft: str, extra: dict[str, str], kv: int) -> None:
    result = source(tmp_path, role, draft=draft, extra=extra, kv=kv)
    assert result.returncode != 0 and 'NBW STOP' in result.stderr, json.dumps(result.stdout)
