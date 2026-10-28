"""PF1-fix.env = PF1-window.env OFF plus the fence sources and one switch; CPU only, no service or remote call."""
from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path

import pytest

from offload.launch.windows.tests import test_pf1_config as base

ROOT = base.ROOT
CONFIG = 'offload/launch/configs/PF1-fix.env'
LIB = '/usr/local/lib/python3.12/dist-packages'
MOE = ('vllm/models/deepseek_v41/nvidia/model.py', 'vllm_exl3_moe/tf_prompt.py', 'vllm_exl3_moe/tf_experts.py')


def source(home: Path, prod_text: str, config: str = CONFIG, **env: str) -> dict[str, str]:
    (home / 'dsv41-deploy').mkdir(parents=True, exist_ok=True)
    (home / 'dsv41-deploy/dsv41.env').write_text(prod_text)
    printer = ' '.join(f'"${{{f}-}}"' for f in base.FIELDS)
    script = (f'set -euo pipefail; set -a; . ./dsv41.env; PORT=8001; . {config}; set +a; test "$PORT" = 8001; '
              f'test -z "$(compgen -v | grep -E "^pf1[cf]?_" || true)"; printf "%s\\0" {printer}')
    result = subprocess.run(['bash', '-c', script], cwd=ROOT, text=True, capture_output=True,
                            env={'HOME': str(home), 'PATH': os.environ['PATH'], **env})
    if result.returncode:
        raise RuntimeError(result.stderr.strip())
    return dict(zip(base.FIELDS, result.stdout.split('\0')))


def fix_root(tmp_path: Path) -> Path:
    out = tmp_path / 'fix'
    subprocess.run(['python3', '-m', 'offload.prefill_cause2.build_fix', '--enabled', '--output-root', str(out)],
                   cwd=ROOT, check=True, capture_output=True, env={**os.environ, 'CUDA_VISIBLE_DEVICES': ''})
    return out


@pytest.mark.parametrize('cap', ['327680', '524288'])
def test_fix_arm_is_the_off_arm_plus_fence_mounts_and_one_switch(tmp_path: Path, cap: str) -> None:
    text = base.production()
    fix = fix_root(tmp_path)
    got = source(tmp_path / 'home', text, PF1_CAP=cap, PF1F_ROOT=str(fix))
    off = source(tmp_path / 'home', text, base.CONFIG, PF1_CAP=cap)
    assert got['EXP_DOCKER_ENV'].split() == off['EXP_DOCKER_ENV'].split() + ['VLLM_DSV41_MOE_FENCE=1']
    assert got['MAX_LEN_OVERRIDE'] == off['MAX_LEN_OVERRIDE'] == cap
    assert got['KV_BYTES_OVERRIDE'] == off['KV_BYTES_OVERRIDE'] and got['EXP_EXTRA_ARGS'] == off['EXP_EXTRA_ARGS'] == ''
    assert off['OVERLAYS'] == ''
    pins = [line.split()[1] for line in (ROOT / 'offload/prefill_cause2/OVERLAY.sha256').read_text().splitlines()]
    assert got['OVERLAYS'].split() == [f'{fix}/{p}={LIB}/{p}' for p in pins] and len(pins) == 4
    kept, before = got['EXTRA_DOCKER_ARGS'].split(), off['EXTRA_DOCKER_ARGS'].split()
    assert kept[::2] == ['-v'] * (len(kept) // 2) and before[::2] == ['-v'] * (len(before) // 2)
    kept, before = kept[1::2], before[1::2]
    removed = [x for x in before if x not in kept]
    assert sorted(r.split(':')[1].removeprefix(LIB + '/') for r in removed) == sorted(MOE)
    assert all(r.endswith(':ro') for r in removed) and [x for x in before if x in kept] == kept
    assert not any('pf1-trace' in x for x in kept) and 'PREFILL_TRACE' not in got['EXP_DOCKER_ENV']
    assert got['LABEL'] == f'PF1-fix-{cap}-fence1'


def test_traced_fix_tampered_source_and_bad_root_refuse(tmp_path: Path) -> None:
    text = base.production()
    fix = fix_root(tmp_path)
    with pytest.raises(RuntimeError, match='fix arm is untraced'):
        source(tmp_path / 'home', text, PF1_TRACE='1', PF1F_ROOT=str(fix))
    target = fix / 'vllm/v1/worker/gpu/dsv41_moe_fence.py'
    target.write_text(target.read_text() + '\n')
    with pytest.raises(RuntimeError, match='fence sources differ'):
        source(tmp_path / 'home', text, PF1F_ROOT=str(fix))
    with pytest.raises(RuntimeError, match='fence source path'):
        source(tmp_path / 'home', text, PF1F_ROOT='a b')
    with pytest.raises(RuntimeError, match='inherited OVERLAYS'):
        source(tmp_path / 'home', text, OVERLAYS='x=y', PF1F_ROOT=str(fix))


def test_production_without_a_moe_mount_or_with_the_switch_refuses(tmp_path: Path) -> None:
    text = base.production()
    fix = fix_root(tmp_path)
    found = re.search(r'-v \S*/vllm_exl3_moe/tf_experts\.py:ro ?', text)
    assert found is not None
    with pytest.raises(RuntimeError, match='exactly three production MoE targets'):
        source(tmp_path / 'a', text[:found.start()] + text[found.end():], PF1F_ROOT=str(fix))
    switched = text.replace('VLLM_EXL3_TF_PROMPT_ALIAS=1', 'VLLM_EXL3_TF_PROMPT_ALIAS=1 VLLM_DSV41_MOE_FENCE=0', 1)
    assert switched != text
    with pytest.raises(RuntimeError, match='already sets VLLM_DSV41_MOE_FENCE'):
        source(tmp_path / 'b', switched, PF1F_ROOT=str(fix))
