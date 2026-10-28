"""PF1-cause.env = PF1-window.env plus the MoE spans / TF-transient discriminator; CPU only, no service or remote call."""
from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path

import pytest

from offload.launch.windows.tests import test_pf1_config as base

ROOT = base.ROOT
CONFIG = 'offload/launch/configs/PF1-cause.env'
LIB = '/usr/local/lib/python3.12/dist-packages'
MOE = ('vllm/models/deepseek_v41/nvidia/model.py', 'vllm_exl3_moe/tf_prompt.py', 'vllm_exl3_moe/tf_experts.py')


def source(home: Path, prod_text: str, config: str = CONFIG, **env: str) -> dict[str, str]:
    (home / 'dsv41-deploy').mkdir(parents=True, exist_ok=True)
    (home / 'dsv41-deploy/dsv41.env').write_text(prod_text)
    printer = ' '.join(f'"${{{f}-}}"' for f in base.FIELDS)
    script = (f'set -euo pipefail; set -a; . ./dsv41.env; PORT=8001; . {config}; set +a; test "$PORT" = 8001; '
              f'test -z "$(compgen -v | grep -E "^pf1c?_" || true)"; printf "%s\\0" {printer}')
    result = subprocess.run(['bash', '-c', script], cwd=ROOT, text=True, capture_output=True,
                            env={'HOME': str(home), 'PATH': os.environ['PATH'], **env})
    if result.returncode:
        raise RuntimeError(result.stderr.strip())
    return dict(zip(base.FIELDS, result.stdout.split('\0')))


def moe_root(tmp_path: Path) -> Path:
    out = tmp_path / 'moe'
    subprocess.run(['python3', '-m', 'offload.prefill_cause.build_overlay', '--enabled', '--output-root', str(out)],
                   cwd=ROOT, check=True, capture_output=True, env={**os.environ, 'CUDA_VISIBLE_DEVICES': ''})
    return out


@pytest.mark.parametrize('cap', ['327680', '524288'])
def test_off_arm_equals_pf1_window_off(tmp_path: Path, cap: str) -> None:
    text = base.production()
    got = source(tmp_path, text, PF1_CAP=cap)
    window = source(tmp_path, text, base.CONFIG, PF1_CAP=cap)
    assert {k: v for k, v in got.items() if k != 'LABEL'} == {k: v for k, v in window.items() if k != 'LABEL'}
    assert got['LABEL'] == f'PF1-cause-{cap}-trace0-tft1'


def test_on_arm_adds_moe_spans_on_top_of_the_pf1_r2_trace(tmp_path: Path) -> None:
    text = base.production()
    trace, moe = base.trace_root(tmp_path), moe_root(tmp_path)
    common = dict(PF1_CAP='524288', PF1_TRACE='1', PF1_TRACE_RUN='cause-524', PF1_TRACE_ROOT=str(trace))
    got = source(tmp_path / 'home', text, PF1C_ROOT=str(moe), **common)
    window = source(tmp_path / 'home', text, base.CONFIG, **common)
    assert got['EXP_DOCKER_ENV'] == window['EXP_DOCKER_ENV']
    pins = [line.split()[1] for line in (ROOT / 'offload/prefill_cause/OVERLAY.sha256').read_text().splitlines()]
    assert got['OVERLAYS'].split() == window['OVERLAYS'].split() + [f'{moe}/{p}={LIB}/{p}' for p in pins]
    targets = [pair.split('=', 1)[1] for pair in got['OVERLAYS'].split()]
    assert len(targets) == len(set(targets)) == 10
    kept, before = got['EXTRA_DOCKER_ARGS'].split(), window['EXTRA_DOCKER_ARGS'].split()
    assert kept[::2] == ['-v'] * (len(kept) // 2) and before[::2] == ['-v'] * (len(before) // 2)
    kept, before = kept[1::2], before[1::2]
    removed = [x for x in before if x not in kept]
    assert sorted(r.split(':')[1].removeprefix(LIB + '/') for r in removed) == sorted(MOE)
    assert all(r.endswith(':ro') for r in removed)
    assert [x for x in before if x in kept] == kept and kept[-1].endswith(':/pf1-trace:rw')
    assert got['LABEL'] == 'PF1-cause-524288-trace1-tft1'


def test_tf_transient_discriminator_changes_exactly_two_switches(tmp_path: Path) -> None:
    text = base.production()
    for trace in ('0', '1'):
        extra = {}
        if trace == '1':
            extra = dict(PF1_TRACE_RUN='cause-t0', PF1_TRACE_ROOT=str(base.trace_root(tmp_path / 't')),
                         PF1C_ROOT=str(moe_root(tmp_path / 'm')))
        on = source(tmp_path / 'home', text, PF1_CAP='524288', PF1_TRACE=trace, **extra)
        t0 = source(tmp_path / 'home', text, PF1_CAP='524288', PF1_TRACE=trace, PF1C_TF_TRANSIENT='0', **extra)
        a, b = on['EXP_DOCKER_ENV'].split(), t0['EXP_DOCKER_ENV'].split()
        assert len(a) == len(b) and [(x, y) for x, y in zip(a, b) if x != y] == [
            ('VLLM_EXL3_TF_PROMPT_TRANSIENT=1', 'VLLM_EXL3_TF_PROMPT_TRANSIENT=0'),
            ('VLLM_EXL3_TF_PROMPT_ALIAS=1', 'VLLM_EXL3_TF_PROMPT_ALIAS=0')]
        assert {k: v for k, v in on.items() if k not in ('EXP_DOCKER_ENV', 'LABEL')} == \
            {k: v for k, v in t0.items() if k not in ('EXP_DOCKER_ENV', 'LABEL')}
        assert t0['LABEL'] == f'PF1-cause-524288-trace{trace}-tft0'


def test_tampered_moe_source_bad_switch_and_inherited_k1_refuse(tmp_path: Path) -> None:
    text = base.production()
    trace, moe = base.trace_root(tmp_path), moe_root(tmp_path)
    target = moe / 'vllm_exl3_moe/tf_experts.py'
    target.write_text(target.read_text() + '\n')
    with pytest.raises(RuntimeError, match='MoE span sources differ'):
        source(tmp_path / 'home', text, PF1_TRACE='1', PF1_TRACE_RUN='x', PF1_TRACE_ROOT=str(trace), PF1C_ROOT=str(moe))
    with pytest.raises(RuntimeError, match='PF1C_TF_TRANSIENT must be 0 or 1'):
        source(tmp_path / 'home', text, PF1C_TF_TRANSIENT='off')
    with pytest.raises(RuntimeError, match='inherited K1_TF_TRANSIENT'):
        source(tmp_path / 'home', text, K1_TF_TRANSIENT='0')
    with pytest.raises(RuntimeError, match='MoE span source path'):
        source(tmp_path / 'home', text, PF1_TRACE='1', PF1_TRACE_RUN='x', PF1_TRACE_ROOT=str(trace), PF1C_ROOT='a b')


def test_production_without_the_moe_mounts_or_switches_refuses(tmp_path: Path) -> None:
    text = base.production()
    trace, moe = base.trace_root(tmp_path), moe_root(tmp_path)
    assert 'VLLM_EXL3_TF_PROMPT_ALIAS=1' in text
    with pytest.raises(RuntimeError, match='lacks VLLM_EXL3_TF_PROMPT_ALIAS=1'):
        source(tmp_path / 'a', text.replace('VLLM_EXL3_TF_PROMPT_ALIAS=1', 'VLLM_EXL3_TF_PROMPT_ALIAS=0', 1),
               PF1C_TF_TRANSIENT='0')
    found = re.search(r'-v \S*/vllm_exl3_moe/tf_prompt\.py:ro ?', text)
    assert found is not None
    without = text[:found.start()] + text[found.end():]
    source(tmp_path / 'b', without, base.CONFIG, PF1_TRACE='1', PF1_TRACE_RUN='x', PF1_TRACE_ROOT=str(trace))
    with pytest.raises(RuntimeError, match='exactly three production MoE targets'):
        source(tmp_path / 'b', without, PF1_TRACE='1', PF1_TRACE_RUN='x', PF1_TRACE_ROOT=str(trace), PF1C_ROOT=str(moe))
