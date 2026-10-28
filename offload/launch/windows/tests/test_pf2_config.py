"""PF2-lowkv.env against production's real 21bed31 dsv41.env; CPU only, no service or remote call."""
from __future__ import annotations

import shlex
from pathlib import Path

import pytest

from offload.launch.windows import pf2_window as p
from offload.launch.windows import sp1_window as s
from offload.launch.windows.tests import test_sp1_config as base
from offload.pf2_lowkv import arith

ROOT = base.ROOT


def runner_env(kv: int) -> dict[str, str]:      # what PF2Live.start passes through pf1_window.Live.start + arm_env
    return {'PF1_CAP': '524288', 'PF1_TRACE': '0', 'PF1_EAGER': '0', 'PF2_KV_BYTES': str(kv)}


@pytest.mark.parametrize('kv', [arith.H_KV, arith.LOW_KV])
def test_arm_is_pf1_off_524_plus_kv_and_sp1_profiler_only(tmp_path: Path, kv: int) -> None:
    off = base.source(tmp_path, 'offload/launch/configs/PF1-window.env', PF1_CAP='524288')
    got = base.source(tmp_path, p.CONFIG, **runner_env(kv))
    for key in ('MAX_LEN_OVERRIDE', 'EXP_DOCKER_ENV', 'OVERLAYS'):
        assert got[key] == off[key]
    assert off['KV_BYTES_OVERRIDE'] == '6442450944' and got['KV_BYTES_OVERRIDE'] == str(kv)
    assert got['MAX_LEN_OVERRIDE'] == '524288' and got['OVERLAYS'] == '' and off['EXP_EXTRA_ARGS'] == ''
    assert got['EXTRA_DOCKER_ARGS'] == off['EXTRA_DOCKER_ARGS'] + ' -v $HOME/dsv41-offload-run/sp1-profile:/sp1-profile:rw'
    assert shlex.split(got['EXP_EXTRA_ARGS']) == ['--profiler-config', s.PROFILER_JSON]
    assert got['LABEL'] == f'PF2-lowkv-524288-kv{kv}'


def test_profiler_json_is_sp1s_byte_for_byte() -> None:
    assert f"SP1_PROFILER_CONFIG='{s.PROFILER_JSON}'" in (ROOT / p.CONFIG).read_text()
    assert f"SP1_PROFILER_CONFIG='{s.PROFILER_JSON}'" in (ROOT / s.CONFIG).read_text()


@pytest.mark.parametrize('env,message', [
    ({'PF2_KV_BYTES': ''}, 'must be set'),
    ({'PF2_KV_BYTES': '3G'}, 'must be set'),
    ({'PF2_KV_BYTES': str(2**30)}, 'outside'),
    ({'PF2_KV_BYTES': str(7 * 2**30)}, 'outside'),
    ({'PF2_KV_BYTES': str(3 * 2**30 + 4096)}, 'whole MiB'),
    ({'PF1_CAP': '327680'}, '524,288 OFF arm'),
    ({'PF1_TRACE': '1'}, '524,288 OFF arm'),
    ({'PF1_EAGER': '1'}, '524,288 OFF arm'),
    ({'PF1_TRACE_RUN': 'x'}, 'inherited PF1 trace'),
    ({'SP1_PROFILER_CONFIG': '{}'}, 'inherited PF1 trace or SP1 profiler'),
    ({'KV_BYTES_OVERRIDE': '1'}, 'inherited KV_BYTES_OVERRIDE'),
    ({'EXP_EXTRA_ARGS': '--enforce-eager'}, 'inherited EXP_EXTRA_ARGS'),
])
def test_config_refuses_anything_but_the_reviewed_arms(tmp_path: Path, env: dict[str, str], message: str) -> None:
    with pytest.raises(RuntimeError, match=message):
        base.source(tmp_path, p.CONFIG, **{**runner_env(arith.LOW_KV), **env})


def test_pins_cover_the_config_and_its_base() -> None:
    assert p.CONFIG in p.PIN_FILES and 'offload/launch/configs/PF1-window.env' in p.PIN_FILES
    assert all((ROOT / f).is_file() for f in p.PIN_FILES)
