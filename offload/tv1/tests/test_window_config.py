from __future__ import annotations

import json
from pathlib import Path
from typing import cast

import pytest

from offload.launch.windows import pf1_window as pf
from offload.launch.windows import sp1_window as sp
from offload.launch.windows import tv1_window as tv
from offload.launch.windows.tests.test_pf1_window import live_identity, prod_identity
from offload.tv1 import build_overlay as b
from offload.tv1.window_config import FLAGS, compose
from offload.prefill_maxlen.evidence import Json, record


@pytest.mark.parametrize('role', ['off', 'candidate', 'compact'])
def test_composition_and_runtime_reject_drift(tmp_path: Path, role: str) -> None:
    tree = f'offload/tv1/generated-test/{role}'
    target = tmp_path / tree
    m = b.build(target, () if role == 'off' else (role,))
    target.with_suffix('.manifest.json').write_text(json.dumps(m))
    raw = '-v $HOME/runner:' + pf.DIST + 'vllm/v1/worker/gpu/model_runner.py:ro -v $HOME/profiles:/sp1-profile:rw'
    kept, mounted, env = compose(role, tree, raw, 'VLLM_ENGRAM_REMOTE_REQUIRED=0', tmp_path)
    assert '/sp1-profile:rw' in kept and '/runner:' not in kept
    assert mounted.count('model_runner.py=') == 1
    assert env.endswith(' '.join(f'{k}={v}' for k,v in FLAGS[role].items()))
    arm = tv.Arm('test', 327680, 0, role=role)
    prod, live = prod_identity(), live_identity(arm, 'r')
    record(live['env'])['VLLM_ENGRAM_REMOTE_REQUIRED'] = '0'
    record(live['mounted']).update({pf.DIST + e['path']: e['sha256'] for e in m['files']})
    record(live['env']).update(cast(dict[str, Json], FLAGS[role]))
    live['argv'] = str(live['argv']) + ' --profiler-config ' + __import__('shlex').quote(sp.PROFILER_JSON)
    tv.runtime(prod, live, arm, m, 'r')
    record(live['env'])['VLLM_TV1_CANDIDATE'] = '2'
    with pytest.raises(ValueError, match='environment'):
        tv.runtime(prod, live, arm, m, 'r')
    with pytest.raises(ValueError, match='inherited'):
        compose(role, tree, raw, env, tmp_path)
    with pytest.raises(ValueError, match='parent'):
        compose(role, tree, raw + ' -v $HOME/x:' + pf.DIST.rstrip('/') + ':ro', '', tmp_path)
    with pytest.raises(ValueError, match='duplicate'):
        compose(role, tree, raw + ' -v $HOME/x:' + pf.DIST + 'vllm/v1/worker/gpu/model_runner.py:ro', '', tmp_path)
    with pytest.raises(ValueError, match='switch'):
        other = 'compact' if role != 'compact' else 'candidate'
        compose(other, tree, raw, '', tmp_path)
    (target / m['files'][0]['path']).write_text('# tampered')
    with pytest.raises(ValueError, match='hash'):
        compose(role, tree, raw, '', tmp_path)


def test_s48_restore_counts_distinct_lines() -> None:
    good = ''.join(f'greedy K1-post vs restored: {i} IDENTICAL 300\n' for i in range(6)) + 'PRODUCTION HOLD OK\nCHECK: DONE\n'
    pf.restored('check', 0, good)
    for bad in (good.replace('5 IDENTICAL', '4 IDENTICAL'), good.replace('5 IDENTICAL', '5 DIVERGES'),
                good.replace('greedy K1-post vs restored: 5 IDENTICAL 300\n', ''), good + 'DIVERGES'):
        with pytest.raises(ValueError, match='6/6'):
            pf.restored('check', 0, bad)
