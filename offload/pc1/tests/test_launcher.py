"""Run actual launcher prefix with fake transport; never start JIT or contact a host."""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess

import pytest

from offload.pc1 import build as b


@pytest.mark.parametrize('arm', [None, *b.ARMS, '8192', 'chunk4096', ''])
def test_launcher_registered_pairs(tmp_path: Path, arm: str | None) -> None:
    # Execute original prefix through its last configuration export, then inspect values.
    source = (b.ROOT / 'offload/launch/serve_exp.sh').read_text().split('JIT="', 1)[0]
    script = tmp_path / 'serve-config.sh'
    # Real location check must see the experiment worktree, not the scratch script location.
    source = source.replace('$(dirname "$0")/../..', str(b.ROOT))
    script.write_text(source + '\nprintf "%s\\n" "$MAX_BATCH" "$EXTRA"\n')
    env = {**os.environ, 'HEAD': 'fake', 'WORKER': 'fake', 'PORT': '8001', 'PC1_REVIEWED': '1'}
    env.pop('PC1_ARM', None)
    if arm is not None:
        env['PC1_ARM'] = arm
    result = subprocess.run(['bash', str(script)], env=env, text=True, capture_output=True)
    if arm not in (None, *b.ARMS):
        assert result.returncode != 0 and 'unregistered' in result.stderr
    else:
        assert result.returncode == 0, result.stderr
        batch, threshold = b.SETTINGS[arm or 'control']
        assert result.stdout.splitlines()[0] == str(batch)
        assert f'--long-prefill-token-threshold {threshold} ' in result.stdout


@pytest.mark.parametrize('field,value', [('PC1_REVIEWED', '0'), ('PORT', '8000')])
def test_unreviewed_or_production_port_refuses(tmp_path: Path, field: str, value: str) -> None:
    source = (b.ROOT / 'offload/launch/serve_exp.sh').read_text().split('JIT="', 1)[0]
    env = {**os.environ, 'HEAD': 'fake', 'WORKER': 'fake', 'PORT': '8001',
           'PC1_REVIEWED': '1', 'PC1_ARM': 'chunk2048', field: value}
    result = subprocess.run(['bash', '-c', source], env=env, text=True, capture_output=True)
    assert result.returncode != 0 and 'reviewed experiment' in result.stderr


def test_production_inputs_and_source_pins() -> None:
    lock = b.verify_inputs()
    assert b.source_pins('control')
    assert all(not p.endswith('.c') for p in b.source_pins('control'))
    assert all(b.source_pins(a) == b.source_pins('control') for a in b.ARMS)
    for name in ('serve.sh', 'start.sh', 'stop.sh', 'dsv41.env'):
        # offload-x86host intentionally carries historical launch files. Preserve
        # those bytes; live production is reconciled against the separate pin.
        initial = subprocess.check_output(['git', 'show', f'5cce699:{name}'], cwd=b.ROOT)
        assert (b.ROOT / name).read_bytes() == initial
    assert json.loads((b.HERE / 'inputs.json').read_text()) == lock
