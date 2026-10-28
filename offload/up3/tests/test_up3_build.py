from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from offload.up3 import build as b


def test_all_arms_exact_and_reproducible() -> None:
    for arm in b.ARMS:
        manifest = b.verify(arm)
        assert set(manifest['files']) == set(b.FILES.get(arm, ()))
        assert manifest['production'] == b.PRODUCTION
    assert b.verify('control')['files'] == {}
    # Every existing K1 file remains unchanged; all modified files were image-resident in production.
    mounted = set(json.loads(b.production_file('offload/kvpack/MANIFEST.json'))['files'][i]['path']
                  for i in range(len(json.loads(b.production_file('offload/kvpack/MANIFEST.json'))['files'])))
    assert not mounted.intersection(name for names in b.FILES.values() for name in names)


def test_drift_is_refused(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (tmp_path / 'inputs.json').write_bytes((b.HERE / 'inputs.json').read_bytes())
    monkeypatch.setattr(b, 'HERE', tmp_path)
    with pytest.raises(FileNotFoundError):
        b.verify_inputs()


def test_config_preserves_current_production(tmp_path: Path) -> None:
    prod = tmp_path / 'dsv41.env'
    prod.write_bytes(b.production_file('dsv41.env'))
    script = 'UP3_ARM=$1 UP3_PROD_ENV=$2; source offload/up3/window.env; printf "%s\\n" "$EXP_DOCKER_ENV" "$EXTRA_DOCKER_ARGS" "$OVERLAYS" "${SPEC_OVERRIDE-}"'
    values = {}
    for arm in b.ARMS:
        rows = subprocess.check_output(['bash', '-eu', '-c', script, 'up3', arm, str(prod)], cwd=b.ROOT, text=True).splitlines()
        values[arm] = rows
    control = values['control']
    assert control[2:] == ['', '']
    assert 'VLLM_ENGRAM_REMOTE_REQUIRED=0' in control[0]
    assert 'VLLM_DSPARK_REMOTE_ASYNC_RESPAWNS=3' in control[0]
    for arm, rows in values.items():
        assert rows[1] == control[1]
        assert rows[0] == control[0] + ''.join(f' {k}={v}' for k, v in b.ENV[arm].items())
        assert rows[3] == (b.SPEC if arm in ('k5', 'fixed2') else '')
        assert len(rows[2].split()) == len(b.FILES.get(arm, ()))
        assert not any(s in rows[2] for s in ('replay', 'scheduler', 'remote_dspark'))
    bad = subprocess.run(['bash', '-eu', '-c', 'UP3_ARM=core; source offload/up3/window.env'], cwd=b.ROOT, capture_output=True)
    assert bad.returncode != 0


def test_k5_noncausal_block_preserved() -> None:
    source = (b.IMAGE / 'vllm/v1/attention/backends/mla/sparse_swa.py').read_text()
    vor = (b.HERE / 'overlay/vor/vllm/v1/attention/backends/mla/sparse_swa.py').read_text()
    def block(text: str) -> str:
        return text.split('            if non_causal:', 1)[1].split('                _COMPUTE_SWA_INDICES_AND_LENS_KERNEL(', 1)[0].rsplit('            el', 1)[0]
    assert block(source) == block(vor)
    assert json.loads(b.SPEC)['num_speculative_tokens_per_batch_size'] == [[1, 4, 5]]
