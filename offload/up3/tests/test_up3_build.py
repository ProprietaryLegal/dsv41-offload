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
        assert rows[0] == control[0] + ''.join(f' {k}={v}' for k, v in b.ENV[arm].items())
        assert rows[3] == (b.SPEC if arm == 'k5' else '')
        if arm == 'replay':
            continue
        assert rows[1] == control[1]
        assert len(rows[2].split()) == len(b.FILES.get(arm, ()))
        assert not any(s in rows[2] for s in ('replay', 'scheduler', 'remote_dspark'))
    # Replay: four per-file mounts plus ONE whole remote_dspark directory, in place of production's directory mount.
    remote = '$HOME/dsv41-offload-prod/dspark/remote_dspark:' + b.DIST + b.REMOTE.rstrip('/') + ':ro'
    replay = values['replay']
    assert control[1].split().count(remote) == 1
    assert replay[1].split() == [a for i, a in enumerate(control[1].split())
                                 if a != remote and not (a == '-v' and control[1].split()[i + 1] == remote)]
    pairs = dict(pair.split('=', 1) for pair in replay[2].split())
    assert pairs == {**{'offload/up3/overlay/replay/' + name: b.DIST + name for name in b.FILES['replay'] if not name.startswith(b.REMOTE)},
                     'offload/up3/overlay/replay/' + b.REMOTE.rstrip('/'): b.DIST + b.REMOTE.rstrip('/')}
    assert len(pairs) == 5
    bad = subprocess.run(['bash', '-eu', '-c', 'UP3_ARM=core; source offload/up3/window.env'], cwd=b.ROOT, capture_output=True)
    assert bad.returncode != 0


def test_k5_noncausal_block_preserved() -> None:
    source = (b.IMAGE / 'vllm/v1/attention/backends/mla/sparse_swa.py').read_text()
    vor = (b.HERE / 'overlay/vor/vllm/v1/attention/backends/mla/sparse_swa.py').read_text()
    def block(text: str) -> str:
        return text.split('            if non_causal:', 1)[1].split('                _COMPUTE_SWA_INDICES_AND_LENS_KERNEL(', 1)[0].rsplit('            el', 1)[0]
    assert block(source) == block(vor)
    assert json.loads(b.SPEC)['num_speculative_tokens_per_batch_size'] == [[1, 4, 5]]


def test_replay_arm_is_production_plus_reviewed_0075() -> None:
    manifest = b.verify('replay')
    files = manifest['files']
    assert len(files) == 24 and set(files) == set(b.FILES['replay'])
    remote = sorted(name.removeprefix(b.REMOTE) for name in files if name.startswith(b.REMOTE))
    assert remote == sorted((*b.REMOTE_PACKAGE, 'suffix_provenance.py'))
    up1 = b.ROOT / 'offload/upstream_adopt/overlay'
    # The measured 2026-10-25 UP1 bytes for every 0075-only file; model_state.py and speculator.py differ from UP1 only
    # by the dormant hunks UP1 also carried (0067; the D2 capture hook production never had).
    same = {'vllm/v1/core/sched/scheduler.py': 'vllm/v1/core/sched/scheduler.py',
            'vllm/v1/worker/gpu/input_batch.py': 'vllm/v1/worker/gpu/input_batch.py',
            'vllm/models/deepseek_v41/replay_skip.py': 'vllm/models/deepseek_v41/replay_skip.py',
            **{b.REMOTE + n: 'offload/dspark/spark/remote_dspark/' + n
               for n in ('gather.py', 'roundcore.py', 'round.py', 'suffix_provenance.py')}}
    for name, source in same.items():
        assert files[name] == b.digest((up1 / source).read_bytes()), name
    untouched = set(b.REMOTE_PACKAGE) - {'gather.py', 'roundcore.py', 'round.py', 'speculator.py'}
    for n in untouched:
        assert files[b.REMOTE + n] == b.digest(b.production_file('offload/dspark/spark/remote_dspark/' + n)), n
    for n in ('speculator.py',):
        assert files[b.REMOTE + n] != b.digest(b.production_file('offload/dspark/spark/remote_dspark/' + n))
    up2 = json.loads((b.ROOT / 'offload/upstream_adopt/MANIFEST.json').read_text())
    assert [e['sha256'] for e in up2['files'] if e['path'] == 'vllm/v1/core/sched/scheduler.py'] == [
        files['vllm/v1/core/sched/scheduler.py']]


def test_replay_refuses_inexact_patch(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    image = tmp_path / 'image'
    for name in b.REPLAY_IMAGE:
        (image / name).parent.mkdir(parents=True, exist_ok=True)
        (image / name).write_bytes((b.IMAGE / name).read_bytes())
    target = image / 'vllm/v1/core/sched/scheduler.py'
    target.write_text(target.read_text().replace('\n', ' \n'))  # every context line drifts
    monkeypatch.setattr(b, 'IMAGE', image)
    with pytest.raises(ValueError, match='not exact'):
        b.replay_tree(tmp_path / 'out')
