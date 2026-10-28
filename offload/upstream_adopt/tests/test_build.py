from __future__ import annotations

import json
from pathlib import Path
import subprocess
from typing import cast

import pytest

from offload.upstream_adopt import build_overlay as b
from offload.upstream_adopt import catalogue


def test_exhaustive_catalogue() -> None:
    body, table = catalogue.generate()
    assert body == (b.HERE/'catalogue.json').read_text()
    assert table == (b.HERE/'PATCH-CATALOGUE.md').read_text()
    inventory = json.loads(body)
    assert len(inventory['changed_paths']) == 72
    assert len(inventory['commits']) == 7
    assert all(row['owners'] for row in inventory['changed_paths'] if row['path'].startswith('docker/files/'))


def test_pinned_rebuild_and_tamper_refusal(tmp_path: Path) -> None:
    tree = tmp_path/'tree'
    manifest = b.build(tree)
    assert manifest == json.loads((b.HERE/'MANIFEST.json').read_text())
    b.verify_tree(b.HERE/'overlay',manifest)
    target = tree/'vllm_exl3_moe/forward.py'
    target.write_text(target.read_text()+'\n# tampered\n')
    with pytest.raises(ValueError,match='overlay hash drift'):
        b.verify_tree(tree,manifest)
    with pytest.raises(ValueError,match='unsafe path'):
        b.safe_path(tree,'../outside')
    with pytest.raises(ValueError,match='must be empty'):
        b.build(tree)


def test_pin_drift_fails_before_patch(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    source = tmp_path/'pins'
    source.mkdir()
    lock = cast(b.Lock,json.loads((b.HERE/'inputs.lock.json').read_text()))
    lock['files'] = {'repo:offload/upstream_adopt/catalogue.py':'0'*64}
    (source/'inputs.lock.json').write_text(json.dumps(lock))
    monkeypatch.setattr(b,'HERE',source)
    with pytest.raises(ValueError,match='input hash drift'):
        b.verify_inputs()


def test_no_duplicate_container_mounts() -> None:
    manifest = cast(b.Manifest,json.loads((b.HERE/'MANIFEST.json').read_text()))
    pairs = b.mounts(manifest)
    assert len({dest for _,dest in pairs}) == len(pairs)
    assert all('upstream_adopt' in source for source,_ in pairs)
    # Every K1 file, including model_runner, must retain its exact bytes.
    k1 = cast(b.K1Manifest,json.loads((b.ROOT/'offload/kvpack/MANIFEST.json').read_text()))
    for entry in k1['files']:
        assert b.sha(b.HERE/'overlay'/entry['path']) == entry['overlay_sha256']


def test_control_keeps_k1_engram_root_and_remote_sources() -> None:
    from offload.upstream_adopt.make_configs import CFG, render
    def env(path: Path) -> list[str]:
        return subprocess.check_output(['bash','-eu','-c','source "$1"; echo "$EXP_DOCKER_ENV"','up1',str(path)],text=True).split()
    baseline = env(CFG/'D-kv6-async-pack.env')
    control = env(CFG/'D-kv6-async-up1-control.env')
    assert control == baseline + ['VLLM_EXL3_MOE_SPLIT_MAX_TOKENS=0','VLLM_DSV41_VOR_FUSED=0','VLLM_DSV41_REPLAY_SKIP=0']
    assert '/opt/dsv41-up1' not in render('control')
    manifest = json.loads((b.HERE/'MANIFEST.json').read_text())
    assert not any(row['path'].startswith(('offload/engram/', 'offload/transport/'))
                   or row['path'].endswith('/x1_engram_lookahead.py') for row in manifest['files'])
    package = 'offload/dspark/spark/remote_dspark'
    changed = {p.name for p in (b.HERE/'overlay'/package).iterdir() if p.is_file() and p.read_bytes() != (b.ROOT/package/p.name).read_bytes()}
    assert changed == {'gather.py', 'roundcore.py', 'round.py', 'speculator.py'}  # only tap-aware0075; round.py includes merged S12 F3


def test_configs_preserve_async_stack_and_isolate_flags() -> None:
    from offload.upstream_adopt.make_configs import CFG, PROFILES, render
    assert set(PROFILES) == {'control','split','vor','replay','k5','dual-ring','cpu-pin','core','all'}
    assert {p.stem.removeprefix('D-kv6-async-up1-') for p in CFG.glob('D-kv6-async-up1-*.env')} == set(PROFILES)
    assert not (b.HERE/'so1.lock.json').exists()
    for profile, defaults in PROFILES.items():
        path = CFG/f'D-kv6-async-up1-{profile}.env'
        assert path.read_text() == render(profile)
        subprocess.run(['bash','-n',str(path)],check=True)
        script = 'source "$1"; printf "%s\\n" "$EXP_DOCKER_ENV" "$OVERLAYS" "$KV_BYTES_OVERRIDE" "${SPEC_OVERRIDE-}"'
        result = subprocess.run(['bash','-eu','-c',script,'up1',str(path)],text=True,capture_output=True,check=True)
        env, overlays, kv, spec = result.stdout.splitlines()
        assert kv == '6442450944'
        for item in ('VLLM_DSPARK_REMOTE_POLICY=capacity','VLLM_DSPARK_REMOTE_ASYNC=1',
                     'VLLM_DSPARK_REMOTE_ASYNC_PROXY=process','VLLM_DSPARK_REMOTE_ASYNC_SPIN_US=50',
                     'VLLM_DSPARK_REMOTE_ASYNC_NAP_US=20','VLLM_ENGRAM_REMOTE_REQUIRED=1',
                     'VLLM_ENGRAM_REMOTE_ROOT=/root/.cache/dsv41-offload'):
            assert item in env.split()
        destinations = [x.split('=',1)[1] for x in overlays.split()]
        assert len(set(destinations)) == len(destinations)
        assert json.loads(spec)['num_speculative_tokens_per_batch_size'] == [[1,4,5]] if defaults.get('K5')=='1' else spec==''
        overridden = subprocess.run(['bash','-eu','-c',
            'UP1_SPLIT=0 UP1_VOR=0 UP1_REPLAY=0 UP1_K5=0 UP1_DUAL_RING=0 UP1_CPU_PIN=0; '+script,
            'up1',str(path)],text=True,capture_output=True,check=True)
        assert 'VLLM_DSV41_REPLAY_SKIP=0' in overridden.stdout
        bad = subprocess.run(['bash','-eu','-c','UP1_SPLIT=4096; source "$1"','up1',str(path)],capture_output=True,text=True)
        assert bad.returncode != 0 and 'invalid UP1_SPLIT' in bad.stderr


def test_no_hidden_installed_source_delta() -> None:
    result = catalogue.audit_image()
    assert result == json.loads((b.HERE/'image-audit.json').read_text())
    assert result['residual'] == []
    assert len(result['identical_files']) == 40


def test_live_hash_checker_refuses_wrong_bytes(tmp_path: Path) -> None:
    from offload.upstream_adopt.verify_runtime import verify
    entry = {'path':'vllm/test.py','sha256':b.sha(b.HERE/'__init__.py')}
    target = tmp_path/'usr/local/lib/python3.12/dist-packages/vllm/test.py'
    target.parent.mkdir(parents=True)
    target.write_bytes((b.HERE/'__init__.py').read_bytes())
    assert verify({'files':[entry]},tmp_path) == 1
    target.write_bytes(b'wrong')
    with pytest.raises(ValueError,match='runtime hash mismatch'):
        verify({'files':[entry]},tmp_path)
    source = subprocess.check_output([__import__('sys').executable,str(b.HERE/'verify_runtime.py'),'--emit'],text=True)
    compile(source,'embedded-runtime-check','exec')
    assert 'UP1_PINNED_EXPECTATIONS = {' in source


def test_runtime_checker_covers_every_retained_mount(tmp_path: Path) -> None:
    from offload.upstream_adopt.verify_runtime import RuntimeManifest, verify
    manifest = cast(RuntimeManifest, json.loads((b.HERE/'MANIFEST.json').read_text()))
    dist = tmp_path/'usr/local/lib/python3.12/dist-packages'
    for entry in manifest['files']:
        name = entry['path']
        target = (dist/name if name.startswith(('vllm/', 'vllm_exl3_moe/')) else
                  dist/'vllm/v1/worker/gpu/spec_decode/remote_dspark'/Path(name).name)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((b.HERE/'overlay'/name).read_bytes())
    assert verify(manifest, tmp_path) == len(manifest['files']) == 42
    with pytest.raises(ValueError, match='unmounted manifest path'):
        verify({'files':[{'path':'offload/engram/spark/engram_remote.cpp','sha256':'0'*64}]}, tmp_path)


def test_config_checker_refuses_held_arm(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import sys
    from offload.upstream_adopt import make_configs as configs
    for path in configs.CFG.glob('D-kv6-async-*.env'):
        (tmp_path/path.name).write_bytes(path.read_bytes())
    (tmp_path/'D-kv6-async-up1-core-so1.env').write_text('# obsolete held arm\n')
    monkeypatch.setattr(configs, 'CFG', tmp_path)
    monkeypatch.setattr(sys, 'argv', ['make_configs', '--check'])
    with pytest.raises(ValueError, match='obsolete or held UP1 configs.*core-so1'):
        configs.main()
