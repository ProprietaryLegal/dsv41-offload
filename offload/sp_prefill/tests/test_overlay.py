"""R-SP overlay build: pinned bases, OFF byte-identical, ON pinned and narrow, mirrored image code unchanged."""
from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest

from offload.sp_prefill import build_overlay as bo
from offload.sp_prefill.tests import toy

K1_MODEL = bo.K1 / bo.MODEL
HELPER_PATH = bo.HERE / 'overlay' / bo.HELPER
# Files whose code the helper mirrors or calls into; a change to any of them needs a re-review of sp_prefill.py.
MIRRORED = {
    toy.IMAGE / 'vllm/models/deepseek_v41/decoder_replay_layers.py':
        '7c7b51754890dde1374c6fa078e72f5168acd8d1fa04cb26c6250c5706e515eb',  # replay(): DecoderReplayLayers.__call__
    toy.IMAGE / 'vllm/models/deepseek_v41/common/engram.py':
        'bf95ad9a61b0f5f211f00c21e4dc1fd46d2404fcfae54ab53690ec2e951b05e5',  # engram_local(): Engram.forward
    bo.REPO / 'offload/engram/spark/engram.py':
        '0bf3030e370d0f144631f4567cc9da06671a131628a92b49f4553bb45adfa9e7',  # O1 remote-Engram copy of the same
    toy.IMAGE / 'vllm/model_executor/layers/fused_moe/runner/moe_runner.py':
        None,  # moe(): skip_final_all_reduce honoured on the late all-reduce path (checked below, not pinned)
}
TOUCHED = {('DeepseekV4DecoderLayer', 'forward'), ('DeepseekV4DecoderLayer', 'write_kv'),
           ('DeepseekV4Model', '__init__'), ('DeepseekV4Model', 'forward'), ('DeepseekV4Model', '_run_layers'),
           ('DeepseekV4Model', '_collapse'), ('DeepseekV4Model', '_run_replay_layers')}


def test_base_is_the_k1_manifest_and_up1_bytes() -> None:
    pins = json.loads((bo.HERE / 'BASES.json').read_text())
    manifest = {f['path']: f['overlay_sha256'] for f in
                json.loads((bo.REPO / 'offload/kvpack/MANIFEST.json').read_text())['files']}
    assert pins == {bo.MODEL: manifest[bo.MODEL]} == {bo.MODEL: bo.sha(K1_MODEL.read_bytes())}
    up1 = bo.REPO / 'offload/upstream_adopt/overlay' / bo.MODEL
    assert bo.sha(up1.read_bytes()) == pins[bo.MODEL], 'UP1 carries the same model.py'


def test_off_is_k1_on_is_pinned(tmp_path: Path) -> None:
    off = bo.build(tmp_path / 'off', False)
    assert off == {bo.MODEL: bo.sha(K1_MODEL.read_bytes())}
    on = bo.build(tmp_path / 'on')
    assert set(on) == {bo.MODEL, bo.HELPER}
    pins = {p: d for d, p in (line.split() for line in (bo.HERE / 'OVERLAY.sha256').read_text().splitlines())}
    assert pins == on, 'OVERLAY.sha256 pins the reviewed ON output'
    for path in on:
        compile((tmp_path / 'on' / path).read_text(), path, 'exec')
    with pytest.raises(ValueError, match='must not exist'):
        bo.build(tmp_path / 'on')
    for forbidden in (bo.K1 / 'x', bo.HERE / 'out'):
        with pytest.raises(ValueError, match='must not modify'):
            bo.build(forbidden)
    assert ' '.join(sorted(on)) in ' '.join(sorted(
        pair.split('=')[1].removeprefix(bo.CONTAINER_PREFIX) for pair in bo.overlays_arg(tmp_path / 'on', on).split()))


def test_drift_refuses(tmp_path: Path) -> None:
    text = K1_MODEL.read_text()
    with pytest.raises(ValueError, match='found 0 times'):
        bo.overlay_model(text.replace('        self.attn.forward_kv(positions, x)\n',
                                      '        self.attn.forward_kv(positions, x )\n'))
    fake = tmp_path / 'k1'
    (fake / bo.MODEL).parent.mkdir(parents=True)
    (fake / bo.MODEL).write_text(text + '\n')
    with pytest.raises(ValueError, match='base drift'):
        bo.build(tmp_path / 'out', root=fake)


def _methods(source: str) -> dict[tuple[str, str], str]:
    out: dict[tuple[str, str], str] = {}
    for node in ast.parse(source).body:
        if isinstance(node, ast.ClassDef):
            for item in node.body:
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    out[(node.name, item.name)] = ast.dump(item)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            out[('', node.name)] = ast.dump(node)
    return out


def test_only_the_dataflow_methods_change() -> None:
    k1 = K1_MODEL.read_text()
    on = bo.overlay_model(k1)
    before, after = _methods(k1), _methods(on)
    assert set(before) == set(after)
    changed = {key for key in before if before[key] != after[key]}
    assert changed == TOUCHED
    imports = [ast.dump(n) for n in ast.parse(on).body if isinstance(n, (ast.Import, ast.ImportFrom))]
    k1_imports = [ast.dump(n) for n in ast.parse(k1).body if isinstance(n, (ast.Import, ast.ImportFrom))]
    assert len(imports) == len(k1_imports) + 1
    assert sum(line.startswith('+') for line in _diff(k1, on)) < 140


def _diff(a: str, b: str) -> list[str]:
    import difflib
    return [line for line in difflib.unified_diff(a.splitlines(), b.splitlines(), lineterm='', n=0)
            if line[:1] in '+-' and not line.startswith(('+++', '---'))]


def test_mirrored_image_code_is_pinned() -> None:
    for path, digest in MIRRORED.items():
        if digest is not None:
            assert bo.sha(path.read_bytes()) == digest, f'{path} changed: re-review sp_prefill.py'


def _engram_launch(source: str, function: str) -> str:
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == function:
            for call in ast.walk(node):
                if (isinstance(call, ast.Call) and isinstance(call.func, ast.Subscript)
                        and '_fused_engram_post_wkv_kernel' in ast.unparse(call.func.value)):
                    return ast.unparse(call).replace('module._fused_engram_post_wkv_kernel',
                                                     '_fused_engram_post_wkv_kernel')
    raise AssertionError(f'no Engram kernel launch in {function}')


@pytest.mark.parametrize('engram_py', [p for p in MIRRORED if p.name == 'engram.py'])
def test_engram_launch_is_engram_forward_s(engram_py: Path) -> None:
    """engram_local's kernel call is Engram.forward's, argument for argument (self -> engram)."""
    ours = _engram_launch(HELPER_PATH.read_text(), '_triton_engram_post')
    theirs = _engram_launch(engram_py.read_text(), 'forward').replace('self.', 'engram.')
    assert ours == theirs


def test_runner_skip_flag_is_the_late_all_reduce_switch() -> None:
    runner = (toy.IMAGE / 'vllm/model_executor/layers/fused_moe/runner/moe_runner.py').read_text()
    tree = ast.parse(runner)
    final = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == '_maybe_reduce_final_output')
    text = ast.unparse(final)
    assert 'not self.moe_config.skip_final_all_reduce' in text and 'tensor_model_parallel_all_reduce(states)' in text
    for name in ('_maybe_reduce_shared_expert_output', '_maybe_reduce_routed_output_before_transform'):
        body = ast.unparse(next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == name))
        assert 'fused_output_is_reduced' in body, f'{name}: reduces only an already-reduced or latent output'


def test_helper_loads_without_vllm_and_defaults_off() -> None:
    helper = toy.load_helper()
    assert helper.ENABLED is False
    assert 'vllm' not in {n.split('.')[0] for n in _top_imports(HELPER_PATH.read_text())}


def _top_imports(source: str) -> list[str]:
    names: list[str] = []
    for node in ast.parse(source).body:
        if isinstance(node, ast.Import):
            names += [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.append(node.module)
    return names
