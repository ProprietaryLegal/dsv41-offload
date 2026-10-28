from __future__ import annotations

import ast
from pathlib import Path

import pytest

from offload.tv1 import build_overlay as b, init_smoke


@pytest.mark.parametrize('switches', [('compact',), ('candidate', 'compact')])
def test_actual_module_bindings_at_real_startup_warmup(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch, switches: tuple[str, ...]) -> None:
    monkeypatch.setenv('VLLM_TV1_COMPACT', '1')
    tree = tmp_path / 'overlay'
    b.build(tree, switches)
    result = init_smoke.check(tree)
    assert result['real_warmup_condition'] is True
    assert result['recorded_production_conf_guard'] is True
    assert result['cuda_or_engine_init_qualified'] is False
    # Simulate the original bug without injecting the missing name into globals.
    path = tree / 'vllm/v1/worker/gpu/model_runner.py'
    path.write_text(path.read_text().replace('\nimport os\n', '\n'))
    with pytest.raises(NameError, match="name 'os' is not defined"):
        init_smoke.check(tree)


def test_smoke_rejects_bad_tau_without_cuda(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    tree = tmp_path / 'overlay'
    b.build(tree, ('compact',))
    monkeypatch.setenv('VLLM_TV1_COMPACT', '1')
    for value in ('nan', '0', '1.01', 'invalid'):
        monkeypatch.setenv('VLLM_TV1_COMPACT_TAU', value)
        with pytest.raises(ValueError):
            init_smoke.check(tree)


def test_old_blanket_depth_refusal_is_caught_before_hold(tmp_path: Path) -> None:
    tree = tmp_path / 'overlay'
    b.build(tree, ('compact',))
    path = tree / 'vllm/v1/worker/gpu/model_runner.py'
    path.write_text(path.read_text().replace('or self.lora_config is not None',
        'or self.depth_policy is not None or self.lora_config is not None'))
    with pytest.raises(ValueError, match='rejects the recorded production'):
        init_smoke.check(tree)


@pytest.mark.parametrize('switches', [('candidate',), ('compact',), ('candidate', 'compact')])
def test_overlay_environment_lookups_have_actual_module_import(tmp_path: Path, switches: tuple[str, ...]) -> None:
    tree = tmp_path / 'overlay'
    b.build(tree, switches)
    source = ast.parse((tree / 'vllm/v1/worker/gpu/model_runner.py').read_text())
    assert any(isinstance(n, ast.Import) and any(a.name == 'os' for a in n.names) for n in source.body)
