from __future__ import annotations

from pathlib import Path
import subprocess

import pytest

from offload.tv1 import build_overlay as b, candidate_init_smoke as smoke


@pytest.mark.parametrize('switches', [('candidate',), ('candidate', 'compact')])
def test_real_composed_imports_and_binding_without_cuda(tmp_path: Path, switches: tuple[str, ...]) -> None:
    tree = tmp_path / 'overlay'
    b.build(tree, switches)
    receipt = smoke.check(tree)
    assert receipt['classification'] == 'SOURCE/CPU ONLY'
    assert receipt['bound_layers'] == [24, 28, 32, 36]
    assert receipt['current_config_absent'] is True
    assert receipt['synthetic_forward_map'] is True
    assert receipt['dummy_metadata_none'] is True
    assert receipt['cached_lookups'] is True
    assert receipt['cuda_initialized'] is False
    assert receipt['cuda_or_engine_init_qualified'] is False
    assert receipt['imports'] == [{'path': 'vllm_exl3_moe/tv1_' + name,
        'sha256': b.digest(tree / 'vllm_exl3_moe' / ('tv1_' + name))}
        for name in ('candidate.py', 'candidate_cases.py', 'candidate_metadata.py')]


def test_original_config_lookup_refused_without_hold(tmp_path: Path) -> None:
    tree = tmp_path / 'overlay'
    b.build(tree, ('candidate',))
    # Restore precisely the failing accessor, leaving all other helper bytes.
    path = tree / 'vllm_exl3_moe/tv1_candidate_metadata.py'
    path.write_text(path.read_text().replace(
        'from vllm.forward_context import get_forward_context',
        'from vllm.config import get_current_vllm_config').replace(
        'context = get_forward_context().no_compile_layers',
        'context = get_current_vllm_config().compilation_config.static_forward_context'))
    with pytest.raises(ValueError, match='Current vLLM config is not set'):
        smoke.check(tree)


@pytest.mark.parametrize('case', ['missing-helper', 'import-failure', 'unqualified-layer'])
def test_candidate_smoke_cannot_hide_import_or_layer_failure(tmp_path: Path, case: str) -> None:
    tree = tmp_path / 'overlay'
    b.build(tree, ('candidate',))
    if case == 'missing-helper':
        (tree / 'vllm_exl3_moe/tv1_candidate.py').unlink()
    elif case == 'import-failure':
        with (tree / 'vllm_exl3_moe/tv1_candidate_metadata.py').open('a') as f:
            f.write('\nraise RuntimeError("synthetic import failure")\n')
    else:
        path = tree / 'vllm_exl3_moe/tv1_candidate_cases.py'
        path.write_text(path.read_text().replace('LAYERS = (24, 28, 32, 36)', 'LAYERS = (24, 28, 32)'))
    with pytest.raises(ValueError, match='TV1 candidate import/binding smoke failed'):
        smoke.check(tree)


def test_smoke_subprocess_is_bounded_and_cpu_only(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def timeout(argv: list[str], *, env: dict[str, str], text: bool, capture_output: bool,
                timeout: int) -> subprocess.CompletedProcess[str]:
        assert env['CUDA_VISIBLE_DEVICES'] == '' and env['PYTHONDONTWRITEBYTECODE'] == '1'
        assert timeout == 30 and text and capture_output
        raise subprocess.TimeoutExpired(argv, timeout)
    monkeypatch.setattr(subprocess, 'run', timeout)
    with pytest.raises(subprocess.TimeoutExpired):
        smoke.check(tmp_path)
