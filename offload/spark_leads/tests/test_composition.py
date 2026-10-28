from __future__ import annotations
import importlib.util
import json
from pathlib import Path

PKG=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('x1_compose',PKG/'compose_check.py')
assert spec is not None and spec.loader is not None
m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)


def test_real_k1_overlay_composes_without_fuzz_or_backups() -> None:
    expected=json.loads((PKG/'patches/K1-composition.json').read_text())
    got=m.check(expected['k1_ref'],Path('/opt/dsv41/data/src-image'))
    assert got==expected
    assert 'vllm/v1/worker/gpu/model_runner.py' in got['combined_files']
    assert not any(name.endswith(('.orig','.rej')) for name in got['combined_files'])
