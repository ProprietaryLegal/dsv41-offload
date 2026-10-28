from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from typing import cast

import pytest

from offload.tv1 import build_overlay as b
from offload.tv1.verify_runtime import Runtime, verify


@pytest.mark.parametrize('switches', [(), ('candidate',), ('compact',), ('candidate', 'compact')])
def test_build_control_hashes_and_runtime(tmp_path: Path, switches: tuple[str, ...]) -> None:
    tree = tmp_path / 'overlay'
    manifest = b.build(tree, switches)
    b.verify(tree, manifest)
    if not switches:
        for name, source in b.BASES.items():
            assert (tree / name).read_bytes() == source.read_bytes()
    root = tmp_path / 'container'
    shutil.copytree(tree, root / b.PREFIX.lstrip('/'))
    runtime = cast(Runtime, {'files': manifest['files']})
    assert verify(runtime, root) == len(manifest['files'])
    first = root / b.PREFIX.lstrip('/') / manifest['files'][0]['path']
    first.write_bytes(first.read_bytes() + b'\n# tampered')
    with pytest.raises(ValueError, match='hash mismatch'):
        verify(runtime, root)
    sidecar = tmp_path / 'manifest.json'
    sidecar.write_text(json.dumps(manifest))
    emitted = subprocess.check_output(['/home/operator/miniconda3/bin/python3', str(b.HERE / 'verify_runtime.py'),
                                      '--emit', '--manifest', str(sidecar)], text=True)
    assert 'TV1_EXPECTED = ' in emitted
    clean = tmp_path / 'clean'
    shutil.copytree(tree, clean / b.PREFIX.lstrip('/'))
    result = subprocess.run(['/home/operator/miniconda3/bin/python3', '-', '--root', str(clean)],
                            input=emitted, text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
    (tree / 'unexpected').write_text('x')
    with pytest.raises(ValueError, match='file set'):
        b.verify(tree, manifest)


def test_bad_switch_and_paths(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match='switch'):
        b.sources(('unknown',))
    with pytest.raises(ValueError, match='unsafe'):
        b.safe(tmp_path, '../escape')
