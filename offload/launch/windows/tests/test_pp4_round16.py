"""PP4 Round 16: the admission client seam must not load torch/vllm into the async proxy (PP4-85b admission cmd/0109).

0109 ran up2_proxy_stats.py against PID 430, the out-of-process async proxy (rank 0's "through proxy process pid 430"
log line), and it refused: "proxy loaded torch/CUDA library". R15's admission overlay inserted
`from vllm.v1.worker.gpu import dsv41_pp4_admission` at the top of remote_dspark/client.py; the proxy imports client.py,
and importing anything under `vllm` runs vllm/__init__.py -> vllm.env_override -> `import torch`. These tests rebuild
that proxy start on CPU (the real BOOT import sequence under `python -I`, a dist-packages stand-in whose vllm imports
torch exactly as the image's does) and run the real up2_proxy_stats.py against it.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from offload.dspark.probabilistic.pp3 import prepare_verifier as pp3
from offload.dspark.spark.remote_dspark import proxy_proc
from offload.launch.windows import pf1_window as pf
from offload.launch.windows import pp4_admission_prepare as p

SOURCE = Path('/opt/dsv41/data/src-image')
REMOTE = pf.ROOT/'offload/dspark/spark/remote_dspark'
STATS = pf.ROOT/'offload/launch/windows/up2_proxy_stats.py'
needs_image = pytest.mark.skipif(not (SOURCE/pp3.UTILS).is_file(), reason='image source snapshot absent')


@pytest.fixture(scope='module')
def prepared(tmp_path_factory: pytest.TempPathFactory) -> Path:
    out = tmp_path_factory.mktemp('r16')/'prepared'
    p.prepare(SOURCE, out)
    return out


def r15_client(remote: Path) -> None:
    """Rewrite a prepared overlay back to R15's client import (the 0109 condition)."""
    client = remote/'client.py'
    text = client.read_text()
    assert text.count(p.CLIENT_IMPORT) == 1
    client.write_text(text.replace(p.CLIENT_IMPORT, p.IMPORT))
    (remote/p.SEAM).unlink()


def test_image_vllm_import_loads_torch() -> None:
    """The mechanism: importing any vllm submodule runs vllm/__init__ -> vllm.env_override -> import torch."""
    if not (SOURCE/'vllm/env_override.py').is_file():
        pytest.skip('image source snapshot absent')
    assert 'import vllm.env_override' in (SOURCE/'vllm/__init__.py').read_text()
    assert '\nimport torch\n' in (SOURCE/'vllm/env_override.py').read_text()


@needs_image
def test_admission_client_imports_the_package_local_seam(prepared: Path) -> None:
    remote = prepared/'remote_dspark'
    client = (remote/'client.py').read_text()
    assert p.CLIENT_IMPORT in client and p.IMPORT not in client
    assert (remote/p.SEAM).read_bytes() == (prepared/p.HELPER).read_bytes() == \
        (pf.ROOT/'offload/launch/windows/pp4_admission_engine.py').read_bytes()
    # Engine-side seams (runner, verifier) still reach the vllm-installed helper: they run in the engine.
    assert p.IMPORT in (prepared/pp3.RUNNER).read_text() and p.IMPORT in (prepared/pp3.UTILS).read_text()
    receipt = json.loads((prepared/'receipt.json').read_text())
    assert 'remote_dspark/'+p.SEAM in receipt['output_sha256']
    assert not (REMOTE/p.SEAM).exists()  # the ordinary overlay is untouched


@needs_image
def test_static_proxy_closure_reproduces_and_clears_0109(prepared: Path, tmp_path: Path) -> None:
    assert p.proxy_torch_free(REMOTE) == []
    assert p.proxy_torch_free(prepared/'remote_dspark') == []
    assert p.SEAM.removesuffix('.py') in p.proxy_closure(prepared/'remote_dspark')
    r15 = tmp_path/'remote_dspark'
    shutil.copytree(prepared/'remote_dspark', r15)
    r15_client(r15)
    assert p.proxy_torch_free(r15) == ['client.py:10 imports vllm.v1.worker.gpu']


@needs_image
def test_prepare_refuses_an_overlay_that_leaks_into_the_proxy(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(p, 'CLIENT_IMPORT', p.IMPORT)
    with pytest.raises(ValueError, match='load torch/vllm in the async proxy'):
        p.prepare(SOURCE, tmp_path/'prepared')


def site_with(remote: Path, helper: Path, root: Path) -> tuple[Path, Path]:
    """dist-packages stand-in: vllm whose import loads torch (as the image's does), the helper at HELPER, the overlay."""
    site = root/'site'
    vllm = site/'vllm'
    for package in (vllm, vllm/'v1', vllm/'v1/worker', vllm/'v1/worker/gpu', vllm/'v1/worker/gpu/spec_decode'):
        package.mkdir(parents=True, exist_ok=True)
        (package/'__init__.py').write_text('')
    (vllm/'__init__.py').write_text('import torch  # image: vllm/__init__ -> vllm.env_override -> import torch\n')
    shutil.copy(helper, site/p.HELPER)
    pkg = vllm/'v1/worker/gpu/spec_decode/remote_dspark'
    shutil.copytree(remote, pkg, ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    return site, pkg


def boot_proxy_and_check(site: Path, pkg: Path) -> tuple[dict[str, bool], subprocess.CompletedProcess[str]]:
    """The proxy's own BOOT under `python -I` (dist-packages on sys.path), stopped after its imports; then 0109's script."""
    stop = 'from remote_dspark.proxy_proc import main\nsys.exit(main(fd))\n'
    assert proxy_proc.BOOT.endswith(stop)
    boot = (f'import sys; sys.path.insert(0, {str(site)!r})\n' + proxy_proc.BOOT.replace(stop,
            'from remote_dspark.proxy_proc import main\nimport json\n'
            'print(json.dumps({"torch": "torch" in sys.modules, "vllm": "vllm" in sys.modules}), flush=True)\n'
            'sys.stdin.read()\n'))
    env = {k: v for k, v in os.environ.items() if k not in ('PYTHONPATH',)}
    child = subprocess.Popen([sys.executable, '-I', '-c', boot, str(pkg), '0'], stdin=subprocess.PIPE,
                             stdout=subprocess.PIPE, text=True, env=env)
    try:
        assert child.stdout is not None
        line = child.stdout.readline()
        assert line, 'proxy boot failed before its imports completed'
        loaded = json.loads(line)
        check = subprocess.run([sys.executable, '-', '--pid', str(child.pid)], input=STATS.read_text(),
                               capture_output=True, text=True, timeout=120, env=env, check=False)
        return loaded, check
    finally:
        assert child.stdin is not None
        child.stdin.close()
        child.wait(timeout=60)


@needs_image
@pytest.mark.parametrize('variant', ['r15', 'r16', 'ordinary'])
def test_0109_stats_check_against_a_real_proxy_boot(prepared: Path, tmp_path: Path, variant: str) -> None:
    remote = tmp_path/'remote_dspark'
    shutil.copytree(REMOTE if variant == 'ordinary' else prepared/'remote_dspark', remote,
                    ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    if variant == 'r15':
        r15_client(remote)
    loaded, check = boot_proxy_and_check(*site_with(remote, prepared/p.HELPER, tmp_path))
    if variant == 'r15':
        assert loaded == {'torch': True, 'vllm': True}
        assert check.returncode != 0 and 'proxy loaded torch/CUDA library' in check.stderr   # exactly cmd/0109
    else:
        assert loaded == {'torch': False, 'vllm': False}
        # Past the torch/CUDA check; this CPU stand-in has no shared-memory region, so the NEXT check stops it.
        assert check.returncode != 0 and 'proxy loaded torch/CUDA library' not in check.stderr
        assert 'expected one distinct proxy memfd' in check.stderr


def test_0109_pid_is_the_async_proxy() -> None:
    """The PID came from rank 0's speculator log of ProxyProcess.pid, which PF1's log filter parses."""
    text = ' '.join((REMOTE/'speculator.py').read_text().split())
    assert 'through proxy process pid %d; rank-0 " "buffers: %.1f MiB device, %.1f MiB shared region", cfg.policy, cfg.endpoint, rnd.proxy.pid' in text
    assert r'through proxy process pid (\d+)' in pf.LOG_FILTER
    assert 'return self.proc.pid' in (REMOTE/'proxy_proc.py').read_text()
