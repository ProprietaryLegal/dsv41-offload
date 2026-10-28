"""CPU-only: the 0085b overlay is a fresh, exact port of the donor patch onto the v5 image and collides with no other mount."""
from __future__ import annotations

import difflib
import json
import os
import re
import subprocess
import tempfile
from collections import Counter
from pathlib import Path, PurePosixPath

import pytest

from offload.upx import build_0085b as b
from offload.upx.tests.test_configs import source

K1_MANIFEST = b.ROOT / 'offload/kvpack/MANIFEST.json'
K1_PROMOTION = b.ROOT / 'offload/promote/k1/dsv41.env.327680.patch'
# Other lanes' branches, read from git objects at the commits this lane composed against (never edited, never merged).
UPFIX_COMMIT = '38c66e6'     # offload/upfix: R-UPF 0091 overlay + MANIFEST-0091.json, top-k overlay
UP3_COMMIT = 'ac23d4f'       # offload/up3: R-UP3 arms incl. the Round 8 CPU-pin startup receipt


def git_show(commit: str, path: str) -> str:
    result = subprocess.run(['git', 'show', f'{commit}:{path}'], cwd=b.ROOT, capture_output=True, text=True)
    if result.returncode != 0:
        pytest.skip(f'git object {commit}:{path} unavailable in this checkout ({result.stderr.strip()})')
    return result.stdout


def changed_lines(before: str, after: str) -> tuple[Counter[str], Counter[str]]:
    plus: Counter[str] = Counter()
    minus: Counter[str] = Counter()
    for line in difflib.unified_diff(before.splitlines(), after.splitlines(), lineterm='', n=0):
        if line.startswith('+') and not line.startswith('+++'):
            plus[line[1:]] += 1
        elif line.startswith('-') and not line.startswith('---'):
            minus[line[1:]] += 1
    return plus, minus


def donor_added(rel: str) -> Counter[str]:
    section = b.sections(b.donor_bytes().decode())[rel]
    return Counter(line[1:] for line in section.splitlines() if line.startswith('+') and not line.startswith('+++'))


def test_committed_overlay_port_patch_and_manifest_are_a_fresh_build() -> None:
    b.check()


def test_manifest_pins_base_overlay_patch_and_port() -> None:
    manifest = json.loads(b.MANIFEST.read_text())
    assert manifest['schema'] == b.SCHEMA and manifest['donor'] == b.DONOR
    assert manifest['patch_sha256'] == b.sha256(b.PATCH.read_bytes()) == b.PATCH_SHA256
    assert manifest['port_patch_sha256'] == b.sha256(b.PORT_PATCH.read_bytes())
    assert [row['path'] for row in manifest['files']] == list(b.FILES)
    for row in manifest['files']:
        assert row['base_sha256'] == b.sha256((b.IMAGE_SRC / row['path']).read_bytes())
        assert row['overlay_sha256'] == b.sha256((b.ROOT / row['overlay']).read_bytes()) != row['base_sha256']
        assert row['container_path'] == b.CONTAINER + row['path']


@pytest.mark.parametrize('rel', b.FILES)
def test_overlay_adds_exactly_the_donor_lines_and_removes_none(rel: str) -> None:
    plus, minus = changed_lines((b.IMAGE_SRC / rel).read_text(), (b.OVERLAY / rel).read_text())
    assert not minus, f'{rel}: the port removed image lines {sorted(minus)[:3]}'
    assert plus == donor_added(rel)


def test_core_applies_with_zero_fuzz_and_zero_offset() -> None:
    section = b.sections(b.donor_bytes().decode())[b.CORE]
    with tempfile.TemporaryDirectory(dir=os.environ.get('TMPDIR')) as tmp:
        target = Path(tmp) / b.CORE
        target.parent.mkdir(parents=True)
        target.write_bytes((b.IMAGE_SRC / b.CORE).read_bytes())
        result = subprocess.run(['patch', '-p1', '--batch', '--fuzz=0', '-r', '-'], cwd=tmp, input=section,
                                text=True, capture_output=True)
        assert result.returncode == 0, result.stdout + result.stderr
        assert 'offset' not in result.stdout and 'fuzz' not in result.stdout, result.stdout
        assert target.read_bytes() == (b.OVERLAY / b.CORE).read_bytes()


def test_reverse_port_patch_restores_the_image_bytes() -> None:
    """Stripping exactly the port patch from the overlay gives the image files back, byte for byte."""
    with tempfile.TemporaryDirectory(dir=os.environ.get('TMPDIR')) as tmp:
        for rel in b.FILES:
            target = Path(tmp) / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes((b.OVERLAY / rel).read_bytes())
        result = subprocess.run(['patch', '-p1', '-R', '--batch', '--fuzz=0', '-r', '-'], cwd=tmp,
                                input=b.PORT_PATCH.read_text(), text=True, capture_output=True)
        assert result.returncode == 0, result.stdout + result.stderr
        for rel in b.FILES:
            assert (Path(tmp) / rel).read_bytes() == (b.IMAGE_SRC / rel).read_bytes(), rel


def test_worker_hunks_land_at_the_v5_positions_equivalent_to_the_donor() -> None:
    text = (b.OVERLAY / b.WORKER).read_text()
    lines = text.splitlines()
    logger = lines.index('logger = init_logger(__name__)')
    assert lines[logger + 1].startswith('# dsv41 patch 0085b: after at least this many prefill tokens')
    assert lines[logger + 10] == '_DSV41_RELEASE_COUNT = _DSV41_RELEASE_TOKENS > 0 or bool(_DSV41_RELEASE_FILE)'
    assert lines[logger + 11: logger + 14] == ['', '', lines[logger + 13]]
    assert lines[logger + 13].startswith('def _num_workspace_lanes(')
    count = lines.index('        if _DSV41_RELEASE_COUNT and forward_pass:')
    assert lines[count - 3:count] == ['                comm_postprocess=comm_postprocess,', '            )', '']
    assert lines[count + 5:count + 7] == ['', '        with self.annotate_profile(scheduler_output):']
    # the count runs inside execute_model, before the model runner, on every rank (same scheduler output on both)
    execute = lines.index('    def execute_model(')
    assert execute < count < lines.index('    def take_draft_token_ids(self) -> DraftTokenIds | None:')
    method = lines.index('    def dsv41_release_if_due(self) -> None:')
    assert lines.index('    def sample_tokens(') < method < execute


def test_missing_or_duplicate_v5_anchor_is_refused() -> None:
    section = b.sections(b.donor_bytes().decode())[b.WORKER]
    base = (b.IMAGE_SRC / b.WORKER).read_text()
    with pytest.raises(ValueError, match='exactly one v5 anchor'):
        b.build_file(b.WORKER, base.replace('logger = init_logger(__name__)\n', 'logger = None\n').encode(), section)
    doubled = base.replace('        with self.annotate_profile(scheduler_output):\n',
                           '        with self.annotate_profile(scheduler_output):\n', 1)
    doubled += '\n                comm_postprocess=comm_postprocess,\n            )\n\n        with self.annotate_profile(scheduler_output):\n'
    with pytest.raises(ValueError, match='exactly one v5 anchor'):
        b.build_file(b.WORKER, doubled.encode(), section)


def test_a_core_hunk_that_does_not_apply_fails_loudly() -> None:
    section = b.sections(b.donor_bytes().decode())[b.CORE]
    base = (b.IMAGE_SRC / b.CORE).read_text().replace('            # Notify callbacks waiting for engine to become idle.\n',
                                                     '            # changed\n', 1)
    with pytest.raises(ValueError, match='do not apply'):
        b.build_file(b.CORE, base.encode(), section)


def test_tampered_donor_bytes_are_refused(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    copy = tmp_path / 'p.patch'
    copy.write_bytes(b.PATCH.read_bytes() + b'\n')
    monkeypatch.setattr(b, 'PATCH', copy)
    with pytest.raises(ValueError, match='patch bytes changed'):
        b.build()


def test_overlays_compile() -> None:
    for rel in b.FILES:
        path = b.OVERLAY / rel
        compile(path.read_text(), str(path), 'exec')


# --- no overlap with any other mount: production K1 + remote drafter + Engram, UP1/UP2 (0075), R-UPF 0091 + top-k, UP3 arms

def k1_manifest_targets() -> set[str]:
    rows = json.loads(K1_MANIFEST.read_text())['files']
    assert len(rows) == 13
    assert K1_MANIFEST.read_text() == git_show('596ba1c', 'offload/kvpack/MANIFEST.json'), 'K1 MANIFEST drifted from production'
    return {row['container_path'] for row in rows}


def production_targets() -> set[str]:
    line = next(s for s in K1_PROMOTION.read_text().splitlines() if s.startswith('+OFFLOAD_DOCKER_ARGS='))
    targets = set(re.findall(r':(/[^: ]+?)(?=:ro)', line))
    assert len(targets) == 16, sorted(targets)     # 13 K1 files, remote_dspark dir, spec_decode __init__, /opt/dsv41-ofl
    return targets


def sourced_targets(config: str, env: dict[str, str]) -> set[str]:
    return {t for _, t in (pair.split('=', 1) for pair in source(config, env)['OVERLAYS'].split())}


def upfix_targets() -> set[str]:
    manifest = json.loads(git_show(UPFIX_COMMIT, 'offload/upfix/MANIFEST-0091.json'))
    topk = json.loads(git_show(UPFIX_COMMIT, 'offload/upfix/topk/MANIFEST-topk.json'))
    return {row['container_path'] for row in manifest['files']} | {topk['overlay']['container_path'],
                                                                   '/opt/dsv41-upfix/dsv41_topk.so'}


def up3_targets() -> set[str]:
    out: set[str] = set()
    for arm in ('split', 'vor', 'cpu-pin'):
        manifest = json.loads(git_show(UP3_COMMIT, f'offload/up3/manifests/{arm}.json'))
        out |= {b.CONTAINER + rel for rel in manifest['files']}
    return out


@pytest.mark.parametrize('name,targets', [
    ('K1 MANIFEST (13)', k1_manifest_targets),
    ('production docker args (16)', production_targets),
    ('UP1/UP2 0075 config', lambda: sourced_targets('offload/launch/configs/UP2-0075.env', {})),
    ('UP1 control config', lambda: sourced_targets('offload/launch/configs/D-kv6-async-up1-control.env', {})),
    ('R-UPF 0091 + top-k', upfix_targets),
    ('R-UP3 split/vor/cpu-pin', up3_targets),
])
def test_no_0085b_file_is_mounted_by_another_package_or_under_one(name: str, targets: object) -> None:
    assert callable(targets)
    found = targets()
    assert isinstance(found, set) and found, name
    for rel in b.FILES:
        mine = PurePosixPath(b.CONTAINER + rel)
        for other in map(PurePosixPath, found):
            assert other != mine and other not in mine.parents and mine not in other.parents, (name, rel, other)


def test_engram_meta_path_overlay_targets_another_module() -> None:
    site = (b.ROOT / 'offload/engram/spark/py/sitecustomize.py').read_text()
    assert '_TARGET = "vllm.models.deepseek_v41.common.engram"' in site
    assert 'vllm.v1.engine.core' not in site and 'gpu_worker' not in site
