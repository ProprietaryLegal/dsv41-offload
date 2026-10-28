from __future__ import annotations

import difflib
import json
import re
from collections import Counter
from pathlib import Path, PurePosixPath

import pytest

from offload.upfix import build_0091 as b

K1_PROMOTION = b.ROOT / 'offload/promote/k1/dsv41.env.327680.patch'
UP1_CONTROL = b.ROOT / 'offload/launch/configs/D-kv6-async-up1-control.env'


def k1_production_mounts() -> set[str]:
    line = next(s for s in K1_PROMOTION.read_text().splitlines() if s.startswith('+OFFLOAD_DOCKER_ARGS='))
    return set(re.findall(r':(/usr/local/lib/python3\.12/dist-packages/[^: ]+?)(?=:ro)', line))


def changed_lines(before: str, after: str) -> tuple[Counter[str], Counter[str]]:
    added: Counter[str] = Counter()
    removed: Counter[str] = Counter()
    for line in difflib.unified_diff(before.splitlines(), after.splitlines(), lineterm='', n=0):
        if line.startswith('+') and not line.startswith('+++'):
            added[line[1:]] += 1
        elif line.startswith('-') and not line.startswith('---'):
            removed[line[1:]] += 1
    return added, removed


def test_committed_overlays_and_manifest_are_a_fresh_build() -> None:
    b.check()


def test_manifest_hashes_every_file_of_both_variants() -> None:
    manifest = json.loads(b.MANIFEST.read_text())
    assert manifest['patch_sha256'] == b.sha256(b.PATCH.read_bytes()) == b.PATCH_SHA256
    assert {(row['variant'], row['path']) for row in manifest['files']} == (
        {('image', rel) for rel in b.FILES} | {('0075', b.SCHEDULER)})
    for row in manifest['files']:
        assert row['overlay_sha256'] == b.sha256((b.ROOT / row['overlay']).read_bytes())
        assert row['overlay_sha256'] != row['base_sha256']
        assert row['container_path'] == b.CONTAINER + row['path']
    image = {row['path']: row for row in manifest['files'] if row['variant'] == 'image'}
    for rel, row in image.items():
        assert row['base_sha256'] == b.sha256((b.IMAGE_SRC / rel).read_bytes())


def test_patch_touches_exactly_four_files() -> None:
    assert sorted(b.sections(b.PATCH.read_text())) == sorted(b.FILES)


def test_no_file_is_a_k1_production_mount_or_under_one() -> None:
    mounts = k1_production_mounts()
    assert len(mounts) == 15, sorted(mounts)
    for rel in b.FILES:
        target = PurePosixPath(b.CONTAINER + rel)
        for mount in map(PurePosixPath, mounts):
            assert mount != target and mount not in target.parents, (rel, mount)


def test_0075_variant_composes_onto_the_mounted_upstream_adopt_scheduler() -> None:
    assert f'offload/upstream_adopt/overlay/{b.SCHEDULER}={b.CONTAINER}{b.SCHEDULER}' in UP1_CONTROL.read_text()
    image = (b.IMAGE_SRC / b.SCHEDULER).read_text()
    adopt = b.ADOPT_SCHEDULER.read_text()
    on_image = changed_lines(image, (b.OVERLAY / b.SCHEDULER).read_text())
    on_adopt = changed_lines(adopt, (b.OVERLAY_0075 / b.SCHEDULER).read_text())
    assert on_image == on_adopt
    added, removed = on_adopt
    assert sum(added.values()) == 83 and sum(removed.values()) == 3


def test_spec_reserve_switch_follows_the_os_import_and_defaults_on() -> None:
    for path in (b.OVERLAY / b.SCHEDULER, b.OVERLAY_0075 / b.SCHEDULER):
        text = path.read_text()
        assert text.count(b.ANCHOR) == 1
        tail = text.split(b.ANCHOR, 1)[1].splitlines()[:6]
        assert tail[3] == '_DSV41_SPEC_RESERVE_PER_GROUP = ('
        assert tail[4].strip() == '_dsv41_os.environ.get("VLLM_DSV41_SPEC_RESERVE_PER_GROUP", "1").strip() != "0"'
        assert text.index(b.ANCHOR) < text.index('_DSV41_SPEC_RESERVE_PER_GROUP and self.num_spec_tokens')


@pytest.mark.parametrize('value,expected', [(None, True), ('1', True), (' 1 ', True), ('0', False), (' 0', False)])
def test_spec_reserve_switch_semantics(value: str | None, expected: bool) -> None:
    text = (b.OVERLAY / b.SCHEDULER).read_text()
    switch = ''.join(b.split_spec_switch(b.sections(b.PATCH.read_text())[b.SCHEDULER])[1])
    assert switch in text

    class Env:
        environ = {} if value is None else {'VLLM_DSV41_SPEC_RESERVE_PER_GROUP': value}

    scope: dict[str, object] = {'_dsv41_os': Env}
    exec(compile(switch, 'switch', 'exec'), scope)
    assert scope['_DSV41_SPEC_RESERVE_PER_GROUP'] is expected


def test_a_hunk_that_does_not_apply_fails_loudly() -> None:
    rel = 'vllm/v1/worker/gpu/block_table.py'
    base = (b.IMAGE_SRC / rel).read_text()
    broken = base.replace('self.input_block_tables: list[torch.Tensor] = [', 'self.input_block_tables = [', 1)
    assert broken != base
    with pytest.raises(ValueError, match='do not apply'):
        b.build_file(rel, broken.encode(), b.sections(b.PATCH.read_text())[rel])


def test_switch_needs_exactly_one_anchor() -> None:
    section = b.sections(b.PATCH.read_text())[b.SCHEDULER]
    base = (b.IMAGE_SRC / b.SCHEDULER).read_text().replace(b.ANCHOR, 'import os as _other_os\n')
    with pytest.raises(ValueError, match='anchor'):
        b.build_file(b.SCHEDULER, base.encode(), section)


def test_wrong_first_scheduler_hunk_is_refused() -> None:
    section = b.sections(b.PATCH.read_text())[b.SCHEDULER]
    with pytest.raises(ValueError, match='not the #59880 switch'):
        b.split_spec_switch(section.replace('VLLM_DSV41_SPEC_RESERVE_PER_GROUP', 'VLLM_OTHER'))


def test_tampered_patch_bytes_are_refused(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    copy = tmp_path / 'p.patch'
    copy.write_bytes(b.PATCH.read_bytes() + b'\n')
    monkeypatch.setattr(b, 'PATCH', copy)
    with pytest.raises(ValueError, match='patch bytes changed'):
        b.build()


def test_overlays_compile() -> None:
    for row in json.loads(b.MANIFEST.read_text())['files']:
        path = b.ROOT / row['overlay']
        compile(path.read_text(), str(path), 'exec')
