"""Immutable export relocation: CPU-only synthetic artifacts and smoke seam."""
from pathlib import Path
import json
import shutil
from unittest.mock import patch

import pytest

from d2 import generation_export as E
from d2.assembly_paths import AssemblyPaths
from d2.tests.test_generation_export import state


def test_export_relocation_preserves_every_byte_and_hash(tmp_path):
    source = tmp_path / 'source'
    source.mkdir()
    gen = state(source)
    smoke = source / 'smoke.json'
    smoke.write_text('{"synthetic":true}')
    with patch.object(E, 'smoke'):
        E.export(gen, source / 'top20', smoke)
    hashes = {str(p.relative_to(source)): E.digest(p) for p in source.rglob('*') if p.is_file()}
    target = tmp_path / 'assembly'
    shutil.copytree(source, target)
    shutil.rmtree(source)  # the original absolute paths cannot accidentally work
    paths = AssemblyPaths(target, {str(source): '.'})
    with patch.object(E, 'smoke'):
        E.verify(target / 'top20', target / 'top20/VERIFIED.json', full=True, paths=paths)
    assert hashes == {str(p.relative_to(target)): E.digest(p) for p in target.rglob('*') if p.is_file()}
    p = target / 'top20/seqs/pA.json'
    p.write_bytes(p.read_bytes() + b' ')
    with patch.object(E, 'smoke'), pytest.raises(ValueError, match='changed'):
        E.verify(target / 'top20', target / 'top20/VERIFIED.json', full=True, paths=paths)


def test_relocation_is_explicit_and_confined(tmp_path):
    root = tmp_path / 'assembly'
    root.mkdir()
    outside = tmp_path / 'outside'
    outside.touch()
    paths = AssemblyPaths(root, {})
    with pytest.raises(ValueError, match='escapes'):
        paths.resolve(str(outside))
    (root / 'link').symlink_to(outside)
    with pytest.raises(ValueError):
        paths.resolve(str(root / 'link'))
    with pytest.raises(ValueError, match='relative'):
        AssemblyPaths(root, {'/old': '../outside'})


def test_context_relocation_keeps_receipt_and_tokenizer_checks(tmp_path):
    from d2.tests.test_sequence_audit import context_fixture
    from d2.post_replay import context_ready
    old = tmp_path / 'original'
    old.mkdir()
    context, manifest = context_fixture(old)
    evidence = old / 'target.json'
    evidence.write_text('{"synthetic":true}')
    sha = E.digest(evidence)
    for key in ('target', 'replay_target'):
        context[key]['receipt_sha256'] = sha
    context['evidence_receipts'] = [{'path': str(evidence), 'sha256': sha}]
    (old / 'context.json').write_text(json.dumps(context))
    (old / 'manifest.json').write_text(json.dumps(manifest))
    (old / 'plan.jsonl').write_text('{"seq_id":"synthetic","kind":"gen"}\n')
    target = tmp_path / 'copied'
    shutil.copytree(old, target)
    shutil.rmtree(old)
    paths = AssemblyPaths(target, {str(old): '.'})
    args = [target / name for name in ('context.json', 'manifest.json', 'plan.jsonl')]
    original_context_hash = E.digest(args[0])
    context_ready(*args, paths=paths)
    assert E.digest(args[0]) == original_context_hash
    (target / 'encoder.py').write_text('# changed\n')
    with pytest.raises(ValueError, match='encoder file hash'):
        context_ready(*args, paths=paths)
