"""Synthetic CPU writer -> real frozen copy -> export -> shards -> real G1(a).

No hosts, sockets or real smoke receipts. Only the external image-smoke gate
is replaced in the success rehearsal; all internal hashes and checks are real.
"""
from __future__ import annotations

import json
import argparse
from pathlib import Path
import shutil
import sys
from unittest.mock import patch

import pytest

from d2 import assemble as A, dataset as D, decode_dataset as DD, generation_export as E
from d2 import generation_top20 as T
from d2.s26.test_s26_resume import C, step
from d2.tests.test_generation_top20 import wire
from offload.dspark.d2 import decode_copy as COPY


class Local(COPY.Remote):
    """Reviewed scan with genuine local O_DIRECT reads; local transfer only."""
    def __init__(self, capture: bool):
        super().__init__(['NEVER-A-HOST'], capture)

    def inventory(self, root: str, nonce: str) -> COPY.Inventory:
        return COPY.scan(root, self.capture, nonce)

    def copy(self, source: str, destination: Path, entry: COPY.Record) -> COPY.Record:
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
        sha = E.digest(destination)
        assert destination.stat().st_size == entry['size']
        if 'sha256' in entry:
            assert sha == entry['sha256']
        return {'size': destination.stat().st_size, 'sha256': sha}


def make_copy(tmp_path: Path, monkeypatch, *, include_synthetic_acap: bool = False) -> tuple[Path, Path, Path, Path]:
    # Deny accidental Remote operations even if the local adapter regresses.
    def forbidden(*args, **kwargs):
        raise AssertionError('CPU rehearsal attempted host access')
    monkeypatch.setattr(COPY.transfer, 'idle', forbidden)
    monkeypatch.setattr(COPY.transfer, 'copy', forbidden)
    source = tmp_path / 'source'
    capture, state = source / 'capture', source / 'generator'
    capture.mkdir(parents=True)
    (state / 'seqs').mkdir(parents=True)
    (state / 'LOCK').touch()
    writer = C.Capture(str(capture))
    prompts = []
    splits = ('train', 'dev', 'heldout_legal', 'heldout_general')
    if include_synthetic_acap:
        # Plumbing fixture only; never a route for relabelling real A-captures.
        splits += ('acap',)
    for i, split in enumerate(splits):
        sid = 'p' + str(i)
        prompt = [10 + i] * 200
        generated = [100, 999, 100, 999, 100, 999, 100, 999]
        tokens = prompt + generated
        raw = {'schema': 'd2.genstate.v1', 'pid': sid, 'status': 'partial', 'split': split,
               'family': {'train': 'holding_summary', 'dev': 'issue_spotting',
                          'heldout_legal': 'chronology', 'heldout_general': 'general_book', 'acap': 'acap'}[split],
               'prompt_sha256': 'hash_' + sid,
               'prompt_ids': prompt, 'gen_ids': generated, 'segments': [],
               'top20': 'generation', 'logprob_rows': [T.rows([t], wire(t, t == 999))[0] for t in generated]}
        (state / 'seqs' / (sid + '.json')).write_text(json.dumps(raw))
        # Real prefill + decode writer rounds, using the S31/S26 CPU technique.
        rid = f'cmpl-d2c.{sid}.0.72.200-0-feed'
        toks = dict(enumerate(tokens))
        step(rid, toks, 0, 200, 0, writer)
        step(rid, toks, 200, 8, 0, writer)
        prompts.append({'pid': sid, 'split': split, 'family': raw['family'],
                        'prompt_sha256': raw['prompt_sha256'], 'source_docs': [sid], 'source': 'synthetic'})
    old_argv = sys.argv
    try:
        sys.argv = ['flush', str(capture), 'assemble-synthetic']
        exec(compile(COPY.FLUSH_CODE, '<reviewed synthetic flush>', 'exec'), {})
    finally:
        sys.argv = old_argv
        writer.close()
    root = tmp_path / 'decode-synthetic'
    COPY.frozen_copy(Local(True), Local(False), str(capture), str(state), root, 'assemble-synthetic')
    shutil.rmtree(source)
    inputs = root / 'inputs'
    inputs.mkdir()
    smoke = inputs / 'SYNTHETIC-smoke.json'
    smoke.write_text('{"synthetic":true}')
    pm = inputs / 'prompt-manifest.json'
    pm.write_text(json.dumps({'schema': 'd2.prompts.synthetic.v1', 'n_prompts': len(prompts),
        'prompts': prompts, 'counts': {p['split']: {p['family']: 1} for p in prompts}}))
    context = inputs / 'sequence-context.json'
    context.write_text('{"synthetic":true}')
    monkeypatch.setenv('D2_COORDINATOR', A.COORDINATOR)
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '')
    return root, smoke, pm, context


def test_synthetic_copy_to_g1a_end_to_end(tmp_path, monkeypatch):
    root, smoke, pm, ctx = make_copy(tmp_path, monkeypatch)
    copy_sha = E.digest(root / 'COPY_RECEIPT.json')
    with patch.object(E, 'smoke'):
        result = A.run(root, smoke, E.digest(smoke), pm, ctx, {})
    assert result['g1a_pass'] is True and result['pilot_cleared'] is False
    assert result['synthetic'] is True
    assert result['copy_receipt_sha256'] == copy_sha
    decode = DD.obj(root / 'decode-plan/DECODE.json')
    assert decode['copy_receipt_sha256'] == copy_sha
    manifest = DD.obj(root / 'shards/MANIFEST.json')
    assert manifest['replay']['copy_receipt_sha256'] == copy_sha
    assert manifest['checks']['sequences'] == 4 and manifest['checks']['skipped_sequences'] == 0
    assert set(manifest['splits']) == {'train', 'dev', 'heldout_legal', 'heldout_general'}
    assert D.verify_dir(str(root / 'shards'))['errors'] == 0
    assert DD.obj(root / 'AGGREGATE.json')['synthetic'] is True
    # Restoring the real gate refuses these synthetic artifacts.
    with pytest.raises(ValueError, match='real image-bound smoke'):
        DD.decode_identity(root / 'decode-plan')


@pytest.mark.parametrize('bad', ['moved', 'hash', 'missing_receipt'])
def test_bad_copy_refused_before_export_and_at_build(tmp_path, monkeypatch, bad):
    root, smoke, pm, ctx = make_copy(tmp_path, monkeypatch)
    with patch.object(E, 'smoke'):
        E.export(root / 'generator', root / 'export', smoke)
        DD.prepare(root / 'export', root / 'capture', root / 'decode-plan', 'generation_verified',
                   root / 'export/VERIFIED.json', assembly_root=root)
    path = root / 'capture/p0/taps.bin'
    if bad == 'moved':
        path.rename(path.with_name('moved.bin'))
    elif bad == 'hash':
        with path.open('r+b') as stream:
            stream.write(b'xx')
    else:
        (root / 'COPY_RECEIPT.json').unlink()
    with patch.object(E, 'smoke'), pytest.raises((ValueError, FileNotFoundError), match='copy|COPY_RECEIPT'):
        D.tree_taps_source(str(root / 'decode-plan'))
    assert not (root / 'shards').exists()


def test_unbound_export_state_refused(tmp_path, monkeypatch):
    root, smoke, pm, ctx = make_copy(tmp_path, monkeypatch)
    with patch.object(E, 'smoke'):
        E.export(root / 'generator', root / 'export', smoke)
        # A different, valid independently exported snapshot must not borrow this copy.
        foreign = tmp_path / 'foreign'
        shutil.copytree(root / 'generator', foreign)
        p = foreign / 'seqs/p0.json'
        raw = json.loads(p.read_text())
        raw['finish_reason'] = 'different_snapshot'
        p.write_text(json.dumps(raw))
        E.export(foreign, root / 'foreign-export', smoke)
        with pytest.raises(ValueError, match='receipt-bound generator'):
            DD.prepare(root / 'foreign-export', root / 'capture', root / 'decode-plan', 'generation_verified',
                       root / 'foreign-export/VERIFIED.json', assembly_root=root)


def test_copied_top20_absolute_paths_relocate_through_shard_build(tmp_path, monkeypatch):
    root, smoke, pm, ctx = make_copy(tmp_path, monkeypatch)
    with patch.object(E, 'smoke'):
        E.export(root / 'generator', root / 'old-export', smoke)
    destination = tmp_path / 'relocated-copy'
    COPY.frozen_copy(Local(True), Local(False), str(root / 'capture'), str(root / 'generator'),
                     destination, 'assemble-synthetic', top20=str(root / 'old-export'))
    shutil.copytree(root / 'inputs', destination / 'inputs')
    original_hashes = {str(p.relative_to(destination / 'top20')): E.digest(p)
                       for p in (destination / 'top20').rglob('*') if p.is_file()}
    mappings = {str(root / 'old-export'): 'top20', str(root / 'inputs'): 'inputs'}
    shutil.rmtree(root)
    with patch.object(E, 'smoke'):
        DD.prepare(destination / 'top20', destination / 'capture', destination / 'decode-plan',
                   'generation_verified', destination / 'top20/VERIFIED.json',
                   assembly_root=destination, path_maps=mappings)
        D.cmd_build(argparse.Namespace(replay=str(destination / 'decode-plan'), out=str(destination / 'shards'),
            prompt_manifest=str(destination / 'inputs/prompt-manifest.json'), sequence_context='',
            taps_source='decode_capture', splits='', allow_skips=False, equiv_gate='',
            override_equiv_gate='', override_plumbing=''))
    assert D.verify_dir(str(destination / 'shards'))['errors'] == 0
    assert DD.obj(destination / 'decode-plan/DECODE.json')['relocation']['mappings'] == mappings
    assert original_hashes == {str(p.relative_to(destination / 'top20')): E.digest(p)
                               for p in (destination / 'top20').rglob('*') if p.is_file()}


def test_coordinator_nonce_mismatch_refuses_before_export(tmp_path, monkeypatch):
    root, smoke, pm, ctx = make_copy(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match='expected nonce'):
        A.run(root, smoke, E.digest(smoke), pm, ctx, {}, expect_nonce='different-run')
    assert not (root / 'export').exists()
    assert not (root / 'ASSEMBLED.json').exists()
