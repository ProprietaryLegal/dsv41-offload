"""Actual async capture -> decode-plan -> shard -> reader; synthetic data only."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import types

import numpy as np
import pytest
import torch

from d2 import dataset as D, decode_dataset as DD, shards as S
from d2.tests.test_dataset import _fake_replay
from d2.tests.test_decode_capture import C


def fixture(tmp_path):
    records = [S.synth_record('pA', 'train', 207, 200, seed=4, family='holding_summary')]
    replay = _fake_replay(tmp_path, records)
    # Older replay tests omit fields; decode assembly deliberately requires actual response identity.
    from d2.replay import read_plan
    for e in read_plan(str(replay)):
        lp = replay / 'logprobs' / (e['seq_id'] + '.json')
        m = json.loads(lp.read_text())
        m.update({k: e[k] for k in ('seq_id', 'request_id', 'from')})
        m['usage'] = {'prompt_tokens_details': {'cached_tokens': 0}}
        lp.write_text(json.dumps(m))
    root = tmp_path / 'capture'
    root.mkdir()
    c = C.Capture(str(root))
    r = records[0]
    ib = types.SimpleNamespace(num_reqs=1, num_tokens=r.L-r.r0, req_ids=[f'd2c.pA.0.{r.r0}'],
                               query_start_loc_np=np.array([0, r.L-r.r0]), positions=torch.arange(r.r0, r.L),
                               input_ids=r.tokens[r.r0:].clone())
    c.enqueue(ib, list(r.taps.view(-1, 3, 5120).unbind(1)), torch.tensor([0]))
    c.close()
    # Write a genuine protocol flush receipt after an orderly close (all file descriptors synchronized).
    (root / 'FLUSH.json').write_text(json.dumps({'nonce': 'synthetic-frozen-copy'}))
    (root / 'FLUSHED.json').write_text(json.dumps({'schema': DD.SCHEMA, 'nonce': 'synthetic-frozen-copy', 'errors': 0,
        'sequences': {'pA': {'ledger_sha256': DD.digest(root / 'pA/ledger.jsonl'),
                             'taps_bytes': (root / 'pA/taps.bin').stat().st_size,
                             'ids_bytes': (root / 'pA/ids.bin').stat().st_size}}}))
    plan = tmp_path / 'decode-plan'
    DD.prepare(replay, root, plan, 'prefill_replay_top20_only')
    args = argparse.Namespace(replay=str(plan), out=str(tmp_path / 'shards'),
        prompt_manifest=str(replay / 'prompt-manifest.json'), taps_source='decode_capture', splits='',
        allow_skips=False, equiv_gate='', override_equiv_gate='', override_plumbing='', sequence_context='')
    return r, root, plan, args


def test_actual_decode_build_without_equivalence_gate(tmp_path):
    r, root, plan, args = fixture(tmp_path)
    assert D.tree_taps_source(str(plan)) == 'decode_capture'
    D.cmd_build(args)
    out = Path(args.out)
    assert D.verify_dir(str(out))['errors'] == 0
    man = json.loads((out / 'MANIFEST.json').read_text())
    assert man['taps_source'] == 'decode_capture'
    assert man['checks']['equivalence_gate'] == {'required': False, 'taps_source': 'decode_capture'}
    sh = S.Shard(str(out / man['splits']['train']['files'][0]['file']), verify_sha256=True)
    got = sh.record(0)
    assert got.L == r.L - 1
    assert torch.equal(got.tokens, r.tokens[:-1])
    assert torch.equal(got.taps, r.taps[:-1])
    assert torch.equal(got.top_lp, r.top_lp[:-1])
    assert got.meta['generation_L'] == r.L
    assert got.meta['decode_capture']['extra_rows_excluded'] == 1
    assert not (root / 'pA/DONE.json').exists()


@pytest.mark.parametrize('bad', ['gap', 'wrong_ids', 'zero', 'stale_flush', 'errors', 'top20', 'header', 'tail'])
def test_decode_failures_never_publish(tmp_path, bad):
    r, root, plan, args = fixture(tmp_path)
    lp = root / 'pA/ledger.jsonl'
    row = json.loads(lp.read_text())
    if bad in ('gap', 'tail'):
        row['rows'] -= 1
        if bad == 'gap':
            row['first'] += 2
            row['rows'] -= 2
        start, n = row['first'] - r.r0, row['rows']
        for key in ('row_sha256_taps', 'row_sha256_ids'):
            row[key] = row[key][start:start+n]
        import hashlib
        row['sha256_taps'] = hashlib.sha256((root / 'pA/taps.bin').read_bytes()[start*30720:(start+n)*30720]).hexdigest()
        row['sha256_ids'] = hashlib.sha256((root / 'pA/ids.bin').read_bytes()[start*4:(start+n)*4]).hexdigest()
        lp.write_text(json.dumps(row)+'\n')
    elif bad == 'wrong_ids':
        p = root / 'pA/ids.bin'
        p.write_bytes(b'\0'*4+p.read_bytes()[4:])
    elif bad == 'zero':
        row['zero_rows'] = 1
        lp.write_text(json.dumps(row)+'\n')
    elif bad == 'stale_flush':
        (root / 'FLUSH.json').write_text(json.dumps({'nonce': 'new-flush-not-acknowledged'}))
    elif bad == 'errors':
        (root / 'ERRORS.jsonl').write_text('{}\n')
    elif bad == 'top20':
        p = tmp_path / 'replay/logprobs/pA.npz'
        p.write_bytes(p.read_bytes() + b'changed')
    elif bad == 'header':
        p = root / 'pA/CAPTURE.json'
        h = json.loads(p.read_text())
        h['taps_source'] = 'prefill_replay'
        p.write_text(json.dumps(h))
    if bad in ('gap', 'zero', 'tail'):
        f = json.loads((root / 'FLUSHED.json').read_text())
        f['sequences']['pA']['ledger_sha256'] = DD.digest(lp)
        (root / 'FLUSHED.json').write_text(json.dumps(f))
    if bad == 'tail':
        # Removing only the extra final token's row is legitimate: L_gen-1 has no required tap.
        D.cmd_build(args)
        assert Path(args.out).exists()
    else:
        with pytest.raises((SystemExit, ValueError), match='integrity|capture|decode'):
            D.cmd_build(args)
        assert not Path(args.out).exists()


def test_decode_label_cannot_bypass_mode_or_top20_verification(tmp_path):
    r, root, plan, args = fixture(tmp_path)
    m = json.loads((plan / 'DECODE.json').read_text())
    m['top20_source'] = 'generation_verified'
    m['verification'] = str(tmp_path / 'unverified.json')
    (plan / 'DECODE.json').write_text(json.dumps(m))
    (tmp_path / 'unverified.json').write_text('{}')
    with pytest.raises(ValueError, match='successful real image-bound smoke'):
        D.tree_taps_source(str(plan))


def test_decode_preserves_served_provenance_gate(tmp_path):
    _, _, _, args = fixture(tmp_path)
    pm = json.loads(Path(args.prompt_manifest).read_text())
    pm['schema'] = 'd2.prompts.v1'
    pm['prompts'][0]['source'] = 'served-fixture'
    Path(args.prompt_manifest).write_text(json.dumps(pm))
    with pytest.raises(SystemExit, match='sequence-context'):
        D.cmd_build(args)
    assert not Path(args.out).exists()
