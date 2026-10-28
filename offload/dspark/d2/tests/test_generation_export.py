"""Offline exporter integrity checks; synthetic rows, never served data."""
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from d2 import generation_export as E
from d2 import generation_top20 as T
from d2.tests.test_generation_top20 import wire


def state(tmp_path):
    root = tmp_path / 'gen'
    (root / 'seqs').mkdir(parents=True)
    (root / 'LOCK').touch()
    tokens = [100, 999, 100, 999]
    raw = {'schema': 'd2.genstate.v1', 'pid': 'pA', 'status': 'partial', 'split': 'train', 'family': 'f',
           'prompt_sha256': 'h', 'prompt_ids': [1, 2, 3], 'gen_ids': tokens, 'segments': [],
           'top20': 'generation', 'logprob_rows': [T.rows([t], wire(t, t == 999))[0] for t in tokens]}
    (root / 'seqs/pA.json').write_text(json.dumps(raw))
    return root


def test_export_private_frozen_tree_and_arrays(tmp_path):
    root = state(tmp_path)
    out = tmp_path / 'export'
    receipt = tmp_path / 'smoke.json'
    receipt.write_text('{}')
    # Unit-only smoke seam: production always requires the real image-bound receipt.
    with patch.object(E, 'smoke'):
        E.export(root, out, receipt)
        E.verify(out, out / 'VERIFIED.json', full=True)
    plan = E.read_plan(str(out))
    assert plan[0]['L'] == 7 and plan[0]['gen_start'] == 3
    assert Path(plan[0]['src']).parent == out / 'seqs'
    with np.load(out / 'logprobs/pA.npz') as z:
        assert z['top_ids'].shape == (4, 20)
        assert z['tok_rank'].tolist() == [1, 21, 1, 21]
        assert z['top_lp'].dtype == np.float32
    assert out.stat().st_mode & 0o777 == 0o700
    assert (out / 'seqs/pA.json').stat().st_mode & 0o777 == 0o600
    (root / 'seqs/pA.json').write_text('{}')  # live original cannot change the frozen export
    with patch.object(E, 'smoke'):
        E.verify(out, out / 'VERIFIED.json', full=True)
    p = out / 'logprobs/pA.npz'
    p.write_bytes(p.read_bytes() + b'changed')
    with patch.object(E, 'smoke'), pytest.raises(ValueError, match='changed'):
        E.verify(out, out / 'VERIFIED.json', full=True)


def test_live_driver_lock_refused(tmp_path):
    import fcntl
    root = state(tmp_path)
    with (root / 'LOCK').open() as lock, patch.object(E, 'smoke'):
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(ValueError, match='driver still holds LOCK'):
            E.export(root, tmp_path / 'export', tmp_path / 'smoke.json')


def test_chat_receipt_alone_cannot_certify_completions(tmp_path):
    root = state(tmp_path)
    smoke = tmp_path / 'smoke.json'
    smoke.write_text('{}')
    with pytest.raises(ValueError, match='successful real image-bound smoke'):
        E.export(root, tmp_path / 'export', smoke)
    assert not (tmp_path / 'export').exists()


@pytest.mark.parametrize('bad', ['missing_row', 'mode', 'running'])
def test_bad_state_never_gets_verified_receipt(tmp_path, bad):
    root = state(tmp_path)
    p = root / 'seqs/pA.json'
    raw = json.loads(p.read_text())
    if bad == 'missing_row':
        raw['logprob_rows'].pop()
    elif bad == 'mode':
        raw['top20'] = 'off'
    else:
        raw['status'] = 'running'
    p.write_text(json.dumps(raw))
    with patch.object(E, 'smoke'), pytest.raises(ValueError):
        E.export(root, tmp_path / 'export', tmp_path / 'smoke.json')
    assert not (tmp_path / 'export/VERIFIED.json').exists()
