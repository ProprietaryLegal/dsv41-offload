"""Named writer failures may be excluded explicitly; integrity failures never may."""
import json
from unittest.mock import patch

import pytest

from d2 import assemble as A, generation_export as E, decode_dataset as DD
from d2.s34 import test_s34_probes as F
from d2.tests.test_pass2_context import attested
from d2 import pass2_context as P


def error_copy(tmp_path, monkeypatch, mode):
    original = F.source_tree
    def source(path):
        capture, state, prompts = original(path)
        seqs = [] if mode == 'global' else ['p0']
        (capture / 'ERRORS.jsonl').write_text(json.dumps({'schema': DD.SCHEMA, 'seq_ids': seqs}) + '\n')
        f = DD.obj(capture / 'FLUSHED.json')
        f['errors'] = 0 if mode == 'flush' else 1
        (capture / 'FLUSHED.json').write_text(json.dumps(f))
        (capture / 'ERROR_COUNT.json').write_text(json.dumps({'schema': 'd2.decode-error-count.v1',
            'errors': 0 if mode == 'count' else 1, 'global_refusal': mode == 'global'}))
        if mode == 'other-bad':
            head = capture / 'p1/CAPTURE.json'
            h = DD.obj(head)
            h['row_bytes'] = 1
            head.write_text(json.dumps(h))
        return capture, state, prompts
    monkeypatch.setattr(F, 'source_tree', source)
    root, prompts = F.copied(tmp_path, monkeypatch)
    receipt, pm = attested(tmp_path, root, prompts)
    stage = P.stage(root, receipt, E.digest(receipt), pm)
    return root, pm, stage


@pytest.mark.parametrize('mode', ['named', 'global', 'other-bad'])
def test_quarantine_only_named_errors(tmp_path, monkeypatch, mode):
    root, pm, stage = error_copy(tmp_path, monkeypatch, mode)
    smoke = root / 'inputs/R6-smoke.json'
    args = (root, smoke, E.digest(smoke), pm, stage['sequence_context'], stage['path_maps'])
    from pathlib import Path
    args = (*args[:4], Path(args[4]), args[5])
    with patch.object(E, 'smoke'):
        if mode != 'named':
            with pytest.raises((ValueError, SystemExit)):
                A.run(*args, quarantine_named_errors=True)
            assert not (root / 'ASSEMBLED.json').exists()
            return
        result = A.run(*args, quarantine_named_errors=True)
    expected = {'enabled': True, 'seq_ids': ['p0'], 'errors_sha256': E.digest(root / 'capture/ERRORS.jsonl')}
    assert result['quarantine'] == DD.obj(root / 'decode-plan/DECODE.json')['quarantine'] == expected
    man = DD.obj(root / 'shards/MANIFEST.json')
    assert man['checks']['sequences'] == 3 and man['checks']['skipped_sequences'] == 0
    assert man['checks']['named_error_quarantine'] == expected
    assert 'p0' not in [json.loads(s)['seq_id'] for s in (root / 'decode-plan/plan.jsonl').read_text().splitlines()]


def test_quarantine_default_off(tmp_path, monkeypatch):
    root, pm, stage = error_copy(tmp_path, monkeypatch, 'named')
    smoke = root / 'inputs/R6-smoke.json'
    from pathlib import Path
    with patch.object(E, 'smoke'), pytest.raises(ValueError, match='quarantine-named-errors'):
        A.run(root, smoke, E.digest(smoke), pm, Path(stage['sequence_context']), stage['path_maps'])
    assert not (root / 'ASSEMBLED.json').exists()


@pytest.mark.parametrize('name', ['FLUSHED.json', 'ERROR_COUNT.json'])
def test_quarantine_count_drift_refused(tmp_path, name):
    capture, _, _ = F.source_tree(tmp_path)
    row = DD.obj(capture / name)
    row['errors'] = 1
    (capture / name).write_text(json.dumps(row))
    with pytest.raises(ValueError, match='flush/error-count mismatch'):
        DD.named_error_policy(capture, True)
