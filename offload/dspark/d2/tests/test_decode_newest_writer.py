"""Segment replacement, restart ownership and tamper refusal using real CPU capture."""
import json

import pytest
import torch

from d2 import decode_dataset as DD
from d2.s26.test_s26_resume import C, R0, run, step, flush, tap


@pytest.mark.parametrize('corruption', ['older_segment', 'prompt', 'newest_bytes', 'retained_hash'])
def test_newest_writer_refuses_invalid_supersession(tmp_path, corruption):
    tokens, length = run(tmp_path, driver_received_step2=False)
    ledger = tmp_path/'pX/ledger.jsonl'
    rows = [json.loads(line) for line in ledger.read_text().splitlines()]
    replaced = next(row for row in rows if row['replaced_rows'])
    assert replaced['prompt_length'] == 304
    assert len(replaced['replaced_rows']) == 3
    if corruption == 'older_segment':
        replaced['replaced_rows'][0]['seg'] = replaced['seg']
    elif corruption == 'prompt':
        replaced['prompt_length'] = 307
    elif corruption == 'retained_hash':
        retained = rows[3]
        retained['row_sha256_taps'][0] = 'a'*64
    else:
        path = tmp_path/'pX/taps.bin'
        with path.open('r+b') as f:
            f.seek((304-R0)*C.ROW_BYTES)
            f.write(b'\0\0')
    ledger.write_text(''.join(json.dumps(row)+'\n' for row in rows))
    receipt = json.loads((tmp_path/'FLUSHED.json').read_text())
    receipt['sequences']['pX']['ledger_sha256'] = DD.digest(ledger)
    (tmp_path/'FLUSHED.json').write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match='capture'):
        DD.capture_rows(tmp_path, 'pX', R0, length-1, tokens)


def test_short_prompt_and_restart_use_explicit_prompt_boundary(tmp_path):
    old = {p:100+p for p in range(12)}
    c = C.Capture(str(tmp_path))
    step('d2c.short.0.0.2', old, 0, 8, 0, c)
    c.close()
    new = {p:old[p] if p<4 else 200+p for p in range(12)}
    c = C.Capture(str(tmp_path))
    step('d2c.short.1.0.4', new, 0, 9, 0, c)
    flush(tmp_path, c)
    tokens = [new[p] for p in range(10)]
    taps, meta = DD.capture_rows(tmp_path, 'short', 0, 9, tokens)
    assert meta['replaced_rows'] == 4
    assert torch.equal(taps, torch.stack([torch.cat([tap(new[p],p,l) for l in range(3)]) for p in range(9)]))
    rows = [json.loads(line) for line in (tmp_path/'short/ledger.jsonl').read_text().splitlines()]
    assert [x['position'] for x in rows[1]['replaced_rows']] == list(range(4,8))
