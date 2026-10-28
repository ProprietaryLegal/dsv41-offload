"""CPU rehearsal: fake completions -> frozen exporter -> synthetic capture -> build_record."""
from __future__ import annotations

import json
import threading
import time
import types
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import torch

from d2 import decode_dataset as DD
from d2 import generate as G
from d2 import generation_export as E
from d2 import generation_top20 as T
from d2 import shards as S
from d2.tests.fakevllm import Fake
from d2.tests.test_decode_capture import C
from d2.tests.test_generate import _args, _check_exact, _prompts


def capture(root, state):
    root.mkdir()
    g = len(state['prompt_ids'])
    tokens = state['prompt_ids'] + state['gen_ids']
    r0 = max(0, g - 128)
    n = len(tokens) - r0
    c = C.Capture(str(root))
    ib = types.SimpleNamespace(num_reqs=1, num_tokens=n, req_ids=[f'd2c.p0.0.{r0}.{g}'],
        query_start_loc_np=np.array([0, n]), positions=torch.arange(r0, len(tokens)),
        input_ids=torch.tensor(tokens[r0:], dtype=torch.int32))
    taps = [torch.full((n, 5120), i + 1, dtype=torch.bfloat16) for i in range(3)]
    c.enqueue(ib, taps, torch.tensor([0]))
    c.close()
    (root / 'FLUSH.json').write_text(json.dumps({'nonce': 'synthetic'}))
    (root / 'FLUSHED.json').write_text(json.dumps({'schema': DD.SCHEMA, 'nonce': 'synthetic', 'errors': 0,
        'sequences': {'p0': {'ledger_sha256': DD.digest(root / 'p0/ledger.jsonl'),
                             'taps_bytes': (root / 'p0/taps.bin').stat().st_size,
                             'ids_bytes': (root / 'p0/ids.bin').stat().st_size}}}))


def test_foreign_cancel_resume_export_prepare_build(tmp_path):
    with Fake(token_delay=.004) as f:
        a = _args(tmp_path, f.url, top20='generation', max_tokens=40, streams=1, idle_s=.01,
                  backoff_s=.03, grace_s=.01)
        d = G.Driver(a, _prompts(1))
        fired = False
        clear = None

        def hook(driver):
            nonlocal fired, clear
            st = driver.streams.get('p0')
            if not fired and st is not None and st.n() >= 7:
                fired = True
                # Exercise an on-disk running checkpoint with rows and IDs under the same lock.
                driver.checkpoint_running()
                snap = json.loads(Path(driver.path('p0')).read_text())
                T.validate_saved(snap['gen_ids'], snap['logprob_rows'])
                f.other = 1
                clear = threading.Timer(.08, lambda: setattr(f, 'other', 0))
                clear.start()
        d.loop(hook)
        if clear:
            clear.join()
        _check_exact(d)
        state = d.st['p0']
        assert fired and d.obs['abort_events'] >= 1 and len(state['segments']) >= 2
        assert len(state['gen_ids']) == len(state['logprob_rows'])
        T.validate_saved(state['gen_ids'], state['logprob_rows'])
        assert {r['tok_rank'] for r in state['logprob_rows']} == {1, 21}
        assert all(r['logprobs'] == 20 and r['return_tokens_as_token_ids'] for r in f.requests)
    out = tmp_path / 'export'
    smoke = tmp_path / 'SYNTHETIC-smoke.json'
    smoke.write_text(json.dumps({'synthetic': True}))
    captures = tmp_path / 'captures'
    capture(captures, state)
    plan = tmp_path / 'decode'
    # Never manufacture a successful real receipt in a CPU rehearsal.
    # Only the external-smoke gate is replaced; all row/hash/assembly gates run normally.
    with patch.object(E, 'smoke'):
        E.export(Path(a.state), out, smoke)
        DD.prepare(out, captures, plan, 'generation_verified', out / 'VERIFIED.json')
        entry = E.read_plan(str(plan))[0]
        rec = DD.build_record(entry, str(plan), {}, None)
        assert not S.check_record(rec)
        assert rec.L == len(state['prompt_ids']) + len(state['gen_ids']) - 1
        assert rec.top_ids.shape[0] == len(state['gen_ids']) - 1
        assert rec.meta['top20_source'] == 'generation_verified'
        assert rec.meta['decode_capture']['extra_rows_excluded'] == 1
        S.write_shard(str(tmp_path / 'synthetic-shard'), 'heldout_legal', [rec], 'decode_capture')
        shard = S.Shard(str(tmp_path / 'synthetic-shard.safetensors'), verify_sha256=True)
        assert torch.equal(shard.record(0).tok_lp, rec.tok_lp)
    # The synthetic receipt seam cannot escape the test and pass the real gate.
    with pytest.raises(ValueError, match='successful real image-bound smoke'):
        DD.decode_identity(plan)


@pytest.mark.parametrize('field', ['malformed_at', 'missing_at'])
def test_bad_row_latches_pause_keeps_only_valid_prefix(tmp_path, field):
    with Fake() as f:
        setattr(f, field, 7)
        a = _args(tmp_path, f.url, top20='generation', streams=1)
        d = G.Driver(a, _prompts(1))
        d.submit('p0')
        st = d.streams['p0']
        st.join(timeout=5)
        assert not st.is_alive()
        d.reap()
        d.close()
        assert Path(a.pause_file).read_text().startswith('generation top20:')
        state = json.loads(Path(d.path('p0')).read_text())
        assert len(state['gen_ids']) == len(state['logprob_rows']) == 6
        assert state['segments'][0]['error']
        assert state['errors']
        events = [json.loads(s) for s in (Path(a.state) / 'events.jsonl').read_text().splitlines()]
        assert any(e['event'] == 'top20_invalid' for e in events)
        assert d.paused() == 'pause_file'


def test_checkpoint_restart_and_mode_refusal(tmp_path):
    with Fake(token_delay=.01) as f:
        a = _args(tmp_path, f.url, top20='generation', streams=1, max_tokens=40, idle_s=.01)
        d = G.Driver(a, _prompts(1))
        d.submit('p0')
        st = d.streams['p0']
        end = time.monotonic() + 5
        while st.n() < 5 and time.monotonic() < end:
            time.sleep(.005)
        d.checkpoint_running()
        snap = Path(d.path('p0')).read_bytes()
        d.abort_all('synthetic crash')
        d.close()
        # Simulate the atomic running checkpoint left by a killed driver (not its later reap).
        Path(d.path('p0')).write_bytes(snap)
        resumed = G.Driver(a, _prompts(1))
        T.validate_saved(resumed.st['p0']['gen_ids'], resumed.st['p0']['logprob_rows'])
        resumed.loop()
        _check_exact(resumed)
        T.validate_saved(resumed.st['p0']['gen_ids'], resumed.st['p0']['logprob_rows'])
        a.top20 = 'off'
        with pytest.raises(ValueError, match='cannot change top20 mode'):
            G.Driver(a, _prompts(1))
