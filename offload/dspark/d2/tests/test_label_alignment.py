"""Actual shard writer -> actual training reader: no model/GPU needed."""
from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from offload.dspark.d2 import shards as S
from offload.dspark.d2.train.data import D2Blocks


@pytest.mark.parametrize('gen', [9, 500_000])
def test_absolute_tap_label_and_teacher_forcing_alignment(tmp_path: Path, gen: int) -> None:
    r = S.synth_record('synthetic', 'dev', gen + 8, gen, seed=17)
    nr = r.L - r.r0
    # Row values identify exactly which taps were consumed, including future rows.
    r.taps[:] = torch.arange(1, nr + 1, dtype=torch.bfloat16)[:, None]
    f = S.write_shard(str(tmp_path / 'dev'), 'dev', [r], 'synthetic')
    manifest = tmp_path / 'MANIFEST.json'
    manifest.write_text(json.dumps({'schema': S.MANIFEST_SCHEMA, 'splits': {'dev': {'files': [f]}}}))
    data = D2Blocks(manifest, 'dev')
    assert len(data) == 7
    first = data[0]
    assert first['inputs']['P'].item() == gen - 1
    assert first['inputs']['positions'][0, -1].item() == gen - 1
    assert first['inputs']['positions'][0, 0].item() == max(0, gen - 128)
    # Taps end at P. The anchor token is P+1; no tap from anchor/future can leak.
    assert torch.equal(first['inputs']['taps'][0], r.taps[:gen-r.r0])
    assert first['inputs']['anchor'].item() == r.tokens[gen].item()
    assert torch.equal(first['targets']['labels'][0], r.tokens[gen+1:gen+6].long())
    assert torch.equal(first['targets']['previous'][0], r.tokens[gen:gen+5].long())
    assert torch.equal(first['targets']['top_ids'][0], r.top_ids[1:6].long())
    assert torch.equal(first['targets']['top_logp'][0], r.top_lp[1:6])
    for index in range(len(data)):
        b = data[index]
        P = int(b['inputs']['P'].item())
        mask = b['targets']['position_mask'][0]
        target_pos = torch.arange(P+2, P+7)
        assert torch.equal(mask, target_pos < r.L)
        assert bool((target_pos[mask] >= gen).all())  # every prompt label masked out
        assert torch.equal(b['targets']['labels'][0, mask], r.tokens[target_pos[mask]].long())
        assert bool((b['targets']['labels'][0, ~mask] == -1).all())
        assert int(b['inputs']['positions'].max()) == P
    assert data[len(data)-1]['targets']['position_mask'].tolist() == [[True, False, False, False, False]]
