"""S23 independent reproduction of D2-FIX1 item 3's writer -> reader alignment.

The lane's test checks the first block's taps and one sequence per shard. Here EVERY block of several sequences in
one shard (cumulative offsets) decodes the absolute position and sequence of each tap row it received, at
gen_start 9 (window clipped at 0), 200, and 500,000, and checks the top-20 row and labels for every draft slot.
"""
from __future__ import annotations

import json
from pathlib import Path

import torch

from offload.dspark.d2 import shards as S
from offload.dspark.d2.train.data import D2Blocks


def tag(r: S.SeqRecord, seq: int) -> None:
    """Exact-in-bf16 identity (each value < 256) in columns 0..3 of every tap layer."""
    pos = torch.arange(r.r0, r.L)
    ident = torch.stack([pos % 256, (pos // 256) % 256, pos // 65536, torch.full_like(pos, seq)], 1).to(torch.bfloat16)
    for layer in range(3):
        r.taps[:, layer * 5120: layer * 5120 + 4] = ident


def decode(taps: torch.Tensor) -> tuple[list[int], set[int]]:
    t = taps.float()
    pos = (t[:, 0] + 256 * t[:, 1] + 65536 * t[:, 2]).long().tolist()
    for layer in (1, 2):
        assert torch.equal(t[:, layer * 5120: layer * 5120 + 4], t[:, :4])
    return pos, set(t[:, 3].long().tolist())


def test_every_block_every_sequence(tmp_path: Path) -> None:
    gens = [9, 200, 500_000]
    recs = []
    for i, g in enumerate(gens):
        r = S.synth_record(f"s23-{i}", "dev", g + 9, g, seed=100 + i)
        r.meta["source_docs"] = [f"doc-{i}"]
        tag(r, i)
        recs.append(r)
    f = S.write_shard(str(tmp_path / "dev"), "dev", recs, "synthetic")
    (tmp_path / "MANIFEST.json").write_text(json.dumps({"schema": S.MANIFEST_SCHEMA,
                                                        "splits": {"dev": {"files": [f]}}}))
    data = D2Blocks(tmp_path / "MANIFEST.json", "dev")
    assert len(data) == sum(r.L - r.gen_start - 1 for r in recs)
    seen = {i: [] for i in range(len(recs))}
    for k in range(len(data)):
        b = data[k]
        P = int(b["inputs"]["P"])
        pos, seqs = decode(b["inputs"]["taps"][0])
        assert len(seqs) == 1
        i = seqs.pop()
        r = recs[i]
        seen[i].append(P)
        assert b["request"] == f"s23-{i}"
        assert pos == list(range(max(0, P - 127), P + 1))                      # rows ..P only, absolute
        assert b["inputs"]["positions"][0].tolist() == pos
        assert int(b["inputs"]["anchor"]) == int(r.tokens[P + 1])
        for j in range(5):
            t = P + 2 + j
            present = bool(b["targets"]["position_mask"][0, j])
            assert present == (t < r.L)
            if present:
                assert t >= r.gen_start + 1                                    # no prompt or first-token label
                assert int(b["targets"]["labels"][0, j]) == int(r.tokens[t])
                assert int(b["targets"]["previous"][0, j]) == int(r.tokens[t - 1])
                assert torch.equal(b["targets"]["top_ids"][0, j], r.top_ids[t - r.gen_start].long())
                assert torch.equal(b["targets"]["top_logp"][0, j], r.top_lp[t - r.gen_start])
            else:
                assert int(b["targets"]["labels"][0, j]) == -1
    for i, r in enumerate(recs):
        assert seen[i] == list(range(r.gen_start - 1, r.L - 2))                 # P = gen_start-1 .. L-3, each once
