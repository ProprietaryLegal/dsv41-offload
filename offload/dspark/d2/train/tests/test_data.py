from __future__ import annotations

import json
import hashlib
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from offload.dspark.d2.train.data import D2Blocks, sha256


def write_shard(root: Path, gen: int = 4, split: str = "train", L: int = 10) -> Path:
    shard = root/(split+".safetensors")
    p = torch.full((L-gen, 20), .045)
    save_file({"seq_index": torch.tensor([[0, L, gen, 0, 0, L, 0, L-gen]]),
               "tokens": torch.arange(L, dtype=torch.int32), "taps": torch.zeros(L, 15360, dtype=torch.bfloat16),
               "row_pos": torch.arange(L, dtype=torch.int32),
               "top_ids": torch.arange(20, dtype=torch.int32).expand(L-gen, -1).contiguous(), "top_lp": p.log(),
               "resid": 1-p.sum(-1), "tok_lp": p[:, 0].log(), "tok_rank": torch.ones(L-gen, dtype=torch.int32)},
              shard, metadata={"schema": "d2.shard.v1", "split": split})
    meta = {"schema": "d2.shard.v1", "split": split, "W": 128, "K": 5, "topk": 20,
            "tap_layers": [37, 38, 39], "hidden": 5120, "noise_token": 128799, "sha256": sha256(shard),
            "sequences": [{"seq_id": "p1", "source_docs": ["doc1"], "family": "argument_section", "L": L, "gen_start": gen,
                           "tokens_sha256": hashlib.sha256(torch.arange(L, dtype=torch.int32).numpy().tobytes()).hexdigest()}]}
    side = root/(split+".json")
    side.write_text(json.dumps(meta))
    manifest = root/"MANIFEST.json"
    manifest.write_text(json.dumps({"schema": "d2.manifest.v1", "checks": {"plumbing_errors": 0},
           "splits": {split: {"files": [{"file": shard.name, "sha256": sha256(shard), "sidecar": side.name,
                                            "sidecar_sha256": sha256(side)}]}}}))
    return manifest


def test_normative_target_offset_and_partial_tail(tmp_path: Path) -> None:
    d = D2Blocks(write_shard(tmp_path), "train")
    assert len(d) == 5
    first, last = d[0], d[len(d)-1]
    assert first["inputs"]["P"].item() == 3
    assert first["inputs"]["anchor"].item() == 4
    assert first["targets"]["labels"].tolist() == [[5, 6, 7, 8, 9]]
    assert first["targets"]["previous"].tolist() == [[4, 5, 6, 7, 8]]
    assert last["inputs"]["P"].item() == 7
    assert last["targets"]["labels"].tolist() == [[9, -1, -1, -1, -1]]
    assert last["targets"]["position_mask"].tolist() == [[True, False, False, False, False]]
    assert last["inputs"]["positions"].tolist() == [list(range(8))]


def test_changed_shard_refused(tmp_path: Path) -> None:
    p = write_shard(tmp_path)
    (tmp_path/"train.json").write_text("{}")
    with pytest.raises(ValueError, match="hash/path mismatch"):
        D2Blocks(p, "train")
