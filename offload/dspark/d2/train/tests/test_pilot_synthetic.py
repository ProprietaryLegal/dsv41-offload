from __future__ import annotations

import json
from pathlib import Path

import torch

from offload.dspark.d2.train.data import D2Blocks, sha256
from offload.dspark.d2.train.pilot import write_json
from offload.dspark.d2.train.pilot_synthetic import build


def test_synthetic_shards_keep_all_captures_eval_only_and_n1_train_only(tmp_path: Path) -> None:
    inputs, reference, out = [tmp_path/n for n in ("inputs", "reference", "shards")]
    inputs.mkdir()
    reference.mkdir()
    files = {}
    for i in range(54):
        name = f"window-{i}.pt"
        window = {"P": torch.tensor([5]), "positions": torch.arange(6)[None],
                  "anchor": torch.tensor([100]), "mask": torch.ones(1, 6, dtype=torch.bool),
                  "has_taps": torch.ones(1, 6, dtype=torch.bool) if i < 4 else torch.zeros(1, 6, dtype=torch.bool),
                  "taps": torch.full((1, 6, 15360), i/100, dtype=torch.bfloat16)}
        torch.save({"key": name, "population": "n1" if i < 4 else "captures", "inputs": window}, inputs/name)
        files[name] = sha256(inputs/name)
        torch.save({"inputs_sha256": files[name], "drafts": torch.tensor([[1, 2, 3, 4, 5]])}, reference/name)
    write_json(inputs/"inputs.json", {"files": files})
    build(inputs, reference, out)
    manifest = json.loads((out/"MANIFEST.json").read_text())
    assert manifest["synthetic"] is True
    assert len(manifest["synthetic_eval_windows"]) == 50
    entry = manifest["synthetic_eval_windows"]["synthetic-acap-000"]
    original = torch.load(out/entry["file"], weights_only=False)
    assert not original["has_taps"].any()
    assert sha256(out/entry["file"]) == entry["sha256"]
    train = D2Blocks(out/"MANIFEST.json", "train")
    assert len(train) == 204
    assert train[0]["targets"]["labels"][0].tolist() == [1, 2, 3, 4, 5]
    train_meta = json.loads((out/"train.json").read_text())
    assert {s["synthetic_window_key"] for s in train_meta["sequences"]} == {f"window-{i}.pt" for i in range(4)}
    captured: list[str] = []
    for split in ("acap", "heldout_legal"):
        d = D2Blocks(out/"MANIFEST.json", split)
        assert len(d.sequences) == 25 and len(d) == 125
        meta = json.loads((out/(split+".json")).read_text())
        captured.extend(s["synthetic_window_key"] for s in meta["sequences"])
    assert len(captured) == len(set(captured)) == 50
    assert set(captured) == {f"window-{i}.pt" for i in range(4, 54)}
    assert json.loads((out/"synthetic-g1.json").read_text())["pass"] is False
