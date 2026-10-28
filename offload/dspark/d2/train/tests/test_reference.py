from __future__ import annotations

import os
import json
from pathlib import Path
from types import SimpleNamespace

import torch

from offload.dspark.d2.train.reference import taps_window
from offload.dspark.d2.train.replay import capture_inputs, load_capture_set
from offload.dspark.d2.train.replay import g1_drafts
from offload.dspark.d2.train.tests.test_data import write_shard
from offload.dspark.d2.train.data import D2Blocks, sha256


def test_d2_data_reader_same_input_contract_and_taps_only_window(tmp_path: Path) -> None:
    manifest = write_shard(tmp_path)
    reader = os.environ.get("D2_DATA_READER", "/opt/dsv41/repos/.worktrees/dsv41-d2-data/offload/dspark/d2/shards.py")
    cs = load_capture_set(str(manifest), reader, "train")
    assert cs.plumbing == []
    sample = cs.samples[0]
    inp, meta = capture_inputs(cs, sample, "cpu")
    assert inp["anchor"].item() == 4
    assert inp["positions"].tolist() == [[0, 1, 2, 3]]
    assert inp["taps"].shape == (1, 4, 15360)
    assert inp["has_taps"].all() and not inp["engine_mx"].any()
    assert meta["engine_mx_rows"] == 0
    saved = []
    ref = SimpleNamespace(main_x=lambda x: x[:5120].float(), stage_kv=lambda p, x: torch.zeros(3, 512),
                          reset_window=lambda: None, fill_window=lambda p, kv: saved.append((p, kv)))
    cache = SimpleNamespace(mx_native={}, kv={})
    positions, sources, mx, source = taps_window(ref, cs, sample, cache)
    assert positions == [0, 1, 2, 3] and sources == ["taps"]*4 and source == "taps"
    assert saved[0][1].shape == (4, 3, 512)
    assert mx.shape == (5120,)
    taps_window(ref, cs, cs.samples[1], cache)
    assert len(cache.mx_native) == len(cache.kv) == 5


def test_empty_sequence_keeps_v1_sequence_key_numbering(tmp_path: Path) -> None:
    entries = []
    for name, gen in (("empty", 9), ("full", 4)):
        root = tmp_path/name
        root.mkdir()
        manifest = write_shard(root, gen)
        spec = json.loads(manifest.read_text())["splits"]["train"]["files"][0]
        sidecar = root/spec["sidecar"]
        side = json.loads(sidecar.read_text())
        side["sequences"][0]["seq_id"] = name
        sidecar.write_text(json.dumps(side))
        spec["sidecar_sha256"] = sha256(sidecar)
        spec.update(file=name+"/"+spec["file"], sidecar=name+"/"+spec["sidecar"])
        entries.append(spec)
    manifest = tmp_path/"MANIFEST.json"
    manifest.write_text(json.dumps({"schema": "d2.manifest.v1", "checks": {"plumbing_errors": 0}, "splits": {"train": {"files": entries}}}))
    data = D2Blocks(manifest, "train")
    assert data.sequence_b == [1]
    reader = os.environ.get("D2_DATA_READER", "/opt/dsv41/repos/.worktrees/dsv41-d2-data/offload/dspark/d2/shards.py")
    captures = load_capture_set(str(manifest), reader, "train")
    assert captures.plumbing == []
    assert captures.samples[0].b == data.sequence_b[0]


def test_actual_g1_jsonl_mapping_and_manifest_without_optional_error_counter(tmp_path: Path) -> None:
    manifest = write_shard(tmp_path)
    spec = json.loads(manifest.read_text())
    spec["checks"] = {"sequences": 1, "skipped_sequences": 0}  # actual dataset.py shape
    manifest.write_text(json.dumps(spec))
    d = D2Blocks(manifest, "train")
    path = tmp_path/"train.k2_drafts.jsonl"
    path.write_text(json.dumps({"seq_id": "p1", "drafts": {str(P): [0]*5 for P in range(3, 8)}})+"\n")
    assert len(g1_drafts(path, d)["p1"]) == 5
    reader = os.environ.get("D2_DATA_READER", "/opt/dsv41/repos/.worktrees/dsv41-d2-data/offload/dspark/d2/shards.py")
    cs = load_capture_set(str(manifest), reader, "train", str(path))
    assert all(not s.engine_drafts_placeholder for s in cs.samples)
