from __future__ import annotations

from pathlib import Path
import hashlib
import json
import os

import pytest
import torch

from offload.dspark.d2.train.package import hardlink_tree, hot_tokens, quantize_fp8
from offload.dspark.d2.train.package import build, training_counts
from offload.dspark.d2.train.tests.test_data import write_shard
from offload.dspark.d2.train.data import sha256
from offload.dspark.d2.train.evaluate import package_markov
from offload.dspark.d2.train.model import Drafter
from offload.dspark.d2.train.synthetic import SyntheticSource, tiny_config
from offload.dspark.native.formats import decode_fp8_block
from offload.dspark.native.official import OfficialShards
from offload.dspark.spark.tools.identity import DRAFTER_FILES, identities
from safetensors.torch import save_file


def test_fp8_block32_ue8m0_roundtrip_and_no_nan() -> None:
    torch.manual_seed(1)
    w = torch.randn(64, 96)*.1
    q, s = quantize_fp8(w)
    got = decode_fp8_block(q, s)
    assert q.shape == w.shape and s.shape == (2, 3)
    assert torch.isfinite(got).all() and bool(((s > 0) & (s < 255)).all())
    error = (got-w).norm()/w.norm()
    assert error < .035
    # Re-encoding decoded values is value-stable, including zero blocks.
    q2, s2 = quantize_fp8(got)
    assert torch.equal(got, decode_fp8_block(q2, s2))
    z, zs = quantize_fp8(torch.zeros(32, 32))
    assert not decode_fp8_block(z, zs).any()


def test_sibling_hardlinks_and_atomic_replacement_preserve_source(tmp_path: Path) -> None:
    original, candidate = tmp_path/"original", tmp_path/"candidate"
    original.mkdir()
    (original/"weights").write_bytes(b"original")
    hardlink_tree(original, candidate)
    assert os.stat(original/"weights").st_ino == os.stat(candidate/"weights").st_ino
    temp = candidate/"weights.new"
    temp.write_bytes(b"adapted")
    os.replace(temp, candidate/"weights")
    assert (original/"weights").read_bytes() == b"original"
    assert (candidate/"weights").read_bytes() == b"adapted"
    with pytest.raises(ValueError):
        hardlink_tree(original, original)


def test_native_identity_uses_canonical_algorithm_without_changing_default() -> None:
    receipt = {"files": [{"file": n, "sha256": f"{i:064x}"} for i, n in enumerate(DRAFTER_FILES)]}
    assert identities(receipt) == identities(receipt, DRAFTER_FILES)
    a = identities({"files": [{"file": "native", "sha256": "a"*64}]}, ("native",))[0]
    b = identities({"files": [{"file": "native", "sha256": "b"*64}]}, ("native",))[0]
    assert a != b
    assert hot_tokens({3: 2, 1: 5, 2: 2}, 3) == [1, 2, 3]


def test_markov_counts_use_normative_training_previous_tokens(tmp_path: Path) -> None:
    assert training_counts(write_shard(tmp_path)) == {4: 1, 5: 2, 6: 3, 7: 4, 8: 5}


def test_complete_synthetic_package_reloads_and_leaves_original_exact(tmp_path: Path) -> None:
    src = SyntheticSource(tiny_config())
    root = tmp_path/"base"
    root.mkdir()
    actual = {}
    for n, t in src.tensors.items():
        dtype = src.dtypes[n]
        actual[n] = t.view({"F8_E4M3": torch.float8_e4m3fn, "F8_E8M0": torch.float8_e8m0fnu, "I8": torch.int8}[dtype]) if dtype in ("F8_E4M3", "F8_E8M0", "I8") else t
    save_file(actual, root/"weights.safetensors")
    (root/"config.json").write_text("{}")
    (root/"model.safetensors.index.json").write_text(json.dumps({"weight_map": {n: "weights.safetensors" for n in actual}}))
    records = [{"name": n, "file": "weights.safetensors", "shape": list(t.shape), "dtype": src.dtypes[n],
                "nbytes": t.numel()*t.element_size(), "sha256": hashlib.sha256(t.view(torch.uint8).numpy().tobytes()).hexdigest()} for n, t in actual.items()]
    manifest = root/"NATIVE_MANIFEST.json"
    manifest.write_text(json.dumps({"tensors": records}))
    original_hash = sha256(root/"weights.safetensors")
    m = Drafter(OfficialShards(str(root), str(manifest)), tiny_config())
    state = {n: p.detach().clone() for n, p in m.named_parameters()}
    state["stages.0.dense.attn__wq_a.b"].fill_(.001)
    state["t.mtp__2__markov_head__embed__weight"].fill_(.123456)
    adapter = tmp_path/"adapter.pt"
    torch.save({"schema": "dsv41.d2.train-checkpoint.v1", "frozen_identity": m.frozen_identity, "trainable": state}, adapter)
    out = tmp_path/"adapted"
    receipt = build(root, out, adapter, {1: 10, 2: 5}, manifest)
    assert receipt["deployed"] is False
    emb, _, verified = package_markov(out, sha256(adapter))
    original_markov = state["t.mtp__2__markov_head__embed__weight"]
    assert torch.equal(emb, original_markov.bfloat16().float())
    assert not torch.equal(emb, original_markov)
    assert verified["drafter_identity"] == receipt["drafter_identity"]
    with pytest.raises(ValueError, match="source adapter"):
        package_markov(out, "wrong-selected-checkpoint")
    assert sha256(root/"weights.safetensors") == original_hash
    loaded = OfficialShards(str(out), str(out/"NATIVE_MANIFEST.json"))
    for n in receipt["changed_tensors"]:
        loaded.verify(n)
    assert loaded.manifest is not None
    for record in records:
        name = str(record["name"])
        if ".experts." in name:
            assert loaded.manifest[name]["sha256"] == record["sha256"]
    bad = dict(state)
    bad["stages.0.dense.attn__wq_a.a"] = bad["stages.0.dense.attn__wq_a.a"][:8]
    bad["stages.0.dense.attn__wq_a.b"] = bad["stages.0.dense.attn__wq_a.b"][:, :8]
    torch.save({"schema": "dsv41.d2.train-checkpoint.v1", "frozen_identity": m.frozen_identity, "trainable": bad}, adapter)
    with pytest.raises(ValueError, match="rank must be 32 or 64"):
        build(root, tmp_path/"out-of-scope", adapter, {1: 10}, manifest)
    assert not (tmp_path/"out-of-scope").exists()
