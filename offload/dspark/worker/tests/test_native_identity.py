"""Native identity covers runtime/arm/weights and cannot negotiate with K2."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from offload.dspark.native.identity import compute, digest, official_hashes
from offload.dspark.spark.tools.identity import DRAFTER_FILES, MARKOV_FILE, identities
from worker import protocol as P
from worker.server import DraftCore, FakeBackend, expected_hello


@pytest.fixture
def delivery(tmp_path):
    source = tmp_path / "source"
    native = tmp_path / "weights"
    (source / "native").mkdir(parents=True)
    (source / "worker").mkdir()
    native.mkdir()
    files = {"config.json": b"{}", "model.safetensors.index.json": b"{}", "shard.safetensors": b"native"}
    for n, b in files.items():
        (native / n).write_bytes(b)
    manifest = {"receipt": "\n".join(f"{hashlib.sha256(b).hexdigest()}  {n}" for n, b in files.items()),
                "tensors": [{"file": "shard.safetensors"}], "official_repo": "official",
                "official_revision": "revision"}
    (source / "native/NATIVE_MANIFEST.json").write_text(json.dumps(manifest))
    (source / "worker/kernel.cu").write_text("kernel")
    receipt = {"files": [{"file": n, "sha256": hashlib.sha256(n.encode()).hexdigest()}
                          for n in (*DRAFTER_FILES, MARKOV_FILE)]}
    d, m, used = identities(receipt)
    checkpoint = {"drafter_identity": d, "markov_identity": m, "files": used}
    return source, native, checkpoint


def test_live_verification_equals_offline_spark_recipe_and_changes_with_sources(delivery):
    source, native, ck = delivery
    arm = "native-hybrid-kx2"
    got = compute(arm, ck, native_root=str(native), source_root=source)
    assert got == compute(arm, ck, source_root=source)
    uppercase = {**ck, "files": {n: h.upper() for n, h in ck["files"].items()}}
    assert got == compute(arm, uppercase, source_root=source)
    assert got["drafter_identity"] != ck["drafter_identity"]
    assert "K6" in got["draft_head"]
    assert got["drafter_identity"] == digest(got["recipe"])
    other = copy.deepcopy(got["recipe"])
    other["arm"]["dense_profile"] = "initial"
    assert digest(other) != got["drafter_identity"]
    (source / "worker/kernel.cu").write_text("different kernel")
    assert compute(arm, ck, source_root=source)["drafter_identity"] != got["drafter_identity"]
    ck["files"]["model-exl3-head.safetensors"] = "a" * 64
    assert compute(arm, ck, source_root=source)["drafter_identity"] != got["drafter_identity"]


@pytest.mark.parametrize("damage", ["missing", "tampered", "redirected"])
def test_native_shard_refusal(delivery, damage, tmp_path):
    source, native, ck = delivery
    p = native / "shard.safetensors"
    p.unlink()
    if damage == "tampered":
        p.write_bytes(b"bad")
    if damage == "redirected":
        target = tmp_path / "outside"
        target.write_bytes(b"native")
        p.symlink_to(target)
    with pytest.raises(ValueError, match="official"):
        compute("native-packed", ck, source_root=source, native_root=str(native))


def test_hello_refuses_every_cross_arm_pair_without_touching_session(delivery):
    source, _, ck = delivery
    arms = [ck] + [compute(a, ck, source_root=source) for a in (
        "native-packed", "native-expanded", "native-hybrid-kx", "native-hybrid-kx2")]
    assert len({a["drafter_identity"] for a in arms}) == 5
    for worker in arms:
        exp = expected_hello(bytes.fromhex(worker["drafter_identity"]), bytes.fromhex(worker["markov_identity"]))
        backend = FakeBackend(exp.window, exp.row_bytes, exp.max_slots, lambda *a: None, exp.vocab_size)
        core = DraftCore(exp, backend)
        for client in arms:
            req = expected_hello(bytes.fromhex(client["drafter_identity"]), bytes.fromhex(client["markov_identity"]))
            frame = P.AppHeader(P.MAGIC, P.VERSION, P.OP_HELLO, 1, 123, len(req.encode()), 0, 0).encode() + req.encode()
            rep = P.AppHeader.decode(core.handle(frame))
            same = client["drafter_identity"] == worker["drafter_identity"]
            assert rep.status == (P.OK if same else P.IDENTITY_MISMATCH)
            assert core.session == (123 if same or core.session == 123 else None)


def test_official_receipt_cannot_omit_tensor_file():
    with pytest.raises(ValueError, match="receipt lacks"):
        official_hashes({"receipt": "", "tensors": [{"file": "absent"}]})
