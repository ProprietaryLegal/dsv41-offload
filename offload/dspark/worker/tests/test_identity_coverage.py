"""CPU regressions for S5R fix 1: the receipt must bind every destination file the loader opens.

Before the fix, identity.compute checked only the files the receipt happened to list in extracted_files, so a manifest
entry redirected to an unlisted (or out-of-root) file passed with the original drafter identity
(/opt/dsv41/repos/.worktrees/dsv41-s5/s5r-evidence/test_s5r_boundaries.py::test_identity_rejects_destination_not_
covered_by_receipt). These tests build a scratch delivery; no real tensor data, no GPU."""

import hashlib
import json
import os

import pytest

from worker.identity import DRAFTER_FILES, VERBATIM, IdentityMismatch, compute, resolve_inside


def _delivery(tmp_path, extracted=("covered.safetensors",)):
    """A minimal delivery: verbatim files hashed into the receipt, the named extracted files covered by it."""
    src = {name: "ab" * 32 for name in DRAFTER_FILES}
    for name in VERBATIM:
        data = ("scratch " + name).encode()
        (tmp_path / name).write_bytes(data)
        src[name] = hashlib.sha256(data).hexdigest()
    ex = []
    for name in extracted:
        data = ("extracted " + name).encode()
        (tmp_path / name).write_bytes(data)
        ex.append({"file": name, "sha256": hashlib.sha256(data).hexdigest()})
    receipt = {"files": [{"file": n, "sha256": d} for n, d in src.items()], "extracted_files": ex}
    (tmp_path / "DSPARK_COPY_DONE").write_text(json.dumps(receipt))
    return src


def _manifest(src, dest):
    return {"tensors": [{"name": "mtp.0.ffn.gate.bias", "destination_file": dest,
                         "source_file": "model-exl3-mtp.safetensors",
                         "source_file_sha256": src["model-exl3-mtp.safetensors"]}]}


def test_covered_destination_accepted(tmp_path):
    src = _delivery(tmp_path)
    r = compute(str(tmp_path), _manifest(src, "covered.safetensors"))
    assert len(r["drafter_identity"]) == 64


def test_destination_not_covered_by_receipt_refused(tmp_path):
    """S5R's reproduction: an unlisted file next to a nonempty receipt."""
    src = _delivery(tmp_path)
    (tmp_path / "uncovered.safetensors").write_bytes(b"different bytes")
    with pytest.raises(IdentityMismatch, match="not one of the receipt"):
        compute(str(tmp_path), _manifest(src, "uncovered.safetensors"))


def test_alias_of_a_covered_file_refused(tmp_path):
    """A second name for a covered file (a symlink inside the root) is still not the receipt's name."""
    src = _delivery(tmp_path)
    os.symlink(tmp_path / "covered.safetensors", tmp_path / "alias.safetensors")
    with pytest.raises(IdentityMismatch, match="not one of the receipt"):
        compute(str(tmp_path), _manifest(src, "alias.safetensors"))


@pytest.mark.parametrize("dest", ["../outside.safetensors", "/etc/passwd", "sub/../covered.safetensors", "",
                                  "./covered.safetensors"])
def test_destination_paths_escaping_or_unnormalised_refused(tmp_path, dest):
    src = _delivery(tmp_path)
    (tmp_path.parent / "outside.safetensors").write_bytes(b"outside")
    with pytest.raises(IdentityMismatch):
        compute(str(tmp_path), _manifest(src, dest))


def test_receipt_entry_symlinked_outside_the_root_refused(tmp_path):
    """The receipt's own extracted file may not resolve outside the delivery (hash would bind foreign bytes)."""
    outside = tmp_path.parent / "elsewhere"
    outside.mkdir()
    data = b"extracted covered.safetensors"
    (outside / "covered.safetensors").write_bytes(data)
    root = tmp_path / "root"
    root.mkdir()
    _delivery(root, extracted=())
    receipt = json.loads((root / "DSPARK_COPY_DONE").read_text())
    receipt["extracted_files"] = [{"file": "covered.safetensors", "sha256": hashlib.sha256(data).hexdigest()}]
    (root / "DSPARK_COPY_DONE").write_text(json.dumps(receipt))
    os.symlink(outside / "covered.safetensors", root / "covered.safetensors")
    with pytest.raises(IdentityMismatch, match="outside the delivery root"):
        compute(str(root))


def test_manifest_without_extracted_check_refused(tmp_path):
    src = _delivery(tmp_path)
    with pytest.raises(IdentityMismatch, match="check_extracted"):
        compute(str(tmp_path), _manifest(src, "covered.safetensors"), check_extracted=False)


def test_resolve_inside_accepts_plain_names_only(tmp_path):
    (tmp_path / "a.safetensors").write_bytes(b"x")
    assert resolve_inside(str(tmp_path), "a.safetensors") == os.path.realpath(tmp_path / "a.safetensors")
    for bad in ("..", ".", "a/../a.safetensors", "/abs", "a\x00b"):
        with pytest.raises(IdentityMismatch):
            resolve_inside(str(tmp_path), bad)
