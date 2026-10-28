"""Run V1's independent official reference/analyze/classify on a D2 candidate.

V1's original HF shard hashes necessarily change for an adapted checkpoint.
This bridge substitutes a verified candidate receipt before permitting that
load. It never disables the official model.py code hash verification. No V1
file is changed. D2 taps-only windows have no engine main_x to compare with.
"""
from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path
import sys
from typing import Any

import torch

from ...spark.tools.identity import identities
from .data import sha256
from .replay import load_capture_set, v1_import


def verify_package(root: Path) -> dict[str, Any]:
    receipt: dict[str, Any] = json.loads((root/"PACKAGE_RECEIPT.json").read_text())
    if receipt["schema"] != "d2.native-package.v1" or receipt["publish"] or receipt["arm"] != "native-fp8":
        raise ValueError("not a local native D2 package receipt")
    for row in receipt["files"]:
        path = (root/row["file"]).resolve()
        if path.parent != root.resolve() or sha256(path) != row["sha256"]:
            raise ValueError("candidate package file identity mismatch")
    drafter, markov, _ = identities(receipt, tuple(receipt["identity_files"]))
    if (drafter, markov) != (receipt["drafter_identity"], receipt["markov_identity"]):
        raise ValueError("candidate package aggregate identities disagree")
    return receipt


def taps_window(ref: Any, cs: Any, sample: Any, caches: Any, fault: str = "") -> tuple[Any, Any, Any, Any]:
    if fault:
        raise ValueError("D2 candidate bridge does not implement injected-state faults")
    positions = cs.window(sample)
    if cs.window_status(sample)["missing"]:
        raise ValueError("D2 reference window is incomplete")
    kv = []
    for p in positions:
        key = sample.req, p
        row = cs.rows[sample.req][p]
        if row.taps is None:
            raise ValueError("D2 reference taps-only window lacks taps")
        if key not in caches.mx_native:
            caches.mx_native[key] = ref.main_x(row.taps)
        if key not in caches.kv:
            caches.kv[key] = ref.stage_kv(p, caches.mx_native[key])
        kv.append(caches.kv[key])
    ref.reset_window()
    ref.fill_window(positions, torch.stack(kv))
    return positions, ["taps"]*len(positions), caches.mx_native[(sample.req, sample.P)], "taps"


def main() -> None:
    ap = argparse.ArgumentParser(__doc__)
    ap.add_argument("--v1-root", required=True)
    ap.add_argument("--tool", choices=("ref", "analyze", "classify"), required=True)
    ap.add_argument("--package-root", type=Path)
    ap.add_argument("--data-manifest", type=Path)
    ap.add_argument("--data-reader")
    ap.add_argument("--split", choices=("dev", "heldout_legal", "heldout_general", "acap"))
    ap.add_argument("--k2-mapping")
    ap.add_argument("v1_args", nargs=argparse.REMAINDER)
    a = ap.parse_args()
    v1_import(a.v1_root)
    runner = importlib.import_module("verify.run_ref")
    module = importlib.import_module("verify."+{"ref": "run_ref", "analyze": "analyze", "classify": "classify"}[a.tool])
    arguments = a.v1_args[1:] if a.v1_args[:1] == ["--"] else a.v1_args
    if a.package_root:
        receipt = verify_package(a.package_root)
        original = runner.NativeRef

        def candidate_reference(*args: Any, **kwargs: Any) -> Any:
            kwargs["verify"] = False
            ref = original(*args, **kwargs)
            ref.provenance["candidate_package"] = receipt
            return ref

        setattr(runner, "NativeRef", candidate_reference)
        if a.tool == "classify":
            # classify imports its constructor inside main, from this module.
            setattr(importlib.import_module("verify.native_ref"), "NativeRef", candidate_reference)
        if "--ckpt" in arguments:
            raise ValueError("--package-root owns --ckpt for candidate verification")
        arguments += ["--ckpt", str(a.package_root)]
    if a.data_manifest:
        if not a.data_reader or not a.split:
            ap.error("D2 reference requires --data-reader and --split")
        if a.tool == "analyze" and not a.k2_mapping:
            raise ValueError("V1 acceptance analysis requires an actual K2 mapping")
        cs = load_capture_set(str(a.data_manifest), a.data_reader, a.split, a.k2_mapping)
        if cs.plumbing:
            raise ValueError(f"D2 V1 plumbing errors: {len(cs.plumbing)}")
        setattr(runner, "CaptureSet", lambda _: cs)
        if a.tool == "classify":
            setattr(module, "CaptureSet", lambda _: cs)
        setattr(runner, "prepare_window", taps_window)
    sys.argv = ["V1-D2-"+a.tool]+arguments
    module.main()
    if a.package_root and a.tool == "classify":
        path = Path(arguments[arguments.index("--out")+1])
        result = json.loads(path.read_text())
        result.update(candidate_reference_identity=receipt["drafter_identity"], candidate_markov_identity=receipt["markov_identity"])
        path.write_text(json.dumps(result, indent=2)+"\n")


if __name__ == "__main__":
    main()
