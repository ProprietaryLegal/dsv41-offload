#!/usr/bin/env python3
"""CPU-only promotion gate, local patch transaction and byte-exact rollback.

No service, ssh or HTTP operations. The S18 receipt is an explicit review
attestation, bound to a report hash, not a substitute for independent review.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
from typing import cast

from profiles import PROFILES, Profile, get_profile

PACKAGE = Path(__file__).resolve().parent
ROOT = PACKAGE.parents[2]
MANIFEST_SHA = "559c5867ba42aa7daa7b4a7c8231d15ee1d35be40ae5b715b013f973193c58de"
FILES = ("dsv41.env", "start.sh")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_object(path: Path) -> dict[str, object]:
    value: object = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return cast(dict[str, object], value)


def gate(receipt: Path, report: Path, profile: Profile) -> None:
    record = read_object(receipt)
    expected: dict[str, object] = {
        "schema": "dsv41-k1-promotion-v1", "reviewer": "S18", "final": True,
        "window6": "PROMOTE", "max_model_len": profile.max_model_len,
        "kv_tokens": profile.kv_tokens, "manifest_sha256": MANIFEST_SHA,
        "report_sha256": sha(report),
    }
    if profile.requires_r5:
        expected["window6b"] = "PROMOTE"  # explicit R5 approval; S18 currently says HOLD
    for key, want in expected.items():
        got = record.get(key)
        if type(got) is not type(want) or got != want:
            raise ValueError(f"S18 final PROMOTE gate: {key}={got!r}, expected {want!r}")


def manifest(root: Path) -> None:
    if sha(root / "offload/kvpack/MANIFEST.json") != MANIFEST_SHA:
        raise ValueError("K1 manifest differs from the reviewed promotion manifest")
    subprocess.run(["bash", str(root / "offload/kvpack/stage_overlay.sh"), "verify-checkout"],
                   cwd=root, check=True)


def git_apply(root: Path, profile: Profile, *, check: bool = False) -> None:
    args = ["git", "apply"] + (["--check"] if check else [])
    patches = [PACKAGE / f"dsv41.env.{profile.name}.patch", PACKAGE / "start.sh.patch"]
    subprocess.run(args + [str(path) for path in patches], cwd=root, check=True)


def apply(root: Path, receipt: Path, report: Path, profile: Profile) -> None:
    gate(receipt, report, profile)
    manifest(root)
    git_apply(root, profile, check=True)
    state = root / "logs/k1-promotion"
    state.mkdir(parents=True, exist_ok=False)  # never overwrite the pre-switch backup
    for name in FILES:
        shutil.copy2(root / name, state / name)
    shutil.copyfile(receipt, state / "S18.receipt.json")
    shutil.copyfile(report, state / "S18.report.md")
    record: dict[str, object] = {
        "profile": profile.name,
        "original": {name: sha(state / name) for name in FILES},
    }
    # git apply is atomic across the two paths; no services are started here.
    git_apply(root, profile)
    record["promoted"] = {name: sha(root / name) for name in FILES}
    (state / "files.json").write_text(json.dumps(record, indent=2) + "\n")
    print(f"K1 patches applied locally; byte-exact rollback saved in {state}. No service changed.")


def hashes(record: dict[str, object], key: str) -> dict[str, str]:
    result = record.get(key)
    if not isinstance(result, dict) or set(result) != set(FILES):
        raise ValueError(f"invalid rollback record: {key}")
    if not all(isinstance(v, str) and len(v) == 64 for v in result.values()):
        raise ValueError(f"invalid rollback hashes: {key}")
    return cast(dict[str, str], result)


def saved(root: Path) -> tuple[Path, dict[str, str], dict[str, str]]:
    state = root / "logs/k1-promotion"
    record = read_object(state / "files.json")
    original, promoted = hashes(record, "original"), hashes(record, "promoted")
    for name in FILES:
        if sha(state / name) != original[name]:
            raise ValueError(f"rollback backup corrupted: {name}")
    return state, original, promoted


def verify(root: Path, selected: str | None = None) -> None:
    state, _, promoted = saved(root)
    value = read_object(state / "files.json").get("profile")
    if not isinstance(value, str):
        raise ValueError("promotion record has no explicit profile")
    profile = get_profile(value)
    if selected is not None and selected != profile.name:
        raise ValueError(f"applied profile {profile.name} differs from configured profile {selected}")
    gate(state / "S18.receipt.json", state / "S18.report.md", profile)
    for name in FILES:
        if sha(root / name) != promoted[name]:
            raise ValueError(f"promoted {name} changed since apply; refusing start")
    manifest(root)
    print(f"K1 profile {profile.name}, promotion record, saved rollback and reviewed checkout verified")


def restore(root: Path) -> None:
    state, original, promoted = saved(root)
    # Refuse to discard edits made after promotion. Validate both before touching either.
    for name in FILES:
        if sha(root / name) not in (original[name], promoted[name]):
            raise ValueError(f"{name} has later edits; refusing to discard them during rollback")
    for name in FILES:
        temporary = root / f".{name}.k1-restore"
        with temporary.open("xb") as stream:
            stream.write((state / name).read_bytes())
        shutil.copymode(state / name, temporary)
        os.replace(temporary, root / name)
        if sha(root / name) != original[name]:
            raise ValueError(f"byte-exact rollback verification failed: {name}")
    print("K1 rollback restored byte-identical dsv41.env and start.sh; backup retained")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("check", "apply", "verify", "restore"))
    parser.add_argument("--profile", choices=tuple(PROFILES))
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--receipt", type=Path)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    try:
        if args.action in ("check", "apply"):
            if args.profile is None:
                raise ValueError("check/apply require --profile (327680 or 524288); no default")
            profile = get_profile(args.profile)
            if args.receipt is None or args.report is None:
                raise ValueError("check/apply require --receipt and --report from S18's final review")
            if args.action == "apply":
                apply(args.root, args.receipt, args.report, profile)
            else:
                gate(args.receipt, args.report, profile)
                manifest(args.root)
                git_apply(args.root, profile, check=True)
                print("K1 final PROMOTE gate and patch check passed; nothing changed")
        elif args.action == "verify":
            verify(args.root, args.profile)
        else:
            restore(args.root)
    except (ValueError, OSError, subprocess.CalledProcessError) as exc:
        parser.exit(1, f"K1 REFUSED: {exc}\n")


if __name__ == "__main__":
    main()
