#!/usr/bin/env python3
"""CPU-only rebuild: image v5 + verified K1 + pinned RP1 patches, fuzz 0.

No upstream script is executed. SRC and DONOR are read-only inputs. Use --write
only to regenerate the committed overlay/ and MANIFEST.json.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import py_compile
import shutil
import subprocess
import tempfile
from typing import Any, cast

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
DONOR_SHA = "afe4d5e41ec6fa4db17c5cc88758af439a106913"
PATCHES: dict[str, str] = {'0056-dsv41-kv-offload-crc.patch': 'd1073571912d229b65af8c070ee0657ad3caa7cce66aa82af20a19fb06a322e4', '0058-dsv41-offload-expandable-segments.patch': '6fb50e3a1b0e4dbf92901011599e0b01b546f7b1a2c532b8cbacd4a569e854ed', '0077-dsv41-kv-offload-two-nodes.patch': '6f818dddd4ecc16bc8e1745914459fd8dbc54a95e19b19bbde2687ac879374e4', '0081-dsv41-kv-offload-load-cap.patch': 'a982833bac6a264b2b119149f283297cdcf61ffa6f2a33c5fcb7f634a07a168a'}
FILES = (
    "vllm/config/vllm.py",
    "vllm/distributed/kv_transfer/kv_connector/v1/offloading_connector.py",
    "vllm/distributed/kv_transfer/kv_connector/v1/offloading/config.py",
    "vllm/distributed/kv_transfer/kv_connector/v1/offloading/worker.py",
    "vllm/distributed/kv_transfer/kv_connector/v1/offloading/scheduler.py",
    "vllm/v1/kv_offload/tiering/fs/io.py",
    "vllm/v1/kv_offload/tiering/fs/manager.py",
    "vllm/v1/kv_offload/tiering/fs/thread_pool.py",
)
# MC1: modules authored in this repo (not patch outputs). Each is a new container path that the image must not
# already have; MANIFEST pins its bytes like the patched files.
NEW_FILES = (
    "vllm/distributed/kv_transfer/kv_connector/v1/offloading/dsv41_blockhash.py",
    "vllm/distributed/kv_transfer/kv_connector/v1/offloading/dsv41_stream.py",
    "vllm/v1/kv_offload/tiering/fs/dsv41_matter.py",
    "vllm/v1/kv_offload/tiering/fs/dsv41_matter_tier.py",
)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(args: list[str], cwd: Path | None = None) -> str:
    return subprocess.run(args, cwd=cwd, check=True, text=True, capture_output=True).stdout.strip()


def verify_checkout() -> dict[str, Any]:
    manifest = json.loads((HERE / "MANIFEST.json").read_text())
    for item in manifest["files"] + manifest.get("new_files", []):
        path = HERE / "overlay" / item["path"]
        if sha(path) != item["overlay_sha256"]:
            raise ValueError(f"overlay SHA mismatch: {path}")
    if [e["path"] for e in manifest.get("new_files", [])] != list(NEW_FILES):
        raise ValueError("MANIFEST new_files differ from NEW_FILES; regenerate with --write")
    if sha(ROOT / "offload/kvpack/MANIFEST.json") != manifest["k1_manifest_sha256"]:
        raise ValueError("K1 manifest changed; rebuild/audit RP1 against that version")
    k1 = json.loads((ROOT / "offload/kvpack/MANIFEST.json").read_text())
    for item in k1["files"]:
        if sha(ROOT / "offload/kvpack/overlay" / item["path"]) != item["overlay_sha256"]:
            raise ValueError(f"K1 overlay SHA mismatch: {item['path']}")
    return cast(dict[str, Any], manifest)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true")
    parser.add_argument("--checkout-only", action="store_true")
    args = parser.parse_args()
    if args.checkout_only:
        m = verify_checkout()
        print(f"verified RP1 checkout: {len(m['files'])} files + K1")
        return
    src = Path(os.environ.get("SRC", "/opt/dsv41/data/src-image"))
    donor = Path(os.environ.get("DONOR", "/opt/dsv41/data/upstream/henryous"))
    if run(["git", "-C", str(donor), "rev-parse", "HEAD"]) != DONOR_SHA:
        raise ValueError("donor HEAD mismatch")
    if run(["git", "-C", str(donor), "status", "--porcelain"]):
        raise ValueError("donor is dirty")
    # Verify K1's 13 committed bytes; its independent rebuild test is also required.
    k1_path = ROOT / "offload/kvpack/MANIFEST.json"
    k1 = json.loads(k1_path.read_text())
    with tempfile.TemporaryDirectory(prefix="rp1-rebuild-") as tmp:
        tree = Path(tmp) / "tree"
        # Rebuild K1 too: the complete v5 -> K1 -> RP1 chain uses fuzz zero.
        for item in k1["files"]:
            path = ROOT / "offload/kvpack/overlay" / item["path"]
            if sha(path) != item["overlay_sha256"]:
                raise ValueError(f"K1 overlay SHA mismatch: {path}")
            base = src / item["path"]
            if item["base_sha256"] is None:
                if base.exists(): raise ValueError(f"K1 new file already exists in base: {base}")
            else:
                if sha(base) != item["base_sha256"]: raise ValueError(f"K1 base SHA mismatch: {base}")
                out = tree / item["path"]
                out.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(base, out)
        for item in k1["patches"]:
            path = donor / "patches" / (item["name"] + ".patch")
            if sha(path) != item["sha256"]: raise ValueError(f"K1 patch SHA mismatch: {path}")
            result = subprocess.run(["patch", "--batch", "--fuzz=0", "-p1", "--no-backup-if-mismatch"],
                                    input=path.read_text(), text=True, cwd=tree, capture_output=True)
            if result.returncode:
                raise RuntimeError(f"K1 fuzz-0 patch failed: {path.name}\n{result.stdout}\n{result.stderr}")
        for item in k1["files"]:
            if sha(tree / item["path"]) != item["overlay_sha256"]:
                raise ValueError(f"K1 fuzz-0 rebuild mismatch: {item['path']}")
        for f in FILES:
            out = tree / f
            if out.exists():
                raise ValueError(f"new K1/RP1 overlap needs audit: {f}")
            out.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src / f, out)
        patches = []
        for name, digest in PATCHES.items():
            path = donor / "patches" / name
            if sha(path) != digest:
                raise ValueError(f"donor patch SHA mismatch: {name}")
            result = subprocess.run(["patch", "--batch", "--fuzz=0", "-p1", "--no-backup-if-mismatch"],
                                    input=path.read_text(), text=True, cwd=tree, capture_output=True)
            if result.returncode:
                raise RuntimeError(f"fuzz-0 patch failed: {name}\n{result.stdout}\n{result.stderr}")
            patches.append({"name": name, "sha256": digest})
        for path in sorted((HERE / "patches").glob("*.patch")):
            result = subprocess.run(["patch", "--batch", "--fuzz=0", "-p1", "--no-backup-if-mismatch"],
                                    input=path.read_text(), text=True, cwd=tree, capture_output=True)
            if result.returncode:
                raise RuntimeError(f"local fuzz-0 patch failed: {path.name}\n{result.stdout}\n{result.stderr}")
            patches.append({"name": "local/" + path.name, "sha256": sha(path)})
        entries = []
        for f in FILES:
            py_compile.compile(str(tree / f), cfile=str(Path(tmp) / "check.pyc"), doraise=True)
            entries.append({"path": f, "container_path": "/usr/local/lib/python3.12/dist-packages/" + f,
                            "base_sha256": sha(src / f), "overlay_sha256": sha(tree / f)})
        new_entries = []
        for f in NEW_FILES:
            if (src / f).exists():
                raise ValueError(f"new MC1 file already exists in base: {f}")
            if f in FILES or any(item["path"] == f for item in k1["files"]):
                raise ValueError(f"new MC1 file overlaps a patched/K1 file: {f}")
            py_compile.compile(str(HERE / "overlay" / f), cfile=str(Path(tmp) / "check.pyc"), doraise=True)
            new_entries.append({"path": f, "container_path": "/usr/local/lib/python3.12/dist-packages/" + f,
                                "base_sha256": None, "overlay_sha256": sha(HERE / "overlay" / f)})
        manifest = {"image": k1["image"], "donor_sha": DONOR_SHA, "patch_fuzz": 0,
                    "k1_manifest_sha256": sha(k1_path), "patches": patches, "files": entries,
                    "new_files": new_entries}
        if args.write:
            for f in FILES:
                out = HERE / "overlay" / f
                out.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(tree / f, out)
            (HERE / "MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n")
            print(f"wrote {len(entries)} RP1 overlay files + MANIFEST ({len(new_entries)} new MC1 files pinned)")
        else:
            if manifest != verify_checkout():
                raise ValueError("fresh fuzz-0 rebuild differs from MANIFEST")
            for item in entries:
                if (tree / item["path"]).read_bytes() != (HERE / "overlay" / item["path"]).read_bytes():
                    raise ValueError(f"rebuild byte mismatch: {item['path']}")
            print(f"verified byte-identical fuzz-0 rebuild: {len(entries)} RP1 files + K1; {len(new_entries)} new MC1 files pinned")


if __name__ == "__main__":
    main()
