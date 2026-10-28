"""drafter_identity / markov_identity of CONTRACT.md 6.7 (v1.1), computed on the worker from what it actually loads.

drafter_identity = sha256 over the lines "<basename> <sha256-hex>\\n", sorted by basename, of the ten files the drafter
reads (DRAFTER_FILES), using the Spark-side SOURCE sha256 that lane S2 recorded in DSPARK_COPY_DONE (files[]): the same
digest lane O3's offload/dspark/spark/tools/identity.py prints for the Spark proxy. The worker then ties that receipt to
the bytes it uses: config.json / exl3_moe_plan.json / markov_top_tokens.json are copied verbatim (their sha256 must equal
the receipt's), the extracted safetensors must match the receipt's extracted_files hashes, and every tensor's
manifest entry names one of the receipt's source files with the same source sha256 (the tensor bytes themselves are
checked against their source ranges by checkpoint.Checkpoint). Any mismatch raises.

Coverage (S5R fix 1): every destination file the manifest makes Checkpoint open must be one of the receipt's
extracted_files, named exactly as the receipt names it, resolving inside the delivery root, and whole-file
sha256-verified in this call. A manifest that redirects a tensor to any other file (an unlisted file, a path outside
the root, a symlink or alias escaping it) is refused, so the tensor bytes Checkpoint serves always come from a file the
receipt binds.
"""

from __future__ import annotations

import hashlib
import json
import os

DRAFTER_FILES = (
    "config.json", "exl3_moe_plan.json",
    "model-00002-of-00046.safetensors", "model-00043-of-00046.safetensors", "model-00044-of-00046.safetensors",
    "model-00045-of-00046.safetensors", "model-00046-of-00046.safetensors",
    "model-exl3-head.safetensors", "model-exl3-mtp.safetensors", "model-exl3-mtp-experts.safetensors",
)
VERBATIM = ("config.json", "exl3_moe_plan.json", "markov_top_tokens.json")
MARKOV_FILE = "markov_top_tokens.json"


class IdentityMismatch(RuntimeError):
    pass


def _sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def resolve_inside(root: str, name: str) -> str:
    """Real path of root/name; raises unless name is a plain relative path whose real path stays inside root."""
    if not isinstance(name, str) or not name or os.path.isabs(name) or "\x00" in name:
        raise IdentityMismatch(f"destination {name!r}: not a plain relative path")
    if any(part in ("", ".", "..") for part in name.replace(os.sep, "/").split("/")):
        raise IdentityMismatch(f"destination {name!r}: '.', '..' or empty path components")
    real_root = os.path.realpath(root)
    real = os.path.realpath(os.path.join(root, name))
    if os.path.commonpath([real_root, real]) != real_root or real == real_root:
        raise IdentityMismatch(f"destination {name!r} resolves to {real}, outside the delivery root {real_root}")
    return real


def compute(root: str, manifest: dict | None = None, check_extracted: bool = True) -> dict:
    with open(os.path.join(root, "DSPARK_COPY_DONE")) as f:
        receipt = json.load(f)
    src = {os.path.basename(e.get("source", e["file"])): e["sha256"].lower() for e in receipt["files"]}
    missing = [n for n in DRAFTER_FILES if n not in src]
    if missing:
        raise IdentityMismatch(f"receipt lacks {missing}")
    lines = "".join(f"{n} {src[n]}\n" for n in sorted(DRAFTER_FILES))
    drafter = hashlib.sha256(lines.encode()).hexdigest()
    for n in VERBATIM:
        path = os.path.join(root, n)
        if n not in src or not os.path.exists(path):
            raise IdentityMismatch(f"{n}: missing from the receipt or the delivery directory")
        got = _sha256_file(path)
        if got != src[n]:
            raise IdentityMismatch(f"{n}: sha256 {got} != receipt {src[n]}")
    markov = src[MARKOV_FILE]
    if manifest is not None and not check_extracted:
        raise IdentityMismatch("a manifest's destinations can only be bound to the receipt with check_extracted")
    covered: dict[str, str] = {}          # receipt name -> real path, whole-file sha256 verified below
    if check_extracted:
        extracted = receipt.get("extracted_files")
        if not extracted:
            raise IdentityMismatch("receipt has no extracted_files: the extracted safetensors cannot be tied to it")
        for e in extracted:
            real = resolve_inside(root, e["file"])
            if not os.path.isfile(real):
                raise IdentityMismatch(f"extracted {e['file']}: {real} is not a regular file")
            got = _sha256_file(real)
            if got != e["sha256"].lower():
                raise IdentityMismatch(f"extracted {e['file']}: sha256 {got} != receipt {e['sha256']}")
            covered[e["file"]] = real
    if manifest is not None:
        for t in manifest["tensors"]:
            dest = t.get("destination_file")
            if dest not in covered:
                raise IdentityMismatch(f"tensor {t['name']}: destination {dest!r} is not one of the receipt's "
                                       f"verified extracted_files {sorted(covered)}")
            if resolve_inside(root, dest) != covered[dest]:
                raise IdentityMismatch(f"tensor {t['name']}: destination {dest!r} changed while being verified")
            want = src.get(os.path.basename(t["source_file"]))
            if want is None or want != t["source_file_sha256"].lower():
                raise IdentityMismatch(f"tensor {t['name']}: source {t['source_file']} sha256 "
                                       f"{t['source_file_sha256']} not the receipt's {want}")
    return {"drafter_identity": drafter, "markov_identity": markov, "files": {n: src[n] for n in sorted(DRAFTER_FILES)}}


if __name__ == "__main__":
    import sys
    r = compute(sys.argv[1] if len(sys.argv) > 1 else "/opt/dsv41/x86host-data/dspark")
    print(f"drafter_identity={r['drafter_identity']}\nmarkov_identity={r['markov_identity']}")
