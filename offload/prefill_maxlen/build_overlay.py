"""Build the PF1 overlay locally from the hash-pinned image source; never deploy."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import tempfile
from pathlib import Path
from typing import cast

HERE = Path(__file__).resolve().parent


def manifest() -> dict[str, str]:
    value = cast(object, json.loads((HERE / "MANIFEST.json").read_text()))
    if not isinstance(value, dict) or not all(
        isinstance(key, str) and isinstance(item, str) for key, item in value.items()
    ):
        raise ValueError("PF1 manifest must contain string keys and values")
    return cast(dict[str, str], value)


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def patched_source(source_root: Path) -> str:
    info = manifest()
    source = source_root / info["path"]
    original = source.read_bytes()
    patch = (HERE / "request-workspace.patch").read_bytes()
    if digest(original) != info["base_sha256"]:
        raise ValueError(f"PF1 base SHA mismatch at {source}; refusing another engine revision")
    if digest(patch) != info["patch_sha256"]:
        raise ValueError("PF1 patch SHA mismatch")
    # Small scratch in the lane tree, never the shared /tmp RAM disk.
    with tempfile.TemporaryDirectory(prefix=".pf1-build-", dir=HERE) as work:
        root = Path(work)
        path = root / info["path"]
        path.parent.mkdir(parents=True)
        path.write_bytes(original)
        applied = subprocess.run(
            ["patch", "--batch", "--fuzz=0", "-p1", "-d", str(root)],
            input=patch, capture_output=True, check=False,
        )
        if applied.returncode:
            raise RuntimeError(
                f"PF1 patch failed ({applied.returncode}): "
                f"{applied.stdout.decode(errors='replace')}"
                f"{applied.stderr.decode(errors='replace')}"
            )
        result = path.read_bytes()
    if digest(result) != info["overlay_sha256"]:
        raise ValueError("PF1 rebuilt overlay SHA mismatch")
    text = result.decode()
    compile(text, info["path"], "exec")
    return text


def build(source_root: Path, output_root: Path) -> Path:
    source_root, output_root = source_root.resolve(), output_root.resolve()
    if output_root == source_root or output_root.is_relative_to(source_root):
        raise ValueError("PF1 output must be outside the read-only image source")
    text = patched_source(source_root)
    info = manifest()
    output = output_root / info["path"]
    if output.exists():
        if output.read_text() != text:
            raise ValueError(f"Refusing to replace different existing overlay: {output}")
    else:
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("x") as stream:
            stream.write(text)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=Path("/opt/dsv41/data/src-image"))
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    output = build(args.source_root, args.output_root)
    print(f"PF1 verified {digest(output.read_bytes())}  {output}")


if __name__ == "__main__":
    main()
