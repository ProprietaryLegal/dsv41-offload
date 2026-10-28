"""Read-only Spark inventory; safetensors bytes always bypass the page cache."""
from __future__ import annotations

import json
import hashlib
import mmap
import os
import struct
from pathlib import Path


def direct_read(fd: int, offset: int, size: int) -> bytes:
    start = offset // 4096 * 4096
    length = ((offset - start + size + 4095) // 4096) * 4096
    with mmap.mmap(-1, length) as buffer:
        count = os.preadv(fd, [buffer], start)
        if count < offset - start + size:
            raise EOFError(f"short O_DIRECT read: {offset=} {size=} {count=}")
        return buffer[offset - start:offset - start + size]


def inventory(root: Path, baseline: Path, manifest: Path) -> dict[str, object]:
    files: dict[str, object] = {}
    for path in sorted(root.glob("*.safetensors")):
        fd = os.open(path, os.O_RDONLY | os.O_DIRECT)
        try:
            length = struct.unpack("<Q", direct_read(fd, 0, 8))[0]
            if length > 64 * 1024 * 1024:
                raise ValueError(f"unreasonable header: {path} {length}")
            header = json.loads(direct_read(fd, 8, length))
        finally:
            os.close(fd)
        stat = path.stat()
        base = baseline / path.name
        files[path.name] = {
            "size": stat.st_size, "device": stat.st_dev, "inode": stat.st_ino,
            "nlink": stat.st_nlink, "header_bytes": length, "header": header,
            "baseline_same_inode": base.exists() and os.path.samefile(path, base),
        }
    metadata = {name: json.loads((root / name).read_text()) for name in (
        "config.json", "exl3_moe_plan.json", "CANDIDATE_MANIFEST.json",
        "model.safetensors.index.json")}
    return {"source_root": str(root), "baseline_root": str(baseline),
            "files": files, "metadata": metadata,
            "metadata_sha256": {name: hashlib.sha256((root/name).read_bytes()).hexdigest()
                for name in (*metadata, "tokenizer.json", "tokenizer_config.json")},
            "manifest": json.loads(manifest.read_text())}


if __name__ == "__main__":
    print(json.dumps(inventory(Path("/srv/dsv41/legal-v2a"), Path("/srv/dsv41/model"),
                               Path.home() / "dsv41-deploy/manifest.json")))
