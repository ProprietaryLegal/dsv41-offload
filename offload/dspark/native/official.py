"""Read-only access to the verified official 1017 shards (x86host /opt/dsv41/x86host-data/native-1017).

Tensors come straight from the safetensors bytes through a memory map: weights stay the official bytes (uint8 views
for I8 / F8_E4M3 / F8_E8M0, bf16 / fp32 otherwise), nothing is converted here. ``verify()`` re-hashes tensors against
NATIVE_MANIFEST.json (sha256 of the tensor bytes)."""
from __future__ import annotations

import hashlib
import json
import os
import struct

import numpy as np
import torch

MANIFEST = os.path.join(os.path.dirname(os.path.abspath(__file__)), "NATIVE_MANIFEST.json")


class OfficialShards:
    def __init__(self, root: str, manifest: str | None = MANIFEST):
        self.root = root
        with open(os.path.join(root, "model.safetensors.index.json")) as f:
            self.wmap = json.load(f)["weight_map"]
        self._hdr: dict = {}
        self._mm: dict = {}
        self.manifest = None
        if manifest and os.path.exists(manifest):
            with open(manifest) as f:
                self.manifest = {t["name"]: t for t in json.load(f)["tensors"]}

    def _file(self, fname: str):
        if fname not in self._hdr:
            path = os.path.join(self.root, fname)
            with open(path, "rb") as f:
                n = struct.unpack("<Q", f.read(8))[0]
                h = json.loads(f.read(n))
            h.pop("__metadata__", None)
            self._hdr[fname] = (h, 8 + n)
            self._mm[fname] = np.memmap(path, dtype=np.uint8, mode="r")
        return self._hdr[fname], self._mm[fname]

    def info(self, name: str) -> dict:
        (h, off), _ = self._file(self.wmap[name])
        return h[name]

    def raw(self, name: str) -> np.ndarray:
        """The tensor's bytes (a memory-mapped uint8 view: no copy)."""
        fname = self.wmap[name]
        (h, off), mm = self._file(fname)
        a, b = h[name]["data_offsets"]
        return mm[off + a:off + b]

    def get(self, name: str) -> torch.Tensor:
        """A CPU tensor (a private copy) in the official dtype: uint8 for I8 / F8_*, bf16, fp32."""
        info = self.info(name)
        t = torch.from_numpy(np.array(self.raw(name)))
        dt = info["dtype"]
        if dt == "F32":
            t = t.view(torch.float32)
        elif dt == "BF16":
            t = t.view(torch.bfloat16)
        elif dt not in ("I8", "F8_E4M3", "F8_E8M0"):
            raise ValueError(f"{name}: unexpected dtype {dt}")
        return t.view(info["shape"])

    def verify(self, name: str) -> None:
        if self.manifest is None:
            raise RuntimeError("no NATIVE_MANIFEST.json to verify against")
        want = self.manifest[name]["sha256"]
        got = hashlib.sha256(memoryview(np.ascontiguousarray(self.raw(name)))).hexdigest()
        if got != want:
            raise RuntimeError(f"{name}: sha256 {got} != manifest {want}")
