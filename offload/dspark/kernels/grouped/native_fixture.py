"""Read-only native tensor fixtures for offline qualification; N1 owns the production loader.

Accept only N1's pinned official receipt. CPU conversion streams a matrix at a time;
no downloads or public export. The decoder is intentionally independent of KX-MXFP4.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import struct

import torch

SHARDS = {
 "model-00044-of-00048.safetensors": "3e8ac315c2bcca3e83c5e6e25c5758a75c03732063a6ff0c1b0e24b9ff49f6e6",
 "model-00045-of-00048.safetensors": "9c73b27c4343236790008979d5e0b565f687aa64440031b9e096fed1ab8039c5",
 "model-00046-of-00048.safetensors": "1dccc19a3741972c661de734572dc7d4b899e6e99bfbdf3ae4e057c707e5b339",
 "model-00002-of-00048.safetensors": "43616eebd88dd4f1f1b642772356fc160478d1d9d04edd240431e7d065d2eda7",
 "model-00043-of-00048.safetensors": "4ae12b941c3facf8616ce62dd45bc6868076c2aeb82106f98123ea6c5bb64182",
}


class NativeFixture:
    def __init__(self, root: Path) -> None:
        self.root = root
        index_bytes = (root/"model.safetensors.index.json").read_bytes()
        if hashlib.sha256(index_bytes).hexdigest() != "74b0686a3d2891980d5e303251b075a3bccae2c2ff650747db2620a649b98fa8":
            raise RuntimeError("native index differs from pinned official 1017")
        self.index: dict[str, str] = json.loads(index_bytes)["weight_map"]
        receipt = (root/"RECEIPT.txt").read_text()
        if "1bae0e1088993b9b82de08f59de5be5a34e42170" not in receipt:
            raise RuntimeError("native receipt is not pinned official 1017")
        self.expected = {line.split()[1]: line.split()[0] for line in receipt.splitlines()
                         if len(line.split()) == 2 and len(line.split()[0]) == 64}
        self.headers: dict[str, tuple[int, dict[str, dict[str, object]]]] = {}
        self.records: list[dict[str, object]] = []

    def raw(self, name: str) -> tuple[torch.Tensor, str]:
        filename = self.index[name]
        path = self.root/filename
        if filename not in self.headers:
            if filename not in SHARDS or self.expected.get(filename) != SHARDS[filename]:
                raise RuntimeError(f"no native receipt hash for {filename}")
            with path.open("rb") as f:
                digest = hashlib.file_digest(f, "sha256").hexdigest()
            if digest != self.expected[filename]:
                raise RuntimeError(f"native shard digest differs: {filename}")
            with path.open("rb") as f:
                size = struct.unpack("<Q", f.read(8))[0]
                if size > 64 << 20:
                    raise RuntimeError("implausible native header size")
                self.headers[filename] = (8+size, json.loads(f.read(size)))
        start, header = self.headers[filename]
        meta = header[name]
        offsets = meta["data_offsets"]
        if not isinstance(offsets, list) or len(offsets) != 2:
            raise RuntimeError("invalid native offsets")
        lo, hi = int(offsets[0]), int(offsets[1])
        with path.open("rb") as f:
            f.seek(start+lo)
            data = bytearray(f.read(hi-lo))
        if len(data) != hi-lo:
            raise RuntimeError(f"short read: {name}")
        self.records.append({"key": name, "shard": filename, "sha256": hashlib.sha256(data).hexdigest(),
                             "shape": meta["shape"], "dtype": meta["dtype"]})
        storage = torch.bfloat16 if meta["dtype"] == "BF16" else torch.uint8
        return torch.frombuffer(data, dtype=storage).reshape(meta["shape"]), str(meta["dtype"])

    def decode(self, name: str) -> torch.Tensor:
        raw, dtype = self.raw(name+".weight")
        if dtype == "BF16":
            return raw.view(torch.bfloat16)
        scale, sdtype = self.raw(name+".scale")
        if sdtype != "F8_E8M0" or bool((scale == 255).any()):
            raise RuntimeError(f"unsupported native scales: {name} {sdtype}")
        sc = torch.exp2(scale.float()-127)
        if dtype == "I8":
            lut = torch.tensor([0, .5, 1, 1.5, 2, 3, 4, 6, -0., -.5, -1, -1.5, -2, -3, -4, -6])
            code = torch.stack((raw&15, raw>>4), -1).reshape(raw.shape[0], -1).long()
            values = lut[code] * sc.repeat_interleave(32, -1)
        elif dtype == "F8_E4M3":
            values = raw.view(torch.float8_e4m3fn).float() * sc.repeat_interleave(32, 0).repeat_interleave(32, 1)
        else:
            raise RuntimeError(f"unsupported native dtype {dtype}: {name}")
        if not bool(values.isfinite().all()):
            raise RuntimeError(f"nonfinite native decoded values: {name}")
        return values

    def narrowed(self, name: str, dtype: torch.dtype) -> torch.Tensor:
        values = self.decode(name)
        narrowed = values.to(dtype)
        if not torch.equal(values.float(), narrowed.float()):
            raise RuntimeError(f"native values not exactly representable in {dtype}: {name}")
        self.records[-1]["decoded_max_abs"] = float(values.abs().max())
        self.records[-1]["decoded_nonzero_min_abs"] = float(values[values != 0].abs().min())
        self.records[-1]["decoded_exact_dtype"] = str(dtype)
        return narrowed

    def expert_shard(self, stage: int, dtype: torch.dtype, device: torch.device, rank: int = 0
                     ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if rank not in (0, 1):
            raise ValueError("rank must be 0 or 1")
        weights = [torch.empty((128, 1152, 5120), dtype=dtype, device=device),
                   torch.empty((128, 1152, 5120), dtype=dtype, device=device),
                   torch.empty((128, 5120, 1152), dtype=dtype, device=device)]
        for e in range(128):
            for i, proj in enumerate(("w1", "w3", "w2")):
                v = self.narrowed(f"mtp.{stage}.ffn.experts.{e}.{proj}", dtype)
                if i < 2:
                    if tuple(v.shape) != (2304, 5120):
                        raise RuntimeError(f"unexpected native {proj} dimensions {v.shape}")
                    v = v[rank*1152:(rank+1)*1152]
                else:
                    if tuple(v.shape) != (5120, 2304):
                        raise RuntimeError(f"unexpected native down dimensions {v.shape}")
                    v = v[:, rank*1152:(rank+1)*1152]
                weights[i][e].copy_(v)
            if e % 16 == 0:
                print(f"native stage {stage}, rank {rank}, expert {e} decoded exactly", flush=True)
        return weights[0], weights[1], weights[2]
