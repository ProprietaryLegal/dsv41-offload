"""Validate offline fixture decoding against N1's separately implemented native decoders.

CPU only: CUDA_VISIBLE_DEVICES must be empty. Streams one tensor at a time and
checks all three official MTP stages; no new weights downloaded or modified.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path

import torch

from native_fixture import NativeFixture


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("native", type=Path)
    p.add_argument("n1_formats", type=Path)
    p.add_argument("output", type=Path)
    a = p.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise RuntimeError("CPU validation requires CUDA_VISIBLE_DEVICES empty")
    torch.set_num_threads(16)
    spec = importlib.util.spec_from_file_location("n1_formats_reference", a.n1_formats)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot import N1 formats reference")
    ref = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ref)
    source = NativeFixture(a.native)
    rows: list[dict[str, object]] = []
    names = sorted({k.removesuffix(".weight") for k in source.index if k.startswith("mtp.") and
                    k.endswith(".weight") and k.removesuffix(".weight")+".scale" in source.index})
    for index, name in enumerate(names):
        got = source.decode(name)
        raw, dtype = source.raw(name+".weight")
        scale, _ = source.raw(name+".scale")
        expected = ref.decode_fp4(raw, scale) if dtype=="I8" else ref.decode_fp8_block(raw, scale)
        if not torch.equal(got, expected):
            raise RuntimeError(f"independent native decode mismatch at {name}")
        for target in (torch.bfloat16, torch.float16):
            if not torch.equal(got, got.to(target).float()):
                raise RuntimeError(f"native values change in {target}: {name}")
        # Signed zeros must also survive both independent decoders.
        if not torch.equal(got.signbit(), expected.signbit()):
            raise RuntimeError(f"native signed zero mismatch at {name}")
        rows.append({"key": name, "shape": list(got.shape), "dtype": dtype,
                     "native_bf16_exact": True, "native_fp16_exact": True,
                     "reference_bit_exact": True, "signed_zero_exact": True})
        if index%32==0 or index==len(names)-1:
            a.output.write_text(json.dumps({"n1_formats_sha256": hashlib.sha256(a.n1_formats.read_bytes()).hexdigest(),
                                           "tensors": rows, "complete": index==len(names)-1}, indent=2)+"\n")
            print(index+1,"/",len(names),"native tensors independently decoded",flush=True)
    print("COMPLETE",len(rows),"CPU decoder/precision checks",flush=True)


if __name__ == "__main__":
    main()
