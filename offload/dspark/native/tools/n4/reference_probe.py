"""Capture pinned CPU reference attention inputs into a private directory."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Callable, cast

import torch


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--verify-root", required=True)
    ap.add_argument("--captures", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--official", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--keys", default="r000008_b0,r000011_b0")
    a = ap.parse_args()
    sys.path.insert(0, a.verify_root)
    from verify.captures import CaptureSet
    from verify.native_ref import NativeRef
    from verify.run_ref import WindowCaches, prepare_window
    torch.set_num_threads(12)
    cs = CaptureSet(a.captures)
    keys = set(a.keys.split(","))
    samples = [s for s in cs.samples if s.key in keys]
    if {s.key for s in samples} != keys:
        raise ValueError("missing requested reference sample")
    ref = NativeRef(a.ckpt, a.official, mode="mirror", max_seq_len=max(s.P for s in samples) + 80)
    caches = cast(Callable[[], Any], WindowCaches)()
    module = cast(Any, ref.M)
    original = module.sparse_attn
    a.out.mkdir(parents=True, exist_ok=False)
    for sample in samples:
        _, _, mx, _ = prepare_window(ref, cs, sample, caches)
        count = 0

        def traced(q: torch.Tensor, kv: torch.Tensor, sink: torch.Tensor, idx: torch.Tensor,
                   scale: float) -> torch.Tensor:
            nonlocal count
            result = cast(torch.Tensor, original(q, kv, sink, idx, scale))
            torch.save(dict(q=q.clone(), kv=kv.clone(), sink=sink.clone(), indices=idx.clone(),
                            scale=scale, out=result.clone()), a.out / f"{sample.key}-s{count}.pt")
            count += 1
            return result

        module.sparse_attn = traced
        ref.draft(sample.P, sample.anchor, mx, keep_full_logits=True)
        module.sparse_attn = original
        print(f"saved {sample.key}: {count} stages", flush=True)


if __name__ == "__main__":
    main()
