"""CPU projection counterfactual using privately captured raw attention.

Only the first raw attention result is injected. All subsequent operations
are the pinned official CPU reference, with the same source window.
"""
from __future__ import annotations

import argparse
import json
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
    ap.add_argument("--raw", type=Path, required=True)
    ap.add_argument("--arm", type=Path, required=True)
    ap.add_argument("--dumps", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--keys", default="r000008_b0,r000011_b0")
    ap.add_argument("--dump-dir", type=Path, help="private V1-format component-control records, never worker receipts")
    a = ap.parse_args()
    sys.path.insert(0, a.verify_root)
    from verify.captures import CaptureSet
    from verify.native_ref import NativeRef
    from verify.run_ref import WindowCaches, prepare_window
    from verify.analyze import relerr
    torch.set_num_threads(12)
    cs = CaptureSet(a.captures)
    keys = set(a.keys.split(","))
    samples = [s for s in cs.samples if s.key in keys]
    if {s.key for s in samples} != keys:
        raise ValueError("missing requested counterfactual sample")
    ref = NativeRef(a.ckpt, a.official, mode="mirror", max_seq_len=max(s.P for s in samples) + 80)
    caches = cast(Callable[[], Any], WindowCaches)()
    module = cast(Any, ref.M)
    original = module.sparse_attn
    rows = []
    if a.dump_dir is not None:
        a.dump_dir.mkdir(parents=True, exist_ok=False)
    for sample in samples:
        _, _, mx, _ = prepare_window(ref, cs, sample, caches)
        call = 0
        raw = torch.load(a.raw / f"{sample.key}-s0.pt", weights_only=False)

        def replaced(q: torch.Tensor, kv: torch.Tensor, sink: torch.Tensor, idx: torch.Tensor,
                     scale: float) -> torch.Tensor:
            nonlocal call
            call += 1
            if call == 1:
                if raw.shape != q.shape:
                    raise ValueError("raw output shape mismatch")
                return cast(torch.Tensor, raw)
            return cast(torch.Tensor, original(q, kv, sink, idx, scale))

        module.sparse_attn = replaced
        cf = ref.draft(sample.P, sample.anchor, mx, keep_full_logits=True)
        module.sparse_attn = original
        baseline = torch.load(a.dumps / f"{sample.key}.pt", weights_only=False)
        actual = torch.load(a.arm / f"{sample.key}.pt", weights_only=False)
        if a.dump_dir is not None:
            positions = list(range(max(0, sample.P - ref.win + 1), sample.P + 1))
            record = dict(cf, n_out=5, main_x_P=mx,
                window=baseline["window"],
                kv_window=torch.stack([st.attn.window_kv_cache[0, [p % ref.win for p in positions]].float()
                                       for st in ref.stages], dim=1),
                meta={"control": "pinned official GPU attention at stage0; official CPU reference elsewhere",
                      "not_worker_serving": True})
            torch.save(record, a.dump_dir / f"{sample.key}.pt")
        rows.append(dict(sample=sample.key,
            s0_injected_vs_reference=relerr(cf["s0"]["attn_out"], baseline["s0"]["attn_out"]),
            s0_arm_vs_reference=relerr(actual["s0"]["attn_out"], baseline["s0"]["attn_out"]),
            s0_arm_vs_injected=relerr(actual["s0"]["attn_out"], cf["s0"]["attn_out"])))
    a.out.write_text(json.dumps(rows, indent=2) + "\n")
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
