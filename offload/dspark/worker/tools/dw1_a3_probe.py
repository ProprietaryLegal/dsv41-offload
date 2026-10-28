"""DW1 item 4 (F-DWAIT option A3), identity probe: can a fused pair run as one exllamav3 multi-matrix GEMM
(exl3_mgemm, as exllamav3's own fused q/g and k/v projections do) with byte-identical fp32 outputs?

The draft step calls the shared expert's gate and up projections (sh_w1, sh_w3: same input, same shape, same K) as two
LinearEXL3.forward calls (exllamav3's coop kernels). This probe runs both ways on the real stage weights for the draft
body's row counts (5 rows per request, 1-4 requests) and compares the fp32 outputs as bytes, then times them with
CUDA events. The coordinator's rule for DW1: byte-identical or off.
Usage (from offload/dspark, GPU-bc62df2a visible): python -m worker.tools.dw1_a3_probe ROOT OUT.json
"""

from __future__ import annotations

import json
import sys

import torch

from .dw1_stages import build_one_card


def main() -> None:
    import exllamav3_ext as ext

    root, out_path = sys.argv[1], sys.argv[2]
    core, m = build_one_card(root)
    gen = torch.Generator(device=m.dev).manual_seed(5)
    rep: dict = {"device": torch.cuda.get_device_name(0), "pairs": {}}
    for si, st in enumerate(m.stages):
        a, b = st.sh_w1.lin, st.sh_w3.lin
        if not (a.K == b.K and a.mcg == b.mcg and a.mul1 == b.mul1 and a.in_features == b.in_features
                and a.out_features == b.out_features):
            rep["pairs"][f"stage{si}"] = "shapes or codebooks differ"
            continue
        dev = m.dev
        tr = torch.tensor([a.trellis.data_ptr(), b.trellis.data_ptr()], dtype=torch.long, device=dev)
        su = torch.tensor([a.suh.data_ptr(), b.suh.data_ptr()], dtype=torch.long, device=dev)
        sv = torch.tensor([a.svh.data_ptr(), b.svh.data_ptr()], dtype=torch.long, device=dev)
        cells = {}
        for rows in (5, 10, 15, 20):
            x = (torch.randn(rows, a.in_features, generator=gen, device=dev) * 0.5).to(torch.float16).contiguous()
            r1 = st.sh_w1(x)
            r3 = st.sh_w3(x)
            C = torch.empty(2, rows, a.out_features, dtype=torch.float32, device=dev)
            had = torch.empty(2, rows, a.in_features, dtype=torch.float16, device=dev)
            ext.exl3_mgemm(x.view(1, rows, a.in_features), tr, C, su, had, sv, None, None, float(a.K), -1,
                           int(a.mcg), int(a.mul1), -1, -1, 0, 1, None, None)
            torch.cuda.synchronize()
            same1 = bool(torch.equal(C[0].view(torch.int32), r1.view(torch.int32)))
            same3 = bool(torch.equal(C[1].view(torch.int32), r3.view(torch.int32)))
            rel = float(((C[0] - r1).norm() / r1.norm()).item())

            def t(fn, n=200):
                s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                for _ in range(20):
                    fn()
                s.record()
                for _ in range(n):
                    fn()
                e.record()
                torch.cuda.synchronize()
                return s.elapsed_time(e) / n * 1e3

            sep = t(lambda: (st.sh_w1(x), st.sh_w3(x)))
            fused = t(lambda: ext.exl3_mgemm(x.view(1, rows, a.in_features), tr, C, su, had, sv, None, None,
                                             float(a.K), -1, int(a.mcg), int(a.mul1), -1, -1, 0, 1, None, None))
            cells[f"rows{rows}"] = {"w1_identical": same1, "w3_identical": same3, "w1_rel_diff": rel,
                                    "separate_us": round(sep, 2), "mgemm_us": round(fused, 2)}
        rep["pairs"][f"stage{si}"] = cells
        print(si, json.dumps(cells), flush=True)
    rep["identical_everywhere"] = all(c["w1_identical"] and c["w3_identical"]
                                      for v in rep["pairs"].values() if isinstance(v, dict) for c in v.values())
    with open(out_path, "w") as fh:
        json.dump(rep, fh, indent=1)
    print("identical_everywhere", rep["identical_everywhere"], flush=True)


if __name__ == "__main__":
    main()
